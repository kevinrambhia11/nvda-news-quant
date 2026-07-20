"""Desk volatility model: forecast NVDA realized volatility from news + price.

Under an efficient market the first moment (direction) is unpredictable by
assumption, but the second moment is not: volatility clusters, and news
arrival drives it. This module forecasts log Garman-Klass volatility at the
horizons in config.VOL_HORIZONS, benchmarked against the standard desk
baselines - random walk, RiskMetrics EWMA, HAR - plus a price-only ablation
so the marginal value of the news features is measured, not assumed.

Forecast output feeds two desk uses: target-volatility position sizing and
a parametric 1-day VaR (normal approximation - understates tail risk; treat
the 99% line as a floor, not a ceiling).
"""
from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

import config
from features.vol import (EVENT_FEATURES, VOL_FEATURES, VOL_TECH_FEATURES,
                          build_vol_dataset, garman_klass_vol)
from trade.calendar import next_trading_day, session_close_hour_et

log = logging.getLogger(__name__)

HAR_FEATURES = ["gk_1", "gk_5", "gk_22"]
EWMA_ALPHA = 0.06  # RiskMetrics lambda = 0.94

# News-vector magnitude features promoted from the magnitude head (its
# holdout verdict: news predicts move SIZE). Candidates carrying them are
# judged with NESTED scorers on the clean window - see model/train.py.
NEWSVEC_FEATURES = ["nn_mag", "nn_conflict", "n2_nvda_mag", "n2_macro_mag",
                    "n2_nvda_count_z"]


def _linreg_imputed():
    # LinearRegression cannot digest NaNs (news-vector coverage starts in
    # 2018; conflict needs 2+ same-day articles); median-impute, keeping
    # all-NaN columns so early refit windows keep a stable feature count.
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        LinearRegression())


def _join_newsvec(df: pd.DataFrame, nested: bool):
    """Join whichever news-vector features exist onto df (entry-day indexed
    on both sides, so the d-1 timing discipline carries over unchanged).
    Returns (frame, joined column names)."""
    from model.news2vec import load_news2_features
    from model.newsnet import load_newsnet_features
    out, cols = df.copy(), []
    for loader in (load_news2_features, load_newsnet_features):
        try:
            f = loader(nested=nested)
        except Exception:
            f = None
        if f is None:
            continue
        for c in NEWSVEC_FEATURES:
            if c in f.columns:
                out[c] = f[c].reindex(out.index).to_numpy()
                cols.append(c)
    return out, cols


def _load_earnings():
    from data.earnings import load_earnings_dates
    try:
        return load_earnings_dates()
    except Exception as exc:
        log.warning("Earnings calendar unavailable (%s); event features "
                    "fall back to neutral values", exc)
        return None


def _gbm() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_depth=3,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=False, random_state=42)


# (name, feature list, model factory) - the pipeline SELECTS the winner by
# out-of-sample QLIKE per horizon rather than assuming the fancier model wins
# (in vol forecasting the parsimonious HAR frequently beats ML). Earnings
# features enter via two candidates so their marginal value is measured.
CANDIDATES = [
    ("GBM news+price", VOL_FEATURES, _gbm),
    ("GBM +events", VOL_FEATURES + EVENT_FEATURES, _gbm),
    ("HAR (classic)", HAR_FEATURES, LinearRegression),
    ("HAR +events", HAR_FEATURES + EVENT_FEATURES, LinearRegression),
]


def _walk_forward(X: pd.DataFrame, y: pd.Series, make_model,
                  embargo: int) -> pd.Series:
    """Expanding-window OOS predictions. `embargo` drops the last rows of
    each training window whose multi-day targets are not yet fully realized
    at (pre-open) refit time: horizon-1 rows."""
    preds = pd.Series(index=X.index, dtype=float)
    for start in range(config.MIN_TRAIN_DAYS, len(X), config.RETRAIN_EVERY):
        stop = min(start + config.RETRAIN_EVERY, len(X))
        cut = max(1, start - embargo)
        model = make_model()
        model.fit(X.iloc[:cut], y.iloc[:cut])
        preds.iloc[start:stop] = model.predict(X.iloc[start:stop])
    return preds


def _metrics(y_log: pd.Series, pred_log: pd.Series,
             realized_var: pd.Series) -> dict:
    err = y_log - pred_log
    rmse = float(np.sqrt((err ** 2).mean()))
    r2 = float(1 - (err ** 2).sum() / ((y_log - y_log.mean()) ** 2).sum())
    ratio = realized_var / np.exp(2 * pred_log)
    qlike = float((ratio - np.log(ratio) - 1).mean())
    return {"rmse": rmse, "qlike": qlike, "r2": r2}


def evaluate_horizon(px: pd.DataFrame, bench: pd.DataFrame,
                     gdelt: pd.DataFrame, horizon: int, earn_dates=None):
    """Walk-forward OOS predictions for every candidate and baseline.

    Returns (data_eval, data_live, preds, oos, candidates, meta). The
    news-vector candidates are EVALUATED on nested-scorer features (so the
    clean judging window is honest) but the live winner is refit on
    production features (data_live); meta is None when no news-vector
    candidates run, and judging then uses the full selection window."""
    from model.train import nested_meta
    ds = build_vol_dataset(px, bench, gdelt, horizon=horizon,
                           earn_dates=earn_dates)
    data = ds.dropna(subset=["gk_22", "tone_1d", "y_vol"]).copy()

    cands = list(CANDIDATES)
    data_eval, data_live, meta = data, data, nested_meta()
    if meta is not None:
        # This dataset's calendar differs from the direction model's, so
        # validate the judging geometry here: the boundary must leave a
        # usable clean window before the (pinned or fraction) split.
        oos_idx = data.index[config.MIN_TRAIN_DAYS:]
        split = None
        if len(oos_idx):
            if oos_idx[0] < meta["holdout_start"] <= oos_idx[-1]:
                split = meta["holdout_start"]
            else:
                split = oos_idx[int(len(oos_idx)
                                    * (1 - config.HOLDOUT_FRACTION))]
            pre = oos_idx[(oos_idx >= meta["inner"]) & (oos_idx < split)]
            if meta["features_through"] is not None:
                pre = pre[pre <= meta["features_through"]]
            if len(pre) < 60:
                log.warning("Nested judging window unusable for horizon %d "
                            "(%d rows) - newsvec candidates skipped",
                            horizon, len(pre))
                meta = None
        else:
            meta = None
    if meta is not None:
        eval_join, nv = _join_newsvec(data, nested=True)
        live_join, nv_live = _join_newsvec(data, nested=False)
        if len(nv) >= 3 and sorted(nv) == sorted(nv_live):
            data_eval, data_live = eval_join, live_join
            cands += [("GBM +ev+newsvec",
                       VOL_FEATURES + EVENT_FEATURES + nv, _gbm),
                      ("HAR +ev+newsvec",
                       HAR_FEATURES + EVENT_FEATURES + nv, _linreg_imputed)]
        else:
            meta = None
    y = data["y_vol"]
    embargo = horizon - 1

    fitted = {name: _walk_forward(data_eval[feats], y, factory, embargo)
              for name, feats, factory in cands}
    oos = next(iter(fitted.values())).notna()

    # Analytic baselines (no fitting; both use info through d-1 only).
    # EWMA smooths the SAME GK variance series the target is built from, so
    # the comparison is not handicapped by a measurement mismatch.
    ewm_var = np.exp(2 * data["gk_1"]).ewm(alpha=EWMA_ALPHA, min_periods=30).mean()
    preds = {"Random walk": data["gk_1"],
             "EWMA (GK var)": 0.5 * np.log(ewm_var), **fitted}
    return data_eval, data_live, preds, oos, cands, meta


def train_and_report() -> str:
    from data.news import load_gdelt_daily
    from data.prices import load_prices
    px, bench = load_prices()
    gdelt = load_gdelt_daily()
    earn_dates = _load_earnings()

    lines = ["=" * 66,
             "  NVDA volatility model - walk-forward comparison",
             "  (targets: log Garman-Klass vol; lower RMSE/QLIKE and higher",
             "   R2 are better; candidates judged on a SELECTION window,",
             "   the winner reported on the untouched HOLDOUT tail)",
             "=" * 66]
    models = {}
    trained_through = None
    for horizon in config.VOL_HORIZONS:
        data, data_live, preds, oos, cands, vmeta = evaluate_horizon(
            px, bench, gdelt, horizon, earn_dates)
        y, rv = data["y_vol"], data["realized_var"]
        oos_idx = data.index[oos]
        split = None
        if vmeta is not None and \
                oos_idx[0] < vmeta["holdout_start"] <= oos_idx[-1]:
            split = vmeta["holdout_start"]
        if split is None:
            split = oos_idx[int(len(oos_idx) * (1 - config.HOLDOUT_FRACTION))]
        judge_start = max(vmeta["inner"], oos_idx[0]) if vmeta is not None \
            else oos_idx[0]
        sel = oos & (data.index >= judge_start) & (data.index < split)
        if vmeta is not None and vmeta["features_through"] is not None:
            sel &= data.index <= vmeta["features_through"]
        hold = oos & (data.index >= split)

        lines.append(f"\n  Horizon {horizon} day(s)  |  judged "
                     f"{judge_start.date()} -> {split.date()}  |  holdout -> "
                     f"{oos_idx[-1].date()}")
        if vmeta is not None:
            lines.append("  (newsvec candidates use NESTED scorers; all "
                         "models judged on the clean window. Newsvec was")
            lines.append("   promoted on the magnitude head's holdout, "
                         "which overlaps this holdout - treat a newsvec")
            lines.append("   holdout QLIKE as second-use; the live paper "
                         "trail is the fresh test.)")
        lines.append(f"  {'model':<22}{'RMSE(log)':>11}{'QLIKE':>9}{'R2':>8}")
        sel_q = {}
        for name, pred in preds.items():
            m = _metrics(y[sel], pred[sel], rv[sel])
            sel_q[name] = m["qlike"]
            lines.append(f"  {name:<22}{m['rmse']:>11.4f}{m['qlike']:>9.4f}"
                         f"{m['r2']:>8.3f}")
        best_name, best_feats, best_factory = min(
            cands, key=lambda c: sel_q[c[0]])
        hm = _metrics(y[hold], preds[best_name][hold], rv[hold])
        lines.append(f"  -> selected: {best_name} (best selection QLIKE "
                     f"among fitted candidates)")
        lines.append(f"     untouched holdout: RMSE {hm['rmse']:.4f} | "
                     f"QLIKE {hm['qlike']:.4f} | R2 {hm['r2']:.3f}")
        if horizon == 1:
            oos_df = data.loc[oos, ["y_vol", "realized_var"]].copy()
            oos_df["pred_selected"] = preds[best_name][oos]
            oos_df.to_csv(config.VOL_OOS_PATH)
        # live refit on PRODUCTION news features (live days postdate every
        # scorer's training data, so contamination cannot arise there)
        models[horizon] = {"name": best_name, "features": best_feats,
                           "model": best_factory().fit(data_live[best_feats],
                                                       data_live["y_vol"])}
        # max across horizons: longer horizons drop more unrealized-target
        # rows, so their dataset ends earlier (ISO strings sort correctly)
        trained_through = max(trained_through or "", str(data.index.max().date()))
    lines.append("=" * 66)

    joblib.dump({"models": models,
                 "trained_through": trained_through}, config.VOL_MODEL_PATH)
    log.info("Volatility models saved -> %s", config.VOL_MODEL_PATH)
    report = "\n".join(lines)
    config.VOL_REPORT_PATH.write_text(report, encoding="utf-8")
    return report


def forecast() -> dict:
    """Today's volatility forecast with target-vol sizing and parametric VaR.
    Uses the shared caches; needs no network when they are current."""
    if not config.VOL_MODEL_PATH.exists():
        raise RuntimeError("No volatility model - run `python main.py vol-train` first")
    bundle = joblib.load(config.VOL_MODEL_PATH)

    from data.news import load_gdelt_daily
    from data.prices import load_prices
    px, bench = load_prices()
    gdelt = load_gdelt_daily()
    today_utc = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    gdelt = gdelt[gdelt.index < today_utc]  # complete UTC buckets only
    now_et = pd.Timestamp.now(tz="America/New_York")
    today_et = now_et.tz_localize(None).normalize()
    if now_et.hour < session_close_hour_et(today_et):
        # drop an in-progress session bar (training parity; early-close aware)
        if len(px) and px.index.max() == today_et:
            px = px.loc[px.index < today_et]
            bench = bench.loc[bench.index < today_et]

    next_day = next_trading_day(px.index.max())
    px_ext = px.reindex(px.index.append(pd.DatetimeIndex([next_day])))
    earn_dates = _load_earnings()

    out = {"generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
           "ticker": config.TICKER, "entry_day": str(next_day.date()),
           "model_trained_through": bundle.get("trained_through"),
           "var_notional": config.VAR_NOTIONAL, "horizons": {}}
    for horizon, spec in bundle["models"].items():
        ds = build_vol_dataset(px_ext, bench, gdelt, horizon=horizon,
                               earn_dates=earn_dates)
        nv_used = [c for c in spec["features"] if c in NEWSVEC_FEATURES]
        if nv_used:
            ds, _ = _join_newsvec(ds, nested=False)
            absent = [c for c in nv_used if c not in ds.columns]
            if absent:
                # feature CSVs missing entirely (fresh clone/deploy): keep
                # the forecast alive on the model's missing-value paths
                log.warning("news-vector feature files unavailable (%s) - "
                            "filling NaN", absent)
                for c in absent:
                    ds[c] = np.nan
            if ds.iloc[-1][nv_used].isna().all():
                log.warning("news-vector features missing for entry day %s "
                            "(article parquet ends before it) - the vol "
                            "model falls back to their missing-value paths",
                            next_day.date())
        row = ds.iloc[[-1]][spec["features"]]
        daily_vol = float(np.exp(spec["model"].predict(row)[0]))
        ann_vol = daily_vol * np.sqrt(252)
        entry = {"model": spec["name"],
                 "daily_vol": round(daily_vol, 5),
                 "annualized_vol": round(ann_vol, 4)}
        if horizon == 1:
            # GK measures intraday range only; scale by the historical
            # total/intraday variance ratio (last ~2y) so sizing and VaR
            # include overnight gap risk. On an earnings entry day the
            # print-night gap dwarfs the ordinary overnight: add the mean
            # squared gap over all PAST prints instead.
            gap = np.log(px["Open"] / px["Close"].shift(1))
            gk_var = garman_klass_vol(px) ** 2
            win_gk, win_gap = gk_var.iloc[-504:], gap.iloc[-504:]
            ratio = float((win_gk + win_gap ** 2).mean() / win_gk.mean())
            print_tonight = bool(float(row["earn_window"].iloc[0]) == 1.0) \
                if "earn_window" in row.columns else False
            if print_tonight and earn_dates is not None and len(earn_dates):
                from features.build import print_gap_sq
                gap_var = float(print_gap_sq(px, earn_dates).mean())
                total_daily = float(np.sqrt(daily_vol ** 2 + gap_var))
            else:
                total_daily = daily_vol * np.sqrt(ratio)
            entry["earnings_print_tonight"] = print_tonight
            total_ann = total_daily * np.sqrt(252)
            entry["total_daily_vol"] = round(total_daily, 5)
            entry["total_annualized_vol"] = round(total_ann, 4)
            entry["var_95"] = round(1.645 * total_daily * config.VAR_NOTIONAL, 0)
            entry["var_99"] = round(2.326 * total_daily * config.VAR_NOTIONAL, 0)
            entry["target_vol_weight"] = round(
                min(1.0, config.VOL_TARGET_ANN / total_ann), 3)
        out["horizons"][str(horizon)] = entry

    path = config.ARTIFACTS / f"vol_forecast_{next_day.strftime('%Y%m%d')}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Vol forecast saved -> %s", path)
    return out


def format_forecast(f: dict) -> str:
    h1 = f["horizons"]["1"]
    lines = ["=" * 62,
             f"  {f['ticker']} volatility forecast for {f['entry_day']}",
             f"  generated {f['generated_at']} | model through "
             f"{f['model_trained_through']}",
             "=" * 62]
    for h, e in f["horizons"].items():
        lines.append(f"  {h}-day intraday vol: {e['daily_vol']:.3%}/day  "
                     f"({e['annualized_vol']:.1%} annualized)  [{e['model']}]")
    print_note = ("  !! EARNINGS PRINT TONIGHT - gap-risk sizing in effect"
                  if h1.get("earnings_print_tonight") else "")
    lines += [
        f"  1-day TOTAL vol (incl. overnight gap): "
        f"{h1['total_daily_vol']:.3%}/day ({h1['total_annualized_vol']:.1%} ann)"
        f"{print_note}",
        "-" * 62,
        f"  Target-vol sizing ({config.VOL_TARGET_ANN:.0%} ann target): "
        f"{h1['target_vol_weight']:.1%} of full position",
        f"  1-day VaR on ${f['var_notional']:,}: "
        f"95% ${h1['var_95']:,.0f} | 99% ${h1['var_99']:,.0f}",
        "  (parametric/normal - real tails are fatter; treat as a floor)",
        "=" * 62,
    ]
    return "\n".join(lines)
