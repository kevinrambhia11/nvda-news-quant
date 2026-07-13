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
from sklearn.linear_model import LinearRegression

import config
from features.vol import (VOL_FEATURES, VOL_TECH_FEATURES, build_vol_dataset,
                          garman_klass_vol)
from trade.calendar import next_trading_day, session_close_hour_et

log = logging.getLogger(__name__)

HAR_FEATURES = ["gk_1", "gk_5", "gk_22"]
EWMA_ALPHA = 0.06  # RiskMetrics lambda = 0.94


def _gbm() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_depth=3,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=False, random_state=42)


# (name, feature list, model factory) - the pipeline SELECTS the winner by
# out-of-sample QLIKE per horizon rather than assuming the fancier model wins
# (in vol forecasting the parsimonious HAR frequently beats ML).
CANDIDATES = [
    ("GBM news+price", VOL_FEATURES, _gbm),
    ("GBM price-only", VOL_TECH_FEATURES, _gbm),
    ("HAR (classic)", HAR_FEATURES, LinearRegression),
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
                     gdelt: pd.DataFrame, horizon: int):
    """Walk-forward OOS comparison of all candidate forecasters."""
    ds = build_vol_dataset(px, bench, gdelt, horizon=horizon)
    data = ds.dropna(subset=["gk_22", "tone_1d", "y_vol"]).copy()
    y = data["y_vol"]
    embargo = horizon - 1

    fitted = {name: _walk_forward(data[feats], y, factory, embargo)
              for name, feats, factory in CANDIDATES}
    oos = next(iter(fitted.values())).notna()

    # Analytic baselines (no fitting; both use info through d-1 only).
    # EWMA smooths the SAME GK variance series the target is built from, so
    # the comparison is not handicapped by a measurement mismatch.
    ewm_var = np.exp(2 * data["gk_1"]).ewm(alpha=EWMA_ALPHA, min_periods=30).mean()
    baselines = {
        "Random walk": data["gk_1"],
        "EWMA (GK var)": 0.5 * np.log(ewm_var),
    }

    results = {}
    for name, pred in {**baselines, **fitted}.items():
        results[name] = _metrics(y[oos], pred[oos], data.loc[oos, "realized_var"])
    return data, results, oos, fitted


def train_and_report() -> str:
    from data.news import load_gdelt_daily
    from data.prices import load_prices
    px, bench = load_prices()
    gdelt = load_gdelt_daily()

    lines = ["=" * 66,
             "  NVDA volatility model - walk-forward out-of-sample comparison",
             "  (targets: log Garman-Klass vol; lower RMSE/QLIKE and higher",
             "   R2 are better; QLIKE is the standard vol-forecast loss)",
             "=" * 66]
    models = {}
    trained_through = None
    for horizon in config.VOL_HORIZONS:
        data, results, oos, fitted = evaluate_horizon(px, bench, gdelt, horizon)
        lines.append(f"\n  Horizon {horizon} day(s)  |  {int(oos.sum())} OOS days "
                     f"({data.index[oos][0].date()} -> {data.index[oos][-1].date()})")
        lines.append(f"  {'model':<22}{'RMSE(log)':>11}{'QLIKE':>9}{'R2':>8}")
        for name, m in results.items():
            lines.append(f"  {name:<22}{m['rmse']:>11.4f}{m['qlike']:>9.4f}"
                         f"{m['r2']:>8.3f}")
        best_name, best_feats, best_factory = min(
            CANDIDATES, key=lambda c: results[c[0]]["qlike"])
        lines.append(f"  -> selected for the desk: {best_name} "
                     f"(best OOS QLIKE among fitted candidates)")
        if horizon == 1:
            oos_df = data.loc[oos, ["y_vol", "realized_var"]].copy()
            oos_df["pred_selected"] = fitted[best_name][oos]
            oos_df.to_csv(config.VOL_OOS_PATH)
        models[horizon] = {"name": best_name, "features": best_feats,
                           "model": best_factory().fit(data[best_feats],
                                                       data["y_vol"])}
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

    out = {"generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
           "ticker": config.TICKER, "entry_day": str(next_day.date()),
           "model_trained_through": bundle.get("trained_through"),
           "var_notional": config.VAR_NOTIONAL, "horizons": {}}
    for horizon, spec in bundle["models"].items():
        ds = build_vol_dataset(px_ext, bench, gdelt, horizon=horizon)
        row = ds.iloc[[-1]][spec["features"]]
        daily_vol = float(np.exp(spec["model"].predict(row)[0]))
        ann_vol = daily_vol * np.sqrt(252)
        entry = {"model": spec["name"],
                 "daily_vol": round(daily_vol, 5),
                 "annualized_vol": round(ann_vol, 4)}
        if horizon == 1:
            # GK measures intraday range only; scale by the historical
            # total/intraday variance ratio (last ~2y) so sizing and VaR
            # include overnight gap risk.
            gap = np.log(px["Open"] / px["Close"].shift(1))
            gk_var = garman_klass_vol(px) ** 2
            win_gk, win_gap = gk_var.iloc[-504:], gap.iloc[-504:]
            ratio = float((win_gk + win_gap ** 2).mean() / win_gk.mean())
            total_daily = daily_vol * np.sqrt(ratio)
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
    lines += [
        f"  1-day TOTAL vol (incl. overnight gap): "
        f"{h1['total_daily_vol']:.3%}/day ({h1['total_annualized_vol']:.1%} ann)",
        "-" * 62,
        f"  Target-vol sizing ({config.VOL_TARGET_ANN:.0%} ann target): "
        f"{h1['target_vol_weight']:.1%} of full position",
        f"  1-day VaR on ${f['var_notional']:,}: "
        f"95% ${h1['var_95']:,.0f} | 99% ${h1['var_99']:,.0f}",
        "  (parametric/normal - real tails are fatter; treat as a floor)",
        "=" * 62,
    ]
    return "\n".join(lines)
