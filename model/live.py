"""The daily learning loop: grade yesterday, retrain the brain, refit heads.

This is the layer Kevin specified: every day the system reads all the news,
puts it through the neural brain (NewsNet over the embedding store), lets
the brain assign each article a learned significance weight for today's
session, builds the feature matrix, and applies the ML heads - AFTER first
checking whether yesterday's prediction was right and retraining on
everything known through yesterday.

Separation of concerns (important):
  * LIVE artifacts (this module) train on ALL history through yesterday -
    the freshest possible brain for today. Their files carry a _live
    suffix and overwrite only the deployed bundles the signal reads.
  * EVALUATION artifacts (nested/production nets, tournament reports) keep
    their pinned boundaries and are rebuilt only by the Saturday retrain.
    The two never share files, so daily learning cannot quietly corrupt
    the honesty machinery - and the tournament + Track Record tab remain
    the judges of whether daily learning actually helps.

Cadence (Kevin's spec, 2026-07-22): the brain is EDUCATED weekly and only
APPLIES its learning daily.
  Daily  (main.py `learn`, between news-topup and signal):
    1. score_yesterday() - grade newly realized forecasts into
       learning_log.csv ("was I right?").
    2. apply_brain()     - pure inference: the frozen brain reads all of
       today's news, assigns each article its learned significance
       weight (today_news_weights.json), and the frozen impact scorers
       rescore; live feature files refresh. No weights change.
  Weekly (main.py `train-brain`, inside the Saturday retrain):
    3. train_brain()     - re-educate: ridge scorers + NewsNet retrain on
       all labeled history, then apply.
    4. refit_heads()     - refit the deployed direction / magnitude / vol
       bundles (same tournament-chosen architectures) on the fresh
       live features.
"""
from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd

import config
from model.news2vec import (ART_PATH, EMB_PATH, _entry_days, _entry_index,
                            build_daily_features, learn_impact)

log = logging.getLogger(__name__)

LIVE_IMPACT_PATH = config.ARTIFACTS / "news2_impact_live.joblib"
LIVE_N2_FEATURES = config.CACHE / "news2_features_live.csv"
LIVE_NET_PATH = config.ARTIFACTS / "newsnet_live.pt"
LIVE_NN_FEATURES = config.CACHE / "newsnet_features_live.csv"
LEARNING_LOG = config.ARTIFACTS / "learning_log.csv"
TODAY_WEIGHTS = config.ARTIFACTS / "today_news_weights.json"


# ---------------------------------------------------------------------------
# 1. Grade yesterday
# ---------------------------------------------------------------------------

def score_yesterday() -> str:
    """Score every archived forecast whose outcome has realized since the
    last scoring run; append to learning_log.csv and return a printable
    grade for the most recent scored day."""
    from data.prices import load_prices
    px, _ = load_prices()
    opens = px["Open"]
    opens = opens[~opens.index.duplicated(keep="last")]

    scored_days = set()
    if LEARNING_LOG.exists():
        prev = pd.read_csv(LEARNING_LOG, parse_dates=["entry_day"])
        scored_days = set(prev["entry_day"].dt.normalize())

    rows = []
    for f in sorted(config.ARTIFACTS.glob("signal_*.json")):
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
            d = pd.Timestamp(s["entry_day"]).normalize()
        except Exception:
            continue
        if d in scored_days or d not in opens.index:
            continue
        pos = opens.index.get_loc(d) + 1
        if pos >= len(opens):
            continue  # outcome not realized yet
        realized = float(opens.iloc[pos] / opens.loc[d] - 1)
        row = {"entry_day": d,
               "prob_up": s.get("model_prob_up"),
               "prob_big": s.get("prob_big_move"),
               "big_thr": s.get("big_move_threshold"),
               "realized_ret": round(realized, 5)}
        if row["prob_up"] is not None:
            row["dir_correct"] = int((row["prob_up"] > 0.5) == (realized > 0))
        if row["prob_big"] is not None and row["big_thr"] is not None:
            row["big_correct"] = int((row["prob_big"] > 0.5)
                                     == (abs(realized) > row["big_thr"]))
        vf = config.ARTIFACTS / f"vol_forecast_{d.strftime('%Y%m%d')}.json"
        if vf.exists():
            try:
                h1 = json.loads(vf.read_text(encoding="utf-8"))["horizons"]["1"]
                fc = h1.get("total_daily_vol", h1.get("daily_vol"))
                row["fc_vol"] = fc
                row["vol_z"] = round(abs(realized) / fc, 3) if fc else None
            except Exception:
                pass
        rows.append(row)

    if rows:
        new = pd.DataFrame(rows).sort_values("entry_day")
        if LEARNING_LOG.exists():
            new = pd.concat([pd.read_csv(LEARNING_LOG,
                                         parse_dates=["entry_day"]), new],
                            ignore_index=True)
        tmp = LEARNING_LOG.with_suffix(".csv.tmp")
        new.to_csv(tmp, index=False)
        import os
        os.replace(tmp, LEARNING_LOG)

    if not rows:
        return "Yesterday's grade: no newly realized outcomes to score."
    last = rows[-1]
    bits = [f"Yesterday's grade ({last['entry_day'].date()}): "
            f"realized {last['realized_ret']:+.2%}"]
    if "dir_correct" in last:
        bits.append(f"direction {'RIGHT' if last['dir_correct'] else 'WRONG'}"
                    f" (P(up) was {last['prob_up']:.0%})")
    if "big_correct" in last:
        bits.append(f"big-move call {'RIGHT' if last['big_correct'] else 'WRONG'}"
                    f" (P(big) was {last['prob_big']:.0%})")
    if last.get("vol_z") is not None:
        bits.append(f"|move|/forecast vol {last['vol_z']:.2f}x")
    return " | ".join(bits) + f"  [{len(rows)} day(s) scored]"


# ---------------------------------------------------------------------------
# 2. Retrain the brain on everything through yesterday
# ---------------------------------------------------------------------------

def _load_corpus(torch):
    """(art, emb, days) aligned against the price index extended by the
    upcoming session - shared by training and daily inference. Drops an
    in-progress price bar (training parity with signal.py) and articles in
    unknown categories (a NaN category maps to a garbage embedding index).
    """
    from data.prices import load_prices
    from model.newsnet import CATS, _aligned, _day_tensors
    from trade.calendar import session_close_hour_et
    art = pd.read_parquet(ART_PATH)
    emb = np.load(EMB_PATH).astype(np.float32)
    keep = art["category"].isin(CATS).to_numpy()
    if not keep.all():
        # filter art and emb by the SAME positions - anything else breaks
        # the row-i == embedding-i invariant
        pos = np.flatnonzero(keep)
        pos = pos[pos < len(emb)]
        art = art.iloc[pos].reset_index(drop=True)
        emb = emb[pos]
        log.warning("Dropped %d articles in unknown categories",
                    int((~keep).sum()))
    px, _ = load_prices()
    now_et = pd.Timestamp.now(tz="America/New_York")
    today_et = now_et.tz_localize(None).normalize()
    if now_et.hour < session_close_hour_et(today_et) and len(px) \
            and px.index.max() == today_et:
        px = px.loc[px.index < today_et]
    art, emb = _aligned(art, emb, _entry_index(px.index))
    return art, emb, _day_tensors(torch, art, emb), px


def train_brain(epochs: int = 40) -> None:
    """WEEKLY education: ridge impact scorers + NewsNet retrain on all
    labeled entry days, then the fresh brain is applied. Daily runs only
    call apply_brain()."""
    from model.newsnet import _build_model, _torch, RET_SCALE

    # ridge scorers: tomorrow is strictly after every labeled day
    tomorrow = (pd.Timestamp.now().normalize() + pd.Timedelta(days=2))
    learn_impact(tomorrow, out_path=LIVE_IMPACT_PATH)

    torch = _torch()
    art, emb, days, px = _load_corpus(torch)
    fwd = (px["Open"].shift(-1) / px["Open"] - 1)
    labeled = [(d, e, s, c, fwd.get(d)) for d, e, s, c in days
               if pd.notna(fwd.get(d))]
    n_val = max(50, int(len(labeled) * 0.1))
    if len(labeled) < n_val + 100:
        raise RuntimeError(f"only {len(labeled)} labeled days - refusing "
                           "to train the brain on degenerate data")
    train_days, val_days = labeled[:-n_val], labeled[-n_val:]
    log.info("Live brain: %d train days, %d val days (through yesterday)",
             len(train_days), len(val_days))

    model = _build_model(torch)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    huber = torch.nn.SmoothL1Loss()
    best, best_state, patience = np.inf, None, 0
    rng = np.random.default_rng(42)
    for epoch in range(epochs):
        model.train()
        for i in rng.permutation(len(train_days)):
            d, e, s, c, y = train_days[i]
            yt = torch.tensor(float(y) * RET_SCALE)
            dir_p, mag_p, _, _ = model(e, s, c)
            loss = huber(dir_p, yt) + huber(mag_p, yt.abs())
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(np.mean([
                (huber(model(e, s, c)[0], torch.tensor(float(y) * RET_SCALE))
                 + huber(model(e, s, c)[1],
                         torch.tensor(abs(float(y)) * RET_SCALE))).item()
                for d, e, s, c, y in val_days]))
        if vloss < best - 1e-4:
            best, best_state, patience = vloss, model.state_dict(), 0
        else:
            patience += 1
        if patience >= 6:
            break
    model.load_state_dict(best_state)
    torch.save({"state": best_state,
                "trained_through": str(px.index.max().date())},
               LIVE_NET_PATH)
    log.info("Brain re-educated on %d labeled days (val %.4f)",
             len(train_days), best)
    apply_brain()


def apply_brain() -> None:
    """DAILY inference: the already-trained brain and frozen scorers read
    the news as it stands now; live feature files and the upcoming
    session's per-article weights refresh. No weights are updated here."""
    from model.newsnet import _build_model, _src_bucket, _torch, CATS, RET_SCALE
    from data.news import atomic_to_csv

    if not (LIVE_NET_PATH.exists() and LIVE_IMPACT_PATH.exists()):
        raise RuntimeError("live brain not trained yet - run "
                           "`python main.py train-brain` once")

    build_daily_features(impact_path=LIVE_IMPACT_PATH,
                         out_path=LIVE_N2_FEATURES)

    torch = _torch()
    art, emb, days, _ = _load_corpus(torch)
    model = _build_model(torch)
    model.load_state_dict(torch.load(LIVE_NET_PATH)["state"])
    model.eval()

    rows = []
    with torch.no_grad():
        for day, e, s, c in days:
            dir_p, mag_p, w, art_dir = model(e, s, c)
            wn = w.numpy()
            rows.append({"date": day,
                         "nn_dir": float(dir_p) / RET_SCALE,
                         "nn_mag": float(mag_p) / RET_SCALE,
                         "nn_conflict": float(art_dir.std()) / RET_SCALE,
                         "nn_attn_ent": float(-(wn * np.log(wn + 1e-9)).sum()),
                         "nn_top_share": float(wn.max())})
    feats = pd.DataFrame(rows).set_index("date").sort_index()
    feats.index.name = "date"
    atomic_to_csv(feats, LIVE_NN_FEATURES)

    # today's article weights: rebuilt explicitly from the upcoming day's
    # articles in frame order (never via _day_tensors, whose >400-article
    # subsampling would silently misalign weight -> headline)
    upcoming = art["entry"].max()
    sub = art[art["entry"] == upcoming]
    today = []
    if len(sub):
        pos = sub.index.to_numpy()
        e_t = torch.tensor(emb[pos], dtype=torch.float32)
        s_t = torch.tensor(sub["source"].fillna("?").map(_src_bucket)
                           .to_numpy(), dtype=torch.long)
        c_t = torch.tensor(sub["category"]
                           .map({c: i for i, c in enumerate(CATS)})
                           .to_numpy(), dtype=torch.long)
        with torch.no_grad():
            _, _, w, art_dir = model(e_t, s_t, c_t)
        wn, an = w.numpy(), art_dir.numpy() / RET_SCALE
        for j in np.argsort(-wn)[:10]:
            r = sub.iloc[int(j)]
            today.append({"headline": str(r["slug"])[:120],
                          "source": str(r["source"]),
                          "category": str(r["category"]),
                          "weight": round(float(wn[j]), 4),
                          "impact": round(float(an[j]), 5)})
    import os
    tmp = TODAY_WEIGHTS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(
        {"entry_day": str(pd.Timestamp(upcoming).date()),
         "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
         "articles": today}, indent=2), encoding="utf-8")
    os.replace(tmp, TODAY_WEIGHTS)
    log.info("Brain applied: %d days of features; %d weighted articles "
             "for %s", len(feats), len(today),
             pd.Timestamp(upcoming).date())


def load_live_features() -> pd.DataFrame | None:
    """Live news features (ridge + brain) joined, or None when either file
    is missing - callers fall back to the weekly production features."""
    if not (LIVE_N2_FEATURES.exists() and LIVE_NN_FEATURES.exists()):
        return None
    n2 = pd.read_csv(LIVE_N2_FEATURES, index_col="date", parse_dates=["date"])
    nn = pd.read_csv(LIVE_NN_FEATURES, index_col="date", parse_dates=["date"])
    return n2.join(nn, how="outer")


# ---------------------------------------------------------------------------
# 3. Refit the deployed heads on data through yesterday
# ---------------------------------------------------------------------------

def _live_dataset() -> pd.DataFrame:
    """Full feature matrix with LIVE news features replacing the weekly
    production ones - what the freshly-retrained heads train and predict on.
    """
    from data.news import load_aux_gdelt, load_gdelt_daily
    from data.prices import load_prices
    from features.build import build_dataset
    from model.news2vec import NEWS2_FEATURES
    from model.newsnet import NEWSNET_FEATURES

    px, bench = load_prices()
    gdelt = load_gdelt_daily()
    aux = load_aux_gdelt()
    earn = None
    try:
        from data.earnings import load_earnings_dates
        earn = load_earnings_dates()
    except Exception as exc:
        log.warning("Earnings calendar unavailable (%s)", exc)
    ds = build_dataset(px, bench, gdelt, earn_dates=earn, aux=aux)
    live = load_live_features()
    if live is None:
        raise RuntimeError("live features missing - run retrain_brain first")
    for c in list(NEWS2_FEATURES) + list(NEWSNET_FEATURES):
        if c in live.columns:
            ds[c] = live[c].reindex(ds.index).to_numpy()
    return ds


def _atomic_dump(bundle: dict, path) -> None:
    import os
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(bundle, tmp)
    os.replace(tmp, path)


def _with_features(frame: pd.DataFrame, feats: list) -> pd.DataFrame:
    """NaN-fill any selected feature the pipeline failed to produce (HGB
    routes NaN to a learned branch) - NEVER silently shrink the deployed
    feature list."""
    missing = [f for f in feats if f not in frame.columns]
    if missing:
        log.warning("Refit: features %s missing today - NaN-filled, list "
                    "preserved", missing)
        frame = frame.copy()
        for c in missing:
            frame[c] = np.nan
    return frame


def refit_heads() -> str:
    """Refit the deployed direction, magnitude and volatility bundles -
    keeping each one's tournament-chosen architecture and FULL feature
    list, with parameters learned from everything through the last close.
    All bundles are fitted first and saved (atomically) only when every
    fit succeeded, so a partial failure can never leave heads from two
    feature generations deployed together. The isotonic calibrator is not
    refit here (it needs honest walk-forward probabilities); the Saturday
    tournament refreshes it.

    Honesty note: the live scorers saw every historical row's own label,
    so these heads train on partially-memorized news features (standard
    online-learning practice for a LIVE model, but it means no backtest
    may ever be run on live features - and none is: only signal.py reads
    them). The learning log and Track Record tab are the only valid
    judges of this configuration."""
    from model.train import _clean
    from sklearn.base import clone

    ds = _live_dataset()
    data = _clean(ds)
    pending, notes = [], []

    if config.MODEL_PATH.exists():
        b = joblib.load(config.MODEL_PATH)
        frame = _with_features(data, b["features"])
        train_slice = frame if b.get("window") is None \
            else frame.iloc[-int(b["window"]):]
        fresh = clone(b["model"])  # clone failure -> abort, never swap
        fresh.fit(train_slice[b["features"]], train_slice["y"])
        b.update({"model": fresh,
                  "trained_through": str(frame.index.max().date()),
                  "live_refit": True})
        pending.append((b, config.MODEL_PATH))
        notes.append(f"direction[{b.get('name', '?')}]")

    if config.MAG_MODEL_PATH.exists():
        mb = joblib.load(config.MAG_MODEL_PATH)
        thresh = (data["fwd_ret"].abs().rolling(63, min_periods=30)
                  .median().shift(1))
        mdata = data[thresh.notna()].copy()
        mdata["y"] = (mdata["fwd_ret"].abs()
                      > thresh[thresh.notna()]).astype(int)
        mframe = _with_features(mdata, mb["features"])
        fresh = clone(mb["model"])
        fresh.fit(mframe[mb["features"]], mframe["y"])
        mb.update({"model": fresh,
                   "trained_through": str(mframe.index.max().date()),
                   "live_refit": True})
        pending.append((mb, config.MAG_MODEL_PATH))
        notes.append("magnitude[price+news2]")

    if config.VOL_MODEL_PATH.exists():
        from features.vol import build_vol_dataset
        from data.news import load_gdelt_daily
        from data.prices import load_prices
        vb = joblib.load(config.VOL_MODEL_PATH)
        px, bench = load_prices()
        gdelt = load_gdelt_daily()
        earn = None
        try:
            from data.earnings import load_earnings_dates
            earn = load_earnings_dates()
        except Exception:
            pass
        live = load_live_features()
        through = None
        for h, spec in vb["models"].items():
            vds = build_vol_dataset(px, bench, gdelt, horizon=int(h),
                                    earn_dates=earn)
            vdata = vds.dropna(subset=["gk_22", "tone_1d", "y_vol"])
            # a newsvec winner's features live outside build_vol_dataset
            if live is not None:
                for c in spec["features"]:
                    if c not in vdata.columns and c in live.columns:
                        vdata = vdata.copy()
                        vdata[c] = live[c].reindex(vdata.index).to_numpy()
            vframe = _with_features(vdata, spec["features"])
            fresh = clone(spec["model"])
            fresh.fit(vframe[spec["features"]], vframe["y_vol"])
            spec["model"] = fresh
            through = str(vdata.index.max().date())
        if through:
            vb["trained_through"] = through
        pending.append((vb, config.VOL_MODEL_PATH))
        notes.append("vol[" + ", ".join(
            s["name"] for s in vb["models"].values()) + "]")

    for bundle, path in pending:  # all fits succeeded - now persist
        _atomic_dump(bundle, path)
    return "Heads refit through last close: " + "; ".join(notes)
