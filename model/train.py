"""Walk-forward training, candidate selection and evaluation for the
direction model.

Refinement discipline: candidates (feature sets x model families) are
compared on strategy Sharpe over a SELECTION window - the first part of the
out-of-sample walk-forward period. The winner's quality is then reported on
the untouched HOLDOUT tail, so model shopping cannot flatter the final
numbers. The winner, refit on all history, is saved for live signals.

Every prediction remains strictly out-of-sample: expanding-window refits
every RETRAIN_EVERY days with a 1-row embargo (the last row's open-to-open
label is not realized at pre-open refit time).
"""
from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from backtest.engine import positions_from_probs
from features.build import (ALL_FEATURES, CROSS_FEATURES, EXTENDED_FEATURES,
                            FULL_FEATURES, QUALITY_FEATURES)

log = logging.getLogger(__name__)


def _gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=3,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=False, random_state=42)


def _gbm_deep() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.04, max_depth=6,
        min_samples_leaf=40, l2_regularization=2.0,
        early_stopping=False, random_state=42)


def _logit():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.5, max_iter=2000))


def _cal_gbm():
    from sklearn.calibration import CalibratedClassifierCV
    return CalibratedClassifierCV(_gbm_deep(), method="isotonic", cv=3)


# (name, feature list, model factory, rolling window in trading days or None
#  for an expanding window). ROLL_3Y tests whether dropping stale history
# helps in the regime the expanding-window model failed in (2023+ holdout).
ROLL_3Y = 756
CANDIDATES = [
    ("GBM deep + vol", EXTENDED_FEATURES, _gbm_deep, None),      # incumbent
    ("GBM deep + events", FULL_FEATURES, _gbm_deep, None),
    ("GBM base + events", FULL_FEATURES, _gbm, None),
    ("GBM deep rolling-3y", FULL_FEATURES, _gbm_deep, ROLL_3Y),
    ("GBM deep calibrated", FULL_FEATURES, _cal_gbm, None),
]
def active_candidates(data: pd.DataFrame) -> list:
    """The competitor/industry candidate uses whichever cross features have
    data (an all-NaN column inside a training window breaks HGB's binning);
    it joins the tournament once at least one aux series is bootstrapped."""
    cands = list(CANDIDATES)
    avail = [c for c in CROSS_FEATURES if not data[c].isna().all()]
    if len(avail) >= 3:
        cands.append(("GBM deep + cross", FULL_FEATURES + avail,
                      _gbm_deep, None))
        log.info("Cross candidate active with features: %s", avail)
    else:
        log.info("Cross features not yet bootstrapped - competitor/industry "
                 "candidate skipped")
    qual = [c for c in QUALITY_FEATURES if not data[c].isna().all()]
    if len(qual) >= 2:
        cands.append(("GBM deep + quality", FULL_FEATURES + qual,
                      _gbm_deep, None))
        log.info("Quality-sources candidate active with features: %s", qual)
    try:
        from model.news2vec import NEWS2_FEATURES
        n2 = [c for c in NEWS2_FEATURES
              if c in data.columns and not data[c].isna().all()]
        if len(n2) >= 6:
            cands.append(("GBM deep + news2", FULL_FEATURES + n2,
                          _gbm_deep, None))
            log.info("news2 candidate active with %d features", len(n2))
    except Exception:
        pass
    try:
        from model.newsnet import NEWSNET_FEATURES
        nn = [c for c in NEWSNET_FEATURES
              if c in data.columns and not data[c].isna().all()]
        if len(nn) >= 3:
            cands.append(("GBM deep + newsnet", FULL_FEATURES + nn,
                          _gbm_deep, None))
            log.info("newsnet candidate active with %d features", len(nn))
    except Exception:
        pass
    return cands


def _clean(dataset: pd.DataFrame) -> pd.DataFrame:
    # sma200_dist has the longest warm-up (200 trading days); dropping its NaN
    # rows removes the whole warm-up period so no feature column is ever
    # entirely NaN inside a training window (which breaks HGB's binning).
    data = dataset.dropna(subset=["tone_1d", "ret_1", "sma200_dist", "y"]).copy()
    data["y"] = data["y"].astype(int)
    return data


def walk_forward_probs(data: pd.DataFrame, features: list[str], factory,
                       window: int | None = None) -> pd.Series:
    """Strictly-OOS P(up) series (NaN before the first refit). `window`
    limits training to the most recent N rows (rolling); None = expanding."""
    X, y = data[features], data["y"]
    n = len(data)
    if n <= config.MIN_TRAIN_DAYS + config.RETRAIN_EVERY:
        raise RuntimeError(f"Not enough rows ({n}) for walk-forward training")
    probs = pd.Series(index=data.index, dtype=float)
    for start in range(config.MIN_TRAIN_DAYS, n, config.RETRAIN_EVERY):
        stop = min(start + config.RETRAIN_EVERY, n)
        # Embargo the last row: its open-to-open label is only realized at
        # the first OOS entry open, unknowable at (pre-open) refit time.
        hi = start - 1
        lo = 0 if window is None else max(0, hi - window)
        model = factory()
        model.fit(X.iloc[lo:hi], y.iloc[lo:hi])
        probs.iloc[start:stop] = model.predict_proba(X.iloc[start:stop])[:, 1]
    return probs


def _strategy_stats(probs: pd.Series, fwd_ret: pd.Series,
                    y: pd.Series) -> dict:
    pos = positions_from_probs(probs)
    turnover = pos.diff().abs()
    if len(turnover):
        turnover.iloc[0] = abs(pos.iloc[0])
    rets = pos * fwd_ret - turnover * config.COST_PER_TURNOVER
    sd = rets.std()
    sharpe = float(rets.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")
    return {"sharpe": sharpe,
            "auc": float(roc_auc_score(y, probs)),
            "acc": float(accuracy_score(y, (probs > 0.5).astype(int))),
            "exposure": float((pos != 0).mean())}


def select_and_train(dataset: pd.DataFrame) -> str:
    """Run all candidates, select on the selection window, report holdout,
    persist the winner's full-OOS predictions and final model. Returns the
    printable selection report."""
    data = _clean(dataset)
    candidates = active_candidates(data)

    all_probs = {}
    for name, feats, factory, window in candidates:
        log.info("walk-forward: %s ...", name)
        all_probs[name] = walk_forward_probs(data, feats, factory, window)

    oos_mask = next(iter(all_probs.values())).notna()
    oos_idx = data.index[oos_mask]
    split = oos_idx[int(len(oos_idx) * (1 - config.HOLDOUT_FRACTION))]
    sel_idx = oos_idx[oos_idx < split]
    hold_idx = oos_idx[oos_idx >= split]

    lines = ["=" * 66,
             "  Direction model - candidate selection (walk-forward OOS)",
             f"  selection window: {sel_idx[0].date()} -> {sel_idx[-1].date()}"
             f"  |  holdout: {hold_idx[0].date()} -> {hold_idx[-1].date()}",
             "=" * 66,
             f"  {'candidate':<20}{'Sharpe':>8}{'AUC':>8}{'acc':>7}{'expo':>7}"]
    sel_stats = {}
    for name, _, _, _ in candidates:
        s = _strategy_stats(all_probs[name][sel_idx],
                            data.loc[sel_idx, "fwd_ret"], data.loc[sel_idx, "y"])
        sel_stats[name] = s
        lines.append(f"  {name:<20}{s['sharpe']:>8.2f}{s['auc']:>8.3f}"
                     f"{s['acc']:>7.3f}{s['exposure']:>7.1%}")

    # nan_to_num: a candidate that never trades has NaN Sharpe - it must
    # lose the selection, not win it via NaN comparison quirks.
    winner, win_feats, win_factory, win_window = max(
        candidates,
        key=lambda c: np.nan_to_num(sel_stats[c[0]]["sharpe"], nan=-np.inf))
    hold = _strategy_stats(all_probs[winner][hold_idx],
                           data.loc[hold_idx, "fwd_ret"], data.loc[hold_idx, "y"])
    bh_rets = data.loc[hold_idx, "fwd_ret"]
    bh_sharpe = float(bh_rets.mean() / bh_rets.std() * np.sqrt(252))
    lines += ["-" * 66,
              f"  WINNER (by selection Sharpe): {winner}",
              f"  Untouched holdout: Sharpe {hold['sharpe']:.2f} "
              f"(buy&hold {bh_sharpe:.2f}) | AUC {hold['auc']:.3f} "
              f"| acc {hold['acc']:.3f} | exposure {hold['exposure']:.1%}",
              "=" * 66]

    oos = data.loc[oos_mask, ["fwd_ret", "y"]].copy()
    oos["prob_up"] = all_probs[winner][oos_mask]
    oos.to_csv(config.OOS_PREDICTIONS_PATH)

    train_slice = data if win_window is None else data.iloc[-win_window:]
    final = win_factory().fit(train_slice[win_feats], train_slice["y"])
    joblib.dump({"model": final, "features": win_feats, "name": winner,
                 "window": win_window,
                 "trained_through": str(data.index.max().date()),
                 "n_rows": len(train_slice)}, config.MODEL_PATH)
    log.info("Winner '%s' trained on %d rows -> %s", winner, len(data),
             config.MODEL_PATH)

    report = "\n".join(lines)
    config.MODEL_SELECTION_PATH.write_text(report, encoding="utf-8")
    return report
