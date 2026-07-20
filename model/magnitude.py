"""Magnitude head: does article-level news predict the SIZE of tomorrow's
move (regardless of direction)?

Target: |open-to-open forward return| above its trailing 63-day median
(median computed through d-1, so the threshold is leak-free and
regime-adaptive - roughly balanced classes by construction).

The DELTA between price-only and price+news candidates is the measured value
of the article-level news. Same walk-forward + untouched holdout discipline
as every other verdict on this desk, with two upgrades:

  * When nested scorers exist (`main.py nested`), news candidates run on
    nested features and ALL candidates are judged on the clean tail of the
    selection window - so the selection columns are honest, not memorized.
  * Isotonic recalibration is fitted on the judged (clean) window and
    applied to the holdout: NewsNet's raw scores are informative but
    miscalibrated, and sizing needs probabilities that mean what they say.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

import config
from features.build import FULL_FEATURES
from model.news2vec import NEWS2_FEATURES
from model.train import (_clean, _gbm_deep, nested_data, nested_meta,
                         walk_forward_probs)

log = logging.getLogger(__name__)

MAGNITUDE_REPORT = config.ARTIFACTS / "magnitude_report.txt"


def evaluate(dataset: pd.DataFrame) -> str:
    data = _clean(dataset)

    thresh = (data["fwd_ret"].abs().rolling(63, min_periods=30)
              .median().shift(1))
    data = data[thresh.notna()].copy()
    data["y"] = (data["fwd_ret"].abs() > thresh[thresh.notna()]).astype(int)

    meta = nested_meta()
    ndata = nested_data(data) if meta is not None else None
    clean_news = ndata is not None

    n2 = [c for c in NEWS2_FEATURES
          if c in data.columns and not data[c].isna().all()]
    news_flags = {"price-only": False, "price + news2": True}
    candidates = [("price-only", FULL_FEATURES),
                  ("price + news2", FULL_FEATURES + n2)]
    try:
        from model.newsnet import NEWSNET_FEATURES
        nn = [c for c in NEWSNET_FEATURES
              if c in data.columns and not data[c].isna().all()]
        if len(nn) >= 3:
            candidates.append(("price + newsnet", FULL_FEATURES + nn))
            candidates.append(("price + both", FULL_FEATURES + n2 + nn))
            news_flags.update({"price + newsnet": True, "price + both": True})
    except Exception:
        pass

    oos_start = config.MIN_TRAIN_DAYS
    oos_idx = data.index[oos_start:]
    # Pinned holdout boundary when nested meta exists (see model/train.py)
    split = None
    if meta is not None and oos_idx[0] < meta["holdout_start"] <= oos_idx[-1]:
        split = meta["holdout_start"]
    if split is None:
        split = oos_idx[int(len(oos_idx) * (1 - config.HOLDOUT_FRACTION))]
    judge_start = meta["inner"] if clean_news else oos_idx[0]
    cap = meta["features_through"] if clean_news else None
    pre_sel = oos_idx[(oos_idx >= judge_start) & (oos_idx < split)]
    if cap is not None:
        pre_sel = pre_sel[pre_sel <= cap]
    if clean_news and (len(pre_sel) < 60
                       or data.loc[pre_sel, "y"].nunique() < 2):
        log.warning("Nested judging window unusable (%d rows) - falling "
                    "back to the full selection window (news columns are "
                    "then memorized there)", len(pre_sel))
        clean_news, judge_start, cap = False, oos_idx[0], None

    lines = ["=" * 72,
             "  Magnitude head - P(|next-day move| > trailing median)",
             f"  base rate {data['y'].mean():.1%} | news2 features: {len(n2)}",
             f"  judged window {max(judge_start, oos_idx[0]).date()} -> "
             f"{split.date()} | holdout -> {oos_idx[-1].date()}",
             ("  (news candidates use NESTED scorers; judged window is clean"
              if clean_news else
              "  (nested artifacts absent: news selection columns are"
              " memorized;"),
             "   hold Brier* = after isotonic recalibration fitted on the"
             " judged window)",
             "=" * 72,
             f"  {'candidate':<16}{'sel AUC':>9}{'sel Brier':>11}"
             f"{'hold AUC':>10}{'hold Brier':>12}{'hold Brier*':>12}"]

    for name, feats in candidates:
        frame = ndata if (clean_news and news_flags.get(name)) else data
        probs = walk_forward_probs(frame, feats, _gbm_deep)
        oos = probs.notna()
        sel = oos & (data.index >= judge_start) & (data.index < split)
        if cap is not None:
            sel &= data.index <= cap
        hold = oos & (data.index >= split)
        row = []
        for m in (sel, hold):
            row += [roc_auc_score(data.loc[m, "y"], probs[m]),
                    brier_score_loss(data.loc[m, "y"], probs[m])]
        if news_flags.get(name) and not clean_news:
            # a calibrator fitted on memorized, saturated probs transfers
            # nothing meaningful to the holdout - suppress rather than print
            cal_txt = f"{'-':>12}"
        else:
            iso = IsotonicRegression(y_min=0.0, y_max=1.0,
                                     out_of_bounds="clip")
            iso.fit(probs[sel].to_numpy(), data.loc[sel, "y"].to_numpy())
            cal = iso.predict(probs[hold].to_numpy())
            cal_txt = f"{brier_score_loss(data.loc[hold, 'y'], cal):>12.4f}"
        lines.append(f"  {name:<16}{row[0]:>9.3f}{row[1]:>11.4f}"
                     f"{row[2]:>10.3f}{row[3]:>12.4f}{cal_txt}")

    lines.append("=" * 72)
    report = "\n".join(lines)
    MAGNITUDE_REPORT.write_text(report, encoding="utf-8")
    return report
