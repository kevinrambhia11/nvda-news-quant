"""Magnitude head: does article-level news predict the SIZE of tomorrow's
move (regardless of direction)?

Target: |open-to-open forward return| above its trailing 63-day median
(median computed through d-1, so the threshold is leak-free and
regime-adaptive - roughly balanced classes by construction).

Two candidates, price-only vs price+news2: the DELTA between them is the
measured value of the article-level news. Same walk-forward + untouched
holdout discipline as every other verdict on this desk.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

import config
from features.build import FULL_FEATURES
from model.news2vec import NEWS2_FEATURES
from model.train import _clean, _gbm_deep, walk_forward_probs

log = logging.getLogger(__name__)

MAGNITUDE_REPORT = config.ARTIFACTS / "magnitude_report.txt"


def evaluate(dataset: pd.DataFrame) -> str:
    data = _clean(dataset)

    thresh = (data["fwd_ret"].abs().rolling(63, min_periods=30)
              .median().shift(1))
    data = data[thresh.notna()].copy()
    data["y"] = (data["fwd_ret"].abs() > thresh[thresh.notna()]).astype(int)

    n2 = [c for c in NEWS2_FEATURES
          if c in data.columns and not data[c].isna().all()]
    candidates = [("price-only", FULL_FEATURES),
                  ("price + news2", FULL_FEATURES + n2)]
    try:
        from model.newsnet import NEWSNET_FEATURES
        nn = [c for c in NEWSNET_FEATURES
              if c in data.columns and not data[c].isna().all()]
        if len(nn) >= 3:
            candidates.append(("price + newsnet", FULL_FEATURES + nn))
            candidates.append(("price + both", FULL_FEATURES + n2 + nn))
    except Exception:
        pass

    oos_start = config.MIN_TRAIN_DAYS
    oos_idx = data.index[oos_start:]
    split = oos_idx[int(len(oos_idx) * (1 - config.HOLDOUT_FRACTION))]

    lines = ["=" * 66,
             "  Magnitude head - P(|next-day move| > trailing median)",
             f"  base rate {data['y'].mean():.1%} | news2 features: {len(n2)}",
             f"  selection {oos_idx[0].date()} -> {split.date()} | "
             f"holdout -> {oos_idx[-1].date()}",
             "  (impact scorers trained pre-holdout: selection numbers are",
             "   optimistic for news2; the holdout column is the verdict)",
             "=" * 66,
             f"  {'candidate':<16}{'sel AUC':>9}{'sel Brier':>11}"
             f"{'hold AUC':>10}{'hold Brier':>12}"]

    for name, feats in candidates:
        probs = walk_forward_probs(data, feats, _gbm_deep)
        oos = probs.notna()
        sel = oos & (data.index < split)
        hold = oos & (data.index >= split)
        row = []
        for m in (sel, hold):
            row += [roc_auc_score(data.loc[m, "y"], probs[m]),
                    brier_score_loss(data.loc[m, "y"], probs[m])]
        lines.append(f"  {name:<16}{row[0]:>9.3f}{row[1]:>11.4f}"
                     f"{row[2]:>10.3f}{row[3]:>12.4f}")

    lines.append("=" * 66)
    report = "\n".join(lines)
    MAGNITUDE_REPORT.write_text(report, encoding="utf-8")
    return report
