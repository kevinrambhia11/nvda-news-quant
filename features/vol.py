"""Volatility-forecast dataset construction.

Same timing discipline as features/build.py: each row is indexed by an entry
day d whose forecast is made BEFORE d's open, so features may only use prices
through the close of d-1 and news through calendar day d-1. The target is
the log Garman-Klass volatility realized DURING session d (horizon 1), or
the average over sessions d..d+H-1 (horizon H) - entirely after decision
time, so no lookahead is possible by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.build import (build_news_features, earnings_calendar_features,
                            garman_klass_vol, news_asof)
from features.build import EVENT_FEATURES  # noqa: F401  (re-export for candidates)

VOL_NEWS_FEATURES = ["news_vol_spike", "news_vol_ratio", "tone_1d",
                     "tone_z60", "tone_shock"]
VOL_TECH_FEATURES = ["gk_1", "gk_5", "gk_22", "ret_1", "abs_ret_1", "ret_5",
                     "cc_vol_21", "bench_gk_1", "bench_ret_1", "dow"]
VOL_FEATURES = VOL_NEWS_FEATURES + VOL_TECH_FEATURES


def build_vol_dataset(px: pd.DataFrame, bench: pd.DataFrame,
                      gdelt: pd.DataFrame, horizon: int = 1,
                      earn_dates=None) -> pd.DataFrame:
    """Dataset indexed by entry day with VOL_FEATURES + [y_vol, realized_var].

    y_vol(d) = ln(mean Garman-Klass vol over sessions d..d+horizon-1);
    realized_var is the squared vol proxy used by the QLIKE loss. Final
    row(s) have NaN targets (not yet realized).
    """
    gk = garman_klass_vol(px)
    log_gk = np.log(gk)
    close = px["Close"]

    tech = pd.DataFrame(index=px.index)
    tech["gk_1"] = log_gk                                # HAR daily leg
    tech["gk_5"] = np.log(gk.rolling(5).mean())          # HAR weekly leg
    tech["gk_22"] = np.log(gk.rolling(22).mean())        # HAR monthly leg
    tech["ret_1"] = close.pct_change()                   # leverage effect
    tech["abs_ret_1"] = tech["ret_1"].abs()
    tech["ret_5"] = close.pct_change(5)
    tech["cc_vol_21"] = tech["ret_1"].rolling(21).std() * np.sqrt(252)
    bench_aligned = bench.reindex(px.index).ffill()
    tech["bench_gk_1"] = np.log(garman_klass_vol(bench_aligned))
    tech["bench_ret_1"] = bench_aligned["Close"].pct_change()

    df = tech.shift(1)  # entry day d sees data through the close of d-1
    df["dow"] = df.index.dayofweek

    news = news_asof(df.index, build_news_features(gdelt))
    for col in ("news_vol_spike", "news_vol_ratio", "tone_1d", "tone_z60"):
        df[col] = news[col].to_numpy()
    df["tone_shock"] = (news["tone_1d"] - news["tone_7d"]).abs().to_numpy()

    ev = earnings_calendar_features(df.index, earn_dates)
    for col in EVENT_FEATURES:
        df[col] = ev[col].to_numpy()

    fwd_mean_vol = gk.rolling(horizon).mean().shift(-(horizon - 1))
    df["y_vol"] = np.log(fwd_mean_vol)
    df["realized_var"] = fwd_mean_vol ** 2
    return df
