"""Feature construction.

Timing convention (chosen so the live signal is leak-free by construction):
each row is indexed by an ENTRY day d (a trading day). The position for day d
is decided BEFORE the market opens on d, so every feature may only use

  * news through calendar day d-1 (GDELT days are UTC; a UTC day ends
    ~13 hours before the next US market open), and
  * prices through the close of the trading day before d.

The target is the open-to-open return Open(d) -> Open(next trading day),
i.e. exactly the return you capture if you trade at the open on day d based
on the previous evening's signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NEWS_FEATURES = ["tone_1d", "tone_3d", "tone_7d", "tone_z60", "tone_chg",
                 "news_vol_spike", "news_vol_ratio"]
TECH_FEATURES = ["ret_1", "ret_5", "ret_21", "vol_21", "rsi_14",
                 "sma50_dist", "sma200_dist", "bench_ret_5", "rel_ret_5", "dow"]
ALL_FEATURES = NEWS_FEATURES + TECH_FEATURES

# Volatility-regime context borrowed from the vol model (HAR lags + tone
# shock) - candidate models may use ALL_FEATURES + VOL_CONTEXT_FEATURES.
VOL_CONTEXT_FEATURES = ["gk_1", "gk_5", "gk_22", "tone_shock"]
EXTENDED_FEATURES = ALL_FEATURES + VOL_CONTEXT_FEATURES

DAILY_VOL_FLOOR = 1e-4  # 1bp-per-day floor so logs never blow up on odd bars


def garman_klass_vol(px: pd.DataFrame) -> pd.Series:
    """Daily Garman-Klass volatility (per day, not annualized). Uses only
    OHLC ratios, so it is invariant to dividend/split adjustment scaling."""
    log_hl = np.log(px["High"] / px["Low"])
    log_co = np.log(px["Close"] / px["Open"])
    var = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    return np.sqrt(var.clip(lower=DAILY_VOL_FLOOR ** 2))


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rsi = 100 - 100 / (1 + gain / loss)  # loss=0, gain>0 -> inf -> RSI 100
    rsi = rsi.mask(gain.eq(0) & loss.eq(0), 50.0)  # flat window -> neutral
    return rsi


def build_news_features(gdelt: pd.DataFrame) -> pd.DataFrame:
    """Daily (calendar) news features. The value stamped at day D uses news
    from day D and earlier only, so rolling windows never look ahead."""
    g = gdelt.sort_index().copy()
    full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
    g = g.reindex(full_idx)
    g["art_count"] = g["art_count"].fillna(0.0)
    g["tone"] = g["tone"].ffill()  # no fresh coverage -> carry last known tone

    out = pd.DataFrame(index=g.index)
    out["tone_1d"] = g["tone"]
    out["tone_3d"] = g["tone"].rolling(3).mean()
    out["tone_7d"] = g["tone"].rolling(7).mean()
    roll = g["tone"].rolling(60)
    out["tone_z60"] = (g["tone"] - roll.mean()) / roll.std()
    out["tone_chg"] = out["tone_1d"] - out["tone_3d"]
    base = g["art_count"].rolling(30).mean().replace(0, np.nan)
    out["news_vol_spike"] = g["art_count"] / base
    out["news_vol_ratio"] = g["art_count"].rolling(7).mean() / base
    out.index.name = "date"
    return out


def build_tech_features(px: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    """Trading-day features stamped at day t using data through the close of t.
    (build_dataset shifts these by one row so entry day d sees t = d-1.)"""
    close = px["Close"]
    out = pd.DataFrame(index=px.index)
    out["ret_1"] = close.pct_change()
    out["ret_5"] = close.pct_change(5)
    out["ret_21"] = close.pct_change(21)
    out["vol_21"] = out["ret_1"].rolling(21).std() * np.sqrt(252)
    out["rsi_14"] = _rsi(close)
    out["sma50_dist"] = close / close.rolling(50).mean() - 1
    out["sma200_dist"] = close / close.rolling(200).mean() - 1
    bench_close = bench["Close"].reindex(px.index).ffill()
    out["bench_ret_5"] = bench_close.pct_change(5)
    out["rel_ret_5"] = out["ret_5"] - out["bench_ret_5"]
    gk = garman_klass_vol(px)
    out["gk_1"] = np.log(gk)
    out["gk_5"] = np.log(gk.rolling(5).mean())
    out["gk_22"] = np.log(gk.rolling(22).mean())
    return out


def news_asof(index: pd.DatetimeIndex, news: pd.DataFrame) -> pd.DataFrame:
    """News feature rows as-of calendar day d-1 for each entry day d in
    index. `index` must be sorted ascending (trading days always are)."""
    cutoffs = pd.DataFrame({"cutoff": index - pd.Timedelta(days=1)})
    news_reset = news.reset_index().rename(columns={"date": "news_date"})
    merged = pd.merge_asof(cutoffs, news_reset.sort_values("news_date"),
                           left_on="cutoff", right_on="news_date",
                           direction="backward")
    merged.index = index
    return merged


def build_dataset(px: pd.DataFrame, bench: pd.DataFrame,
                  gdelt: pd.DataFrame) -> pd.DataFrame:
    """Full modelling dataset indexed by entry day, with columns
    ALL_FEATURES + [fwd_ret, y]. The final row(s) may have NaN targets."""
    tech = build_tech_features(px, bench).shift(1)  # entry day d sees close(d-1)
    df = tech.copy()
    df["dow"] = df.index.dayofweek

    # News through calendar day d-1: as-of merge on cutoff = d minus 1 day.
    merged = news_asof(df.index, build_news_features(gdelt))
    for col in NEWS_FEATURES:
        df[col] = merged[col].to_numpy()
    df["tone_shock"] = (merged["tone_1d"] - merged["tone_7d"]).abs().to_numpy()

    open_ = px["Open"]
    df["fwd_ret"] = open_.shift(-1) / open_ - 1  # held open(d) -> open(d+1)
    df["y"] = np.where(df["fwd_ret"].isna(), np.nan,
                       (df["fwd_ret"] > 0).astype(float))
    return df
