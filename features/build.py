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

# Earnings calendar (dates are scheduled weeks ahead, so knowable pre-open):
# earn_window = the print lands inside this entry day's holding window (AMC);
# post_earn = the gap session immediately after the print.
EVENT_FEATURES = ["days_to_earn", "days_since_earn", "earn_window", "post_earn"]
EARN_CAP = 30  # trading days; distances beyond this carry no signal

# Market-regime state (both computed strictly from data through d-1)
REGIME_FEATURES = ["dd_252", "vol_regime_z"]

FULL_FEATURES = EXTENDED_FEATURES + EVENT_FEATURES + REGIME_FEATURES

# Cross-name context: competitor and industry news also moves NVDA.
# comp_tone_rel (NVDA tone minus competitor tone) is the relative-sentiment
# signal; the volume spikes capture sector-wide attention shocks.
CROSS_FEATURES = ["comp_tone_1d", "comp_tone_rel", "comp_vol_spike",
                  "ind_tone_1d", "ind_vol_spike"]

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


def earnings_calendar_features(index: pd.DatetimeIndex,
                               earn_dates) -> pd.DataFrame:
    """Trading-day distances to/from earnings announcements for each entry
    day. AMC convention: the announcement on trading day E belongs to entry
    day E (its open->open holding window contains the print). Future dates
    beyond the index are approximated with business-day counts (they are
    capped at EARN_CAP anyway). With no calendar available, features take
    their neutral capped values."""
    out = pd.DataFrame(index=index)
    n = len(index)
    days_to = np.full(n, float(EARN_CAP))
    days_since = np.full(n, float(EARN_CAP))
    if earn_dates is not None and len(earn_dates):
        earn = pd.DatetimeIndex(earn_dates).sort_values()
        pos = index.searchsorted(earn)          # index position holding each print
        pos_in = np.unique(pos[pos < n])
        future = earn[pos >= n]
        i_arr = np.arange(n)
        if len(pos_in):
            nxt = np.searchsorted(pos_in, i_arr, side="left")
            valid = nxt < len(pos_in)
            days_to[valid] = pos_in[nxt[valid]] - i_arr[valid]
            prv = np.searchsorted(pos_in, i_arr, side="right") - 1
            valid = prv >= 0
            days_since[valid] = i_arr[valid] - pos_in[prv[valid]]
        if len(future):
            from trade.calendar import NYSEHolidays
            hols = NYSEHolidays().holidays(
                index[-1], future[0]).values.astype("datetime64[D]")
            extra = float(np.busday_count(np.datetime64(index[-1].date()),
                                          np.datetime64(future[0].date()),
                                          holidays=hols))
            approx = (n - 1 - i_arr) + extra
            days_to = np.minimum(days_to, np.maximum(approx, 0))
    out["days_to_earn"] = np.clip(days_to, 0, EARN_CAP)
    out["days_since_earn"] = np.clip(days_since, 0, EARN_CAP)
    out["earn_window"] = (out["days_to_earn"] == 0).astype(float)
    out["post_earn"] = (out["days_since_earn"] == 1).astype(float)
    return out


def print_gap_sq(px: pd.DataFrame, earn_dates) -> pd.Series:
    """Squared overnight gap of the session AFTER each earnings print (AMC on
    entry day E -> gap into session E+1), stamped at entry day E. Callers
    derive risk estimates: expanding().mean().shift(1) for backtests (past
    prints only), plain .mean() for a live forecast."""
    gap = np.log(px["Open"] / px["Close"].shift(1))
    earn = pd.DatetimeIndex(earn_dates).sort_values()
    pos = px.index.searchsorted(earn)
    pos = pos[pos < len(px.index) - 1]
    return pd.Series(gap.to_numpy()[pos + 1] ** 2, index=px.index[pos])


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
    out["dd_252"] = close / close.rolling(252).max() - 1
    roll = out["gk_22"].rolling(252)
    out["vol_regime_z"] = (out["gk_22"] - roll.mean()) / roll.std()
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
                  gdelt: pd.DataFrame, earn_dates=None,
                  aux: dict | None = None) -> pd.DataFrame:
    """Full modelling dataset indexed by entry day, with columns
    FULL_FEATURES + CROSS_FEATURES + [fwd_ret, y]. The final row(s) may have
    NaN targets; cross features are NaN when aux series are unavailable."""
    tech = build_tech_features(px, bench).shift(1)  # entry day d sees close(d-1)
    df = tech.copy()
    df["dow"] = df.index.dayofweek

    # News through calendar day d-1: as-of merge on cutoff = d minus 1 day.
    merged = news_asof(df.index, build_news_features(gdelt))
    for col in NEWS_FEATURES:
        df[col] = merged[col].to_numpy()
    df["tone_shock"] = (merged["tone_1d"] - merged["tone_7d"]).abs().to_numpy()

    ev = earnings_calendar_features(df.index, earn_dates)
    for col in EVENT_FEATURES:
        df[col] = ev[col].to_numpy()

    # Competitor/industry cross features (same d-1 cutoff discipline)
    for col in CROSS_FEATURES:
        df[col] = np.nan
    aux = aux or {}
    if "competitors" in aux:
        m = news_asof(df.index, build_news_features(aux["competitors"]))
        df["comp_tone_1d"] = m["tone_1d"].to_numpy()
        df["comp_vol_spike"] = m["news_vol_spike"].to_numpy()
        df["comp_tone_rel"] = df["tone_1d"] - df["comp_tone_1d"]
    if "industry" in aux:
        m = news_asof(df.index, build_news_features(aux["industry"]))
        df["ind_tone_1d"] = m["tone_1d"].to_numpy()
        df["ind_vol_spike"] = m["news_vol_spike"].to_numpy()

    open_ = px["Open"]
    df["fwd_ret"] = open_.shift(-1) / open_ - 1  # held open(d) -> open(d+1)
    df["y"] = np.where(df["fwd_ret"].isna(), np.nan,
                       (df["fwd_ret"] > 0).astype(float))
    return df
