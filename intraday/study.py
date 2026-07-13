"""Intraday news-reaction event study (60-day, 5-minute pilot).

The question that decides whether intraday news trading is worth building:
once the FIRST 5-minute bar has priced a news burst, is there exploitable
drift afterwards - and does the burst's tone predict its direction?

Method:
  1. Detect bursts in GDELT's 15-minute NVIDIA article counts (robust
     z-score vs a trailing 7-day baseline, absolute article floor, 1-hour
     cooldown between episodes).
  2. Align each burst to the first tradable 5-minute bar with bar start
     >= the 15-minute bin's END (no lookahead: the bin only exists once it
     has closed). Bursts outside regular hours form an "overnight" cohort
     whose reaction is the next session's first bar.
  3. Measure NVDA market-adjusted (minus QQQ) returns: the immediate
     reaction bar, then drift from the reaction bar's close to +15m, +30m,
     +1h, +3h (same session only).
  4. Split by burst tone terciles and t-test the drifts.

This is a pilot on ~60 days of free data - power is limited and the report
says so. Its job is to decide whether the full intraday build is warranted.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

ET = "America/New_York"
HORIZONS = {"+15m": 3, "+30m": 6, "+1h": 12, "+3h": 36}  # in 5-min bars
COOLDOWN_BINS = 4          # 1 hour between distinct burst episodes
BASELINE_BINS = 672        # 7 days of 15-min bins
MIN_BASELINE_BINS = 192    # 2 days minimum before a z-score is trusted


def _load_bars(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
    df = df.set_index(ts).drop(columns=["timestamp"])
    df.columns = [c.lower() for c in df.columns]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # regular trading hours only
    df = df[(df.index.time >= pd.Timestamp("09:30").time())
            & (df.index.time < pd.Timestamp("16:00").time())]
    return df[["open", "high", "low", "close", "volume"]].dropna()


def _load_news() -> pd.DataFrame:
    g = pd.read_csv(config.INTRADAY_GDELT_15M)
    ts = pd.to_datetime(g["utc_timestamp"], utc=True)
    g = g.set_index(ts).drop(columns=["utc_timestamp"])
    g = g[~g.index.duplicated(keep="last")].sort_index()
    # Materialize the complete 15-min grid so row-based rolling windows and
    # cooldowns are time-true even if quiet bins are absent from the cache.
    full = pd.date_range(g.index[0], g.index[-1], freq="15min")
    g = g.reindex(full)
    g["art_count"] = g["art_count"].fillna(0.0)
    return g


def detect_bursts(news: pd.DataFrame) -> pd.DataFrame:
    """Burst = article-count robust z >= BURST_Z with an absolute floor,
    keeping only the first bin of each episode."""
    counts = news["art_count"].astype(float)
    med = counts.rolling(BASELINE_BINS, min_periods=MIN_BASELINE_BINS).median()
    mad = (counts - med).abs().rolling(
        BASELINE_BINS, min_periods=MIN_BASELINE_BINS).median()
    # MAD floor of 1 article: a spike over a flat baseline must register as a
    # huge z, not NaN (warm-up NaNs still propagate through clip).
    z = (counts - med) / (1.4826 * mad.clip(lower=1.0))
    hot = (z >= config.BURST_Z) & (counts >= config.BURST_MIN_ARTICLES)

    bursts = []
    last_pos = -10_000
    positions = np.flatnonzero(hot.to_numpy())
    for pos in positions:
        if pos - last_pos <= COOLDOWN_BINS:
            last_pos = pos  # extend the episode without emitting a new event
            continue
        last_pos = pos
        bursts.append({"bin_start_utc": news.index[pos],
                       "z": float(z.iloc[pos]),
                       "art_count": float(counts.iloc[pos]),
                       "tone": float(news["tone"].iloc[pos])})
    out = pd.DataFrame(bursts)
    log.info("Detected %d burst episodes from %d hot bins", len(out),
             len(positions))
    return out


def _abnormal(nvda: pd.DataFrame, qqq: pd.DataFrame, i0: int, i1: int,
              from_open: bool) -> float:
    """NVDA-minus-QQQ return between bar positions (same session assumed)."""
    base = nvda["open"].iloc[i0] if from_open else nvda["close"].iloc[i0]
    r_n = nvda["close"].iloc[i1] / base - 1
    t0, t1 = nvda.index[i0], nvda.index[i1]
    try:
        q0 = qqq.index.get_loc(t0)
        q1 = qqq.index.get_loc(t1)
    except KeyError:
        return np.nan
    qbase = qqq["open"].iloc[q0] if from_open else qqq["close"].iloc[q0]
    r_q = qqq["close"].iloc[q1] / qbase - 1
    return float(r_n - r_q)


def build_events(nvda: pd.DataFrame, qqq: pd.DataFrame,
                 bursts: pd.DataFrame) -> pd.DataFrame:
    """One row per burst with reaction-bar return and post-reaction drifts."""
    sessions = pd.Series(nvda.index.date, index=nvda.index)
    rows = []
    for _, b in bursts.iterrows():
        bin_end = b["bin_start_utc"] + pd.Timedelta(minutes=15)
        t_react = bin_end.tz_convert(ET)
        i0 = int(nvda.index.searchsorted(t_react))
        if i0 >= len(nvda):
            continue
        # Intraday = burst bin started during regular hours AND the reaction
        # bar follows immediately; pre-open/evening/weekend bursts react at
        # the next session's first bar (overnight cohort).
        bin_start_et = b["bin_start_utc"].tz_convert(ET)
        in_rth = (pd.Timestamp("09:30").time() <= bin_start_et.time()
                  < pd.Timestamp("16:00").time())
        same_bin_gap = (nvda.index[i0] - t_react) <= pd.Timedelta(minutes=15)
        cohort = "intraday" if (same_bin_gap and in_rth) else "overnight"
        session = sessions.iloc[i0]
        sess_end = int(np.searchsorted(sessions.to_numpy(), session,
                                       side="right")) - 1
        row = {"bin_start_utc": b["bin_start_utc"], "cohort": cohort,
               "z": b["z"], "art_count": b["art_count"], "tone": b["tone"],
               "react_bar": nvda.index[i0],
               "r_react": _abnormal(nvda, qqq, i0, i0, from_open=True)}
        for name, k in HORIZONS.items():
            i1 = i0 + k
            row[f"drift{name}"] = (_abnormal(nvda, qqq, i0, i1, from_open=False)
                                   if i1 <= sess_end else np.nan)
        rows.append(row)
    cols = (["bin_start_utc", "cohort", "z", "art_count", "tone",
             "react_bar", "r_react"] + [f"drift{h}" for h in HORIZONS])
    return pd.DataFrame(rows, columns=cols)  # schema survives zero events


def _tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 3 or x.std() == 0:
        return np.nan
    return float(x.mean() / (x.std() / np.sqrt(len(x))))


def run_study() -> str:
    for p in (config.INTRADAY_NVDA_5M, config.INTRADAY_QQQ_5M,
              config.INTRADAY_GDELT_15M):
        if not p.exists():
            raise RuntimeError(f"Missing intraday input {p.name} - bootstrap "
                               "the 5-minute/15-minute caches first")
    nvda = _load_bars(config.INTRADAY_NVDA_5M)
    qqq = _load_bars(config.INTRADAY_QQQ_5M)
    if nvda.empty or qqq.empty:
        raise RuntimeError("Intraday bar cache parsed to zero RTH rows - "
                           "re-bootstrap nvda_5m.csv / qqq_5m.csv")
    news = _load_news()
    bursts = detect_bursts(news)
    events = build_events(nvda, qqq, bursts)
    events.to_csv(config.INTRADAY_EVENTS_PATH, index=False)

    intra = events[events["cohort"] == "intraday"].copy()
    # Complete-case filter on the CORE horizons (+15m/+30m/+1h) so table and
    # plot cover identical events; requiring +3h too would discard every
    # burst after ~13:00 ET. +3h is reported separately with its own n.
    core_h = list(HORIZONS)[:3]
    core_cols = [f"drift{h}" for h in core_h]
    n_dropped = int(len(intra) - len(intra.dropna(subset=core_cols)))
    intra = intra.dropna(subset=core_cols).copy()

    # Time-of-day-matched abnormal baseline (bursts cluster near the open,
    # where every bar is wilder - an unconditional baseline would flatter
    # the reaction).
    q = qqq.reindex(nvda.index)
    adj_abs = ((nvda["close"] / nvda["open"] - 1)
               - (q["close"] / q["open"] - 1)).abs()
    tod_med = adj_abs.groupby(adj_abs.index.time).median()

    lines = ["=" * 66,
             "  Intraday news-reaction pilot (5-min bars vs 15-min GDELT)",
             f"  bars {nvda.index[0].date()} -> {nvda.index[-1].date()} "
             f"({len(nvda)} bars) | {len(events)} burst events "
             f"({len(intra)} intraday, {len(events) - len(intra)} overnight)",
             "  returns are NVDA minus QQQ (market-adjusted)",
             "=" * 66]

    if n_dropped:
        lines.append(f"  ({n_dropped} late-session events dropped so all "
                     "horizons cover identical events)")

    if len(intra) >= 10 and intra["tone"].notna().sum() >= 3:
        matched = intra["react_bar"].map(
            lambda t: tod_med.get(t.time(), np.nan))
        lines.append(f"\n  Reaction bar |abnormal return|: median "
                     f"{intra['r_react'].abs().median():.3%} vs "
                     f"time-of-day-matched baseline "
                     f"{float(matched.median()):.3%} "
                     f"({intra['r_react'].abs().median() / float(matched.median()):.1f}x)")
        # rank breaks tone ties so tercile edges are always unique
        intra["tone_bucket"] = pd.qcut(intra["tone"].rank(method="first"), 3,
                                       labels=["negative", "neutral", "positive"])
        lines.append(f"\n  {'tone bucket':<12}{'n':>4}{'react':>9}"
                     + "".join(f"{h:>9}" for h in core_h))
        for bucket, grp in intra.groupby("tone_bucket", observed=True):
            cells = "".join(
                f"{grp[f'drift{h}'].mean():>9.3%}" for h in core_h)
            lines.append(f"  {str(bucket):<12}{len(grp):>4}"
                         f"{grp['r_react'].mean():>9.3%}" + cells)
        lines.append("\n  Post-reaction drift t-stats (positive minus "
                     "negative tone):")
        for h in HORIZONS:
            pos = intra.loc[intra["tone_bucket"] == "positive", f"drift{h}"]
            neg = intra.loc[intra["tone_bucket"] == "negative", f"drift{h}"]
            spread = pos.mean() - neg.mean()
            # two-sample t on the spread
            n1, n2 = pos.notna().sum(), neg.notna().sum()
            se = np.sqrt(pos.var() / max(n1, 1) + neg.var() / max(n2, 1))
            t = spread / se if se > 0 else np.nan
            lines.append(f"    {h:<5} spread {spread:>8.3%}  t = {t:>5.2f}  "
                         f"(n={n1}/{n2})")
        lines.append("\n  Read |t| >= 2 as suggestive, not conclusive: with "
                     "~60 days of")
        lines.append("  data this pilot is powered to find large effects "
                     "only.")
    else:
        lines.append(f"\n  Only {len(intra)} intraday events - too few to "
                     "analyze. Loosen BURST_Z or extend the sample.")
    lines.append("=" * 66)

    _maybe_plot(intra)
    report = "\n".join(lines)
    config.INTRADAY_REPORT_PATH.write_text(report, encoding="utf-8")
    return report


def _maybe_plot(intra: pd.DataFrame) -> None:
    if len(intra) < 10 or "tone_bucket" not in intra.columns:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        core_h = list(HORIZONS)[:3]
        xs = [0] + [HORIZONS[h] for h in core_h]
        labels = ["react"] + core_h
        for bucket, grp in intra.groupby("tone_bucket", observed=True):
            ys = [grp["r_react"].mean()] + [
                grp["r_react"].mean() + grp[f"drift{h}"].mean()
                for h in core_h]
            ax.plot(xs, [y * 100 for y in ys], marker="o", label=str(bucket))
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xticks(xs, labels)
        ax.set_ylabel("cumulative abnormal return (%)")
        ax.set_title("NVDA reaction to GDELT news bursts by tone tercile")
        ax.legend(title="burst tone")
        fig.tight_layout()
        fig.savefig(config.INTRADAY_PLOT_PATH, dpi=120)
        plt.close(fig)
        log.info("Reaction plot -> %s", config.INTRADAY_PLOT_PATH)
    except Exception as exc:
        log.info("Skipping plot (%s)", exc)
