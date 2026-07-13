"""NVDA earnings-announcement calendar with a local cache.

NVDA reports after market close (AMC), so the announcement on day E lands
inside the open(E) -> open(E+1) holding window: entry day E holds through
the print, and session E+1 is the gap/spike session. The feature builder
encodes both (earn_window and post_earn).

Cache format: artifacts/cache/earnings_NVDA.csv with a single `date` column
of ISO announcement dates (past and scheduled future). The cache is
considered fresh while it still contains a future scheduled date; it can
also be maintained by hand when Yahoo is rate-limited.
"""
from __future__ import annotations

import logging

import pandas as pd

import config
from data.news import atomic_to_csv

log = logging.getLogger(__name__)


def fetch_earnings_dates(ticker: str = config.TICKER) -> pd.DatetimeIndex:
    import yfinance as yf
    df = yf.Ticker(ticker).get_earnings_dates(limit=60)
    if df is None or df.empty:
        raise RuntimeError(f"No earnings dates returned for {ticker}")
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    dates = pd.DatetimeIndex(sorted(set(idx.normalize())))
    return dates[dates >= pd.Timestamp(config.TRAIN_START)]


def load_earnings_dates(refresh: bool = False) -> pd.DatetimeIndex:
    """Cached earnings dates; refetches only when no future date remains."""
    cache_file = config.CACHE / f"earnings_{config.TICKER}.csv"
    cached = None
    if cache_file.exists():
        cached = pd.DatetimeIndex(
            pd.read_csv(cache_file, parse_dates=["date"])["date"]).sort_values()
        if not refresh and len(cached) and \
                cached.max() >= pd.Timestamp.now().normalize():
            return cached
    try:
        dates = fetch_earnings_dates()
        atomic_to_csv(pd.DataFrame(index=dates.rename("date")), cache_file)
        return dates
    except Exception as exc:
        if cached is not None and len(cached):
            log.warning("Earnings refetch failed (%s); using cached calendar "
                        "through %s", exc, cached.max().date())
            return cached
        raise
