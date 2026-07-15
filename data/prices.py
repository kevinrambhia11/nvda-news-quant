"""Daily OHLCV price data with a local cache.

Source chain (first that works wins):
  1. yfinance - full-featured, but its cookie/crumb flow gets rate-limited
     on corporate/shared IPs;
  2. Yahoo's raw v8 chart API with a browser User-Agent - same data, and the
     rate limit usually does not apply to it;
  3. Stooq's daily CSV (split- but not dividend-adjusted; last resort - it
     serves a JavaScript challenge to some networks).
"""
from __future__ import annotations

import logging
import time
from io import StringIO

import pandas as pd
import requests

import config
from data.news import atomic_to_csv

log = logging.getLogger(__name__)

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"
HEADERS = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/126.0 Safari/537.36")}


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    return df[~df.index.duplicated(keep="last")].sort_index()


def _fetch_yahoo(ticker: str, start: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"Yahoo returned no price data for {ticker}")
    return _normalize_index(df)[["Open", "High", "Low", "Close", "Volume"]]


def _fetch_yahoo_chart(ticker: str, start: str) -> pd.DataFrame:
    """Yahoo's raw chart endpoint; back-adjusts OHLC by adjclose/close so the
    result matches yfinance's auto_adjust=True convention."""
    params = {
        "period1": int(pd.Timestamp(start, tz="America/New_York").timestamp()),
        "period2": int(time.time()),
        "interval": "1d",
        "events": "div,splits",
    }
    resp = requests.get(YAHOO_CHART_URL.format(symbol=ticker), params=params,
                        headers=HEADERS, timeout=60)
    resp.raise_for_status()
    result = (resp.json().get("chart", {}).get("result") or [None])[0]
    if not result or not result.get("timestamp"):
        raise RuntimeError(f"Yahoo chart API returned no data for {ticker}")
    quote = result["indicators"]["quote"][0]
    adj = result["indicators"]["adjclose"][0]["adjclose"]
    idx = (pd.to_datetime(result["timestamp"], unit="s", utc=True)
           .tz_convert("America/New_York"))
    df = pd.DataFrame({"Open": quote["open"], "High": quote["high"],
                       "Low": quote["low"], "Close": quote["close"],
                       "AdjClose": adj, "Volume": quote["volume"]}, index=idx)
    # Yahoo's chart payload can carry null O/H/L on odd rows; a NaN there
    # poisons Garman-Klass features and crashes the linear vol model.
    df = df.dropna(subset=["Open", "High", "Low", "Close", "AdjClose"])
    factor = df["AdjClose"] / df["Close"]
    for col in ("Open", "High", "Low", "Close"):
        df[col] = df[col] * factor
    df = _normalize_index(df.drop(columns=["AdjClose"]))
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _fetch_stooq(ticker: str, start: str) -> pd.DataFrame:
    sym = f"{ticker.lower()}.us"
    resp = requests.get(STOOQ_URL.format(symbol=sym), headers=HEADERS, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    if df.empty or "Date" not in df.columns:
        raise RuntimeError(f"Stooq returned no price data for {sym}")
    df = df.set_index(pd.to_datetime(df["Date"])).drop(columns=["Date"])
    df = _normalize_index(df)
    df = df.loc[df.index >= pd.Timestamp(start)]
    return df[["Open", "High", "Low", "Close", "Volume"]]


def fetch_prices(ticker: str, start: str = config.TRAIN_START) -> pd.DataFrame:
    """Adjusted daily OHLCV, indexed by naive trading dates. Walks the source
    chain (yfinance -> Yahoo chart API -> Stooq) until one succeeds."""
    sources = [("yfinance", _fetch_yahoo),
               ("yahoo-chart", _fetch_yahoo_chart),
               ("stooq", _fetch_stooq)]
    last_exc: Exception | None = None
    for name, fn in sources:
        try:
            df = fn(ticker, start)
            log.info("%s: %d rows from %s (through %s)",
                     ticker, len(df), name, df.index.max().date())
            return df
        except Exception as exc:
            last_exc = exc
            log.warning("%s failed for %s (%s); trying next source", name, ticker, exc)
    raise RuntimeError(f"All price sources failed for {ticker}") from last_exc


def load_prices(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached (NVDA, benchmark) price frames; refetches when stale.

    Note: cached prices are adjusted as of the fetch date. After a split or
    dividend, refresh so the whole history is re-adjusted consistently.
    """
    out = []
    for ticker in (config.TICKER, config.BENCHMARK):
        cache_file = config.CACHE / f"prices_{ticker}.csv"
        df = None
        if cache_file.exists() and not refresh:
            cached = pd.read_csv(cache_file, index_col="date", parse_dates=["date"])
            if not cached.empty:
                # Fresh if it holds the last completed business day (weekends
                # and Mondays pre-open must not trigger a refetch).
                last_bday = pd.Timestamp.now().normalize() - pd.tseries.offsets.BDay(1)
                if cached.index.max() >= last_bday:
                    df = cached
        if df is None:
            df = fetch_prices(ticker)
            atomic_to_csv(df, cache_file)
        out.append(df)
    return out[0], out[1]
