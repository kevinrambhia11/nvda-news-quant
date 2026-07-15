"""News collection.

Historical daily sentiment comes from the GDELT Project's DOC 2.0 API, which
monitors and tone-scores global online news coverage back to January 2017 -
this is the "scrape the whole web" backbone, done through a free, stable API
instead of brittle per-site scrapers.

Live headlines additionally come from Google News RSS, Yahoo Finance RSS,
Finviz, StockTwits and Reddit; every source is optional and failures in one
never block the rest.
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

import config


def atomic_to_csv(df: pd.DataFrame, path) -> None:
    """Write-then-rename so an interrupted write never leaves a torn cache."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp)
    os.replace(tmp, path)

log = logging.getLogger(__name__)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) nvda-news-quant/1.0",
    "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
}

session = requests.Session()
session.headers.update(HEADERS)


# ---------------------------------------------------------------------------
# Historical daily sentiment (GDELT)
# ---------------------------------------------------------------------------

class GdeltRateLimited(RuntimeError):
    """GDELT throttles per source IP; on shared/corporate networks the limit
    can already be exhausted by other users. Completed chunks are cached, so
    simply re-run `python main.py fetch` later to resume where you left off."""


def _gdelt_timeline_chunk(mode: str, start: datetime, end: datetime,
                          query: str) -> pd.Series:
    """One GDELT timeline request. mode: 'timelinetone' or 'timelinevolraw'.

    GDELT signals throttling either as HTTP 429 or as an HTTP-200 text page
    starting with 'Please limit requests' - both are retried with patient
    backoff, then surfaced as GdeltRateLimited.
    """
    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d000000"),
        "enddatetime": end.strftime("%Y%m%d235959"),
    }
    backoffs = [75, 150, 300]
    timeline = None
    for attempt in range(len(backoffs) + 1):
        throttled = False
        try:
            resp = session.get(GDELT_URL, params=params, timeout=60)
            if resp.status_code == 429 or resp.text.lstrip().startswith("Please limit"):
                throttled = True
            else:
                resp.raise_for_status()
                payload = resp.json()
                timeline = (payload or {}).get("timeline") or []
                if not timeline or not timeline[0].get("data"):
                    # Empty payloads happen during GDELT hiccups; treating
                    # them as retryable prevents silent holes in the history.
                    raise ValueError(
                        f"GDELT returned an empty {mode} timeline for "
                        f"{start.date()} -> {end.date()}")
                break
        except (requests.RequestException, ValueError) as exc:
            if attempt == len(backoffs):
                raise
            log.warning("GDELT request failed (%s); retrying...", exc)
        if throttled and attempt == len(backoffs):
            raise GdeltRateLimited(
                "GDELT is rate-limiting this network. Re-run "
                "`python main.py fetch` later; completed chunks are cached.")
        wait = backoffs[attempt]
        log.warning("GDELT throttled/failed; waiting %ds before retry "
                    "(%d/%d)...", wait, attempt + 1, len(backoffs))
        time.sleep(wait)
    df = pd.DataFrame(timeline[0]["data"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%SZ")
    return df.set_index("date")["value"].astype(float)


def fetch_gdelt_daily(start: str, end: str | None = None,
                      query: str = config.GDELT_QUERY,
                      cache_prefix: str = "gdelt") -> pd.DataFrame:
    """Daily average tone and raw article count for a GDELT query.

    Returns a calendar-day (UTC) indexed DataFrame with columns
    [tone, art_count]. Chunked into <=300-day requests so GDELT always
    returns daily granularity; each completed historical chunk is cached on
    disk (namespaced by cache_prefix), so an interrupted fetch resumes
    instead of restarting.
    """
    start_dt = pd.Timestamp(start).to_pydatetime()
    end_dt = (pd.Timestamp(end).to_pydatetime() if end
              else datetime.now(timezone.utc).replace(tzinfo=None))
    today = datetime.now(timezone.utc).date()
    parts = []
    cursor = start_dt
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=300), end_dt)
        chunk_cache = (config.CACHE / (f"{cache_prefix}_chunk_"
                                       f"{cursor:%Y%m%d}_{chunk_end:%Y%m%d}.csv"))
        if chunk_cache.exists():
            part = pd.read_csv(chunk_cache, index_col="date", parse_dates=["date"])
        else:
            log.info("GDELT fetch [%s] %s -> %s", cache_prefix, cursor.date(),
                     chunk_end.date())
            tone = _gdelt_timeline_chunk("timelinetone", cursor, chunk_end, query)
            time.sleep(6.5)  # GDELT allows max one request per 5 seconds
            vol = _gdelt_timeline_chunk("timelinevolraw", cursor, chunk_end, query)
            time.sleep(6.5)
            part = pd.DataFrame({"tone": tone, "art_count": vol})
            part.index.name = "date"
            # Never cache empty frames or today's still-partial UTC bucket
            if not part.empty and chunk_end.date() < today:
                atomic_to_csv(part, chunk_cache)
        parts.append(part)
        cursor = chunk_end + timedelta(days=1)
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "date"
    if df.empty:
        raise RuntimeError(
            f"GDELT returned no data for {start} -> {end or 'now'} "
            f"(query: {query!r})")
    return df


def load_gdelt_daily(refresh: bool = False, name: str = "gdelt_daily",
                     query: str = config.GDELT_QUERY) -> pd.DataFrame:
    """Cached GDELT daily series; incrementally extends the cache when stale."""
    cache_file = config.CACHE / f"{name}.csv"
    today_utc = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    if cache_file.exists() and not refresh:
        cached = pd.read_csv(cache_file, index_col="date", parse_dates=["date"])
        if not cached.empty:  # empty/corrupt cache falls through to a refetch
            last = cached.index.max()
            if last >= today_utc - pd.Timedelta(days=1):
                return cached
            try:
                fresh = fetch_gdelt_daily(
                    (last - pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
                    query=query, cache_prefix=name)
            except Exception as exc:
                # A scheduled run must degrade to slightly-stale data, not
                # die: tone is ffilled downstream anyway. Cold bootstraps
                # (no cache) still raise below.
                log.warning("GDELT top-up failed (%s); using cached %s "
                            "through %s", exc, name, last.date())
                return cached
            merged = pd.concat([cached, fresh])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            atomic_to_csv(merged, cache_file)
            return merged
    df = fetch_gdelt_daily(config.TRAIN_START, query=query, cache_prefix=name)
    atomic_to_csv(df, cache_file)
    return df


def load_aux_gdelt(refresh: bool = False) -> dict:
    """Competitor/industry daily series per config.AUX_SERIES, routed to
    each series' permanent source. Failures degrade to omission (features go
    NaN) rather than blocking the pipeline."""
    out = {}
    for series, spec in config.AUX_SERIES.items():
        # Negative cache: a cold bootstrap against a throttled network burns
        # ~10 minutes of retries; don't repeat that more than every 3 hours.
        marker = config.CACHE / f"gdelt_{series}.unavailable"
        if not refresh and marker.exists():
            # st_mtime is epoch (UTC); compare against a UTC clock, not local
            age = (pd.Timestamp.now(tz="UTC").tz_localize(None)
                   - pd.Timestamp(marker.stat().st_mtime, unit="s"))
            if age < pd.Timedelta(hours=3):
                log.info("Aux series %r skipped (marked unavailable %s ago)",
                         series, str(age).split(".")[0])
                continue
        try:
            if spec["source"] == "bigquery":
                from data.bigquery_gdelt import load_bq_daily
                out[series] = load_bq_daily(f"gdelt_{series}", spec["terms"],
                                            refresh)
            else:
                out[series] = load_gdelt_daily(refresh,
                                               name=f"gdelt_{series}",
                                               query=spec["query"])
            marker.unlink(missing_ok=True)
        except Exception as exc:
            log.warning("Aux series %r unavailable (%s)", series, exc)
            marker.touch()
    return out


# ---------------------------------------------------------------------------
# Live headline scrapers
# ---------------------------------------------------------------------------

def google_news_headlines(query: str = "NVIDIA stock", max_items: int = 50) -> list[dict]:
    import feedparser
    from urllib.parse import quote_plus
    url = (f"https://news.google.com/rss/search?q={quote_plus(query)}"
           "&hl=en-US&gl=US&ceid=US:en")
    feed = feedparser.parse(url)
    return [{"source": "google_news", "title": e.get("title", ""),
             "published": e.get("published", ""), "url": e.get("link", "")}
            for e in feed.entries[:max_items]]


def yahoo_finance_headlines(ticker: str = config.TICKER, max_items: int = 50) -> list[dict]:
    import feedparser
    url = (f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}"
           "&region=US&lang=en-US")
    feed = feedparser.parse(url)
    return [{"source": "yahoo_rss", "title": e.get("title", ""),
             "published": e.get("published", ""), "url": e.get("link", "")}
            for e in feed.entries[:max_items]]


def finviz_headlines(ticker: str = config.TICKER, max_items: int = 50) -> list[dict]:
    from bs4 import BeautifulSoup
    resp = session.get(f"https://finviz.com/quote.ashx?t={ticker}", timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", id="news-table")
    items: list[dict] = []
    if table is None:
        return items
    for row in table.find_all("tr")[:max_items]:
        link = row.find("a")
        if link is not None:
            first_td = row.find("td")
            items.append({"source": "finviz", "title": link.get_text(strip=True),
                          "published": first_td.get_text(strip=True) if first_td else "",
                          "url": link.get("href", "")})
    return items


def stocktwits_messages(ticker: str = config.TICKER, max_items: int = 30) -> list[dict]:
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    msgs = resp.json().get("messages", [])
    items = []
    for m in msgs[:max_items]:
        declared = (m.get("entities") or {}).get("sentiment") or {}
        items.append({"source": "stocktwits", "title": m.get("body", ""),
                      "published": m.get("created_at", ""),
                      "declared_sentiment": declared.get("basic")})
    return items


def reddit_headlines(query: str = config.TICKER, max_items: int = 40) -> list[dict]:
    """NOT in the default collector list: since Reddit's 2023 API lockdown the
    unauthenticated search.json endpoint returns 403 for non-browser clients.
    To use it, register a Reddit script app and swap this for an OAuth call."""
    url = ("https://www.reddit.com/r/stocks+investing+wallstreetbets+StockMarket/"
           f"search.json?q={query}&restrict_sr=1&sort=new&t=week&limit={max_items}")
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    posts = resp.json().get("data", {}).get("children", [])
    return [{"source": "reddit", "title": p["data"].get("title", ""),
             "published": "",
             "url": "https://reddit.com" + p["data"].get("permalink", "")}
            for p in posts]


def collect_live_headlines() -> list[dict]:
    """Pull fresh headlines from every source and de-duplicate by title."""
    collectors = [
        ("google_news", google_news_headlines),
        ("yahoo_rss", yahoo_finance_headlines),
        ("finviz", finviz_headlines),
        ("stocktwits", stocktwits_messages),
        # reddit_headlines omitted: needs OAuth since 2023 (see its docstring)
    ]
    items: list[dict] = []
    for name, fn in collectors:
        try:
            got = fn()
            log.info("%s: %d items", name, len(got))
            items.extend(got)
        except Exception as exc:
            log.warning("%s failed: %s", name, exc)
    seen: set[str] = set()
    unique: list[dict] = []
    for it in items:
        key = re.sub(r"\W+", " ", (it.get("title") or "").lower()).strip()[:120]
        if key and key not in seen:
            seen.add(key)
            unique.append(it)
    return unique
