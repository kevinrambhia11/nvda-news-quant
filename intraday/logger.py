"""Append-only timestamped headline logger.

Run `python main.py log-headlines` periodically (e.g. every 30-60 minutes
during US market hours, via Windows Task Scheduler) to accumulate the desk's
own high-resolution, timestamped news history - the raw material for future
intraday sentiment models that the free daily feeds cannot provide
retroactively. Each run appends new (deduplicated) headlines with a UTC
timestamp and a VADER score.
"""
from __future__ import annotations

import csv
import logging

import pandas as pd

import config

log = logging.getLogger(__name__)

_DEDUP_WINDOW = 3000  # compare against this many most-recent logged titles


def _norm(title: str) -> str:
    return " ".join((title or "").lower().split())[:150]


def log_once() -> int:
    """Scrape all live sources, score, append new rows. Returns rows added."""
    from data.news import collect_live_headlines
    from sentiment.analyzer import SentimentAnalyzer

    items = collect_live_headlines()
    if not items:
        log.warning("No headlines scraped; nothing logged")
        return 0
    scores = SentimentAnalyzer(prefer_finbert=False).score(
        [h.get("title", "") for h in items])

    seen: set[str] = set()
    if config.HEADLINE_LOG_PATH.exists():
        try:
            prev = pd.read_csv(config.HEADLINE_LOG_PATH,
                               usecols=["title"]).tail(_DEDUP_WINDOW)
            seen = {_norm(t) for t in prev["title"].astype(str)}
        except Exception as exc:
            log.warning("Could not read existing log for dedup (%s)", exc)

    now_utc = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    new_file = not config.HEADLINE_LOG_PATH.exists()
    added = 0
    with open(config.HEADLINE_LOG_PATH, "a", newline="",
              encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["logged_at_utc", "source", "score", "title"])
        for item, score in zip(items, scores):
            key = _norm(item.get("title", ""))
            if not key or key in seen:
                continue
            seen.add(key)
            writer.writerow([now_utc, item.get("source", ""),
                             round(float(score), 3),
                             " ".join(str(item.get("title", "")).split())])
            added += 1
    log.info("Logged %d new headlines -> %s", added, config.HEADLINE_LOG_PATH)
    return added
