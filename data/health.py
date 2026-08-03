"""Desk health: run outcomes and data freshness, surfaced not buried.

Two silent-failure classes motivated this module:

1. A scheduled task dies and nothing says so. On 2026-07-21 a signal run
   failed and the desk simply showed the previous day's numbers; the only
   evidence was a line in scheduler.log nobody reads. Every `main.py`
   command now records its outcome here, and the dashboard turns the
   record into a banner - including on Streamlit Cloud, because the
   status file is committed (it holds no article text, only counters).

2. A stale cache masquerading as calm markets. The GDELT tone series has
   been blocked by this office network since 2026-07-13; downstream
   features fill missing days with 0.0, which the model reads as "a
   normal, quiet news day" rather than "no data". The fill stays (the
   model was trained with it), but the staleness is now stated on the
   signal and the dashboard, so a reader can discount it.

Deliberately dependency-light and never raising: health reporting must
not be able to break the pipeline it reports on.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime

import config

log = logging.getLogger(__name__)

STATUS_PATH = config.ARTIFACTS / "desk_status.json"
LOG_PATH = config.ARTIFACTS / "scheduler.log"
MAX_LOG_BYTES = 5_000_000

# Age (in days of data) past which a cache is called stale. Business-day
# aware thresholds are deliberate: prices roll every trading day, GDELT
# tone lands a day behind, earnings only change quarterly.
STALE_LIMITS = {
    "prices_NVDA.csv": 4,
    # The PRIMARY tone cache follows config.TONE_SOURCE. The other corpus is
    # deliberately NOT age-checked: the frozen DOC cache would otherwise
    # raise a permanent staleness alarm nobody can act on.
    config.TONE_CACHE_NAME: 4,
    "news2_features.csv": 4,
    "news2_daily.csv": 4,
    "earnings_NVDA.csv": 400,
}


def _read(path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _write_atomic(path, obj) -> None:
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        log.warning("status write failed (%s)", exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def record_run(stage: str, ok: bool, note: str | None = None,
               started: float | None = None) -> None:
    """Record one command's outcome. Never raises."""
    try:
        status = _read(STATUS_PATH)
        stages = status.setdefault("stages", {})
        prev = stages.get(stage, {})
        now = datetime.now().isoformat(timespec="seconds")
        entry = {
            "last_run": now,
            "ok": bool(ok),
            "note": (note or "")[:300] or None,
            "secs": round(time.monotonic() - started, 1) if started else None,
            "last_ok": now if ok else prev.get("last_ok"),
            # consecutive failures: one flaky evening is not an outage,
            # three in a row is
            "fails_since_ok": 0 if ok else int(prev.get("fails_since_ok", 0)) + 1,
        }
        stages[stage] = entry
        status["updated"] = now
        _write_atomic(STATUS_PATH, status)
    except Exception as exc:            # never break the caller
        log.warning("record_run failed (%s)", exc)


def data_warnings() -> list[str]:
    """Human-readable staleness warnings for the caches the desk reads.
    Empty list means everything is current. Never raises."""
    import pandas as pd
    out = []
    today = pd.Timestamp.now().normalize()
    for name, limit in STALE_LIMITS.items():
        path = config.CACHE / name
        if not path.exists():
            out.append(f"{name} is MISSING - features fall back to defaults")
            continue
        try:
            df = pd.read_csv(path, nrows=None, usecols=[0])
            col = df.columns[0]
            last = pd.to_datetime(df[col], errors="coerce").max()
            if pd.isna(last):
                continue
            age = int((today - last.normalize()).days)
            if age > limit:
                out.append(f"{name} last has {last.date()} ({age} days old)")
        except Exception:
            continue
    return out


def record_data_snapshot() -> list[str]:
    """Store the freshness warnings alongside the run record so the hosted
    dashboard can show them without the caches themselves."""
    warns = data_warnings()
    try:
        status = _read(STATUS_PATH)
        status["data_warnings"] = warns
        status["data_checked"] = datetime.now().isoformat(timespec="seconds")
        _write_atomic(STATUS_PATH, status)
    except Exception as exc:
        log.warning("data snapshot failed (%s)", exc)
    return warns


def rotate_log() -> None:
    """Keep scheduler.log bounded: three runs a day append forever, and an
    unbounded log is both a disk risk and unreadable when it matters."""
    try:
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > MAX_LOG_BYTES:
            prev = LOG_PATH.with_suffix(".log.1")
            try:
                prev.unlink(missing_ok=True)
            except OSError:
                pass
            LOG_PATH.replace(prev)
            log.info("scheduler.log rotated (kept as %s)", prev.name)
    except OSError as exc:
        log.warning("log rotation failed (%s)", exc)


def summarize(status: dict | None = None, now=None) -> dict:
    """Turn the raw record into display-ready alerts. Pure function so the
    dashboard can render it and tests can drive it.

    Returns {"level": ok|warn|alert, "alerts": [...], "stages": {...}}.
    """
    import pandas as pd
    status = status if status is not None else _read(STATUS_PATH)
    stages = status.get("stages", {}) or {}
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now()
    alerts, level = [], "ok"

    for stage, e in sorted(stages.items()):
        if not isinstance(e, dict):
            continue
        if not e.get("ok"):
            n = int(e.get("fails_since_ok", 1))
            last_ok = e.get("last_ok") or "never"
            alerts.append(f"`{stage}` failed "
                          + (f"{n} runs in a row" if n > 1 else "on its last run")
                          + f" (last success: {last_ok})"
                          + (f" - {e['note']}" if e.get("note") else ""))
            level = "alert" if n > 1 or level == "alert" else "warn"

    # The daily signal is the product: if it has not succeeded within the
    # last ~2 calendar days (weekend-tolerant), say so loudly.
    sig = stages.get("signal", {})
    if sig.get("last_ok"):
        try:
            age_h = (now - pd.Timestamp(sig["last_ok"])).total_seconds() / 3600
            grace = 96 if now.dayofweek in (5, 6, 0) else 48
            if age_h > grace:
                alerts.append(f"no successful `signal` run in "
                              f"{age_h / 24:.1f} days - the numbers on this "
                              "page are stale")
                level = "alert"
        except Exception:
            pass
    elif "signal" in stages:
        # attempted but never succeeded. A status file that simply has no
        # signal entry yet (fresh clone, or only other commands have run)
        # must stay silent - absence of evidence is not an outage.
        alerts.append("no successful `signal` run has ever been recorded")
        level = "alert"

    for w in status.get("data_warnings", []) or []:
        alerts.append(f"stale input: {w}")
        level = "alert" if level == "alert" else "warn"

    return {"level": level, "alerts": alerts, "stages": stages,
            "updated": status.get("updated")}
