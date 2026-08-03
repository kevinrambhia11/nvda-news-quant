"""GDELT via Google BigQuery (service-account path, no claude.ai connector).

The full GDELT archive is a public BigQuery dataset (`gdelt-bq.gdeltv2`).
With a service-account key in GOOGLE_APPLICATION_CREDENTIALS this module can
build the same daily tone/article-count series as the rate-limited DOC API -
instantly, from the office network.

Two important disciplines:
  * COST: the BigQuery sandbox allows 1 TB of scanned bytes per month. Every
    query here is dry-run priced first and hard-capped with
    maximum_bytes_billed, so a mistake cannot burn the quota.
  * CONSISTENCY: the GKG tables are a related-but-different corpus from the
    DOC API (organization-tagged records vs full-text search), so BigQuery
    series are cached under their own names and must not be spliced into
    DOC-API history. Switching source means rebuilding a series wholesale
    and retraining - `probe()` measures the cost of that before anyone
    commits to it.
"""
from __future__ import annotations

import logging
import os
import time

import pandas as pd

import config

log = logging.getLogger(__name__)

GKG_TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"
PROBE_CAP_BYTES = 60 * 1024 ** 3          # 60 GB ceiling for probe/sample runs
DEFAULT_CAP_BYTES = 900 * 1024 ** 3       # stay inside the 1 TB/month sandbox

# Organization-name fragments per series (lowercase substring match against
# GKG's V2Organizations field). Competitor terms live in config.AUX_SERIES.
ORG_TERMS = {
    "nvidia": ["nvidia"],
    "competitors": config.AUX_SERIES["competitors"]["terms"],
}


def _client():
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set - point it at the "
            "service-account JSON (stored outside the repo)")
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise RuntimeError("pip install google-cloud-bigquery") from exc
    return bigquery, bigquery.Client()


def _daily_sql(n_terms: int, with_domains: bool = False) -> str:
    likes = " OR ".join(f"LOWER(V2Organizations) LIKE @term{i}"
                        for i in range(n_terms))
    domain_filter = ("AND LOWER(SourceCommonName) IN UNNEST(@domains)"
                     if with_domains else "")
    return f"""
        SELECT DATE(_PARTITIONTIME) AS date,
               COUNT(*) AS art_count,
               AVG(SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) AS tone
        FROM `{GKG_TABLE}`
        WHERE _PARTITIONTIME >= TIMESTAMP(@start)
          AND _PARTITIONTIME < TIMESTAMP(@end)
          AND ({likes})
          {domain_filter}
        GROUP BY date
        ORDER BY date
    """


def daily_series(terms: list[str], start: str, end: str,
                 dry_run: bool = False,
                 cap_bytes: int = PROBE_CAP_BYTES,
                 domains: list[str] | None = None):
    """Daily [tone, art_count] for records mentioning any of `terms`,
    optionally restricted to a whitelist of source domains (the
    reliable-sources experiment). dry_run=True returns the bytes the query
    WOULD scan, without running it."""
    bigquery, client = _client()
    params = ([bigquery.ScalarQueryParameter("start", "TIMESTAMP",
                                             pd.Timestamp(start, tz="UTC")),
               bigquery.ScalarQueryParameter("end", "TIMESTAMP",
                                             pd.Timestamp(end, tz="UTC"))]
              + [bigquery.ScalarQueryParameter(f"term{i}", "STRING",
                                               f"%{t.lower()}%")
                 for i, t in enumerate(terms)])
    if domains:
        params.append(bigquery.ArrayQueryParameter(
            "domains", "STRING", [d.lower() for d in domains]))
    job_config = bigquery.QueryJobConfig(query_parameters=params,
                                         dry_run=dry_run)
    if not dry_run:  # dry runs bill nothing; a None cap is rejected upstream
        job_config.maximum_bytes_billed = int(cap_bytes)
    job = client.query(_daily_sql(len(terms), with_domains=bool(domains)),
                       job_config=job_config)
    if dry_run:
        return int(job.total_bytes_processed)
    df = job.to_dataframe()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()[["tone", "art_count"]]
    log.info("BigQuery: %d daily rows for %s (%s -> %s), %.2f GB billed",
             len(df), terms[0], start, end,
             (job.total_bytes_billed or 0) / 1024 ** 3)
    return df


# Units are GiB (matching total_bytes_billed / 1024^3). Google's sandbox
# ceiling ("free query bytes scanned") is nominally 1 TiB/month but floats
# in practice: on 2026-08-03 a 233 GiB rebuild was 403-rejected with only
# ~428 GiB tracked this month (July's overshoot appears to carry over).
# 700 keeps OUR guard tripping before Google's does in a normal month;
# quota rejections degrade to stale caches either way.
MONTH_BUDGET_GB = 700


def month_usage_gb() -> float:
    """Bytes billed this calendar month, from the project's job history."""
    _, client = _client()
    sql = ("SELECT IFNULL(SUM(total_bytes_billed), 0) / POW(1024, 3) AS gb "
           "FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT "
           "WHERE creation_time >= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), "
           "MONTH) AND job_type = 'QUERY'")
    return float(list(client.query(sql).result())[0].gb)


def load_bq_daily(name: str, terms: list[str], refresh: bool = False,
                  domains: list[str] | None = None,
                  allow_rebuild: bool = False) -> pd.DataFrame:
    """Cached BigQuery daily series; tiny incremental top-ups (a few GB a
    month). Top-up failures degrade to the stale cache.

    Full-history rebuilds run ONLY when allow_rebuild=True, i.e. from an
    explicit backfill command - never implicitly because a load path found
    the cache missing. Learned the expensive way on 2026-08-03: the boot
    catch-up ran the industry backfill, and two minutes later the GDELT
    top-up task's aux loader saw the not-yet-written cache and launched
    its own identical 230 GiB rebuild. A cross-process lock additionally
    serializes explicit rebuilds so even two backfill commands cannot
    double-bill."""
    from data.news import atomic_to_csv
    cache_file = config.CACHE / f"{name}.csv"
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    if cache_file.exists() and not refresh:
        cached = pd.read_csv(cache_file, index_col="date", parse_dates=["date"])
        if not cached.empty:
            last = cached.index.max()
            if last >= today - pd.Timedelta(days=1):
                return cached
            try:
                # end = today (exclusive in SQL): the current UTC partition
                # is still filling and must never be aggregated as a day.
                # Committed proof of the old behavior: every boundary row of
                # gdelt_competitors.csv was later revised 2.2-3.8x in count
                # (07-27 committed as 225 articles, final 503) - and the
                # freshness gate then served that half-day for a full day.
                fresh = daily_series(terms,
                                     str((last - pd.Timedelta(days=3)).date()),
                                     str(today.date()),
                                     domains=domains)
            except Exception as exc:
                log.warning("BigQuery top-up failed for %s (%s); serving "
                            "cache through %s", name, exc, last.date())
                return cached
            merged = pd.concat([cached, fresh])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            atomic_to_csv(merged, cache_file)
            return merged
    if not allow_rebuild:
        # refuse BEFORE any query (even a dry-run): absent caches on load
        # paths are a degrade-and-report situation, not a spend trigger
        raise RuntimeError(
            f"{name} cache is missing and full rebuilds only run from an "
            f"explicit backfill command (main.py industry-backfill etc.) - "
            f"a load path must never spend hundreds of GiB implicitly")
    lock = config.CACHE / f"{name}.rebuild.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            stale = time.time() - lock.stat().st_mtime > 3600
        except OSError:
            stale = False
        if not stale:
            raise RuntimeError(
                f"a rebuild of {name} is already running in another "
                f"process (remove {lock.name} if that process is dead)")
        lock.unlink()
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        end = str(today.date())   # exclusive: complete UTC partitions only
        cost_gb = daily_series(terms, config.TRAIN_START, end, dry_run=True,
                               domains=domains) / 1024 ** 3
        used_gb = month_usage_gb()
        if used_gb + cost_gb > MONTH_BUDGET_GB:
            raise RuntimeError(
                f"full rebuild of {name} needs {cost_gb:.0f} GiB with "
                f"{used_gb:.0f} GiB already billed this month "
                f"(budget {MONTH_BUDGET_GB}) - deferred to the next reset")
        df = daily_series(terms, config.TRAIN_START, end,
                          cap_bytes=DEFAULT_CAP_BYTES, domains=domains)
        if df.empty:
            raise RuntimeError(
                f"BigQuery returned no rows for {name} ({terms})")
        atomic_to_csv(df, cache_file)
        return df
    finally:
        try:
            os.close(fd)
            lock.unlink()
        except OSError:
            pass


def probe() -> str:
    """Validate the credential end-to-end and price the big rebuilds:
    a 7-day sample query plus dry-run costs for full-history rebuilds."""
    lines = ["=" * 62, "  BigQuery GDELT probe", "=" * 62]
    end = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    wk_start, wk_end = str((end - pd.Timedelta(days=8)).date()), str(
        (end - pd.Timedelta(days=1)).date())

    sample_cost = daily_series(ORG_TERMS["nvidia"], wk_start, wk_end,
                               dry_run=True)
    lines.append(f"  7-day NVIDIA sample would scan: "
                 f"{sample_cost / 1024 ** 3:.2f} GB")
    sample = daily_series(ORG_TERMS["nvidia"], wk_start, wk_end)
    lines.append(f"  sample rows: {len(sample)}  (tone range "
                 f"{sample['tone'].min():+.2f} .. {sample['tone'].max():+.2f}, "
                 f"count range {int(sample['art_count'].min())} .. "
                 f"{int(sample['art_count'].max())})")

    for name, terms in ORG_TERMS.items():
        full_cost = daily_series(terms, config.TRAIN_START, str(end.date()),
                                 dry_run=True)
        lines.append(f"  full 2017->now rebuild [{name}]: "
                     f"{full_cost / 1024 ** 3:.1f} GB "
                     f"({full_cost / (1024 ** 4) * 100:.1f}% of the 1 TB/month "
                     f"sandbox quota)")
    lines += ["-" * 62,
              "  Next step if costs fit: rebuild series wholesale from",
              "  BigQuery (own cache names), compare against the DOC-API",
              "  series, then retrain before switching the live source.",
              "=" * 62]
    return "\n".join(lines)
