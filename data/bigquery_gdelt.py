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


def _daily_sql(n_terms: int) -> str:
    likes = " OR ".join(f"LOWER(V2Organizations) LIKE @term{i}"
                        for i in range(n_terms))
    return f"""
        SELECT DATE(_PARTITIONTIME) AS date,
               COUNT(*) AS art_count,
               AVG(SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) AS tone
        FROM `{GKG_TABLE}`
        WHERE _PARTITIONTIME >= TIMESTAMP(@start)
          AND _PARTITIONTIME < TIMESTAMP(@end)
          AND ({likes})
        GROUP BY date
        ORDER BY date
    """


def daily_series(terms: list[str], start: str, end: str,
                 dry_run: bool = False,
                 cap_bytes: int = PROBE_CAP_BYTES):
    """Daily [tone, art_count] for records mentioning any of `terms`.
    dry_run=True returns the bytes the query WOULD scan, without running it."""
    bigquery, client = _client()
    params = ([bigquery.ScalarQueryParameter("start", "TIMESTAMP",
                                             pd.Timestamp(start, tz="UTC")),
               bigquery.ScalarQueryParameter("end", "TIMESTAMP",
                                             pd.Timestamp(end, tz="UTC"))]
              + [bigquery.ScalarQueryParameter(f"term{i}", "STRING",
                                               f"%{t.lower()}%")
                 for i, t in enumerate(terms)])
    job_config = bigquery.QueryJobConfig(query_parameters=params,
                                         dry_run=dry_run)
    if not dry_run:  # dry runs bill nothing; a None cap is rejected upstream
        job_config.maximum_bytes_billed = int(cap_bytes)
    job = client.query(_daily_sql(len(terms)), job_config=job_config)
    if dry_run:
        return int(job.total_bytes_processed)
    df = job.to_dataframe()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()[["tone", "art_count"]]
    log.info("BigQuery: %d daily rows for %s (%s -> %s), %.2f GB billed",
             len(df), terms[0], start, end,
             (job.total_bytes_billed or 0) / 1024 ** 3)
    return df


def load_bq_daily(name: str, terms: list[str],
                  refresh: bool = False) -> pd.DataFrame:
    """Cached BigQuery daily series; full rebuild once, tiny incremental
    top-ups thereafter (a few GB/month of the 1 TB quota)."""
    from data.news import atomic_to_csv
    cache_file = config.CACHE / f"{name}.csv"
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    if cache_file.exists() and not refresh:
        cached = pd.read_csv(cache_file, index_col="date", parse_dates=["date"])
        if not cached.empty:
            last = cached.index.max()
            if last >= today - pd.Timedelta(days=1):
                return cached
            fresh = daily_series(terms,
                                 str((last - pd.Timedelta(days=3)).date()),
                                 str((today + pd.Timedelta(days=1)).date()))
            merged = pd.concat([cached, fresh])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            atomic_to_csv(merged, cache_file)
            return merged
    df = daily_series(terms, config.TRAIN_START,
                      str((today + pd.Timedelta(days=1)).date()),
                      cap_bytes=DEFAULT_CAP_BYTES)
    if df.empty:
        raise RuntimeError(f"BigQuery returned no rows for {name} ({terms})")
    atomic_to_csv(df, cache_file)
    return df


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
