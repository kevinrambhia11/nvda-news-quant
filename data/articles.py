"""Incremental top-up of the article-level news corpus from BigQuery.

The original 707k-article archive (news2_articles.parquet + exact per-day
aggregates in news2_daily.csv) was pulled from the GDELT GKG in one unified
scan: for each of six categories, up to 60 sampled articles per day plus
full-corpus daily count/tone aggregates. This module reproduces that pull
INCREMENTALLY so the live news features stay fresh: each run fetches only
the days missing from the parquet (partition-pruned - a few hundred MB per
day of quota), appends, extends the embedding matrix, and rebuilds the
production feature files.

Category definitions were reconstructed from the archive's own fingerprint
(per-day volumes, tone profiles, multi-category membership, the 60/day cap)
and validated by re-pulling days the archive already covers and comparing
distributions - see validate_overlap(). An article may match several
categories (16k slugs in the archive do).

ORDER INVARIANT: new rows are strictly APPENDED to the parquet, never
re-sorted - embedding row i must forever correspond to parquet row i.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

TOPUP_CAP_BYTES = 60 * 1024 ** 3   # hard ceiling per top-up run
PER_DAY_CAP = 60                   # matches the original archive's sampling

# SQL boolean conditions per category, over lowercased V2Organizations
# (`orgs`) and raw V2Themes (`themes`).
CATEGORY_CONDITIONS = {
    "nvda": "orgs LIKE '%nvidia%'",
    "competitors": ("orgs LIKE '%advanced micro devices%'"
                    " OR orgs LIKE '%intel corp%'"
                    " OR orgs LIKE '%taiwan semiconductor%'"
                    " OR orgs LIKE '%broadcom%'"
                    " OR orgs LIKE '%qualcomm%'"),
    "hyperscalers": ("orgs LIKE '%microsoft%'"
                     " OR orgs LIKE '%google%'"
                     " OR orgs LIKE '%amazon%'"
                     " OR orgs LIKE '%meta platforms%'"
                     " OR orgs LIKE '%oracle corp%'"),
    "ai_companies": ("orgs LIKE '%openai%'"
                     " OR orgs LIKE '%anthropic%'"
                     " OR orgs LIKE '%deepmind%'"
                     " OR orgs LIKE '%hugging face%'"
                     " OR orgs LIKE '%mistral ai%'"),
    "brokers": ("orgs LIKE '%goldman sachs%'"
                " OR orgs LIKE '%morgan stanley%'"
                " OR orgs LIKE '%jpmorgan%'"
                " OR orgs LIKE '%bank of america%'"
                " OR orgs LIKE '%citigroup%'"
                " OR orgs LIKE '%wells fargo%'"
                " OR orgs LIKE '%deutsche bank%'"),
    "macro": ("themes LIKE '%TARIFF%'"
              " OR themes LIKE '%SANCTION%'"
              " OR themes LIKE '%ECON_CENTRALBANK%'"),
}


def _base_cte() -> str:
    return f"""
    WITH matched AS (
      SELECT DATE(_PARTITIONTIME) AS date,
             LOWER(SourceCommonName) AS source,
             DocumentIdentifier AS url,
             SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS tone,
             LOWER(IFNULL(V2Organizations, '')) AS orgs,
             IFNULL(V2Themes, '') AS themes
      FROM `gdelt-bq.gdeltv2.gkg_partitioned`
      WHERE _PARTITIONTIME >= TIMESTAMP(@start)
        AND _PARTITIONTIME < TIMESTAMP(@end)
        AND SourceCommonName IS NOT NULL
    ),
    labeled AS (
      SELECT m.*, category
      FROM matched m
      CROSS JOIN UNNEST([
        {", ".join(f"IF({cond}, '{cat}', NULL)"
                   for cat, cond in CATEGORY_CONDITIONS.items())}
      ]) AS category
      WHERE category IS NOT NULL AND m.tone IS NOT NULL
    )"""


def _articles_sql() -> str:
    # slug: last URL path segment, separators -> spaces, lowercased -
    # matching the archive's existing slug format
    return _base_cte() + """
    , slugged AS (
      SELECT date, category, source, tone,
             LOWER(REGEXP_REPLACE(
                 REGEXP_EXTRACT(url, r'/([^/?#]+)/?(?:[?#].*)?$'),
                 r"[-_']", ' ')) AS slug,
             url
      FROM labeled
    ),
    deduped AS (
      SELECT *,
             ROW_NUMBER() OVER (PARTITION BY category, date, url
                                ORDER BY url) AS dup_rank
      FROM slugged
      WHERE slug IS NOT NULL AND LENGTH(slug) >= 1
    ),
    -- sampling prefers headline-like (multi-word) slugs and falls back to
    -- ID-style slugs only when the day runs short - matching the archive's
    -- observed ~0-6% single-token rate without excluding sparse days
    sampled AS (
      SELECT date, category, source, slug, tone,
             ROW_NUMBER() OVER (
                 PARTITION BY category, date
                 ORDER BY IF(STRPOS(slug, ' ') > 0, 0, 1),
                          FARM_FINGERPRINT(url)) AS rk
      FROM deduped WHERE dup_rank = 1
    )
    SELECT date, category, source, slug, tone
    FROM sampled WHERE rk <= @per_day_cap
    ORDER BY date, category, slug
    """


def _agg_sql() -> str:
    return _base_cte() + """
    SELECT date, category,
           COUNT(*) AS day_n,
           AVG(tone) AS day_tone,
           STDDEV(tone) AS day_tone_sd
    FROM labeled
    GROUP BY date, category
    ORDER BY date, category
    """


def _run(sql: str, start: str, end: str, dry_run: bool = False):
    from data.bigquery_gdelt import _client
    bigquery, client = _client()
    params = [
        bigquery.ScalarQueryParameter("start", "TIMESTAMP",
                                      pd.Timestamp(start, tz="UTC")),
        bigquery.ScalarQueryParameter("end", "TIMESTAMP",
                                      pd.Timestamp(end, tz="UTC")),
        bigquery.ScalarQueryParameter("per_day_cap", "INT64", PER_DAY_CAP),
    ]
    job_config = bigquery.QueryJobConfig(query_parameters=params,
                                         dry_run=dry_run)
    if not dry_run:
        job_config.maximum_bytes_billed = TOPUP_CAP_BYTES
    job = client.query(sql, job_config=job_config)
    if dry_run:
        return int(job.total_bytes_processed)
    df = job.to_dataframe()
    log.info("BigQuery article query: %d rows (%s -> %s), %.2f GB billed",
             len(df), start, end, (job.total_bytes_billed or 0) / 1024 ** 3)
    return df


def fetch_window(start: str, end: str):
    """(sampled articles, exact daily aggregates) for [start, end)."""
    cost = _run(_articles_sql(), start, end, dry_run=True) \
        + _run(_agg_sql(), start, end, dry_run=True)
    log.info("Top-up window %s -> %s will scan %.2f GB", start, end,
             cost / 1024 ** 3)
    if cost > TOPUP_CAP_BYTES:
        raise RuntimeError(
            f"Top-up would scan {cost / 1024 ** 3:.0f} GB "
            f"(> {TOPUP_CAP_BYTES / 1024 ** 3:.0f} GB cap) - run in "
            "smaller windows")
    art = _run(_articles_sql(), start, end)
    agg = _run(_agg_sql(), start, end)
    for df in (art, agg):
        df["date"] = pd.to_datetime(df["date"])
    return art, agg


def topup() -> dict:
    """Fetch the days missing from the archive, append (never re-sort),
    extend embeddings, rebuild production news features. Returns a summary
    dict; raises on failure (callers decide whether that is fatal)."""
    from model.news2vec import ART_PATH, DAILY_PATH

    art = pd.read_parquet(ART_PATH)
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    # start = the OLDER of the two files' coverage (a crash between the two
    # writes must be healed by re-pulling, never leave a permanent hole),
    # re-pulling the last archived day too: it may have been mid-day
    start = min(art["date"].max(), daily["date"].max()).normalize()
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    if start >= end:
        log.info("Article archive already current through %s", start.date())
        return {"new_articles": 0, "through": str(start.date())}

    fresh_art, fresh_agg = fetch_window(str(start.date()), str(end.date()))

    # Exact daily aggregates FIRST: this write is idempotent (days >= start
    # are replaced wholesale), so a crash before the parquet write just
    # repeats the same window next run. The parquet write moves `start`
    # forward, so it must come last.
    daily = daily[daily["date"] < start]
    daily = pd.concat([daily, fresh_agg], ignore_index=True) \
        .sort_values(["date", "category"])
    tmp = DAILY_PATH.with_suffix(".csv.tmp")
    daily.to_csv(tmp, index=False)
    os.replace(tmp, DAILY_PATH)

    # dedup against the archive on (date, category, source, slug)
    key_cols = ["date", "category", "source", "slug"]
    existing = set(map(tuple, art.loc[art["date"] >= start, key_cols]
                       .astype(str).to_numpy()))
    fresh_keys = fresh_art[key_cols].astype(str).apply(tuple, axis=1)
    fresh_art = fresh_art.loc[~fresh_keys.isin(existing)]

    if len(fresh_art):
        fresh_art = fresh_art[art.columns.tolist()]
        merged = pd.concat([art, fresh_art], ignore_index=True)
        merged["date"] = merged["date"].astype("datetime64[ms]")
        tmp = ART_PATH.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp, index=False)
        os.replace(tmp, ART_PATH)
        log.info("Archive: +%d articles -> %d total (through %s)",
                 len(fresh_art), len(merged),
                 merged["date"].max().date())

    from model.news2vec import embed_new
    n_emb = embed_new()

    # rebuild production features with the frozen scorers (fast: predict
    # + net forward only). Nested features refresh on the weekly retrain.
    from model.news2vec import build_daily_features
    from model.newsnet import build_features
    build_daily_features()
    try:
        build_features()
    except Exception as exc:
        log.warning("newsnet feature rebuild failed (%s) - news2 features "
                    "are current, nn_* stay at previous coverage", exc)
    return {"new_articles": int(len(fresh_art)), "embedded": int(n_emb),
            "through": str(end.date())}


def validate_overlap(days: int = 4) -> str:
    """Re-pull a window the archive already covers and compare the
    reconstruction against ground truth: exact daily counts/tone vs
    news2_daily.csv, sampled volume vs the parquet, slug-format match rate.
    """
    from model.news2vec import ART_PATH, DAILY_PATH
    art = pd.read_parquet(ART_PATH)
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    end = art["date"].max().normalize()
    start = end - pd.Timedelta(days=days)
    new_art, new_agg = fetch_window(str(start.date()), str(end.date()))

    win = (daily["date"] >= start) & (daily["date"] < end)
    old_agg = daily.loc[win].groupby("category").agg(
        n=("day_n", "mean"), tone=("day_tone", "mean"))
    got_agg = new_agg.groupby("category").agg(
        n=("day_n", "mean"), tone=("day_tone", "mean"))
    aw = (art["date"] >= start) & (art["date"] < end)
    old_n = art.loc[aw].groupby("category").size()
    new_n = new_art.groupby("category").size()

    old_slugs = set(art.loc[aw & (art["category"] == "nvda"), "slug"])
    new_slugs = set(new_art.loc[new_art["category"] == "nvda", "slug"])
    overlap = len(old_slugs & new_slugs) / max(1, len(old_slugs))

    lines = [f"  overlap window {start.date()} -> {end.date()}",
             f"  {'category':<14}{'day_n old':>10}{'day_n new':>10}"
             f"{'tone old':>10}{'tone new':>10}{'arts old':>9}{'arts new':>9}"]
    for cat in sorted(CATEGORY_CONDITIONS):
        oa = old_agg.loc[cat] if cat in old_agg.index else None
        ga = got_agg.loc[cat] if cat in got_agg.index else None
        lines.append(
            f"  {cat:<14}"
            f"{(oa['n'] if oa is not None else float('nan')):>10.0f}"
            f"{(ga['n'] if ga is not None else float('nan')):>10.0f}"
            f"{(oa['tone'] if oa is not None else float('nan')):>10.2f}"
            f"{(ga['tone'] if ga is not None else float('nan')):>10.2f}"
            f"{old_n.get(cat, 0):>9}{new_n.get(cat, 0):>9}")
    lines.append(f"  nvda slug exact-match rate vs archive: {overlap:.0%}")
    return "\n".join(lines)
