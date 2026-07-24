"""Article-level news intelligence: embeddings, learned impact, daily features.

Pipeline (all local, all free):
  1. embed_articles()  - encode 785k headline-slugs (6 categories, 2017->now)
     with a sentence-transformer into a float16 matrix on disk.
  2. learn_impact()    - supervised impact scorers: ridge regressions from
     embedding space onto NVDA's next-day return (direction) and absolute
     return (magnitude), trained ONLY on articles before the tournament's
     holdout boundary. Frozen thereafter, so holdout verdicts stay honest.
     (Selection-window numbers for news2 candidates are consequently
     optimistic - the holdout is the only honest read, as always.)
  3. build_daily_features() - per entry day: impact-weighted category scores,
     conflict (disagreement between same-day articles), novelty (semantic
     distance from the trailing month), tone dispersion, volume anomalies.

Timing discipline: an article published on calendar day D feeds the first
trading day strictly AFTER D - identical to the tone pipeline's d-1 cutoff.
"""
from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

ART_PATH = config.CACHE / "news2_articles.parquet"
DAILY_PATH = config.CACHE / "news2_daily.csv"
EMB_PATH = config.CACHE / "news2_emb.npy"
IMPACT_PATH = config.ARTIFACTS / "news2_impact.joblib"
FEATURES_PATH = config.CACHE / "news2_features.csv"
# Nested variant: scorers trained only on the early selection era, so the
# late selection era is clean judging ground (see config.NESTED_FRACTION).
IMPACT_NESTED_PATH = config.ARTIFACTS / "news2_impact_nested.joblib"
FEATURES_NESTED_PATH = config.CACHE / "news2_features_nested.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

NEWS2_FEATURES = [
    "n2_nvda_dir", "n2_nvda_mag", "n2_nvda_conflict", "n2_nvda_novelty",
    "n2_macro_mag", "n2_macro_tone_sd", "n2_hyper_dir", "n2_brokers_dir",
    "n2_cross_divergence", "n2_nvda_count_z", "n2_macro_count_z",
]


def embed_articles(batch_size: int = 512) -> None:
    """One-time (then incremental) encoding of article slugs to vectors."""
    from sentence_transformers import SentenceTransformer
    art = pd.read_parquet(ART_PATH)
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    log.info("Embedding %d slugs with %s ...", len(art), MODEL_NAME)
    emb = model.encode(art["slug"].tolist(), batch_size=batch_size,
                       show_progress_bar=True, normalize_embeddings=True)
    np.save(EMB_PATH, emb.astype(np.float16))
    log.info("Saved %s embeddings -> %s", emb.shape, EMB_PATH)


def embed_new(batch_size: int = 512) -> int:
    """Encode only the articles appended since the last embedding run and
    extend the matrix on disk. Relies on the archive's append-only order
    invariant (embedding row i == parquet row i, forever)."""
    import os
    art = pd.read_parquet(ART_PATH)
    emb = np.load(EMB_PATH)
    if len(emb) >= len(art):
        log.info("Embeddings already cover all %d articles", len(art))
        return 0
    from sentence_transformers import SentenceTransformer
    new_slugs = art["slug"].iloc[len(emb):].tolist()
    log.info("Embedding %d new slugs (%d -> %d) ...", len(new_slugs),
             len(emb), len(art))
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    add = model.encode(new_slugs, batch_size=batch_size,
                       show_progress_bar=False, normalize_embeddings=True)
    merged = np.concatenate([emb, add.astype(emb.dtype)])
    # np.save appends ".npy" to any path not already ending in it - name
    # the temp file with the suffix so the atomic replace targets reality
    tmp = EMB_PATH.with_name("news2_emb_tmp.npy")
    np.save(tmp, merged)
    os.replace(f"{tmp}", EMB_PATH)
    return len(new_slugs)


def _entry_days(article_dates: pd.Series, trading_index: pd.DatetimeIndex):
    """Map each article's calendar day D to the first trading day > D."""
    pos = trading_index.searchsorted(article_dates.to_numpy(), side="right")
    valid = pos < len(trading_index)
    entry = pd.Series(pd.NaT, index=article_dates.index)
    entry[valid] = trading_index[pos[valid]]
    return entry


def _entry_index(px_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Price index extended by the NEXT trading session. Without this, no
    article dated on/after the last cached close can map to any entry day,
    so the feature files would never contain the row the live signal
    predicts on (permanent train/serve skew for news-candidate models)."""
    from trade.calendar import next_trading_day
    return px_index.append(
        pd.DatetimeIndex([next_trading_day(px_index.max())]))


def learn_impact(holdout_start: pd.Timestamp, nested: bool = False,
                 out_path=None) -> dict:
    """Ridge scorers from embedding space onto next-day signed and absolute
    NVDA returns, trained strictly before `holdout_start`. `out_path`
    overrides the artifact location (used by the LIVE daily-learning layer,
    which trains through yesterday and must never touch evaluation files).
    """
    from sklearn.linear_model import Ridge
    from data.prices import load_prices

    art = pd.read_parquet(ART_PATH)
    emb = np.load(EMB_PATH).astype(np.float32)
    px, _ = load_prices()
    fwd = (px["Open"].shift(-1) / px["Open"] - 1)

    art["entry"] = _entry_days(art["date"], px.index)
    art["fwd"] = art["entry"].map(fwd)
    mask = art["fwd"].notna() & (art["entry"] < holdout_start)
    # embeddings may lag a fresh parquet - only rows with a vector count
    mask &= art.index < len(emb)
    X, y = emb[mask.to_numpy()[: len(emb)]], art.loc[mask, "fwd"].to_numpy()
    log.info("Impact training set: %d articles (< %s)", len(y),
             holdout_start.date())

    dir_model = Ridge(alpha=50.0).fit(X, y)
    mag_model = Ridge(alpha=50.0).fit(X, np.abs(y))
    bundle = {"dir": dir_model, "mag": mag_model,
              "trained_before": str(holdout_start.date()),
              "n_articles": int(len(y))}
    dest = out_path or (IMPACT_NESTED_PATH if nested else IMPACT_PATH)
    joblib.dump(bundle, dest)
    return bundle


def build_daily_features(nested: bool = False, impact_path=None,
                         out_path=None) -> pd.DataFrame:
    """Apply the frozen impact scorers to every article and aggregate to
    entry-day features. Writes news2_features.csv and returns it. Path
    overrides route the LIVE daily-learning layer to its own files."""
    from data.prices import load_prices

    art = pd.read_parquet(ART_PATH)
    emb = np.load(EMB_PATH).astype(np.float32)
    bundle = joblib.load(impact_path
                         or (IMPACT_NESTED_PATH if nested else IMPACT_PATH))
    px, _ = load_prices()
    entry_idx = _entry_index(px.index)

    art["entry"] = _entry_days(art["date"], entry_idx)
    art = art[art["entry"].notna()].copy()
    keep = art.index.to_numpy()
    keep = keep[keep < len(emb)]  # embeddings may lag a fresh parquet
    art = art.loc[keep]
    emb = emb[keep]
    art = art.reset_index(drop=True)

    art["dir_score"] = bundle["dir"].predict(emb)
    art["mag_score"] = bundle["mag"].predict(emb)

    nvda = art[art["category"] == "nvda"]
    g = nvda.groupby("entry")
    feats = pd.DataFrame({
        "n2_nvda_dir": g["dir_score"].mean(),
        "n2_nvda_mag": g["mag_score"].mean(),
        "n2_nvda_conflict": g["dir_score"].std(),
    })

    # novelty: cosine distance of today's NVDA centroid vs trailing 30 days
    cent = pd.DataFrame(emb[nvda.index.to_numpy()], index=nvda["entry"]) \
        .groupby(level=0).mean()
    trail = cent.rolling(30, min_periods=10).mean().shift(1)
    num = (cent * trail).sum(axis=1)
    den = np.sqrt((cent ** 2).sum(axis=1) * (trail ** 2).sum(axis=1))
    feats["n2_nvda_novelty"] = 1 - (num / den)

    macro = art[art["category"] == "macro"].groupby("entry")
    feats["n2_macro_mag"] = macro["mag_score"].mean()
    hyper = art[art["category"] == "hyperscalers"].groupby("entry")
    feats["n2_hyper_dir"] = hyper["dir_score"].mean()
    brok = art[art["category"] == "brokers"].groupby("entry")
    feats["n2_brokers_dir"] = brok["dir_score"].mean()

    # exact daily aggregates: dispersion + volume anomalies + divergence
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    daily["entry"] = _entry_days(daily["date"], entry_idx)
    dv = daily.pivot_table(index="entry", columns="category",
                           values=["day_tone", "day_tone_sd", "day_n"])
    feats["n2_macro_tone_sd"] = dv[("day_tone_sd", "macro")]
    feats["n2_cross_divergence"] = dv["day_tone"].std(axis=1)
    for cat, col in (("nvda", "n2_nvda_count_z"),
                     ("macro", "n2_macro_count_z")):
        n = dv[("day_n", cat)]
        roll = n.rolling(60, min_periods=20)
        feats[col] = (n - roll.mean()) / roll.std()

    feats.index.name = "date"
    feats = feats.sort_index()
    from data.news import atomic_to_csv
    dest = out_path or (FEATURES_NESTED_PATH if nested else FEATURES_PATH)
    atomic_to_csv(feats, dest)
    log.info("news2 daily features: %d rows x %d cols -> %s",
             len(feats), feats.shape[1], dest.name)
    return feats


def load_news2_features(nested: bool = False) -> pd.DataFrame | None:
    path = FEATURES_NESTED_PATH if nested else FEATURES_PATH
    if not path.exists():
        return None
    return pd.read_csv(path, index_col="date", parse_dates=["date"])
