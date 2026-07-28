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

# News-decay layer: the directional/magnitude LEVEL features carried
# forward with a fading half-life (config.DECAY_HALFLIVES). Dispersion,
# novelty and count-z features are not decayed (they are already
# windowed/relative). Judged against the same-day pulse in the tournament.
_DECAY_BASE = ["n2_nvda_dir", "n2_nvda_mag", "n2_macro_mag", "n2_hyper_dir",
               "n2_brokers_dir"]
NEWS2_DECAY_FEATURES = [f"{b}_d{int(h)}" for b in _DECAY_BASE
                        for h in config.DECAY_HALFLIVES]


# Article eligibility for FEATURE construction (the archive itself is never
# filtered). Three rules, from the data-quality audit:
#   form    - the slug must read like a headline: >=3 tokens with a real
#             word, not a bare article ID ("132451732.cms", hex strings);
#   anchor  - the slug must visibly relate to its category (GKG tags orgs
#             from article FULL TEXT, but our models only see the slug, so
#             an unanchored slug is pure noise to them - local-news weather
#             roundups were landing in hyperscalers this way);
#   dedup   - one copy per (entry day, slug): syndicated wire stories
#             otherwise overweight day means and suppress the conflict
#             feature, and duplicates surfaced in the brain's top-weights.
CATEGORY_ANCHORS = {
    "nvda": (r"nvidia|nvda|jensen|geforce|rtx|blackwell|hopper|rubin|cuda"
             r"|h100|h200|b200|gb200|\bgpu"),
    "competitors": (r"\bamd\b|advanced micro|\bintel\b|tsmc|taiwan semi"
                    r"|broadcom|qualcomm|semiconductor|chipmaker|\bchips?\b"),
    "hyperscalers": (r"microsoft|google|alphabet|amazon|\baws\b|\bmeta\b"
                     r"|facebook|oracle|azure|\bcloud\b|datacenter"
                     r"|data center|hyperscaler"),
    "ai_companies": (r"openai|anthropic|deepmind|chatgpt|claude|gemini"
                     r"|mistral|hugging face|\bai\b"
                     r"|artificial intelligence|\bllms?\b"),
    # macro and brokers are deliberately absent: they are THEME categories
    # whose relevant vocabulary cannot be enumerated (a central-bank
    # resignation in Jakarta is macro news with none of the obvious
    # keywords). They get the form and dedup rules only - an anchor list
    # tried here kept just 5% of macro, ~2.5 articles/day, too thin for
    # stable day-level features.
}


def informative_mask(art: pd.DataFrame) -> np.ndarray:
    """Boolean mask over `art` rows (which must carry `entry`) selecting
    articles worth reading for features. Position-aligned with `art`, so
    callers can subset the embedding matrix by the same positions."""
    slugs = art["slug"].fillna("").astype(str)
    formish = (slugs.str.count(" ").ge(2)
               & slugs.str.contains(r"[a-z]{3}", regex=True))
    # categories without an anchor list (macro, brokers) pass by default
    anchored = ~art["category"].isin(CATEGORY_ANCHORS)
    for cat, pat in CATEGORY_ANCHORS.items():
        m = (art["category"] == cat).to_numpy()
        if m.any():
            anchored[m] = slugs[m].str.contains(pat, regex=True)
    ok = (formish & anchored).to_numpy()
    # Dedup ONLY among eligible rows, and per (entry, CATEGORY, slug):
    #  - dedup over all rows would let a rejected first copy permanently
    #    shadow an eligible later copy (measured: 3,494 stories would
    #    vanish from every category);
    #  - a key without category would strip dual-tagged stories from all
    #    but their parquet-first category (measured: nvda would lose
    #    6,344 articles, mostly the NVDA-vs-competitor head-to-heads).
    #    One copy per category matches the archive's multi-tag semantics.
    keep = np.zeros(len(art), dtype=bool)
    pos = np.flatnonzero(ok)
    if len(pos):
        sub = art.iloc[pos]
        first = ~sub.assign(_slug=slugs.iloc[pos]).duplicated(
            subset=["entry", "category", "_slug"])
        keep[pos[first.to_numpy()]] = True
    return keep


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

    # trim to the embedding frontier BEFORE the informative mask, so a
    # not-yet-embedded copy can never shadow an embedded duplicate out of
    # the training set
    if len(emb) < len(art):
        art = art.iloc[: len(emb)].copy()
    art["entry"] = _entry_days(art["date"], px.index)
    art["fwd"] = art["entry"].map(fwd)
    mask = art["fwd"].notna() & (art["entry"] < holdout_start)
    mask &= pd.Series(informative_mask(art), index=art.index)
    X, y = emb[: len(art)][mask.to_numpy()], art.loc[mask, "fwd"].to_numpy()
    log.info("Impact training set: %d articles (< %s)", len(y),
             holdout_start.date())

    dir_model = Ridge(alpha=50.0).fit(X, y)
    mag_model = Ridge(alpha=50.0).fit(X, np.abs(y))
    bundle = {"dir": dir_model, "mag": mag_model,
              "trained_before": str(holdout_start.date()),
              "n_articles": int(len(y))}
    dest = out_path or (IMPACT_NESTED_PATH if nested else IMPACT_PATH)
    from data.news import atomic_dump
    atomic_dump(bundle, dest)
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
    inf = informative_mask(art)
    art = art.loc[inf].reset_index(drop=True)
    emb = emb[inf]
    log.info("Informative filter: %d of %d articles feed features",
             len(art), len(inf))

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

    # news-decay layer: fade each level feature forward (leak-free EWMA
    # over entry days). Built here so the nested, production and live
    # feature files all carry identical decay columns.
    from features.build import decay_ewma
    for base in _DECAY_BASE:
        if base in feats.columns:
            for h in config.DECAY_HALFLIVES:
                feats[f"{base}_d{int(h)}"] = decay_ewma(feats[base], h)

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
