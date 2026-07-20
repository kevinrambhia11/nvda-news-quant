"""NewsNet: a neural network over the news itself.

Architecture (all trained locally, CPU):
  frozen sentence-encoder (MiniLM transformer, 384-d vectors per headline)
    -> per-article projection + learned SOURCE embedding + CATEGORY embedding
    -> day-level ATTENTION POOLING (the learned per-article weights: which
       news mattered today)
    -> two heads: next-day signed return (direction) and |return| (magnitude)

What falls out, per entry day, as features for the tournaments:
  nn_dir        day-level direction score
  nn_mag        day-level magnitude score
  nn_conflict   std of per-article direction predictions (3-4 conflicting
                stories -> high value)
  nn_attn_ent   attention entropy (one dominant story vs diffuse chatter)
  nn_top_share  largest single-article attention weight

And per source, the learned answer to "which sources move the stock":
  artifacts/newsnet_sources.csv - mean |attention x contribution| by outlet.

Honesty: trained ONLY on entry days before the tournament holdout boundary
(last 15% of that as time-ordered validation for early stopping). Applied
frozen everywhere else. Selection-window numbers for nn features are
therefore optimistic; holdout columns are the verdict.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config
from model.news2vec import ART_PATH, EMB_PATH, _entry_days

log = logging.getLogger(__name__)

NET_PATH = config.ARTIFACTS / "newsnet.pt"
FEATURES_PATH = config.CACHE / "newsnet_features.csv"
SOURCES_PATH = config.ARTIFACTS / "newsnet_sources.csv"

NEWSNET_FEATURES = ["nn_dir", "nn_mag", "nn_conflict", "nn_attn_ent",
                    "nn_top_share"]

CATS = ["nvda", "competitors", "hyperscalers", "ai_companies", "macro",
        "brokers"]
N_SRC_BUCKETS = 512
DIM = 128
MAX_ARTICLES = 400
RET_SCALE = 25.0  # ~2-4% daily moves -> O(1) targets


def _torch():
    import torch
    torch.manual_seed(42)
    return torch


def _build_model(torch):
    import torch.nn as nn

    class NewsNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(nn.Linear(384, DIM), nn.GELU(),
                                      nn.LayerNorm(DIM))
            self.src_emb = nn.Embedding(N_SRC_BUCKETS, DIM)
            self.cat_emb = nn.Embedding(len(CATS), DIM)
            self.attn = nn.Sequential(nn.Linear(DIM, 64), nn.Tanh(),
                                      nn.Linear(64, 1))
            self.head_dir = nn.Linear(DIM, 1)
            self.head_mag = nn.Linear(DIM, 1)

        def forward(self, emb, src, cat):
            x = self.proj(emb) + self.src_emb(src) + self.cat_emb(cat)
            a = self.attn(x).squeeze(-1)
            w = a.softmax(dim=0)
            day = (w.unsqueeze(-1) * x).sum(dim=0)
            art_dir = self.head_dir(x).squeeze(-1)  # per-article direction
            return (self.head_dir(day).squeeze(-1),
                    self.head_mag(day).squeeze(-1), w, art_dir)

    return NewsNet()


def _day_tensors(torch, art, emb):
    """Group articles by entry day -> list of (day, emb_t, src_t, cat_t)."""
    src_ids = (art["source"].fillna("?")
               .map(lambda s: hash(s) % N_SRC_BUCKETS).to_numpy())
    cat_ids = art["category"].map({c: i for i, c in enumerate(CATS)}).to_numpy()
    days = []
    for day, idx in art.groupby("entry").indices.items():
        if len(idx) > MAX_ARTICLES:
            idx = np.random.default_rng(7).choice(idx, MAX_ARTICLES,
                                                  replace=False)
        days.append((pd.Timestamp(day),
                     torch.tensor(emb[idx], dtype=torch.float32),
                     torch.tensor(src_ids[idx], dtype=torch.long),
                     torch.tensor(cat_ids[idx], dtype=torch.long)))
    days.sort(key=lambda t: t[0])
    return days


def train(holdout_start: pd.Timestamp, epochs: int = 60) -> None:
    torch = _torch()
    from data.prices import load_prices

    art = pd.read_parquet(ART_PATH)
    emb = np.load(EMB_PATH).astype(np.float32)
    px, _ = load_prices()
    fwd = (px["Open"].shift(-1) / px["Open"] - 1)
    art["entry"] = _entry_days(art["date"], px.index)
    art = art[art["entry"].notna()].reset_index(drop=True)
    emb = emb[: len(art)] if len(emb) > len(art) else emb

    days = _day_tensors(torch, art, emb)
    labeled = [(d, e, s, c, fwd.get(d)) for d, e, s, c in days
               if d < holdout_start and pd.notna(fwd.get(d))]
    n_val = max(50, int(len(labeled) * 0.15))
    train_days, val_days = labeled[:-n_val], labeled[-n_val:]
    log.info("NewsNet: %d train days, %d val days (< %s)",
             len(train_days), len(val_days), holdout_start.date())

    model = _build_model(torch)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    huber = torch.nn.SmoothL1Loss()
    best, best_state, patience = np.inf, None, 0
    rng = np.random.default_rng(42)

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_days))
        for i in order:
            d, e, s, c, y = train_days[i]
            yt = torch.tensor(float(y) * RET_SCALE)
            dir_p, mag_p, _, _ = model(e, s, c)
            loss = huber(dir_p, yt) + huber(mag_p, yt.abs())
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(np.mean([
                (huber(model(e, s, c)[0], torch.tensor(float(y) * RET_SCALE))
                 + huber(model(e, s, c)[1],
                         torch.tensor(abs(float(y)) * RET_SCALE))).item()
                for d, e, s, c, y in val_days]))
        if vloss < best - 1e-4:
            best, best_state, patience = vloss, model.state_dict(), 0
        else:
            patience += 1
        if epoch % 10 == 0:
            log.info("epoch %d | val %.4f | best %.4f", epoch, vloss, best)
        if patience >= 8:
            log.info("early stop at epoch %d", epoch)
            break

    model.load_state_dict(best_state)
    torch.save({"state": best_state,
                "trained_before": str(holdout_start.date())}, NET_PATH)
    log.info("NewsNet saved -> %s (val %.4f)", NET_PATH.name, best)


def build_features() -> pd.DataFrame:
    """Apply the frozen net to every day; emit daily features + the learned
    source-impact table."""
    torch = _torch()
    from data.news import atomic_to_csv
    from data.prices import load_prices

    art = pd.read_parquet(ART_PATH)
    emb = np.load(EMB_PATH).astype(np.float32)
    px, _ = load_prices()
    art["entry"] = _entry_days(art["date"], px.index)
    art = art[art["entry"].notna()].reset_index(drop=True)
    emb = emb[: len(art)] if len(emb) > len(art) else emb

    model = _build_model(torch)
    model.load_state_dict(torch.load(NET_PATH)["state"])
    model.eval()

    rows, src_rows = [], []
    with torch.no_grad():
        for day, e, s, c in _day_tensors(torch, art, emb):
            dir_p, mag_p, w, art_dir = model(e, s, c)
            wn = w.numpy()
            rows.append({"date": day,
                         "nn_dir": float(dir_p) / RET_SCALE,
                         "nn_mag": float(mag_p) / RET_SCALE,
                         "nn_conflict": float(art_dir.std()) / RET_SCALE,
                         "nn_attn_ent": float(-(wn * np.log(wn + 1e-9)).sum()),
                         "nn_top_share": float(wn.max())})
            src_rows.append(pd.DataFrame(
                {"w_contrib": wn * np.abs(art_dir.numpy())},
                index=s.numpy()))
    feats = pd.DataFrame(rows).set_index("date").sort_index()
    atomic_to_csv(feats, FEATURES_PATH)

    # learned source impact: map hash buckets back to the most common name
    src_ids = art["source"].fillna("?").map(
        lambda x: hash(x) % N_SRC_BUCKETS)
    names = art.groupby(src_ids.to_numpy())["source"] \
        .agg(lambda g: g.mode().iat[0])
    impact = (pd.concat(src_rows).groupby(level=0)["w_contrib"]
              .agg(["mean", "count"]))
    impact["source"] = names
    impact = (impact[impact["count"] >= 200]
              .sort_values("mean", ascending=False).head(30)
              [["source", "mean", "count"]])
    impact.to_csv(SOURCES_PATH, index=False)
    log.info("NewsNet features: %d days; top source by learned impact: %s",
             len(feats), impact.iloc[0]["source"] if len(impact) else "n/a")
    return feats


def load_newsnet_features() -> pd.DataFrame | None:
    if not FEATURES_PATH.exists():
        return None
    return pd.read_csv(FEATURES_PATH, index_col="date", parse_dates=["date"])
