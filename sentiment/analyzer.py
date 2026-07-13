"""Headline sentiment scoring.

Uses FinBERT (ProsusAI/finbert) when the optional transformers/torch stack is
installed; otherwise falls back to VADER, which is lightweight and always
available. Scores are mapped to [-1, +1] in both cases.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class SentimentAnalyzer:
    def __init__(self, prefer_finbert: bool = True):
        self.backend = "vader"
        self._finbert = None
        self._vader = None
        if prefer_finbert:
            try:
                from transformers import pipeline
                self._finbert = pipeline(
                    "sentiment-analysis", model="ProsusAI/finbert", truncation=True
                )
                self.backend = "finbert"
                log.info("Using FinBERT for sentiment scoring")
            except Exception as exc:
                log.info("FinBERT unavailable (%s); falling back to VADER", exc)
        if self._finbert is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()

    def score(self, texts: list[str]) -> list[float]:
        """Score each text in [-1, +1]; positive = bullish tone."""
        texts = [t if isinstance(t, str) else "" for t in texts]
        if not texts:
            return []
        if self._finbert is not None:
            results = self._finbert(texts, batch_size=16)
            out = []
            for r in results:
                label = r["label"].lower()
                if label == "positive":
                    out.append(float(r["score"]))
                elif label == "negative":
                    out.append(-float(r["score"]))
                else:
                    out.append(0.0)
            return out
        return [self._vader.polarity_scores(t)["compound"] for t in texts]
