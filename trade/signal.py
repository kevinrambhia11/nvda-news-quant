"""Live daily signal generation.

The traded ACTION comes from the trained ML model's P(next open-to-open
return > 0), applied through the exact thresholds that were backtested
(LONG_ENTER / LONG_EXIT). Freshly scraped headlines (Google News, Yahoo
Finance, Finviz, StockTwits; FinBERT if installed, VADER otherwise) are
scored and blended into an ADVISORY composite for context - they are not
part of the backtested rule, so they never decide the action.

Feature parity with training is enforced: the in-progress GDELT UTC day and
any in-progress intraday price bar are dropped, so the model only ever sees
the complete daily buckets it was trained on. Best run after 20:00 ET (00:00
UTC) and before the next US open.

Output is a recommendation, never an order. Nothing here touches a broker.
"""
from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd

import config
from data import news as news_mod
from data.prices import load_prices
from features.build import build_dataset
from sentiment.analyzer import SentimentAnalyzer
from trade.calendar import next_trading_day, session_close_hour_et

log = logging.getLogger(__name__)


def _load_model_bundle() -> tuple:
    if not config.MODEL_PATH.exists():
        raise RuntimeError("No trained model found - run `python main.py train` first")
    bundle = joblib.load(config.MODEL_PATH)
    trained_through = bundle.get("trained_through")
    if trained_through:
        age_days = (pd.Timestamp.now().normalize()
                    - pd.Timestamp(trained_through)).days
        if age_days > config.MAX_MODEL_AGE_DAYS:
            raise RuntimeError(
                f"Model trained through {trained_through} is {age_days} days old "
                f"(limit {config.MAX_MODEL_AGE_DAYS}) - re-run `python main.py train`")
        if age_days > 35:
            log.warning("Model is %d days old (trained through %s); "
                        "consider retraining", age_days, trained_through)
    return bundle["model"], bundle["features"], trained_through


def generate_signal(prefer_finbert: bool = True) -> dict:
    model, feat_names, trained_through = _load_model_bundle()

    # Windows: ~430 calendar days of prices covers the 200-day SMA warm-up;
    # 120 days of GDELT covers the 60-day tone z-score warm-up. load_prices
    # serves the shared cache when it already holds the last completed
    # business day, so a blocked network never stops signal generation.
    px_start = (pd.Timestamp.now() - pd.Timedelta(days=430)).strftime("%Y-%m-%d")
    gd_start = (pd.Timestamp.now() - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    px_full, bench_full = load_prices()
    px = px_full.loc[px_full.index >= pd.Timestamp(px_start)]
    bench = bench_full.loc[bench_full.index >= pd.Timestamp(px_start)]

    # Training rows only ever see settled closes; drop an in-progress bar
    # (early-close aware: 13:00 sessions around July 4th/Thanksgiving/Xmas).
    now_et = pd.Timestamp.now(tz="America/New_York")
    today_et = now_et.tz_localize(None).normalize()
    if now_et.hour < session_close_hour_et(today_et):
        if len(px) and px.index.max() == today_et:
            log.warning("Run before today's close - dropping the "
                        "in-progress price bar")
            px = px.loc[px.index < today_et]
            bench = bench.loc[bench.index < today_et]

    # Same discipline for news: only fully elapsed UTC-day buckets, matching
    # training. Intraday news is covered by the scraped-headline overlay.
    # load_gdelt_daily serves the shared cache when it is current and only
    # tops it up incrementally otherwise.
    gdelt = news_mod.load_gdelt_daily()
    today_utc = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    gdelt = gdelt[(gdelt.index >= pd.Timestamp(gd_start)) & (gdelt.index < today_utc)]

    # Append the next actual trading day (NYSE calendar) so the last row's
    # features use everything known now, exactly as a training row would.
    next_day = next_trading_day(px.index.max())
    px_ext = px.reindex(px.index.append(pd.DatetimeIndex([next_day])))
    try:
        from data.earnings import load_earnings_dates
        earn_dates = load_earnings_dates()
    except Exception as exc:
        log.warning("Earnings calendar unavailable (%s); event features "
                    "fall back to neutral values", exc)
        earn_dates = None
    aux = news_mod.load_aux_gdelt()
    ds = build_dataset(px_ext, bench, gdelt, earn_dates=earn_dates, aux=aux)
    missing = [f for f in feat_names if f not in ds.columns]
    if missing:
        raise RuntimeError(
            f"Saved model expects features {missing} that the current pipeline "
            "no longer produces - re-run `python main.py train`")
    row = ds.iloc[[-1]][feat_names]
    prob_up = float(model.predict_proba(row)[0, 1])

    # The traded action uses the model probability - the rule that was
    # actually backtested.
    if prob_up > config.LONG_ENTER:
        action = "BUY / HOLD LONG"
    elif config.ALLOW_SHORT and prob_up < 1 - config.LONG_ENTER:
        action = "GO SHORT"
    elif prob_up < config.LONG_EXIT:
        action = "EXIT / STAY FLAT"
    else:
        action = "HOLD (keep current position)"

    # Freshly scraped headlines, scored now - advisory context only.
    headlines = news_mod.collect_live_headlines()
    analyzer = SentimentAnalyzer(prefer_finbert=prefer_finbert)
    scores = analyzer.score([h.get("title", "") for h in headlines])
    for h, sc in zip(headlines, scores):
        h["score"] = round(float(sc), 3)
    if scores:
        headline_mean = float(np.mean(scores))
        headline_prob = (headline_mean + 1) / 2  # map [-1, 1] -> [0, 1]
        advisory = (config.MODEL_WEIGHT * prob_up
                    + config.HEADLINE_WEIGHT * headline_prob)
        degraded = False
    else:
        headline_mean = 0.0
        advisory = prob_up  # never fabricate a neutral headline score
        degraded = True
        log.warning("All headline sources failed; advisory equals model-only")

    st = [h for h in headlines
          if h.get("source") == "stocktwits" and h.get("declared_sentiment")]
    bulls = sum(1 for h in st if h["declared_sentiment"] == "Bullish")

    ranked = sorted((h for h in headlines if "score" in h),
                    key=lambda h: h["score"])
    signal = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "ticker": config.TICKER,
        "entry_day": str(next_day.date()),
        "last_close": round(float(px["Close"].iloc[-1]), 2),
        "model_trained_through": trained_through,
        "model_prob_up": round(prob_up, 4),
        "action": action,
        "headline_sentiment": round(headline_mean, 4),
        "headline_count": len(scores),
        "headlines_degraded": degraded,
        "sentiment_backend": analyzer.backend,
        "stocktwits_bulls": bulls,
        "stocktwits_bears": len(st) - bulls,
        "advisory_composite": round(float(advisory), 4),
        "most_negative": [{k: h.get(k) for k in ("title", "source", "score")}
                          for h in ranked[:3]],
        "most_positive": [{k: h.get(k) for k in ("title", "source", "score")}
                          for h in ranked[-3:][::-1]],
    }
    out_path = config.ARTIFACTS / f"signal_{next_day.strftime('%Y%m%d')}.json"
    out_path.write_text(json.dumps(signal, indent=2), encoding="utf-8")
    log.info("Signal saved -> %s", out_path)
    return signal


def format_signal(signal: dict) -> str:
    degraded_note = ("  [DEGRADED: no live headlines scraped]"
                     if signal["headlines_degraded"] else "")
    lines = [
        "=" * 62,
        f"  {signal['ticker']} signal for entry day {signal['entry_day']}",
        f"  generated {signal['generated_at']}  |  last close ${signal['last_close']}",
        "=" * 62,
        f"  Model P(up)          : {signal['model_prob_up']:.1%} "
        f"(long > {config.LONG_ENTER}, exit < {config.LONG_EXIT})",
        f"  Headline sentiment   : {signal['headline_sentiment']:+.3f} "
        f"({signal['headline_count']} items, {signal['sentiment_backend']})"
        f"{degraded_note}",
        f"  StockTwits bull/bear : {signal['stocktwits_bulls']}/{signal['stocktwits_bears']}",
        f"  Advisory composite   : {signal['advisory_composite']:.3f} "
        f"(context only - not the traded rule)",
        "-" * 62,
        f"  ADVISORY direction (no demonstrated holdout edge): "
        f"{signal['action']}",
        "-" * 62,
        "  Most positive headlines:",
        *[f"    {h['score']:+.2f} [{h['source']}] {h['title'][:70]}"
          for h in signal["most_positive"]],
        "  Most negative headlines:",
        *[f"    {h['score']:+.2f} [{h['source']}] {h['title'][:70]}"
          for h in signal["most_negative"]],
        "=" * 62,
        "  Educational tool - not financial advice. You decide the trade.",
        "=" * 62,
    ]
    return "\n".join(lines)
