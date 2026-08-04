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


def _gap_context() -> dict | None:
    """Yesterday's close vs the latest traded price (pre-market included):
    the live answer to 'how is it opening compared to yesterday's close?'.
    This is deliberately the TAPE, not a model - by signal time the US
    pre-market has been trading for hours and prices the gap in real time;
    a model frozen on yesterday's features cannot compete with that.
    Degrades to omission when the quote source throttles."""
    import requests
    from data.prices import HEADERS, YAHOO_CHART_URL
    try:
        r = requests.get(YAHOO_CHART_URL.format(symbol=config.TICKER),
                         params={"range": "1d", "interval": "5m",
                                 "includePrePost": "true"},
                         headers=HEADERS, timeout=15)
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        prev_chart = float(res["meta"].get("chartPreviousClose") or 0)
        reg_price = float(res["meta"].get("regularMarketPrice") or 0)
        ts = pd.to_datetime(res["timestamp"], unit="s",
                            utc=True).tz_convert("America/New_York")
        closes = pd.Series(res["indicators"]["quote"][0]["close"],
                           index=ts, dtype=float).dropna()
        if not len(closes):
            return None
        last_t, last_p = closes.index[-1], float(closes.iloc[-1])
        hm = (last_t.hour, last_t.minute)
        session = ("pre-market" if hm < (9, 30)
                   else "live session" if hm < (16, 0) else "after-hours")
        # "yesterday's close" must be the most recent REGULAR close.
        # During a live session Yahoo's chartPreviousClose is exactly
        # that, but outside regular hours it lags one session (during the
        # 7/30 pre-market it still pointed at 7/28's close and printed a
        # -1.8% gap when the tape was really +1.9%). regularMarketPrice
        # holds the last regular trade - the close a gap forms against.
        prev = (prev_chart if session == "live session"
                else (reg_price or prev_chart))
        if not prev:
            return None
        return {"prev_close": round(prev, 2), "latest": round(last_p, 2),
                "gap_pct": round(last_p / prev - 1, 4), "session": session,
                "as_of_et": last_t.strftime("%H:%M ET")}
    except Exception as exc:
        log.warning("Gap context unavailable (%s) - line omitted", exc)
        return None


def _data_warnings() -> list[str]:
    """Input-staleness notes for the artifact; never fails the signal."""
    try:
        from data.health import data_warnings
        return data_warnings()
    except Exception as exc:
        log.warning("Freshness check unavailable (%s)", exc)
        return []


def _load_magnitude_bundle() -> dict | None:
    """The calibrated P(big move) head - the desk's validated news edge.
    Absent or stale bundles degrade to omission, never to a crash."""
    if not config.MAG_MODEL_PATH.exists():
        return None
    try:
        bundle = joblib.load(config.MAG_MODEL_PATH)
        trained_through = bundle.get("trained_through")
        if trained_through:
            age = (pd.Timestamp.now().normalize()
                   - pd.Timestamp(trained_through)).days
            if age > config.MAX_MODEL_AGE_DAYS:
                log.warning("Magnitude bundle is %d days old - line "
                            "omitted; re-run `python main.py magnitude`",
                            age)
                return None
        return bundle
    except Exception as exc:
        log.warning("Magnitude bundle unreadable (%s) - line omitted", exc)
        return None


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
    # A stale price frontier must fail LOUDLY, never sign a backdated bet.
    # On 2026-08-04 a malformed Monday bar left the cache ending Friday:
    # this line resolved the entry day to Monday - a session already
    # traded - and the on-time-record guard rightly refused the artifact
    # write, so the run "succeeded" while publishing nothing and nobody
    # was told. Refusing here instead routes the failure to the health
    # banner. 19:00 IST is the same open convention the tracker uses
    # (conservative by an hour in US winter, like everything else).
    if pd.Timestamp.now() >= next_day + pd.Timedelta(hours=19):
        raise RuntimeError(
            f"computed entry day {next_day.date()} has already opened - "
            f"the price cache ends {px.index.max().date()}, so its next "
            f"session is in the past. Refresh prices (data source likely "
            f"served a malformed latest bar) and rerun before an open.")
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
    # News-vector features are needed when the direction winner carries
    # them OR when the magnitude bundle (always price+news2) will run.
    # They are built from articles strictly before each entry day - same
    # convention as training rows.
    mag_bundle = _load_magnitude_bundle()
    need_news = any(f.startswith(("n2_", "nn_")) for f in feat_names) \
        or mag_bundle is not None
    if need_news:
        try:
            # LIVE features first (weekly-educated brain applied daily) -
            # but ONLY if they cover this entry day: a failed `learn` run
            # must never silently serve today from yesterday's news.
            from model.live import load_live_features
            extra = load_live_features()
            if extra is not None and extra.index.max() >= next_day:
                ds = ds.join(extra, how="left")
                log.info("Using LIVE news features (brain applied today)")
            else:
                if extra is not None:
                    log.warning("Live features end %s < entry day %s - "
                                "falling back to weekly production features",
                                extra.index.max().date(), next_day.date())
                from model.news2vec import load_news2_features
                from model.newsnet import load_newsnet_features
                for loader in (load_news2_features, load_newsnet_features):
                    weekly = loader()
                    if weekly is not None:
                        ds = ds.join(weekly, how="left")
        except Exception as exc:
            log.warning("news-vector features unavailable (%s)", exc)
    missing = [f for f in feat_names if f not in ds.columns]
    newsish = [f for f in missing if f.startswith(("n2_", "nn_"))]
    if newsish:
        # degrade, don't die: HGB routes NaN to a learned missing branch
        log.warning("News-vector feature files absent (%s) - filling NaN so "
                    "the signal still generates", newsish)
        for c in newsish:
            ds[c] = np.nan
        missing = [f for f in missing if f not in newsish]
    if missing:
        raise RuntimeError(
            f"Saved model expects features {missing} that the current pipeline "
            "no longer produces - re-run `python main.py train`")
    nv = [f for f in feat_names if f.startswith(("n2_", "nn_"))]
    if nv and ds.iloc[-1][nv].isna().all():
        log.warning("News-vector features are all NaN for entry day %s - "
                    "the feature files predate this session (they refresh "
                    "with the weekly retrain: article top-up + `news2` + "
                    "`newsnet`). The model runs on its missing-value paths "
                    "meanwhile; treat the direction line with extra "
                    "skepticism today", next_day.date())
    row = ds.iloc[[-1]][feat_names]
    prob_up = float(model.predict_proba(row)[0, 1])

    # Calibrated P(big move) - the validated news-magnitude edge. Threshold
    # parity note: training uses the trailing 63-day median of |open-open|
    # moves through d-1, whose newest term needs today's open; pre-open we
    # serve the same median one day older (a 63-day median barely moves in
    # a day). Failures degrade to omission.
    prob_big, big_thr = None, None
    if mag_bundle is not None:
        try:
            mfeats = mag_bundle["features"]
            mag_missing = [f for f in mfeats if f not in ds.columns]
            if mag_missing:
                log.warning("Magnitude features %s absent - NaN-filled",
                            mag_missing)
                for c in mag_missing:
                    ds[c] = np.nan
            raw = float(mag_bundle["model"]
                        .predict_proba(ds.iloc[[-1]][mfeats])[0, 1])
            prob_big = float(mag_bundle["iso"].predict(
                np.array([raw]))[0])
            oo = (px["Open"].shift(-1) / px["Open"] - 1).abs()
            med = oo.rolling(63, min_periods=30).median().dropna()
            big_thr = float(med.iloc[-1]) if len(med) else None
        except Exception as exc:
            log.warning("Magnitude line failed (%s) - omitted", exc)
            prob_big, big_thr = None, None

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
    raw_scores = analyzer.score([h.get("title", "") for h in headlines])
    for h, sc in zip(headlines, raw_scores):
        h["score"] = round(float(sc), 3)
    # sentiment is averaged over STORIES, not articles: a wire story
    # syndicated by four outlets gets one vote, not four
    stories = news_mod.cluster_headlines(headlines)

    # Read the article bodies, not just the headlines. Blocked sites fall
    # back to another outlet's copy of the same story (cluster members),
    # then to a same-story search. Sentiment then blends the title score
    # 50/50 with a "title + opening of the body" score - the opening only,
    # because FinBERT verifiably drifts to neutral on long multi-topic
    # text (a -0.92 headline washed to 0.0 with a 2000-char lead).
    fulltext_stats = None
    if config.FULLTEXT_ENABLED:
        try:
            from data import fulltext
            stats = fulltext.read_stories(stories)
            read = [s for s in stories if s.get("body")]
            units = [f"{s.get('title', '').rstrip('. ')}. "
                     f"{s['body'][:config.FULLTEXT_SCORE_CHARS]}"
                     for s in read]
            for s, body_sc in zip(read, analyzer.score(units)):
                s["title_score"] = s.get("score")
                s["body_score"] = round(float(body_sc), 3)
                if s.get("score") is not None:
                    s["score"] = round(0.5 * s["score"]
                                       + 0.5 * s["body_score"], 3)
                else:
                    s["score"] = s["body_score"]
            # published only after the blend succeeds: the artifact must
            # never claim bodies were read while scores are title-only
            fulltext_stats = stats
            # feed the permanent body corpus (phase 1 of the body-aware
            # brain) - best-effort, never fails the signal
            try:
                fulltext.append_bodies([
                    {"date": str(next_day.date()), "kind": "live_story",
                     "category": None, "source": s.get("source"),
                     "slug": None, "title": s.get("title"),
                     "url": s.get("read_url"),
                     "status": s.get("read_status"),
                     "chars": s.get("body_chars"), "text": s.get("body"),
                     # the text may come from ANOTHER outlet's copy
                     # (cluster fallback / alternate search) - keep the
                     # provenance so training can weigh that
                     "read_via": s.get("read_via")}
                    for s in read])
            except Exception as exc:
                log.warning("body corpus append failed (%s)", exc)
        except Exception as exc:
            log.warning("Full-article reading failed (%s); scoring "
                        "headlines only", exc)
            for s in stories:   # scrub per-story claims for the same reason
                if "title_score" in s:   # un-blend any partially blended
                    s["score"] = s.pop("title_score")
                for k in ("read_status", "read_from", "read_via",
                          "body", "body_chars", "body_score"):
                    s.pop(k, None)

    scores = [s["score"] for s in stories if s.get("score") is not None]
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

    # what the daily-trained brain weighted for this entry day (written by
    # `main.py learn`; omitted when absent or for a different session)
    brain_news = []
    try:
        from model.live import TODAY_WEIGHTS
        if TODAY_WEIGHTS.exists():
            tw = json.loads(TODAY_WEIGHTS.read_text(encoding="utf-8"))
            if tw.get("entry_day") == str(next_day.date()):
                brain_news = tw.get("articles", [])[:5]
    except Exception as exc:
        log.warning("Brain weights unavailable (%s)", exc)

    # The most-positive/negative display only admits headlines that are
    # (a) linkable, (b) from news sources rather than StockTwits chatter
    # (posts have no URL and VADER misreads their sarcasm; the bull/bear
    # tally covers that crowd), and (c) visibly about NVDA - Yahoo's
    # ticker feed in particular carries general-market stories that would
    # otherwise top the list on sentiment alone. The full unfiltered
    # scrape still feeds the advisory composite above.
    import re
    relevant = re.compile(
        r"(?i)\b(nvidia|nvda|jensen|geforce|rtx|blackwell|hopper|rubin"
        r"|cuda|h100|h200|b200|gb200|gpu)\b")
    ranked = sorted((h for h in stories
                     if "score" in h and h.get("url")
                     and h.get("source") != "stocktwits"
                     and relevant.search(str(h.get("title", "")))),
                    key=lambda h: h["score"])
    signal = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "ticker": config.TICKER,
        "entry_day": str(next_day.date()),
        "last_close": round(float(px["Close"].iloc[-1]), 2),
        "model_trained_through": trained_through,
        "gap_context": _gap_context(),
        "model_prob_up": round(prob_up, 4),
        "prob_big_move": round(prob_big, 4) if prob_big is not None else None,
        "big_move_threshold": round(big_thr, 4) if big_thr is not None else None,
        "action": action,
        "headline_sentiment": round(headline_mean, 4),
        "headline_count": len(scores),       # stories after clustering
        "article_count": len(headlines),     # raw articles before clustering
        "fulltext": fulltext_stats,          # body-reading counters (no text)
        # stale inputs are stated on the artifact itself: downstream fills
        # missing tone days with 0.0, which reads as "a quiet news day"
        # rather than "no data" - a reader must be able to discount that
        "data_warnings": _data_warnings(),
        "headlines_degraded": degraded,
        "sentiment_backend": analyzer.backend,
        "stocktwits_bulls": bulls,
        "stocktwits_bears": len(st) - bulls,
        "advisory_composite": round(float(advisory), 4),
        "brain_top_news": brain_news,
        # sign is required, not just rank: with a small eligible pool the
        # old top/bottom-3 slicing could list one headline on both sides,
        # or crown a negative headline "most positive"
        "most_negative": [{k: h.get(k) for k in ("title", "source", "score",
                                                 "url", "n_sources",
                                                 "also_from", "read_status",
                                                 "read_from")}
                          for h in ranked if h["score"] < -0.1][:3],
        "most_positive": [{k: h.get(k) for k in ("title", "source", "score",
                                                 "url", "n_sources",
                                                 "also_from", "read_status",
                                                 "read_from")}
                          for h in reversed(ranked) if h["score"] > 0.1][:3],
    }
    out_path = config.ARTIFACTS / f"signal_{next_day.strftime('%Y%m%d')}.json"
    from model.volatility import save_entry_artifact
    save_entry_artifact(out_path, signal, next_day)
    return signal


def format_signal(signal: dict) -> str:
    degraded_note = ("  [DEGRADED: no live headlines scraped]"
                     if signal["headlines_degraded"] else "")
    lines = [
        "=" * 62,
        f"  {signal['ticker']} signal for entry day {signal['entry_day']}",
        f"  generated {signal['generated_at']}  |  last close ${signal['last_close']}",
        *([
            f"  Vs yesterday's close : ${signal['gap_context']['latest']} "
            f"{signal['gap_context']['session']} "
            f"({signal['gap_context']['gap_pct']:+.1%} vs "
            f"${signal['gap_context']['prev_close']}) "
            f"as of {signal['gap_context']['as_of_et']}",
            "  (that gap belongs to YESTERDAY's bet - tonight's bet starts "
            "at the open)",
        ] if signal.get("gap_context") else []),
        "=" * 62,
        f"  Model P(up)          : {signal['model_prob_up']:.1%} "
        f"(long > {config.LONG_ENTER}, exit < {config.LONG_EXIT}; "
        f"bet = entry open -> next open, graded at the next open)",
        *([f"  P(move > {signal['big_move_threshold']:.2%}) : "
           f"{signal['prob_big_move']:.1%} calibrated - the news-magnitude "
           "edge (size, not direction)"]
          if signal.get("prob_big_move") is not None
          and signal.get("big_move_threshold") is not None else []),
        f"  Headline sentiment   : {signal['headline_sentiment']:+.3f} "
        f"({signal['headline_count']} stories"
        + (f" from {signal['article_count']} articles"
           if signal.get("article_count") else "")
        + f", {signal['sentiment_backend']}){degraded_note}",
        *([f"  Articles read        : {ft['read_full']} in full, "
           f"{ft['read_partial']} partially, {ft['headline_only']} "
           f"headline-only ({ft['elapsed_s']:.0f}s)"]
          if (ft := signal.get("fulltext")) else []),
        *([f"  !! STALE INPUT       : {w}"
           for w in signal.get("data_warnings") or []]),
        f"  StockTwits bull/bear : {signal['stocktwits_bulls']}/{signal['stocktwits_bears']}",
        f"  Advisory composite   : {signal['advisory_composite']:.3f} "
        f"(context only - not the traded rule)",
        "-" * 62,
        f"  ADVISORY direction (no demonstrated holdout edge): "
        f"{signal['action']}",
        *([
            "-" * 62,
            "  What the brain weighted for this session:",
            *[f"    w={a['weight']:.3f} [{a['category']}/{a['source']}] "
              f"{a['headline'][:60]}"
              for a in signal.get("brain_top_news", [])[:5]],
        ] if signal.get("brain_top_news") else []),
        *([
            "-" * 62,
            "  Most positive stories:",
            *[f"    {h['score']:+.2f} [{h['source']}"
              + (f" +{h['n_sources'] - 1} more" if h.get("n_sources", 1) > 1
                 else "") + f"] {h['title'][:70]}"
              for h in signal["most_positive"]],
        ] if signal.get("most_positive") else []),
        *([
            "  Most negative stories:",
            *[f"    {h['score']:+.2f} [{h['source']}"
              + (f" +{h['n_sources'] - 1} more" if h.get("n_sources", 1) > 1
                 else "") + f"] {h['title'][:70]}"
              for h in signal["most_negative"]],
        ] if signal.get("most_negative") else []),
        "=" * 62,
        "  Educational tool - not financial advice. You decide the trade.",
        "=" * 62,
    ]
    return "\n".join(lines)
