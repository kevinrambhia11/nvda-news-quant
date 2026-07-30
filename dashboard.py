"""NVDA Quant Desk - Streamlit dashboard.

Run:  streamlit run dashboard.py   (or via .claude/launch.json, port 8502)

Read-only over the pipeline's artifacts, plus on-demand refresh buttons that
call the same functions as the CLI. Works fully offline once the data caches
are current. Educational tool - not financial advice.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402

st.set_page_config(page_title="NVDA Quant Desk", page_icon="ðŸ“ˆ", layout="wide")


@st.cache_data(ttl=300)
def fetch_intraday(sym: str):
    """Today's 5-minute closes + previous close, live from Yahoo's chart
    API. Cached 5 minutes; returns None when the source is throttled."""
    import requests
    from data.prices import HEADERS, YAHOO_CHART_URL
    try:
        r = requests.get(YAHOO_CHART_URL.format(symbol=sym),
                         params={"range": "1d", "interval": "5m"},
                         headers=HEADERS, timeout=15)
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        ts = (pd.to_datetime(res["timestamp"], unit="s", utc=True)
              .tz_convert("America/New_York"))
        closes = pd.Series(res["indicators"]["quote"][0]["close"],
                           index=ts, dtype=float).dropna()
        prev = float(res["meta"].get("chartPreviousClose")
                     or res["meta"].get("previousClose") or float("nan"))
        if closes.empty:
            return None
        return closes, prev
    except Exception:
        return None


def fitted_lines(df: pd.DataFrame, y_title: str = "price ($)",
                 height: int = 280, fmt: str = ".2f"):
    """Multi-series line chart with the y-axis fitted to the data (Streamlit's
    native chart anchors at zero, which flattens a $200 stock)."""
    data = (df.rename_axis("date").reset_index()
            .melt("date", var_name="series", value_name="value").dropna())
    lo, hi = float(data["value"].min()), float(data["value"].max())
    pad = (hi - lo) * 0.06 or max(abs(hi), 1) * 0.02
    return (alt.Chart(data).mark_line()
            .encode(x=alt.X("date:T", title=None),
                    y=alt.Y("value:Q", title=y_title,
                            scale=alt.Scale(domain=[lo - pad, hi + pad])),
                    color=alt.Color("series:N",
                                    legend=alt.Legend(orient="top", title=None)),
                    tooltip=[alt.Tooltip("date:T"), alt.Tooltip("series:N"),
                             alt.Tooltip("value:Q", format=fmt)])
            .properties(height=height))


def headline_lines(items):
    for h in items:
        prefix = f"`{h.get('score', 0):+.2f}` [{h.get('source', '')}] "
        title = str(h.get("title", ""))[:160]
        url = h.get("url")
        line = prefix + (f"[{title}]({url})" if url else title)
        extra = h.get("also_from") or []
        if extra:
            shown = ", ".join(dict.fromkeys(extra[:3]))
            more = len(extra) - 3
            line += (f"  &middot; *also: {shown}"
                     + (f" +{more} more" if more > 0 else "") + "*")
        st.markdown(line)


# ---------------------------------------------------------------------------
# Cached artifact readers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def read_latest_json(pattern: str):
    files = sorted(config.ARTIFACTS.glob(pattern))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


@st.cache_data(ttl=30)
def read_csv(path_str: str):
    p = Path(path_str)
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0, parse_dates=[0])


@st.cache_data(ttl=30)
def read_text(path_str: str):
    p = Path(path_str)
    return p.read_text(encoding="utf-8") if p.exists() else None


st.title("NVDA Quant Desk")
st.caption("News-sentiment direction signal + EMH-consistent volatility desk. "
           "Educational tool - not financial advice.")


@st.cache_data(ttl=120, show_spinner=False)
def desk_health():
    """Run outcomes + input staleness, from the committed status file (so
    the hosted app alerts too, not just the desk machine)."""
    try:
        from data.health import summarize
        return summarize()
    except Exception:
        return None


# Health banner ABOVE the tabs: a failed scheduled task or a stale input
# must be impossible to miss, not buried in an expander.
_h = desk_health()
if _h and _h["alerts"]:
    _box = st.error if _h["level"] == "alert" else st.warning
    _box("**Desk health**\n\n"
         + "\n".join(f"- {a}" for a in _h["alerts"])
         + (f"\n\nStatus recorded {_h['updated']}." if _h.get("updated")
            else ""))

(tab_today, tab_vol, tab_links, tab_track, tab_arch) = st.tabs(
    ["Desk today", "Volatility", "Today's news", "Track record",
     "Architecture"])


@st.cache_data(ttl=3600, show_spinner=False)
def load_pup_calibration():
    """P(up) honesty table: what the direction model said out-of-sample
    vs how often the next open-to-open was actually up. Computed from
    oos_predictions.csv (walk-forward, appended by every retrain) so the
    table stays current as evidence accumulates. Returns (table, base
    rate, n_days) or None when the file is absent."""
    try:
        df = pd.read_csv(config.ARTIFACTS / "oos_predictions.csv",
                         parse_dates=[0], index_col=0)
        assert {"y", "prob_up"} <= set(df.columns) and len(df) >= 200
    except Exception:
        return None
    edges = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ["under 40%", "40-50%", "50-60%", "60-70%", "70-80%",
              "80-90%", "90-100%"]
    cut = pd.cut(df["prob_up"], edges, labels=labels)
    g = df.groupby(cut, observed=False)
    tab = pd.DataFrame({
        "model said": labels,
        "days": g.size().reindex(labels).fillna(0).astype(int).values,
        "actually up next day": [
            f"{v:.0%}" if pd.notna(v) else "-"
            for v in g["y"].mean().reindex(labels).values],
    })

    return tab, float(df["y"].mean()), len(df)


# module-level (NOT inside the cached loader: st.cache_data pickles
# return values and a closure would not survive that)
_PUP_EDGES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
_PUP_LABELS = ["under 40%", "40-50%", "50-60%", "60-70%", "70-80%",
               "80-90%", "90-100%"]


def _pup_bucket(p: float) -> str:
    for hi, lab in zip(_PUP_EDGES, _PUP_LABELS):
        if p <= hi:
            return lab
    return _PUP_LABELS[-1]


@st.cache_data(ttl=900, show_spinner="Collecting today's news...")
def todays_news():
    from data.news import cluster_headlines, collect_live_headlines
    from sentiment.analyzer import SentimentAnalyzer
    items = collect_live_headlines()
    scores = SentimentAnalyzer(prefer_finbert=False).score(
        [h.get("title", "") for h in items])
    for h, s in zip(items, scores):
        h["score"] = round(float(s), 3)
    # one entry per STORY: outlets rewriting the same event are grouped,
    # so a story syndicated four times gets one vote, not four; its score
    # is the mean of its rewrites (steadier than any single one)
    stories = cluster_headlines(items)
    return stories, len(items)


# ---------------------------------------------------------------------------
# Tab 1: Desk today
# ---------------------------------------------------------------------------
with tab_today:
    st.caption("**What this page is:** the desk's daily output. The green "
               "banner is the product - how much NVDA to hold so your risk "
               "stays constant. The blue banner is the news-based direction "
               "read (advisory - it has no proven edge). Below: today's "
               "numbers, live charts, and what the news is saying.")
    signal = read_latest_json("signal_*.json")
    vol = read_latest_json("vol_forecast_*.json")

    c1, c2 = st.columns([3, 1])
    with c2:
        if st.button("Regenerate signal", use_container_width=True):
            try:
                with st.spinner("Scraping headlines and scoring..."):
                    from trade.signal import generate_signal
                    generate_signal(prefer_finbert=False)
            except Exception as exc:
                st.error(f"Signal generation failed: {exc}")
            else:
                st.cache_data.clear()
                st.rerun()
        if st.button("Refresh vol forecast", use_container_width=True):
            try:
                with st.spinner("Forecasting volatility..."):
                    from model.volatility import forecast
                    forecast()
            except Exception as exc:
                st.error(f"Vol forecast failed: {exc}")
            else:
                st.cache_data.clear()
                st.rerun()

    with c1:
        if vol:
            h1 = vol["horizons"]["1"]
            st.success(
                f"**DESK POSITION (primary): hold NVDA at "
                f"{h1['target_vol_weight']:.0%} of full size** - vol-sized "
                f"long, {config.VOL_TARGET_ANN:.0%} annual target vs "
                f"{h1['total_annualized_vol']:.1%} forecast vol"
                + ("  |  :red[**EARNINGS PRINT TONIGHT**]"
                   if h1.get("earnings_print_tonight") else ""))
        if signal is None:
            st.info("No direction signal yet - run `python main.py signal` "
                    "or click Regenerate signal.")
        else:
            st.info(f"Advisory direction (no demonstrated holdout edge): "
                    f"**{signal['action']}** - entry day "
                    f"{signal['entry_day']}, generated "
                    f"{signal['generated_at']}")
            st.caption(f"This bet covers **{signal['entry_day']}'s open "
                       f"&rarr; the NEXT session's open** and is graded at "
                       "that next open - not before. A red or green open "
                       "this morning belongs to YESTERDAY's bet (which the "
                       "Track record grades automatically). No single day "
                       "grades a probability either way - only the running "
                       "score does.")
            gc = signal.get("gap_context")
            if gc:
                st.caption(f"Heading into the open: **${gc['latest']}** "
                           f"({gc['session']}, {gc['gap_pct']:+.1%} vs "
                           f"yesterday's close ${gc['prev_close']}, as of "
                           f"{gc['as_of_et']}). The live tape prices this "
                           "gap in real time - the desk reports it rather "
                           "than modeling it; tonight's bet starts at "
                           "whatever the open turns out to be.")

    if signal:
        m = st.columns(5)
        m[0].metric("Model P(up)", f"{signal['model_prob_up']:.1%}",
                    help=f"Long > {config.LONG_ENTER}, exit < {config.LONG_EXIT}")
        m[0].caption("Model's lean that the OPEN-to-OPEN return (today's "
                     "open to tomorrow's open) is positive. NOT a literal "
                     "probability - see 'What does this percentage really "
                     "mean?' below the metrics.")
        m[1].metric("Headline sentiment", f"{signal['headline_sentiment']:+.3f}",
                    help=f"{signal['headline_count']} stories"
                         + (f" from {signal['article_count']} articles"
                            if signal.get("article_count") else "")
                         + f", {signal['sentiment_backend']}")
        _ft = signal.get("fulltext") or {}
        m[1].caption("Average mood of live news stories, from -1 (bearish) "
                     "to +1 (bullish). Syndicated copies of the same story "
                     "are grouped first, so each story gets one vote."
                     + (f" Article bodies were read for "
                        f"{_ft['read_full']} stories in full and "
                        f"{_ft['read_partial']} partially (blocked sites "
                        f"fall back to another outlet's copy); "
                        f"{_ft['headline_only']} scored headline-only."
                        if _ft else ""))
        m[2].metric("StockTwits bulls/bears",
                    f"{signal['stocktwits_bulls']}/{signal['stocktwits_bears']}")
        m[2].caption("How the retail crowd says it is positioned right now.")
        m[3].metric("Last close", f"${signal['last_close']}")
        m[3].caption("Latest verified daily closing price the desk "
                     "computed from.")
        if vol:
            h1 = vol["horizons"]["1"]
            m[4].metric("1-day total vol", f"{h1['total_daily_vol']:.2%}",
                        help="Includes overnight gap risk")
            m[4].caption("Expected size of tomorrow's move (either "
                         "direction), incl. overnight gap.")

    if vol:
        h1 = vol["horizons"]["1"]
        v = st.columns(4)
        v[0].metric("Annualized vol", f"{h1['total_annualized_vol']:.1%}")
        v[0].caption("The same expected movement, scaled to a yearly rate.")
        v[1].metric("Target-vol sizing", f"{h1['target_vol_weight']:.0%}",
                    help=f"{config.VOL_TARGET_ANN:.0%} annual target")
        v[1].caption(f"Position size that keeps risk at the "
                     f"{config.VOL_TARGET_ANN:.0%} target - THE desk "
                     "recommendation.")
        v[2].metric("VaR 95% ($1M)", f"${h1['var_95']:,.0f}")
        v[2].caption("A bad day (worst 1-in-20) loses about this much "
                     "per $1M held.")
        v[3].metric("VaR 99% ($1M)", f"${h1['var_99']:,.0f}")
        v[3].caption("A very bad day (worst 1-in-100) - treat as a floor, "
                     "tails are fatter.")

    if signal:
        with st.expander("What does the P(up) percentage really mean?"):
            st.markdown(
                "**What it is.** The direction model's raw score that the "
                "open-to-open return will be positive, from a gradient-"
                "boosted tree trained walk-forward (each day's forecast "
                "uses only data available before it). The traded rule only "
                "checks thresholds: above "
                f"{config.LONG_ENTER} → long, below {config.LONG_EXIT} "
                "→ flat. **A 93% and a 60% reading produce the exact "
                "same action** - the size of the number beyond the "
                "threshold changes nothing.\n\n"
                "**What it is not: a real probability.** The table below "
                "is the model's own out-of-sample record - every forecast "
                "it made on days it had never seen, against what actually "
                "happened next:")
            cal = load_pup_calibration()
            if cal is not None:
                tab, base, n_days = cal
                p_now = signal.get("model_prob_up")
                if p_now is not None:
                    tab = tab.copy()
                    tab.loc[tab["model said"] == _pup_bucket(p_now),
                            "model said"] += "  ← today"
                st.table(tab.set_index("model said")[
                    ["days", "actually up next day"]])
                st.markdown(
                    f"Base rate over these {n_days:,} days: NVDA opened-to-"
                    f"opened up **{base:.0%}** of the time regardless of "
                    "what any model said. Readings in the 90-100% zone "
                    "have historically been followed by up-days LESS often "
                    "than that base rate - extreme confidence has been "
                    "noise, not extra signal. This is why the desk calls "
                    "the number advisory, sizes positions with the "
                    "volatility model instead, and lets the Track Record "
                    "tab keep score.")
            else:
                st.caption("Out-of-sample history unavailable in this "
                           "deployment - the number remains advisory.")

    st.subheader("Charts")
    c_live, c_hist = st.columns(2)
    with c_live:
        live = fetch_intraday(config.TICKER)
        if live is None:
            st.info("Live intraday feed unavailable right now (source "
                    "throttled) - try again in a few minutes.")
        else:
            closes, prev = live
            last = float(closes.iloc[-1])
            st.caption(f"{config.TICKER} - 5-minute prices, latest session, "
                       "live (times in ET)")
            intraday_df = pd.DataFrame(
                {"price": closes.tz_localize(None), "prev close": prev})
            st.altair_chart(fitted_lines(intraday_df, height=260),
                            use_container_width=True)
            if prev == prev:  # NaN-safe
                st.metric("Latest", f"${last:,.2f}",
                          f"{(last / prev - 1) * 100:+.2f}% vs prev close")
    with c_hist:
        px_hist = read_csv(str(config.CACHE / f"prices_{config.TICKER}.csv"))
        if px_hist is not None and len(px_hist) > 60:
            st.caption(f"{config.TICKER} daily close - ~6 months, with "
                       "50-day average")
            st.altair_chart(fitted_lines(pd.DataFrame({
                "close": px_hist["Close"].iloc[-126:],
                "SMA50": px_hist["Close"].rolling(50).mean().iloc[-126:],
            }), height=260), use_container_width=True)

    if signal:
        # Only linkable, NVDA-relevant news headlines are shown here (the
        # signal filters at generation time; the url guard also covers
        # archived signals written before that filter existed).
        pos = [h for h in signal.get("most_positive", []) if h.get("url")]
        neg = [h for h in signal.get("most_negative", []) if h.get("url")]
        if pos or neg:
            cpos, cneg = st.columns(2)
            with cpos:
                st.subheader("Most positive stories")
                headline_lines(pos)
                if not pos:
                    st.caption("No clearly positive NVDA story in the "
                               "latest scrape.")
            with cneg:
                st.subheader("Most negative stories")
                headline_lines(neg)
                if not neg:
                    st.caption("No clearly negative NVDA story in the "
                               "latest scrape.")


    with st.expander("Data freshness (all caches)"):
        st.caption("Last data day and file modification time for every "
                   "cache the desk reads. If a number above ever looks "
                   "stale, this shows why.")
        _rows = []
        for _f in sorted(config.CACHE.glob("*.csv")):
            try:
                _last = pd.read_csv(_f, index_col=0,
                                    parse_dates=[0]).index.max()
            except Exception:
                _last = None
            _rows.append({
                "file": _f.name,
                "last data day": str(_last.date()) if _last is not None
                else "?",
                "modified": datetime.fromtimestamp(
                    _f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")})
        if _rows:
            st.dataframe(pd.DataFrame(_rows), use_container_width=True,
                         hide_index=True)


# ---------------------------------------------------------------------------
# Tab 3: Volatility
# ---------------------------------------------------------------------------
with tab_vol:
    st.caption("**What this page is:** the desk's workhorse. Volatility - "
               "how much the stock moves, not which way - is genuinely "
               "predictable, and this model's forecasts drive the position "
               "size and VaR on the front page. Here you can see its "
               "out-of-sample accuracy vs standard benchmarks and translate "
               "risk into dollars for any position.")
    vol_report = read_text(str(config.VOL_REPORT_PATH))
    if vol_report:
        st.subheader("Model comparison (walk-forward OOS)")
        st.code(vol_report, language=None)

    vol_oos = read_csv(str(config.VOL_OOS_PATH))
    if vol_oos is not None:
        st.subheader("1-day forecast vs realized (annualized)")
        window = st.slider("Show last N trading days", 60, len(vol_oos),
                           min(500, len(vol_oos)), 20)
        chart = pd.DataFrame({
            "realized": np.exp(vol_oos["y_vol"]) * np.sqrt(252),
            "forecast": np.exp(vol_oos["pred_selected"]) * np.sqrt(252),
        }).iloc[-window:]
        st.line_chart(chart)

    vol = read_latest_json("vol_forecast_*.json")
    if vol:
        st.subheader("VaR calculator")
        h1 = vol["horizons"]["1"]
        notional = st.number_input("Position notional ($)", 10_000,
                                   100_000_000, 1_000_000, 10_000)
        daily = h1["total_daily_vol"]
        c = st.columns(3)
        c[0].metric("1-day vol (total)", f"{daily:.2%}")
        c[1].metric("VaR 95%", f"${1.645 * daily * notional:,.0f}")
        c[2].metric("VaR 99%", f"${2.326 * daily * notional:,.0f}")
        st.caption("Parametric/normal - real tails are fatter; treat as a floor.")


# ---------------------------------------------------------------------------
# Tab 5: Today's news (all collected links, grouped by category)
# ---------------------------------------------------------------------------
with tab_links:
    st.caption("**What this page is:** every article the desk can see right "
               "now, grouped by category, scored for sentiment, and linked "
               "to the original source. Refreshes at most every 15 minutes. "
               "Scores here are a quick live read: lexicon-based (VADER), "
               "headlines only. The Desk today number is a different, "
               "deeper measurement - FinBERT over each story's title plus "
               "the opening of its article body, frozen at signal time - "
               "so the two will not match, by design.")
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("Refresh now"):
            todays_news.clear()
            st.rerun()
    try:
        items, n_raw = todays_news()
    except Exception as exc:
        items, n_raw = [], 0
        st.warning(f"Collection failed: {exc}")
    if items:
        with c2:
            st.metric("Stories collected", len(items),
                      f"mean headline tone {np.mean([h['score'] for h in items]):+.3f}")
            st.caption(f"{n_raw} articles grouped into {len(items)} "
                       "stories - outlets rewriting the same event count "
                       "once, and every mean is over stories, not copies. "
                       "Tone is VADER on headlines; the desk's own "
                       "sentiment (Desk today) reads article bodies with "
                       "FinBERT and will differ.")
        order = ["finance", "ai", "semiconductors", "hyperscalers", "macro",
                 "brokers"]
        groups: dict = {}
        for h in items:
            groups.setdefault(h.get("vertical") or "general", []).append(h)
        for cat in order + [c for c in groups if c not in order]:
            if cat not in groups:
                continue
            grp = sorted(groups[cat], key=lambda h: -abs(h.get("score", 0)))
            mean_s = np.mean([h.get("score", 0) for h in grp])
            n_arts = sum(h.get("n_sources", 1) for h in grp)
            with st.expander(f"{cat}  -  {len(grp)} stories "
                             f"({n_arts} articles), "
                             f"mean sentiment {mean_s:+.3f}",
                             expanded=(cat in ("finance", "macro"))):
                headline_lines(grp[:25])


# ---------------------------------------------------------------------------
# Tab: Track record (live forecasts vs realized outcomes)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def load_track_record():
    """Join every archived daily forecast against what actually happened.
    Returns a frame indexed by entry day; rows whose outcome is not yet
    realized carry NaN outcomes and are shown as pending."""
    import glob
    import json as _json

    def _late(entry: pd.Timestamp, generated) -> bool:
        """True when the forecast was published after its session opened.
        Timestamps are naive IST (the desk machine's clock); the US open is
        19:00 IST during daylight time - a late row would be scored on an
        outcome window it could partly see, so it must never count."""
        if generated is None:
            return True
        try:
            g = pd.Timestamp(generated)
        except Exception:
            return True
        return g >= entry + pd.Timedelta(hours=19)

    rows = {}
    for path in sorted(glob.glob(str(config.ARTIFACTS / "signal_*.json"))):
        try:
            s = _json.loads(Path(path).read_text(encoding="utf-8"))
            d = pd.Timestamp(s["entry_day"])
            rows.setdefault(d, {})
            rows[d].update({
                "prob_up": s.get("model_prob_up"),
                "action": s.get("action"),
                "prob_big": s.get("prob_big_move"),
                "big_thr": s.get("big_move_threshold"),
                "late": _late(d, s.get("generated_at")),
            })
        except Exception:
            continue
    for path in sorted(glob.glob(str(config.ARTIFACTS / "vol_forecast_*.json"))):
        try:
            v = _json.loads(Path(path).read_text(encoding="utf-8"))
            d = pd.Timestamp(v["entry_day"])
            h1 = v.get("horizons", {}).get("1", {})
            rows.setdefault(d, {})
            late_v = _late(d, v.get("generated_at"))
            rows[d].update({
                "fc_vol": h1.get("total_daily_vol", h1.get("daily_vol")),
                "weight": h1.get("target_vol_weight"),
                "var95": h1.get("var_95"),
                "late": bool(rows[d].get("late", False)) or late_v,
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    track = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    # fixed schema: a deploy that has only ever produced one artifact
    # family must degrade, not KeyError on a missing column
    track = track.reindex(columns=["prob_up", "action", "prob_big",
                                   "big_thr", "fc_vol", "weight", "var95",
                                   "late"])
    track["late"] = track["late"].fillna(True).astype(bool)

    px = read_csv(str(config.CACHE / f"prices_{config.TICKER}.csv"))
    if px is None or "Open" not in px.columns:
        track["realized_ret"] = float("nan")
        return track
    opens = px["Open"]
    entry_open = opens.reindex(track.index)
    pos = opens.index.searchsorted(track.index) + 1
    next_open = pd.Series(
        [opens.iloc[p] if p < len(opens) else float("nan") for p in pos],
        index=track.index)
    track["realized_ret"] = next_open / entry_open - 1
    return track


with tab_track:
    st.caption("**What this page is:** the fresh test. Every scored row is "
               "a forecast the desk published BEFORE its session opened - "
               "none of these outcomes existed when any model was chosen, "
               "so this is the only evidence immune to backtest flattery. "
               "Rows published late (e.g. a catch-up run after a stale "
               "cache) are flagged and excluded from every metric. Judge "
               "the desk here.")
    track = load_track_record()
    if track.empty:
        st.info("No archived forecasts yet - the daily 17:00 task fills "
                "this page in automatically.")
    else:
        scored = track[track["realized_ret"].notna() & ~track["late"]]
        c1, c2, c3, c4 = st.columns(4)
        n = len(scored)
        c1.metric("Sessions scored", f"{n}",
                  help="Days with a published forecast AND a realized "
                       "open-to-open outcome")
        dirs = scored.dropna(subset=["prob_up"])
        if len(dirs):
            hit = ((dirs["prob_up"] > 0.5)
                   == (dirs["realized_ret"] > 0)).mean()
            c2.metric("Direction hit rate", f"{hit:.0%}",
                      help="Advisory line - expect ~coin flip; that is the "
                           "honest holdout verdict")
        vols = scored.dropna(subset=["fc_vol"])
        if len(vols):
            z = vols["realized_ret"].abs() / vols["fc_vol"]
            c3.metric("|move| / forecast vol", f"{z.median():.2f}x",
                      help="Median absolute move over forecast vol - "
                           "~0.7x is healthy (median of |N(0,1)| = 0.67); "
                           ">1 means vol is underforecast")
            var_breach = (vols["realized_ret"] < -1.645 * vols["fc_vol"]).mean()
            c4.metric("VaR95 breaches", f"{var_breach:.0%}",
                      help="Share of days the loss exceeded the 95% VaR - "
                           "should settle near 5%")
        mags = scored.dropna(subset=["prob_big", "big_thr"])
        if len(mags) >= 5:
            realized_big = (mags["realized_ret"].abs()
                            > mags["big_thr"]).astype(int)
            brier = float(((mags["prob_big"] - realized_big) ** 2).mean())
            base = float(realized_big.mean())
            st.metric("Magnitude head Brier (live)", f"{brier:.4f}",
                      help=f"P(big move) vs outcomes; base rate {base:.0%}. "
                           "Compare against 0.25 (coin flip). This is the "
                           "news edge on trial.")
        pending = int(track["realized_ret"].isna().sum())
        late_n = int((track["late"] & track["realized_ret"].notna()).sum())
        st.caption(f"{pending} forecast(s) pending outcomes"
                   + (f"; {late_n} excluded as late-published." if late_n
                      else "."))

        show = track.copy().sort_index(ascending=False)
        show.index = show.index.date
        st.dataframe(show.rename(columns={
            "prob_up": "P(up)", "prob_big": "P(big move)",
            "big_thr": "big-move bar", "fc_vol": "forecast vol",
            "weight": "position size", "var95": "VaR95 $",
            "late": "late (unscored)",
            "realized_ret": "realized move"}), use_container_width=True)
        vols_chart = scored.dropna(subset=["fc_vol"]).reset_index(
            names="entry")
        if len(vols_chart) >= 3:
            vols_chart["+band"] = vols_chart["fc_vol"]
            vols_chart["-band"] = -vols_chart["fc_vol"]
            base_ch = alt.Chart(vols_chart).encode(
                x=alt.X("entry:T", title=None))
            band = base_ch.mark_area(opacity=0.25).encode(
                y=alt.Y("-band:Q", title="daily move",
                        axis=alt.Axis(format="%")),
                y2="+band:Q")
            pts = base_ch.mark_circle(size=60).encode(
                y="realized_ret:Q",
                color=alt.condition(
                    "abs(datum.realized_ret) > datum.fc_vol",
                    alt.value("#b0563a"), alt.value("#2e7d5b")),
                tooltip=["entry:T", alt.Tooltip("realized_ret:Q", format=".2%"),
                         alt.Tooltip("fc_vol:Q", format=".2%")])
            st.altair_chart((band + pts).properties(height=260),
                            use_container_width=True)
            st.caption("Realized open-to-open move (dots) vs the "
                       "previous evening's +-1-sigma vol forecast (band). "
                       "Red dots landed outside the band - roughly a "
                       "third should, if the forecast is honest.")


# ---------------------------------------------------------------------------
# Tab 7: Architecture
# ---------------------------------------------------------------------------
# Streamlit embeds a component in a FIXED-height iframe, which gives the
# guide its own scrollbar on top of the dashboard's - two scrollbars. Fix:
# report the real content height outward so the frame (and the wrapper
# Streamlit sizes from the height= argument) grows to fit exactly, leaving
# one scrollbar. Lives here rather than in architecture.html so the shared
# file stays clean for the published artifact, which sizes itself.
ARCH_AUTOFIT = """
<script>
(function () {
  function fit() {
    try {
      var fr = window.frameElement;
      var b = document.body;
      if (!fr || !b) return;
      // Measure the BODY box, never documentElement.scrollHeight - the
      // latter is floored by the frame's own height, so the panel could
      // grow but never shrink back to a snug fit.
      var content = Math.max(b.scrollHeight,
                             Math.ceil(b.getBoundingClientRect().height));
      if (content < 200) return;          // tab still hidden, nothing to size
      var target = content + 4;
      // Hysteresis is essential: resizing the frame re-measures content
      // inside it, so an unconditional set plus any buffer becomes a
      // runaway loop that creeps taller on every observer tick.
      if (Math.abs(target - (fr.clientHeight || 0)) > 12) {
        fr.style.height = target + 'px';
        fr.setAttribute('height', target);
        var box = fr.closest('[data-testid="stElementContainer"]');
        if (box) box.style.setProperty('height', target + 'px', 'important');
      }
      // Suppress the frame's own scrollbar ONLY once it genuinely fits.
      // Setting 'no' while short would make the tail of the guide
      // unreachable - always prefer a temporary scrollbar to clipping.
      fr.setAttribute('scrolling', fr.clientHeight >= content ? 'no' : 'auto');
    } catch (e) {}
  }
  function refit() {
    fit();
    requestAnimationFrame(fit);
    setTimeout(fit, 150);
    setTimeout(fit, 500);   // re-measure after reflow settles
  }
  refit();
  window.addEventListener('load', refit);
  window.addEventListener('resize', refit);
  if (window.ResizeObserver) {
    // Observe BODY, not documentElement: this panel is first rendered
    // while its tab is hidden (zero height, nothing to measure), and
    // documentElement's height is pinned to the frame so it never
    // reports the change when the tab becomes visible.
    var ro = new ResizeObserver(fit);
    if (document.body) ro.observe(document.body);
  }
  document.addEventListener('toggle', fit, true);   // <details> sections
  document.addEventListener('visibilitychange', refit);
  setInterval(fit, 2000);   // cheap safety net; a no-op once fitted

  // In-page anchors (the section nav pills) must be intercepted here: a
  // srcdoc iframe has no URL of its own, so a plain click on href="#data"
  // resolves against the INHERITED base URL and navigates this frame to
  // the Streamlit app itself - rendering the whole dashboard inside its
  // own Architecture tab. Scroll to the target instead of navigating.
  document.addEventListener('click', function (ev) {
    var a = ev.target && ev.target.closest
        ? ev.target.closest('a[href^="#"]') : null;
    if (!a) return;
    ev.preventDefault();
    var el = document.getElementById(a.getAttribute('href').slice(1));
    var fr = window.frameElement;
    if (!el || !fr) return;
    try {
      // scrollIntoView stops at the iframe boundary here, so scroll the
      // dashboard's own scroll container (same-origin, reachable). The
      // frame is full-height/non-scrolling, so an element's viewport
      // offset inside it IS its offset from the frame's top.
      var pdoc = window.parent.document;
      var main = pdoc.querySelector('[data-testid="stMain"]')
          || pdoc.scrollingElement;
      var top = el.getBoundingClientRect().top
              + fr.getBoundingClientRect().top
              + main.scrollTop - 16;
      // instant, not smooth: smooth scrolling is animation-frame driven
      // and silently stalls in background/non-composited tabs
      main.scrollTop = Math.max(0, top);
    } catch (e) {
      el.scrollIntoView({block: 'start'});
    }
  }, true);
})();
</script>
"""

with tab_arch:
    arch_path = ROOT / "assets" / "architecture.html"
    if arch_path.exists():
        import streamlit.components.v1 as components
        # height=/scrolling= are only the pre-script fallback: the shim
        # grows the frame to the true content height. The fallback must sit
        # BELOW typical content, not above it: Streamlit reserves this many
        # pixels for the panel wrapper and that reservation resists inline
        # overrides, so a too-large fallback leaves thousands of pixels of
        # dead scroll after the footer (the frame growing LARGER than the
        # wrapper is fine - ancestors are overflow:visible and the page
        # scrolls to the frame's real bottom). scrolling=True keeps the
        # no-JS degradation a scrollbar rather than clipped content.
        components.html(arch_path.read_text(encoding="utf-8") + ARCH_AUTOFIT,
                        height=4200, scrolling=True)
    else:
        st.info("assets/architecture.html missing - pull the latest repo")
