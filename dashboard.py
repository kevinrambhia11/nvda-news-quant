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

st.set_page_config(page_title="NVDA Quant Desk", page_icon="📈", layout="wide")


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
        if url:
            st.markdown(prefix + f"[{title}]({url})")
        else:
            st.markdown(prefix + title)


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


# ---------------------------------------------------------------------------
# Local backtest helpers (parameterized, so sliders never mutate config)
# ---------------------------------------------------------------------------

def positions(probs: pd.Series, enter: float, exit_: float,
              allow_short: bool) -> pd.Series:
    pos = np.zeros(len(probs))
    current = 0.0
    for i, p in enumerate(probs.to_numpy()):
        if p > enter:
            current = 1.0
        elif allow_short and p < 1 - enter:
            current = -1.0
        elif current > 0 and p < exit_:
            current = 0.0
        elif allow_short and current < 0 and p > 1 - exit_:
            current = 0.0
        pos[i] = current
    return pd.Series(pos, index=probs.index)


def quick_backtest(oos: pd.DataFrame, enter: float, exit_: float,
                   allow_short: bool, cost: float):
    df = oos.copy()
    df["pos"] = positions(df["prob_up"], enter, exit_, allow_short)
    df["turnover"] = df["pos"].diff().abs()
    df.iloc[0, df.columns.get_loc("turnover")] = abs(df["pos"].iloc[0])
    df["strat_ret"] = df["pos"] * df["fwd_ret"] - df["turnover"] * cost
    df["strategy"] = (1 + df["strat_ret"]).cumprod()
    df["buy_hold"] = (1 + df["fwd_ret"]).cumprod()
    years = len(df) / 252
    sd = df["strat_ret"].std()
    stats = {
        "cagr": df["strategy"].iloc[-1] ** (1 / years) - 1,
        "sharpe": df["strat_ret"].mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
        "maxdd": (df["strategy"] / df["strategy"].cummax() - 1).min(),
        "exposure": (df["pos"] != 0).mean(),
        "bh_cagr": df["buy_hold"].iloc[-1] ** (1 / years) - 1,
        "bh_sharpe": df["fwd_ret"].mean() / df["fwd_ret"].std() * np.sqrt(252),
        "bh_maxdd": (df["buy_hold"] / df["buy_hold"].cummax() - 1).min(),
        "changes": int((df["turnover"] > 0).sum()),
    }
    return stats, df


st.title("NVDA Quant Desk")
st.caption("News-sentiment direction signal + EMH-consistent volatility desk. "
           "Educational tool - not financial advice.")

tab_today, tab_dir, tab_vol, tab_tech, tab_news = st.tabs(
    ["Desk today", "Direction model", "Volatility", "Technical charts",
     "News & data"])


# ---------------------------------------------------------------------------
# Tab 1: Desk today
# ---------------------------------------------------------------------------
with tab_today:
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

    if signal:
        m = st.columns(5)
        m[0].metric("Model P(up)", f"{signal['model_prob_up']:.1%}",
                    help=f"Long > {config.LONG_ENTER}, exit < {config.LONG_EXIT}")
        m[1].metric("Headline sentiment", f"{signal['headline_sentiment']:+.3f}",
                    help=f"{signal['headline_count']} items, "
                         f"{signal['sentiment_backend']}")
        m[2].metric("StockTwits bulls/bears",
                    f"{signal['stocktwits_bulls']}/{signal['stocktwits_bears']}")
        m[3].metric("Last close", f"${signal['last_close']}")
        if vol:
            h1 = vol["horizons"]["1"]
            m[4].metric("1-day total vol", f"{h1['total_daily_vol']:.2%}",
                        help="Includes overnight gap risk")

    if vol:
        h1 = vol["horizons"]["1"]
        v = st.columns(4)
        v[0].metric("Annualized vol", f"{h1['total_annualized_vol']:.1%}")
        v[1].metric("Target-vol sizing", f"{h1['target_vol_weight']:.0%}",
                    help=f"{config.VOL_TARGET_ANN:.0%} annual target")
        v[2].metric("VaR 95% ($1M)", f"${h1['var_95']:,.0f}")
        v[3].metric("VaR 99% ($1M)", f"${h1['var_99']:,.0f}")

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
        cpos, cneg = st.columns(2)
        with cpos:
            st.subheader("Most positive headlines")
            headline_lines(signal["most_positive"])
        with cneg:
            st.subheader("Most negative headlines")
            headline_lines(signal["most_negative"])


# ---------------------------------------------------------------------------
# Tab 2: Direction model
# ---------------------------------------------------------------------------
with tab_dir:
    selection = read_text(str(config.MODEL_SELECTION_PATH))
    if selection:
        st.subheader("Candidate selection (selection window vs holdout)")
        st.code(selection, language=None)

    oos = read_csv(str(config.OOS_PREDICTIONS_PATH))
    if oos is None:
        st.info("No out-of-sample predictions - run `python main.py train`.")
    else:
        st.subheader("Threshold explorer (out-of-sample, recomputed live)")
        s1, s2, s3, s4 = st.columns(4)
        enter = s1.slider("Long enter (P(up) >)", 0.50, 0.70,
                          float(config.LONG_ENTER), 0.01)
        exit_ = s2.slider("Long exit (P(up) <)", 0.30, 0.55,
                          float(config.LONG_EXIT), 0.01)
        cost = s3.slider("Cost per change (bps)", 0, 30,
                         int(config.COST_PER_TURNOVER * 10_000), 1) / 10_000
        allow_short = s4.checkbox("Allow shorts", value=config.ALLOW_SHORT)
        if exit_ >= enter:
            st.error("Exit threshold must be below the entry threshold.")
        else:
            stats, bt = quick_backtest(oos, enter, exit_, allow_short, cost)
            k = st.columns(6)
            k[0].metric("CAGR", f"{stats['cagr']:.1%}",
                        f"{stats['cagr'] - stats['bh_cagr']:+.1%} vs B&H")
            k[1].metric("Sharpe", f"{stats['sharpe']:.2f}",
                        f"{stats['sharpe'] - stats['bh_sharpe']:+.2f} vs B&H")
            k[2].metric("Max drawdown", f"{stats['maxdd']:.1%}",
                        f"{stats['maxdd'] - stats['bh_maxdd']:+.1%} vs B&H",
                        delta_color="inverse")
            k[3].metric("Exposure", f"{stats['exposure']:.0%}")
            k[4].metric("Position changes", stats["changes"])
            k[5].metric("B&H Sharpe", f"{stats['bh_sharpe']:.2f}")

            st.caption("Growth of $1 (log scale)")
            st.line_chart(np.log10(bt[["strategy", "buy_hold"]]))
            st.caption("21-day average model probability")
            st.line_chart(oos["prob_up"].rolling(21).mean())

    fused_report = read_text(str(config.FUSED_REPORT_PATH))
    if fused_report:
        st.subheader("Direction x volatility fusion")
        st.code(fused_report, language=None)
        fused = read_csv(str(config.FUSED_CURVE_PATH))
        if fused is not None:
            st.caption("Growth of $1 (log scale)")
            st.line_chart(np.log10(fused))


# ---------------------------------------------------------------------------
# Tab 3: Volatility
# ---------------------------------------------------------------------------
with tab_vol:
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
# Tab 4: Technical charts
# ---------------------------------------------------------------------------
with tab_tech:
    px_full = read_csv(str(config.CACHE / f"prices_{config.TICKER}.csv"))
    if px_full is None or len(px_full) < 260:
        st.info("Price cache not ready - run `python main.py fetch`.")
    else:
        window = st.radio("Window", ["6M", "1Y", "2Y"], index=1,
                          horizontal=True)
        n = {"6M": 126, "1Y": 252, "2Y": 504}[window]
        c, v = px_full["Close"], px_full["Volume"]

        # --- price with SMAs and Bollinger band (full width) ---------------
        mid, sd = c.rolling(20).mean(), c.rolling(20).std()
        tail = c.index[-n:]
        band = pd.DataFrame({"date": tail,
                             "upper": (mid + 2 * sd).iloc[-n:].to_numpy(),
                             "lower": (mid - 2 * sd).iloc[-n:].to_numpy()})
        price_df = pd.DataFrame({"close": c.iloc[-n:],
                                 "SMA50": c.rolling(50).mean().iloc[-n:],
                                 "SMA200": c.rolling(200).mean().iloc[-n:]})
        lo = float(min(band["lower"].min(), price_df.min().min()))
        hi = float(max(band["upper"].max(), price_df.max().max()))
        pad = (hi - lo) * 0.05
        area = (alt.Chart(band).mark_area(opacity=0.15)
                .encode(x=alt.X("date:T", title=None),
                        y=alt.Y("lower:Q", title="price ($)",
                                scale=alt.Scale(domain=[lo - pad, hi + pad])),
                        y2="upper:Q"))
        st.caption("Price with SMA50 / SMA200 and Bollinger band (20, 2 sd)")
        st.altair_chart(area + fitted_lines(price_df, height=340),
                        use_container_width=True)

        col_l, col_r = st.columns(2)

        with col_l:
            # --- RSI(14) with 30/70 guides ---------------------------------
            delta = c.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rsi = (100 - 100 / (1 + gain / loss)).iloc[-n:]
            st.caption("RSI (14) - overbought > 70, oversold < 30")
            rsi_data = rsi.rename_axis("date").reset_index(name="RSI")
            rsi_line = (alt.Chart(rsi_data).mark_line()
                        .encode(x=alt.X("date:T", title=None),
                                y=alt.Y("RSI:Q",
                                        scale=alt.Scale(domain=[0, 100])),
                                tooltip=["date:T",
                                         alt.Tooltip("RSI:Q", format=".1f")]))
            guides = (alt.Chart(pd.DataFrame({"y": [30, 70]}))
                      .mark_rule(strokeDash=[4, 4], opacity=0.6)
                      .encode(y="y:Q"))
            st.altair_chart((rsi_line + guides).properties(height=220),
                            use_container_width=True)

            # --- realized volatility ---------------------------------------
            from features.build import garman_klass_vol
            gk_ann = (garman_klass_vol(px_full).rolling(21).mean()
                      * np.sqrt(252) * 100).iloc[-n:]
            cc_ann = (c.pct_change().rolling(21).std()
                      * np.sqrt(252) * 100).iloc[-n:]
            st.caption("Realized volatility, 21-day, annualized (%)")
            st.altair_chart(fitted_lines(pd.DataFrame(
                {"Garman-Klass": gk_ann, "close-to-close": cc_ann}),
                y_title="vol (%)", height=220, fmt=".1f"),
                use_container_width=True)

        with col_r:
            # --- MACD (12, 26, 9) ------------------------------------------
            ema12 = c.ewm(span=12, adjust=False).mean()
            ema26 = c.ewm(span=26, adjust=False).mean()
            macd = (ema12 - ema26).iloc[-n:]
            sig = (ema12 - ema26).ewm(span=9, adjust=False).mean().iloc[-n:]
            hist = (macd - sig)
            st.caption("MACD (12, 26, 9)")
            hist_data = hist.rename_axis("date").reset_index(name="hist")
            bars = (alt.Chart(hist_data).mark_bar(opacity=0.4)
                    .encode(x=alt.X("date:T", title=None),
                            y=alt.Y("hist:Q", title="MACD"),
                            color=alt.condition(alt.datum.hist > 0,
                                                alt.value("#2a9d64"),
                                                alt.value("#c0504d"))))
            macd_lines = fitted_lines(pd.DataFrame(
                {"MACD": macd, "signal": sig}), y_title="MACD", height=220)
            st.altair_chart(bars + macd_lines, use_container_width=True)

            # --- drawdown ---------------------------------------------------
            dd = (c / c.cummax() - 1).iloc[-n:] * 100
            st.caption("Drawdown from all-time high (%)")
            dd_data = dd.rename_axis("date").reset_index(name="drawdown")
            st.altair_chart(
                (alt.Chart(dd_data).mark_area(opacity=0.5, color="#c0504d")
                 .encode(x=alt.X("date:T", title=None),
                         y=alt.Y("drawdown:Q", title="drawdown (%)"),
                         tooltip=["date:T",
                                  alt.Tooltip("drawdown:Q", format=".1f")])
                 .properties(height=220)), use_container_width=True)

        # --- volume (full width) --------------------------------------------
        st.caption("Volume with 20-day average")
        vol_data = (v.iloc[-n:] / 1e6).rename_axis("date").reset_index(name="volume")
        vol_bars = (alt.Chart(vol_data).mark_bar(opacity=0.5)
                    .encode(x=alt.X("date:T", title=None),
                            y=alt.Y("volume:Q", title="shares (millions)"),
                            tooltip=["date:T",
                                     alt.Tooltip("volume:Q", format=".0f")]))
        vol_avg = ((v.rolling(20).mean() / 1e6).iloc[-n:]
                   .rename_axis("date").reset_index(name="avg20"))
        avg_line = (alt.Chart(vol_avg).mark_line(color="#e8a838")
                    .encode(x="date:T", y="avg20:Q"))
        st.altair_chart((vol_bars + avg_line).properties(height=200),
                        use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 5: News & data
# ---------------------------------------------------------------------------
with tab_news:
    g = read_csv(str(config.CACHE / "gdelt_daily.csv"))
    if g is not None:
        st.subheader("GDELT news tone (30-day average) and article volume")
        st.line_chart(g["tone"].rolling(30).mean())
        st.area_chart(g["art_count"].rolling(7).mean())

    st.subheader("Competitors & industry")
    st.caption("Cross-name news series feeding the direction model's "
               "'GBM deep + cross' candidate. Parameters live in "
               "config.AUX_SERIES.")
    tone_panel = {}
    for series, spec in config.AUX_SERIES.items():
        a = read_csv(str(config.CACHE / f"gdelt_{series}.csv"))
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown(f"**{series}**")
            st.caption(spec["label"])
        with c2:
            if a is None or a.empty:
                st.info("Series not bootstrapped yet - history download "
                        "pending (GDELT rate limits).")
            else:
                m = st.columns(3)
                m[0].metric("Last tone", f"{a['tone'].iloc[-1]:+.2f}")
                m[1].metric("Tone 7d avg",
                            f"{a['tone'].rolling(7).mean().iloc[-1]:+.2f}")
                base = a["art_count"].rolling(30).mean().iloc[-1]
                spike = (a["art_count"].iloc[-1] / base) if base else np.nan
                m[2].metric("Volume vs 30d", f"{spike:.1f}x")
                tone_panel[series] = a["tone"].rolling(7).mean()
    if tone_panel:
        nv = read_csv(str(config.CACHE / "gdelt_daily.csv"))
        if nv is not None:
            tone_panel["NVIDIA"] = nv["tone"].rolling(7).mean()
        st.caption("7-day average tone: NVIDIA vs competitors vs industry")
        st.line_chart(pd.DataFrame(tone_panel))

    st.subheader("Data caches")
    rows = []
    for f in sorted(config.CACHE.glob("*.csv")):
        try:
            head = pd.read_csv(f, index_col=0, parse_dates=[0], nrows=0)
            last = pd.read_csv(f, index_col=0, parse_dates=[0]).index.max()
        except Exception:
            last = None
        rows.append({"file": f.name,
                     "last data day": str(last.date()) if last is not None else "?",
                     "modified": datetime.fromtimestamp(
                         f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)

    st.subheader("Live headline scan")
    if st.button("Scrape and score headlines now"):
        with st.spinner("Scraping Google News, Yahoo, Finviz, StockTwits..."):
            from data.news import collect_live_headlines
            from sentiment.analyzer import SentimentAnalyzer
            items = collect_live_headlines()
            scores = SentimentAnalyzer(prefer_finbert=False).score(
                [h.get("title", "") for h in items])
        if not items:
            st.warning("No headlines returned - sources may be blocked "
                       "on this network.")
        else:
            df = pd.DataFrame({
                "score": scores,
                "source": [h["source"] for h in items],
                "headline": [h["title"] for h in items],
                "link": [h.get("url", "") for h in items],
            }).sort_values("score")
            st.metric("Mean sentiment", f"{df['score'].mean():+.3f}",
                      help=f"{len(df)} unique items")
            st.dataframe(df, use_container_width=True, hide_index=True,
                         column_config={
                             "link": st.column_config.LinkColumn("link")})
