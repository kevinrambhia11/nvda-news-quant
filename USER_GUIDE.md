# NVDA Quant Desk — Operator's Guide

*For someone sitting down at this desk for the first time. The technical
companion is [README.md](README.md).*

## What this is

An automated research desk for one stock — NVIDIA. Every day it reads the
world (news volume and tone from thousands of global sources via GDELT,
prices, the earnings calendar), runs that through statistical models, and
answers one practical question: **"How much NVDA should I be holding
tonight, and how much could it cost me if tomorrow goes badly?"**

It *recommends*; it never trades. You are the only one who places orders.

## The one number that matters

Open the dashboard (hosted on Streamlit Cloud, or locally via the
`quant-desk` launch config on port 8502). The green banner at the top is
the desk's product:

> **DESK POSITION (primary): hold NVDA at NN% of full size**

That number comes from the **volatility model**. You pick a risk budget
once (`VOL_TARGET_ANN` in config.py, currently 30% annualized volatility)
and the desk sizes the position daily so your *risk* stays constant as the
market's wildness changes. Calm market → bigger position; after a violent
day → it automatically trims you down.

Next to it, **VaR** translates the position into money: "on $1M of NVDA, a
bad day (worst 5%) loses about $X." If that number scares you, the position
is too big — that is the point of showing it. VaR is a parametric floor;
real tails are fatter.

Around earnings (the desk knows NVIDIA's report dates), the banner flags
**EARNINGS PRINT TONIGHT** and cuts size hard — print nights have the
largest overnight gaps.

## The blue banner — a weather vane, not a compass

Below the position is the **advisory direction** (BUY / HOLD / EXIT with a
probability). Full honesty: this desk tested whether news can predict
NVDA's daily *direction* — three ML toolchains, competitor news, earnings
features, intraday data — and the answer, on held-out data the models never
saw during selection, is **no meaningful edge**. Markets price public news
within seconds. The direction line stays because it usefully summarizes the
news mood; it is labeled "no demonstrated holdout edge" so nobody mistakes
it for the product.

**Volatility is predictable; direction is not.** The desk is built on the
first fact and honest about the second.

## The rest of the dashboard

Every metric carries a one-line explanation beneath it, and every tab opens
with a note on what it shows and why it matters.

- **Desk today**: the banners and metrics above, a live intraday price
  chart (5-minute, vs previous close), six months of price context, and
  the day's most bullish/bearish headlines with links to the articles.
- **Direction model**: tournament results (selection window vs untouched
  holdout) and a threshold explorer that recomputes the backtest live.
- **Volatility**: the vol model's out-of-sample record vs standard
  benchmarks (random walk, EWMA, HAR), plus a VaR calculator.
- **Technical charts**: price with SMA50/200 and Bollinger bands, RSI,
  MACD, realized volatility (the raw material behind position sizing),
  drawdown, and volume — with a 6M/1Y/2Y window selector. Context, not
  signal.
- **News & data**: NVIDIA news-tone history, competitor/industry series,
  cache freshness, and an on-demand scored headline scan.
- **Buttons**: *Regenerate signal* re-runs the whole pipeline with a fresh
  headline scrape (useful when big news breaks mid-day); *Refresh vol
  forecast* re-runs just sizing/VaR. Never required — the schedule owns the
  official record. On the hosted app, button output is ephemeral.

## What happens without you

Three Windows scheduled tasks on the desk machine:

| Task | When | Purpose |
|---|---|---|
| Headline Logger | every 30 min, 24/7 | archives timestamped news |
| Daily Signal | weekdays 17:00 IST (pre-US-open) | position + VaR + advisory; pushes so the hosted dashboard self-updates |
| Weekly Retrain | Saturdays 14:00 IST | re-runs both model tournaments on the week's data |

Every run appends to `artifacts\scheduler.log`. When a data source fails
(they do), the desk degrades to slightly-stale data and *says so* in the
output rather than crashing or pretending.

Retraining uses a tournament with an untouched holdout: a model is only
promoted if it earns it on data it never saw. Expect the winner to change
occasionally — and expect the honest conclusion (direction ≈ no edge) to
keep reasserting itself.

## Rules of engagement

1. **Paper first.** Track the desk's calls against outcomes for 4-6 weeks —
   every daily signal is archived automatically for exactly this purpose.
   Evidence, not backtests, decides whether real money follows.
2. **The desk proposes; you dispose.** It cannot place trades.
3. **Read the labels.** "Data through <older date>" means degraded mode —
   the desk never pretends.
4. **This is risk management, not a money printer.** The honest backtest:
   roughly market-like returns at about two-thirds of the drawdown.

Daily commitment: thirty seconds after 17:00 — green banner, VaR, done.

## Running it by hand

```cmd
cd /d "<project folder>"
<venv>\Scripts\python.exe main.py signal
```

Commands: `signal`, `vol-forecast`, `fetch`, `train`, `vol-train`,
`backtest`, `fuse`, `intraday-study`, `log-headlines`, `bq-probe`, `bqml`.
Manual runs write the same artifacts as scheduled ones, so prefer
`vol-forecast` for experiments and let the 17:00 task own the daily record.

---

*Educational software — not financial advice. Past performance does not
predict future results. Single-stock strategies carry concentration risk.*
