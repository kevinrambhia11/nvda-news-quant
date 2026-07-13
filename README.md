# NVDA News-Sentiment Quant System

An end-to-end pipeline that turns web-wide news sentiment about NVIDIA into a
daily machine-learning trading signal, with an honest walk-forward backtest.

> **Disclaimer** — educational software, not financial advice. Past
> performance does not predict future results. Single-stock strategies carry
> concentration risk; you alone decide whether to place any trade.

## How it works

```
GDELT (global news, daily tone     Yahoo Finance (daily OHLCV,
+ article counts since 2017)       NVDA + QQQ benchmark)
        │                                 │
        └────────► feature builder ◄──────┘
                        │  17 features: tone level/momentum/z-score,
                        │  news-volume spikes, returns, RSI, SMA
                        │  distances, volatility, relative strength
                        ▼
        walk-forward gradient boosting (expanding window,
        retrained monthly, all predictions out-of-sample)
                        │
          ┌─────────────┴──────────────┐
          ▼                            ▼
   backtest engine               live signal
   (open-to-open returns,        ACTION from the backtested model rule;
   costs, hysteresis band,       headlines scraped NOW from Google News,
   vs buy & hold)                Yahoo, Finviz, StockTwits (scored by
                                 FinBERT/VADER) shown as advisory context
```

**Why GDELT for history?** Training an ML model needs *years* of aligned
news sentiment. Scraping news sites retroactively is impossible; GDELT has
already monitored and tone-scored worldwide online news since 2017 and
exposes it through a free API. The live scrapers then add up-to-the-minute
headlines the daily series hasn't absorbed yet.

## No-lookahead design

Each dataset row is an **entry day** `d`. Features may only use news through
calendar day `d-1` (a UTC news day ends ~13 h before the next US open) and
prices through the close of the prior trading day. The target is the
open-to-open return `Open(d) → Open(d+1)` — exactly what you capture trading
at the open on a signal generated the previous evening. Every backtested
prediction comes from a model trained only on earlier data.

## Setup

```powershell
# venv outside OneDrive (OneDrive sync chokes on venvs)
python -m venv $env:USERPROFILE\venvs\nvda-quant
& $env:USERPROFILE\venvs\nvda-quant\Scripts\Activate.ps1
pip install -r requirements.txt
# optional, better sentiment (large): pip install torch transformers
```

## Usage

```powershell
python main.py all        # first run: fetch + train + backtest + today's signal
python main.py signal     # daily: regenerate the live signal (run pre-open)
python main.py train      # periodic: refresh data and retrain
python main.py backtest   # re-print the out-of-sample report
```

Outputs land in `artifacts/`: `model.joblib`, `oos_predictions.csv`,
`equity_curve.csv/.png`, `backtest_report.txt`, `signal_YYYYMMDD.json`.

Key knobs live in `config.py`: entry/exit probability thresholds
(`LONG_ENTER`/`LONG_EXIT`), transaction costs, shorting on/off, walk-forward
cadence, and the model-vs-headlines blend weights.

## Daily routine

Run `python main.py signal` between 20:00 ET (00:00 UTC, when the news day
completes) and the next US open. The signal names the entry day, the model
probability, fresh headline sentiment with the most positive and negative
stories, and a BUY / HOLD / FLAT action. The action comes from the
backtested model rule; the headline blend is advisory context. Run at other
times, the pipeline automatically drops in-progress price bars and partial
news days so the model only ever sees what it saw in training. Execute (or
ignore) the trade at the open with your US-equities broker.

## Desk volatility model (EMH-consistent)

Under an efficient market the direction model above is a null benchmark by
assumption — but the *second moment* is still predictable, and that is what
a desk models first. `python main.py vol-train` walk-forward-evaluates
forecasts of log Garman-Klass volatility at 1- and 5-day horizons against
the desk classics (random walk, RiskMetrics EWMA, HAR) plus a price-only
ablation, then saves whichever candidate wins on out-of-sample QLIKE.
`python main.py vol-forecast` produces today's vol forecast, target-vol
position sizing, and a parametric 1-day VaR (normal — treat as a floor).

Empirical result on NVDA 2018-2026: vol is highly predictable (OOS R² 0.37
at 1 day, 0.45 at 5 days), the parsimonious HAR beats gradient boosting,
and daily-granularity news features add ~nothing once HAR vol lags are in —
consistent with prices impounding news arrival within the day.

`python main.py backtest --frictionless` re-runs the direction backtest
with zero transaction costs (the idealized-execution bound).

## Dashboard

`streamlit run dashboard.py` (or start "quant-desk" from `.claude/launch.json`,
port 8502): today's signal and vol forecast with regenerate buttons, the
candidate-selection report, a live threshold explorer over the OOS
predictions, forecast-vs-realized vol charts, a VaR calculator, GDELT tone
history, cache health, and on-demand live headline scans.

## Model selection discipline

`python main.py train` now compares four candidates (GBM base, GBM + vol
context, deep GBM, logistic) walk-forward, selects by strategy Sharpe on a
SELECTION window (first 60% of OOS), and reports the winner on the untouched
HOLDOUT tail. Current honest verdict: the winner's selection-window Sharpe
(1.51) collapses to 0.97 on the 2023-2026 holdout with AUC below 0.5 —
the direction edge does not demonstrably survive the recent regime. Treat
the direction signal as context; the volatility model is the workhorse.

## Honest limitations

- **Daily bars only.** News is absorbed into prices within minutes; a daily
  system captures slower sentiment drift, not the initial jump.
- **Backtest ≠ future.** Even a strictly out-of-sample backtest overstates
  live results (data revisions, fill quality, regime change).
- **GDELT tone is generic**, not finance-tuned. The live overlay can use
  FinBERT, but the trained features use GDELT's tone score.
- **Free data sources** (Yahoo, Finviz, StockTwits, Reddit) rate-limit or
  change markup occasionally; each source degrades gracefully.

## Extension ideas

Intraday bars + FinBERT on raw headlines with timestamps; earnings-date
features; options-implied sentiment (put/call skew); position sizing by
Kelly fraction of the model's edge; ensemble with a mean-reversion sleeve.
