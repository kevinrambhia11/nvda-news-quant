"""NVDA news-sentiment quant system - command line entry point.

Usage (from the project root, with the venv active):
    python main.py fetch      # download & cache news sentiment + prices
    python main.py train      # build features, walk-forward train, save model
    python main.py backtest   # evaluate out-of-sample performance
    python main.py signal     # generate today's live trading signal
    python main.py all        # fetch -> train -> backtest -> signal
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

import config

# Scraped headlines contain emoji; redirected Windows stdout defaults to
# cp1252 and would raise UnicodeEncodeError on print without this.
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("main")


def _load_earnings():
    from data.earnings import load_earnings_dates
    try:
        return load_earnings_dates()
    except Exception as exc:
        log.warning("Earnings calendar unavailable (%s); event features "
                    "fall back to neutral values", exc)
        return None


def cmd_fetch(refresh: bool = False) -> None:
    from data.news import load_gdelt_daily
    from data.prices import load_prices
    px, bench = load_prices(refresh=refresh)
    log.info("Prices: %s %d rows (through %s), %s %d rows", config.TICKER,
             len(px), px.index.max().date(), config.BENCHMARK, len(bench))
    gdelt = load_gdelt_daily(refresh=refresh)
    log.info("GDELT daily sentiment: %d days (%s -> %s)", len(gdelt),
             gdelt.index.min().date(), gdelt.index.max().date())
    earn = _load_earnings()
    if earn is not None:
        log.info("Earnings calendar: %d dates (%s -> %s)", len(earn),
                 earn.min().date(), earn.max().date())


def _build() -> pd.DataFrame:
    from data.news import load_aux_gdelt, load_gdelt_daily
    from data.prices import load_prices
    from features.build import build_dataset
    px, bench = load_prices()
    gdelt = load_gdelt_daily()
    aux = load_aux_gdelt()
    ds = build_dataset(px, bench, gdelt, earn_dates=_load_earnings(), aux=aux)
    ds.to_csv(config.FEATURES_PATH)
    log.info("Dataset: %d rows x %d cols (aux series: %s) -> %s", len(ds),
             ds.shape[1], sorted(aux) or "none", config.FEATURES_PATH)
    return ds


def cmd_train() -> None:
    from model.train import select_and_train
    ds = _build()
    print(select_and_train(ds))


def cmd_backtest(frictionless: bool = False) -> None:
    from backtest.engine import format_report, run_backtest
    if not config.OOS_PREDICTIONS_PATH.exists():
        sys.exit("No OOS predictions found - run `python main.py train` first")
    if frictionless:
        config.COST_PER_TURNOVER = 0.0
    oos = pd.read_csv(config.OOS_PREDICTIONS_PATH, index_col=0, parse_dates=[0])
    stats, _ = run_backtest(oos)
    report = format_report(stats)
    if frictionless:
        report += "\n  NOTE: frictionless run - zero transaction costs assumed."
    config.REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)


def cmd_fuse() -> None:
    from backtest.engine import format_fused_report, run_fused_backtest
    from data.prices import load_prices
    for path, cmd in ((config.OOS_PREDICTIONS_PATH, "train"),
                      (config.VOL_OOS_PATH, "vol-train")):
        if not path.exists():
            sys.exit(f"Missing {path.name} - run `python main.py {cmd}` first")
    oos = pd.read_csv(config.OOS_PREDICTIONS_PATH, index_col=0, parse_dates=[0])
    # Report only the direction model's untouched holdout tail: earlier days
    # were used to SELECT both models, so including them would flatter the
    # fusion numbers.
    split = oos.index[int(len(oos) * (1 - config.HOLDOUT_FRACTION))]
    oos = oos.loc[oos.index >= split]
    vol_oos = pd.read_csv(config.VOL_OOS_PATH, index_col=0, parse_dates=[0])
    px, _ = load_prices()
    stats, curves = run_fused_backtest(oos, vol_oos, px,
                                       earn_dates=_load_earnings())
    report = format_fused_report(stats, curves.index)
    config.FUSED_REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)


def cmd_bq_probe() -> None:
    from data.bigquery_gdelt import probe
    print(probe())


def cmd_intraday_study() -> None:
    from intraday.study import run_study
    print(run_study())


def cmd_log_headlines() -> None:
    from intraday.logger import log_once
    print(f"{log_once()} new headlines logged -> {config.HEADLINE_LOG_PATH}")


def cmd_vol_train() -> None:
    from model.volatility import train_and_report
    print(train_and_report())


def cmd_vol_forecast() -> None:
    from model.volatility import forecast, format_forecast
    print(format_forecast(forecast()))


def cmd_signal(prefer_finbert: bool = True) -> None:
    from trade.signal import format_signal, generate_signal
    signal = generate_signal(prefer_finbert=prefer_finbert)
    print(format_signal(signal))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command",
                        choices=["fetch", "train", "backtest", "signal",
                                 "vol-train", "vol-forecast", "fuse",
                                 "intraday-study", "log-headlines",
                                 "bq-probe", "all"])
    parser.add_argument("--refresh", action="store_true",
                        help="force re-download of cached data")
    parser.add_argument("--no-finbert", action="store_true",
                        help="use VADER even if transformers is installed")
    parser.add_argument("--frictionless", action="store_true",
                        help="backtest with zero transaction costs")
    args = parser.parse_args()

    if args.command == "fetch":
        cmd_fetch(refresh=args.refresh)
    elif args.command == "train":
        cmd_fetch(refresh=args.refresh)
        cmd_train()
    elif args.command == "backtest":
        cmd_backtest(frictionless=args.frictionless)
    elif args.command == "signal":
        cmd_signal(prefer_finbert=not args.no_finbert)
    elif args.command == "vol-train":
        cmd_fetch(refresh=args.refresh)
        cmd_vol_train()
    elif args.command == "vol-forecast":
        cmd_vol_forecast()
    elif args.command == "fuse":
        cmd_fuse()
    elif args.command == "intraday-study":
        cmd_intraday_study()
    elif args.command == "log-headlines":
        cmd_log_headlines()
    elif args.command == "bq-probe":
        cmd_bq_probe()
    elif args.command == "all":
        cmd_fetch(refresh=args.refresh)
        cmd_train()
        cmd_backtest(frictionless=args.frictionless)
        cmd_signal(prefer_finbert=not args.no_finbert)


if __name__ == "__main__":
    main()
