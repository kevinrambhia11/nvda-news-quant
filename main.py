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
    try:
        from model.news2vec import load_news2_features
        n2 = load_news2_features()
        if n2 is not None:
            ds = ds.join(n2, how="left")
    except Exception as exc:
        log.warning("news2 features unavailable (%s)", exc)
    try:
        from model.newsnet import load_newsnet_features
        nn = load_newsnet_features()
        if nn is not None:
            ds = ds.join(nn, how="left")
    except Exception as exc:
        log.warning("newsnet features unavailable (%s)", exc)
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
    # fusion numbers. The boundary comes from the sidecar written at train
    # time - a fraction of the saved file would misplace it whenever the
    # file starts at the nested boundary (news-candidate winners).
    split = None
    try:
        import json
        m = json.loads(config.OOS_META_PATH.read_text(encoding="utf-8"))
        split = pd.Timestamp(m["holdout_start"])
    except Exception:
        pass
    if split is None:
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


def cmd_bqml() -> None:
    from model.bqml import run_experiments
    print(run_experiments())


def _holdout_split(data: pd.DataFrame) -> pd.Timestamp:
    """The tournament holdout boundary: pinned via nested_meta when it
    exists (so production scorers train up to the SAME date every refresh),
    fraction rule otherwise."""
    from model.train import nested_meta
    oos_idx = data.index[config.MIN_TRAIN_DAYS:]
    m = nested_meta()
    if m is not None and oos_idx[0] < m["holdout_start"] <= oos_idx[-1]:
        return m["holdout_start"]
    return oos_idx[int(len(oos_idx) * (1 - config.HOLDOUT_FRACTION))]


def cmd_newsnet() -> None:
    """Train the attention-pooling NewsNet on pre-holdout days, build its
    daily features, and re-run the magnitude evaluation."""
    from model.magnitude import evaluate
    from model.newsnet import build_features, train
    from model.train import _clean
    data = _clean(_build())
    train(_holdout_split(data))
    build_features()
    print(evaluate(_build()))


def cmd_nested() -> None:
    """Nested split: retrain the news2 impact scorers and NewsNet on only
    the EARLY part of the selection window, leaving its tail genuinely clean
    for judging - which makes the news candidates eligible for the
    production crown in both tournaments. Writes nested_meta.json LAST so a
    crashed run never leaves the boundary pointing at missing artifacts."""
    import json
    from model.news2vec import build_daily_features, learn_impact
    from model.newsnet import build_features, train
    from model.train import _clean, nested_meta
    data = _clean(_build())
    oos_idx = data.index[config.MIN_TRAIN_DAYS:]
    prior = nested_meta()
    if prior is not None:
        # Boundaries are PINNED: a refresh retrains scorers and extends
        # feature coverage WITHOUT moving the goalposts - a drifting inner
        # would train new scorers on days earlier runs judged as clean, and
        # a drifting split would quietly consume reported holdout.
        inner, split = prior["inner"], prior["holdout_start"]
        log.info("Reusing pinned nested boundaries: inner %s, holdout %s",
                 inner.date(), split.date())
    else:
        split = oos_idx[int(len(oos_idx) * (1 - config.HOLDOUT_FRACTION))]
        sel_idx = oos_idx[oos_idx < split]
        inner = sel_idx[int(len(sel_idx) * config.NESTED_FRACTION)]
        log.info("Nested boundary %s | clean selection %s -> %s | holdout "
                 "from %s", inner.date(), inner.date(), sel_idx[-1].date(),
                 split.date())
    learn_impact(inner, nested=True)
    n2 = build_daily_features(nested=True)
    train(inner, nested=True)
    nn = build_features(nested=True)
    through = min(pd.Timestamp(n2.index.max()), pd.Timestamp(nn.index.max()))
    config.NESTED_META_PATH.write_text(
        json.dumps({"inner": str(inner.date()),
                    "holdout_start": str(split.date()),
                    "features_through": str(through.date())}, indent=2),
        encoding="utf-8")
    log.info("Nested artifacts complete (coverage through %s) -> %s",
             through.date(), config.NESTED_META_PATH.name)


def cmd_magnitude() -> None:
    from model.magnitude import evaluate, train_final
    ds = _build()
    print(evaluate(ds))
    print(train_final(ds))


def cmd_news_topup() -> None:
    """Incremental article-archive refresh from BigQuery: pull missing
    days, extend embeddings, rebuild production news features. Never fails
    the caller - the signal must run on stale coverage rather than not run.
    """
    from data.articles import topup
    try:
        info = topup()
        print(f"News archive: +{info['new_articles']} articles, "
              f"{info.get('embedded', 0)} embedded, through {info['through']}")
    except Exception as exc:
        log.warning("News top-up failed (%s) - live features keep their "
                    "previous coverage", exc)


def cmd_industry_backfill() -> None:
    """One-shot BigQuery backfill for the industry series (scheduled for a
    fresh monthly quota; the weekly retrain retries it until it lands).
    Refuses to run without ~300 GB of headroom; once the cache exists,
    config auto-routes the series to BigQuery forever."""
    from data.bigquery_gdelt import daily_series, load_bq_daily, month_usage_gb
    cache = config.CACHE / "bq_industry.csv"
    if cache.exists():
        print("Industry series already migrated to BigQuery - nothing to do")
        return
    terms = config.AUX_SERIES["industry"]["terms"]
    end = str((pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
               + pd.Timedelta(days=1)).date())
    cost = daily_series(terms, config.TRAIN_START, end,
                        dry_run=True) / 1024 ** 3
    used = month_usage_gb()
    print(f"Month usage {used:.0f} GB; backfill needs {cost:.0f} GB")
    if used + cost > 700:
        sys.exit("Not enough quota headroom this month - re-run after "
                 "the monthly reset")
    df = load_bq_daily("bq_industry", terms, refresh=True)
    print(f"Industry series backfilled: {len(df)} days "
          f"({df.index.min().date()} -> {df.index.max().date()}); the "
          "series now routes to BigQuery permanently")


def cmd_news2() -> None:
    """Learn article-impact scorers (pre-holdout only), build the daily
    news2 feature matrix, and run the magnitude-head evaluation."""
    from model.magnitude import evaluate
    from model.news2vec import build_daily_features, learn_impact
    ds = _build()
    from model.train import _clean
    data = _clean(ds)
    learn_impact(_holdout_split(data))
    build_daily_features()
    ds = _build()  # rebuild with news2 columns joined
    print(evaluate(ds))


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
    """Desk output, vol-first: the validated product (vol-sized long) leads;
    the direction model prints as advisory context."""
    from trade.signal import format_signal, generate_signal
    try:
        from model.volatility import forecast, format_forecast
        vol = forecast()
        h1 = vol["horizons"]["1"]
        print("=" * 62)
        print(f"  DESK POSITION (primary): hold NVDA at "
              f"{h1['target_vol_weight']:.0%} of full size (vol-sized long)")
        print(format_forecast(vol))
    except Exception as exc:
        log.warning("Vol forecast unavailable (%s) - advisory only", exc)
    signal = generate_signal(prefer_finbert=prefer_finbert)
    print(format_signal(signal))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command",
                        choices=["fetch", "train", "backtest", "signal",
                                 "vol-train", "vol-forecast", "fuse",
                                 "intraday-study", "log-headlines",
                                 "bq-probe", "bqml", "news2", "newsnet",
                                 "nested", "magnitude", "news-topup",
                                 "industry-backfill", "all"])
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
    elif args.command == "bqml":
        cmd_bqml()
    elif args.command == "news2":
        cmd_news2()
    elif args.command == "newsnet":
        cmd_newsnet()
    elif args.command == "nested":
        cmd_nested()
    elif args.command == "magnitude":
        cmd_magnitude()
    elif args.command == "news-topup":
        cmd_news_topup()
    elif args.command == "industry-backfill":
        cmd_industry_backfill()
    elif args.command == "all":
        cmd_fetch(refresh=args.refresh)
        cmd_train()
        cmd_backtest(frictionless=args.frictionless)
        cmd_signal(prefer_finbert=not args.no_finbert)


if __name__ == "__main__":
    main()
