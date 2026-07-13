"""Vectorized backtest over the walk-forward out-of-sample predictions.

Position for entry day d is decided from P(up) known before d's open and is
applied to the open(d) -> open(d+1) return, minus transaction costs on every
position change. Compared against buy-and-hold over the same dates.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)


def positions_from_probs(probs: pd.Series) -> pd.Series:
    """Long/flat (optionally short) with a hysteresis band to avoid churn."""
    pos = np.zeros(len(probs))
    current = 0.0
    for i, p in enumerate(probs.to_numpy()):
        if p > config.LONG_ENTER:
            current = 1.0
        elif config.ALLOW_SHORT and p < 1 - config.LONG_ENTER:
            current = -1.0
        elif current > 0 and p < config.LONG_EXIT:
            current = 0.0
        elif current < 0 and p > 1 - config.LONG_EXIT:
            current = 0.0
        pos[i] = current
    return pd.Series(pos, index=probs.index, name="pos")


def run_backtest(oos: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    df = oos.copy()
    df["pos"] = positions_from_probs(df["prob_up"])
    df["turnover"] = df["pos"].diff().abs()
    df.iloc[0, df.columns.get_loc("turnover")] = abs(df["pos"].iloc[0])
    df["strat_ret"] = df["pos"] * df["fwd_ret"] - df["turnover"] * config.COST_PER_TURNOVER
    df["strat_equity"] = (1 + df["strat_ret"]).cumprod()
    df["bh_equity"] = (1 + df["fwd_ret"]).cumprod()

    years = len(df) / 252

    def _stats(rets: pd.Series, equity: pd.Series) -> tuple:
        cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
        vol = rets.std() * np.sqrt(252)
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else np.nan
        maxdd = (equity / equity.cummax() - 1).min()
        return cagr, vol, sharpe, maxdd

    s: dict = {"days": len(df), "years": years,
               "start": str(df.index.min().date()), "end": str(df.index.max().date())}
    (s["strat_cagr"], s["strat_vol"], s["strat_sharpe"],
     s["strat_maxdd"]) = _stats(df["strat_ret"], df["strat_equity"])
    (s["bh_cagr"], s["bh_vol"], s["bh_sharpe"],
     s["bh_maxdd"]) = _stats(df["fwd_ret"], df["bh_equity"])
    s["strat_total"] = df["strat_equity"].iloc[-1] - 1
    s["bh_total"] = df["bh_equity"].iloc[-1] - 1
    active = df[df["pos"] != 0]
    s["exposure"] = float((df["pos"] != 0).mean())
    s["hit_rate"] = (float((np.sign(active["pos"]) == np.sign(active["fwd_ret"])).mean())
                     if len(active) else np.nan)
    s["position_changes"] = int((df["turnover"] > 0).sum())

    df.to_csv(config.EQUITY_CURVE_PATH)
    _maybe_plot(df)
    return s, df


def format_report(s: dict) -> str:
    lines = [
        "=" * 62,
        f"  NVDA news-sentiment strategy - out-of-sample backtest",
        f"  {s['start']} -> {s['end']}  ({s['days']} trading days, "
        f"{s['years']:.1f} years)",
        "=" * 62,
        f"  {'':24}{'Strategy':>14}{'Buy & hold':>14}",
        f"  {'Total return':24}{s['strat_total']:>13.1%}{s['bh_total']:>13.1%}",
        f"  {'CAGR':24}{s['strat_cagr']:>13.1%}{s['bh_cagr']:>13.1%}",
        f"  {'Ann. volatility':24}{s['strat_vol']:>13.1%}{s['bh_vol']:>13.1%}",
        f"  {'Sharpe (rf=0)':24}{s['strat_sharpe']:>13.2f}{s['bh_sharpe']:>13.2f}",
        f"  {'Max drawdown':24}{s['strat_maxdd']:>13.1%}{s['bh_maxdd']:>13.1%}",
        "-" * 62,
        f"  Market exposure:    {s['exposure']:.1%} of days",
        f"  Hit rate (active):  {s['hit_rate']:.1%}",
        f"  Position changes:   {s['position_changes']} "
        f"(cost {config.COST_PER_TURNOVER:.2%} per change)",
        "=" * 62,
    ]
    return "\n".join(lines)


def _maybe_plot(df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
        axes[0].plot(df.index, df["strat_equity"], label="News-sentiment strategy")
        axes[0].plot(df.index, df["bh_equity"], label="Buy & hold NVDA", alpha=0.7)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Growth of $1 (log)")
        axes[0].legend()
        axes[0].set_title("NVDA news-sentiment strategy vs buy & hold (out-of-sample)")
        axes[1].fill_between(df.index, df["pos"], step="mid", alpha=0.5)
        axes[1].set_ylabel("Position")
        fig.tight_layout()
        fig.savefig(config.EQUITY_PLOT_PATH, dpi=120)
        plt.close(fig)
        log.info("Equity curve plot -> %s", config.EQUITY_PLOT_PATH)
    except Exception as exc:  # matplotlib genuinely optional
        log.info("Skipping plot (%s)", exc)
