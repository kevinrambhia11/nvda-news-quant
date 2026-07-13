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


def _leg_stats(weights: pd.Series, fwd_ret: pd.Series) -> tuple[dict, pd.Series]:
    turnover = weights.diff().abs()
    if len(turnover):
        turnover.iloc[0] = abs(weights.iloc[0])
    rets = weights * fwd_ret - turnover * config.COST_PER_TURNOVER
    equity = (1 + rets).cumprod()
    years = len(rets) / 252
    sd = rets.std()
    return ({"cagr": equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
             "sharpe": rets.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
             "maxdd": (equity / equity.cummax() - 1).min(),
             "avg_exposure": float(weights.abs().mean())}, equity)


def run_fused_backtest(oos: pd.DataFrame, vol_oos: pd.DataFrame,
                       px: pd.DataFrame, earn_dates=None) -> tuple[dict, pd.DataFrame]:
    """Item-2 fusion: the direction model sets the SIGN, the vol model sets
    the SIZE (target-vol weight on total next-day vol, capped at 1 - no
    leverage). Benchmarked against direction-only and a vol-sized always-long
    leg, all on the same days with identical costs.

    The intraday GK vol forecast is scaled to total vol with a trailing-2y
    (intraday+overnight)/intraday variance ratio known at d-1; on earnings
    entry days the print-night gap variance (expanding mean over strictly
    PAST prints) replaces the ordinary overnight component, so target-vol
    sizing actually derisks the ~4 print nights per year."""
    from features.build import garman_klass_vol, print_gap_sq
    df = oos.join(vol_oos[["pred_selected"]], how="inner")
    if df.empty:
        raise RuntimeError("Direction and volatility OOS windows do not overlap")

    gap = np.log(px["Open"] / px["Close"].shift(1))
    gk_var = garman_klass_vol(px) ** 2
    ratio = ((gk_var + gap ** 2).rolling(504).mean()
             / gk_var.rolling(504).mean()).shift(1)
    ratio = ratio.reindex(df.index).ffill()
    var_intra = np.exp(2 * df["pred_selected"])
    total_var = var_intra * ratio
    if earn_dates is not None and len(earn_dates):
        past_gap_var = (print_gap_sq(px, earn_dates)
                        .expanding().mean().shift(1).reindex(df.index))
        mask = past_gap_var.notna()
        total_var[mask] = var_intra[mask] + past_gap_var[mask]
    total_ann = np.sqrt(total_var) * np.sqrt(252)
    size = (config.VOL_TARGET_ANN / total_ann).clip(upper=1.0)

    sign = positions_from_probs(df["prob_up"])
    legs = {
        "direction-only": sign,
        "fused (sign x vol size)": sign * size,
        "vol-sized long": size,
        "buy & hold": pd.Series(1.0, index=df.index),
    }
    stats, curves = {}, pd.DataFrame(index=df.index)
    for name, w in legs.items():
        stats[name], curves[name] = _leg_stats(w, df["fwd_ret"])
    curves.to_csv(config.FUSED_CURVE_PATH)
    return stats, curves


def format_fused_report(stats: dict, index) -> str:
    lines = ["=" * 66,
             "  Direction x volatility fusion - untouched holdout only",
             "  (days used to select either model are excluded)",
             f"  {index.min().date()} -> {index.max().date()}  "
             f"({len(index)} trading days)",
             f"  (sizing: {config.VOL_TARGET_ANN:.0%} ann target vol, "
             f"no leverage, costs {config.COST_PER_TURNOVER:.2%}/change)",
             "=" * 66,
             f"  {'leg':<26}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'AvgExp':>8}"]
    for name, s in stats.items():
        lines.append(f"  {name:<26}{s['cagr']:>9.1%}{s['sharpe']:>8.2f}"
                     f"{s['maxdd']:>9.1%}{s['avg_exposure']:>8.1%}")
    lines.append("=" * 66)
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
