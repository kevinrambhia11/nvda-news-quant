"""BigQuery ML experiments: train in the warehouse, judge locally.

Two models, per the desk's standing question list:
  * BOOSTED_TREE_CLASSIFIER (XGBoost) on the direction dataset - trained on
    everything BEFORE the tournament's holdout boundary, scored on the same
    untouched holdout with the same strategy stats as every local candidate.
    Single-fit (no monthly refits), a slight handicap vs the walk-forward
    tournament - noted in the report.
  * ARIMA_PLUS_XREG on 1-day log Garman-Klass volatility with the earnings
    calendar as external regressors. BQML evaluates ARIMA multi-step, which
    structurally handicaps it against one-step HAR - also noted.

Everything is cost-guarded: the feature tables are <1 MB, so training and
prediction cost effectively nothing against the sandbox quota.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config
from model.train import _clean, _strategy_stats
from features.build import CROSS_FEATURES, EVENT_FEATURES, FULL_FEATURES

log = logging.getLogger(__name__)

DATASET = "nvda_quant"
QUERY_CAP = 2 * 1024 ** 3  # 2 GB cap; real scans here are kilobytes


def _bq():
    from data.bigquery_gdelt import _client
    return _client()


def _ensure_dataset(bigquery, client) -> str:
    ds_id = f"{client.project}.{DATASET}"
    try:
        client.get_dataset(ds_id)
    except Exception:
        ds = bigquery.Dataset(ds_id)
        ds.location = "US"
        client.create_dataset(ds)
        log.info("Created BigQuery dataset %s", ds_id)
    return ds_id


def _run(client, bigquery, sql: str, cap: bool = True):
    """cap=False for CREATE MODEL statements - BigQuery rejects
    maximum_bytes_billed on ML training jobs (our tables are <1 MB anyway)."""
    cfg = (bigquery.QueryJobConfig(maximum_bytes_billed=QUERY_CAP)
           if cap else None)
    return client.query(sql, job_config=cfg).result()


def _upload(client, table_id: str, df: pd.DataFrame) -> None:
    job = client.load_table_from_dataframe(
        df.reset_index().rename(columns={"index": "date"}), table_id)
    job.result()
    log.info("Uploaded %d rows -> %s", len(df), table_id)


def run_experiments() -> str:
    bigquery, client = _bq()
    ds_id = _ensure_dataset(bigquery, client)

    # ----- shared local data, same boundary as the tournament --------------
    from data.earnings import load_earnings_dates
    from data.news import load_aux_gdelt, load_gdelt_daily
    from data.prices import load_prices
    from features.build import build_dataset
    from features.vol import build_vol_dataset

    px, bench = load_prices()
    gdelt = load_gdelt_daily()
    aux = load_aux_gdelt()
    earn = load_earnings_dates()
    data = _clean(build_dataset(px, bench, gdelt, earn_dates=earn, aux=aux))
    oos_idx = data.index[config.MIN_TRAIN_DAYS:]
    split = oos_idx[int(len(oos_idx) * (1 - config.HOLDOUT_FRACTION))]

    lines = ["=" * 66,
             "  BigQuery ML experiments (trained in-warehouse, judged locally)",
             f"  holdout boundary (same as tournament): {split.date()}",
             "=" * 66]

    # ----- 1. XGBoost direction model ---------------------------------------
    feats = [f for f in FULL_FEATURES + CROSS_FEATURES
             if not data[f].isna().all()]
    dir_df = data[feats + ["y", "fwd_ret"]].copy()
    dir_df["is_hold"] = dir_df.index >= split
    dir_table = f"{ds_id}.direction_features"
    _upload(client, dir_table, dir_df)

    try:
        _run(client, bigquery, f"""
            CREATE OR REPLACE MODEL `{ds_id}.dir_xgb`
            OPTIONS(model_type='BOOSTED_TREE_CLASSIFIER',
                    input_label_cols=['y'], max_iterations=100,
                    max_tree_depth=6, subsample=0.9, l2_reg=1.0,
                    early_stop=FALSE, data_split_method='NO_SPLIT')
            AS SELECT {', '.join(feats)}, y
            FROM `{dir_table}` WHERE NOT is_hold
        """, cap=False)
        rows = _run(client, bigquery, f"""
            SELECT date,
                   (SELECT p.prob FROM UNNEST(predicted_y_probs) p
                    WHERE p.label = 1) AS prob_up
            FROM ML.PREDICT(MODEL `{ds_id}.dir_xgb`,
                            (SELECT * FROM `{dir_table}` WHERE is_hold))
            ORDER BY date
        """)
        # BQ returns UTC-aware timestamps; the local index is naive dates
        probs = pd.Series({
            pd.Timestamp(r.date).tz_localize(None).normalize(): r.prob_up
            for r in rows}).sort_index()
        hold = data.loc[probs.index]
        s = _strategy_stats(probs, hold["fwd_ret"], hold["y"])

        local = pd.read_csv(config.OOS_PREDICTIONS_PATH, index_col=0,
                            parse_dates=[0])
        lh = local.loc[local.index >= split]
        s_local = _strategy_stats(lh["prob_up"], lh["fwd_ret"], lh["y"])
        bh = hold["fwd_ret"]
        bh_sharpe = float(bh.mean() / bh.std() * np.sqrt(252))

        lines += [
            "\n  1) BOOSTED_TREE_CLASSIFIER (XGBoost) - direction, holdout only",
            "     (single fit at the boundary; local winner refits monthly)",
            f"     {'model':<28}{'Sharpe':>8}{'AUC':>8}{'acc':>7}{'expo':>7}",
            f"     {'BQML XGBoost':<28}{s['sharpe']:>8.2f}{s['auc']:>8.3f}"
            f"{s['acc']:>7.3f}{s['exposure']:>7.1%}",
            f"     {'local tournament winner':<28}{s_local['sharpe']:>8.2f}"
            f"{s_local['auc']:>8.3f}{s_local['acc']:>7.3f}"
            f"{s_local['exposure']:>7.1%}",
            f"     {'buy & hold':<28}{bh_sharpe:>8.2f}{'':>8}{'':>7}{'100%':>7}",
        ]
    except Exception as exc:
        lines.append(f"\n  1) XGBoost: FAILED - {str(exc).splitlines()[0][:90]}")
        lines.append("     (boosted trees may require billing enabled on "
                     "the project; ARIMA below usually works in sandbox)")

    # ----- 2. ARIMA_PLUS_XREG on volatility ---------------------------------
    try:
        vol = build_vol_dataset(px, bench, gdelt, horizon=1, earn_dates=earn)
        vol = vol.dropna(subset=["gk_22", "y_vol"])
        xregs = EVENT_FEATURES + ["dow"]
        vol_df = vol[["y_vol"] + xregs].copy()
        vol_df["is_hold"] = vol_df.index >= split
        vol_table = f"{ds_id}.vol_features"
        _upload(client, vol_table, vol_df)

        _run(client, bigquery, f"""
            CREATE OR REPLACE MODEL `{ds_id}.vol_arima_xreg`
            OPTIONS(model_type='ARIMA_PLUS_XREG',
                    time_series_timestamp_col='date',
                    time_series_data_col='y_vol', horizon=60)
            AS SELECT date, y_vol, {', '.join(xregs)}
            FROM `{vol_table}` WHERE NOT is_hold ORDER BY date
        """, cap=False)
        ev = list(_run(client, bigquery, f"""
            SELECT * FROM ML.EVALUATE(MODEL `{ds_id}.vol_arima_xreg`,
                (SELECT date, y_vol, {', '.join(xregs)}
                 FROM `{vol_table}` WHERE is_hold ORDER BY date LIMIT 60),
                STRUCT(TRUE AS perform_aggregation, 60 AS horizon))
        """))[0]

        vol_oos = pd.read_csv(config.VOL_OOS_PATH, index_col=0, parse_dates=[0])
        h60 = vol_oos.loc[vol_oos.index >= split].iloc[:60]
        har_mae = float((h60["y_vol"] - h60["pred_selected"]).abs().mean())
        lines += [
            "\n  2) ARIMA_PLUS_XREG - 1d log-vol, first 60 holdout days",
            "     (BQML evaluates multi-step; HAR is one-step - structural",
            "      handicap for ARIMA, so read the gap, not the sign)",
            f"     BQML ARIMA_PLUS_XREG  MAE(log-vol) = "
            f"{float(ev.mean_absolute_error):.4f}",
            f"     local HAR (one-step)  MAE(log-vol) = {har_mae:.4f}",
        ]
    except Exception as exc:
        lines.append(f"\n  2) ARIMA_PLUS_XREG: FAILED - "
                     f"{str(exc).splitlines()[0][:90]}")

    lines.append("=" * 66)
    report = "\n".join(lines)
    (config.ARTIFACTS / "bqml_report.txt").write_text(report, encoding="utf-8")
    return report
