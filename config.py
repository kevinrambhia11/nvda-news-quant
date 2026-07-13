"""Central configuration for the NVDA news-sentiment trading system."""
from pathlib import Path

TICKER = "NVDA"
BENCHMARK = "QQQ"

# GDELT full-text search query (GDELT monitors global online news media)
GDELT_QUERY = "NVIDIA sourcelang:english"

# Data range (GDELT DOC 2.0 API coverage begins January 2017)
TRAIN_START = "2017-01-01"

# Paths
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
CACHE = ARTIFACTS / "cache"
MODEL_PATH = ARTIFACTS / "model.joblib"
FEATURES_PATH = ARTIFACTS / "features.csv"
OOS_PREDICTIONS_PATH = ARTIFACTS / "oos_predictions.csv"
EQUITY_CURVE_PATH = ARTIFACTS / "equity_curve.csv"
EQUITY_PLOT_PATH = ARTIFACTS / "equity_curve.png"
REPORT_PATH = ARTIFACTS / "backtest_report.txt"

# Walk-forward training
MIN_TRAIN_DAYS = 400        # minimum history before the first out-of-sample prediction
RETRAIN_EVERY = 21          # retrain cadence in trading days (~monthly)
HOLDOUT_FRACTION = 0.4      # OOS tail reserved for post-selection reporting
MODEL_SELECTION_PATH = ARTIFACTS / "model_selection.txt"

# Signal thresholds (hysteresis band reduces churn around 0.50)
LONG_ENTER = 0.55           # go long when P(up) exceeds this
LONG_EXIT = 0.48            # drop back to flat when P(up) falls below this
ALLOW_SHORT = False         # shorting off by default

# Backtest assumptions
COST_PER_TURNOVER = 0.0005  # 5 bps per unit of position change (slippage + fees)

# Live signal: advisory blend of model probability and freshly scraped
# headline sentiment (the traded ACTION uses the backtested model rule only)
MODEL_WEIGHT = 0.7
HEADLINE_WEIGHT = 0.3
MAX_MODEL_AGE_DAYS = 45    # refuse to generate signals from a stale model

# Desk volatility model (EMH-consistent: forecasts the second moment)
VOL_MODEL_PATH = ARTIFACTS / "vol_model.joblib"
VOL_OOS_PATH = ARTIFACTS / "vol_oos_predictions.csv"
VOL_REPORT_PATH = ARTIFACTS / "vol_report.txt"
VOL_HORIZONS = (1, 5)       # forecast horizons in trading days
VOL_TARGET_ANN = 0.30       # annualized vol anchor for target-vol sizing
VAR_NOTIONAL = 1_000_000    # $ notional used in the VaR lines

ARTIFACTS.mkdir(exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
