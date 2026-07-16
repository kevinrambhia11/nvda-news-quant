"""Central configuration for the NVDA news-sentiment trading system."""
from pathlib import Path

TICKER = "NVDA"
BENCHMARK = "QQQ"

# GDELT full-text search query (GDELT monitors global online news media)
GDELT_QUERY = "NVIDIA sourcelang:english"

# Curated live-headline feeds: reliable publishers per vertical, fetched as
# one simple Google News query per (query, publisher) pair - OR-chained
# source: filters are unreliable in the RSS endpoint. Edit to taste.
CURATED_FEEDS = {
    "finance": {"query": "NVDA OR NVIDIA",
                "sources": ["Reuters", "Bloomberg", "CNBC",
                            "The Wall Street Journal", "Barron's",
                            "Financial Times", "MarketWatch",
                            "The Economist"]},
    "ai": {"query": "NVIDIA AI",
           "sources": ["Reuters", "Bloomberg", "TechCrunch", "The Verge",
                       "BBC News", "Financial Times"]},
    "semiconductors": {"query": "semiconductor OR chipmaker OR TSMC",
                       "sources": ["Reuters", "CNBC", "Tom's Hardware",
                                   "Bloomberg", "Nikkei Asia",
                                   "Financial Times"]},
    "hyperscalers": {"query": ('(Microsoft OR Google OR Amazon OR Meta OR '
                               'Oracle) (AI OR datacenter OR cloud OR GPU)'),
                     "sources": ["Reuters", "Bloomberg", "CNBC"]},
    "macro": {"query": ('tariffs OR "export controls" OR "Federal Reserve" '
                        'OR sanctions OR geopolitics'),
              "sources": ["Reuters", "BBC News", "Financial Times",
                          "The Wall Street Journal"]},
    "brokers": {"query": ('NVDA (upgrade OR downgrade OR "price target" OR '
                          'analyst)'),
                "sources": ["Reuters", "Barron's", "CNBC", "Bloomberg"]},
}

# Vetted source domains for the QUALITY tone series (matched against GKG's
# SourceCommonName in BigQuery) - the "reliable sources only" experiment.
QUALITY_DOMAINS = [
    "reuters.com", "bloomberg.com", "cnbc.com", "wsj.com", "barrons.com",
    "ft.com", "bbc.com", "bbc.co.uk", "apnews.com", "economist.com",
    "nikkei.com", "asia.nikkei.com", "marketwatch.com", "forbes.com",
    "fortune.com", "techcrunch.com", "theverge.com", "tomshardware.com",
]

# Auxiliary news series: competitor and industry coverage also moves NVDA.
# Each becomes a cached daily tone/volume series and cross features for the
# direction model. Sources are permanent per series (never spliced):
#   bigquery  - GKG organization matching via the service account (instant)
#   gdelt_api - the DOC API full-text search (rate-limited on this network)
AUX_SERIES = {
    "competitors": {
        "source": "bigquery",
        "terms": ["advanced micro devices", "intel corp",
                  "taiwan semiconductor", "broadcom", "qualcomm"],
        "label": "AMD / Intel / TSMC / Broadcom / Qualcomm (BigQuery GKG)",
    },
    "industry": {
        "source": "gdelt_api",
        "query": ('(semiconductors OR chipmakers OR "AI chips") '
                  'sourcelang:english'),
        "label": "semiconductor industry coverage (GDELT DOC API)",
    },
    "quality": {
        "source": "bigquery",
        "terms": ["nvidia"],
        "domains": QUALITY_DOMAINS,
        "label": "NVIDIA coverage from vetted outlets only (BigQuery GKG)",
    },
}

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

# Direction x volatility fusion backtest
FUSED_CURVE_PATH = ARTIFACTS / "fused_equity.csv"
FUSED_REPORT_PATH = ARTIFACTS / "fused_report.txt"

# Intraday news-reaction pilot (60 days of 5-minute bars vs 15-min GDELT)
INTRADAY_NVDA_5M = CACHE / "nvda_5m.csv"
INTRADAY_QQQ_5M = CACHE / "qqq_5m.csv"
INTRADAY_GDELT_15M = CACHE / "gdelt_15min.csv"
INTRADAY_REPORT_PATH = ARTIFACTS / "intraday_report.txt"
INTRADAY_EVENTS_PATH = ARTIFACTS / "intraday_events.csv"
INTRADAY_PLOT_PATH = ARTIFACTS / "intraday_reaction.png"
BURST_Z = 3.0               # article-count z-score to call a news burst
BURST_MIN_ARTICLES = 20     # absolute floor so low-base spikes don't count
HEADLINE_LOG_PATH = ARTIFACTS / "headline_log.csv"

ARTIFACTS.mkdir(exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
