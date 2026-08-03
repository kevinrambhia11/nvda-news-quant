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

# Paths are needed below for the industry-source switch. The BigQuery
# cache carries a bq_ prefix: gdelt_industry.csv stays reserved for any
# DOC-API data, so an off-office DOC bootstrap can never masquerade as the
# BigQuery series (the two corpora must never splice).
_ROOT = Path(__file__).resolve().parent
_INDUSTRY_BQ_CACHE = _ROOT / "artifacts" / "cache" / "bq_industry.csv"

# Auxiliary news series: competitor and industry coverage also moves NVDA.
# Each becomes a cached daily tone/volume series and cross features for the
# direction model. Sources are permanent per series (never spliced):
#   bigquery  - GKG organization matching via the service account (instant)
#   gdelt_api - the DOC API full-text search (rate-limited on this network)
# The industry series MIGRATES to BigQuery the moment its BQ cache exists
# (created by `main.py industry-backfill` - scheduled for the Aug 1 quota
# reset). Until then it stays on the blocked-at-office DOC API and degrades
# to stale caches as before. The series has no DOC history, so this is a
# clean source assignment, not a splice.
AUX_SERIES = {
    "competitors": {
        "source": "bigquery",
        "terms": ["advanced micro devices", "intel corp",
                  "taiwan semiconductor", "broadcom", "qualcomm"],
        "label": "AMD / Intel / TSMC / Broadcom / Qualcomm (BigQuery GKG)",
    },
    "industry": {
        "source": ("bigquery" if _INDUSTRY_BQ_CACHE.exists()
                   else "gdelt_api"),
        "bq_name": "bq_industry",
        "terms": ["semiconductor", "chipmaker", "micron technology",
                  "asml", "sk hynix"],
        "query": ('(semiconductors OR chipmakers OR "AI chips") '
                  'sourcelang:english'),
        "label": "semiconductor industry coverage",
    },
    "quality": {
        "source": "bigquery",
        "terms": ["nvidia"],
        "domains": QUALITY_DOMAINS,
        "label": "NVIDIA coverage from vetted outlets only (BigQuery GKG)",
    },
}

# ---------------------------------------------------------------------------
# PRIMARY tone series source
# ---------------------------------------------------------------------------
# The desk's main NVDA tone/volume series. Two permanent, never-spliced
# corpora exist:
#   "doc" - GDELT DOC 2.0 full-text search -> artifacts/cache/gdelt_daily.csv
#           FROZEN at 2026-07-13: the free endpoint has been throttled from
#           every network tried since, so that cache stopped advancing.
#   "bq"  - GDELT GKG organization tagging via BigQuery -> bq_nvda_tone.csv
#           The same service-account machinery that has served the
#           competitors series reliably for weeks.
# Switched to "bq" on 2026-08-03 after a wholesale full-history rebuild and
# a like-for-like comparison (artifacts/bq_tone_comparison.txt: daily tone
# r=+0.90, art_count r=+0.82 across the 2017->2026-07-13 overlap, and
# between +0.82 and +0.95 in every individual year).
# Flipping back to "doc" reproduces the pre-switch desk; the DOC cache stays
# in the repo as frozen legacy history. The two must NEVER be concatenated:
# BigQuery carries ~1.6x the article volume, so a spliced column would show
# a structural break that the models would happily read as news.
TONE_SOURCE = "bq"                  # "bq" | "doc"
TONE_BQ_NAME = "bq_nvda_tone"       # cache stem under artifacts/cache/
TONE_BQ_TERMS = ["nvidia"]          # GKG V2Organizations match, mirroring
                                    # articles.CATEGORY_CONDITIONS["nvda"]
TONE_DOC_NAME = "gdelt_daily"       # frozen legacy DOC-corpus cache stem
TONE_CACHE_NAME = (f"{TONE_BQ_NAME}.csv" if TONE_SOURCE == "bq"
                   else f"{TONE_DOC_NAME}.csv")

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

# Nested split: news2/newsnet scorers memorize whatever era they train on,
# so a second scorer set is trained on only the FIRST part of the selection
# window; the remainder stays clean and every candidate is judged there.
# The fraction is where the inner boundary sits inside the selection window.
NESTED_FRACTION = 0.6
NESTED_META_PATH = ARTIFACTS / "nested_meta.json"
OOS_META_PATH = ARTIFACTS / "oos_meta.json"

# Signal thresholds (hysteresis band reduces churn around 0.50)
LONG_ENTER = 0.55           # go long when P(up) exceeds this
LONG_EXIT = 0.48            # drop back to flat when P(up) falls below this
ALLOW_SHORT = False         # shorting off by default

# Backtest assumptions
COST_PER_TURNOVER = 0.0005  # 5 bps per unit of position change (slippage + fees)

# News-decay ("news memory") layer: instead of news influencing only the
# next session and vanishing, a story's weight fades with a half-life (in
# TRADING days). Two half-lives are built so the tournament can judge which
# timescale, if any, beats the same-day pulse - near-term attention decay
# vs slower post-event drift. Leak-free (EWMA over the already-d-1 series).
DECAY_HALFLIVES = (3, 10)

# Live magnitude head: P(|next-day move| > trailing median), the desk's
# validated news edge, served calibrated in the daily signal
MAG_MODEL_PATH = ARTIFACTS / "magnitude_model.joblib"

# Live signal: advisory blend of model probability and freshly scraped
# headline sentiment (the traded ACTION uses the backtested model rule only)
MODEL_WEIGHT = 0.7
HEADLINE_WEIGHT = 0.3
MAX_MODEL_AGE_DAYS = 45    # refuse to generate signals from a stale model

# Full-article reading for the live signal (data/fulltext.py): fetch each
# story's body so sentiment sees more than the headline. Bounded hard so a
# slow website can never hang the 17:00 task; on any failure a story simply
# degrades to headline-only scoring.
FULLTEXT_ENABLED = True
FULLTEXT_TIMEOUT = 10          # seconds per HTTP request (also the TOTAL
                               # streamed-read cap - see fetch_and_extract)
FULLTEXT_MAX_BYTES = 2_000_000  # per-page download cap
FULLTEXT_WORKERS = 8           # parallel fetch threads
FULLTEXT_BUDGET_S = 300        # wall-clock budget for the whole enrichment
FULLTEXT_LEAD_CHARS = 8000     # article text kept per story (local
                               # cache/corpus only, never the repo)
FULLTEXT_SCORE_CHARS = 600     # body chars fed to sentiment (FinBERT drifts
                               # neutral on long multi-topic text - verified)
FULLTEXT_MAX_ALT_SEARCHES = 12  # same-story-elsewhere lookups per run
FULLTEXT_CACHE_DIR = CACHE / "fulltext"   # NOT gitignore-whitelisted:
FULLTEXT_CACHE_TTL_DAYS = 3               # article text never enters the repo

# Permanent LOCAL body corpus (phase 1 of a body-aware brain): every body
# fetched for live stories and daily top-up articles is appended here -
# old pages die, so each unfetched day is training data lost forever.
# Local-only (copyright); a future body-trained scorer must still earn a
# seat through the tournaments / live paper trail like everything else.
FULLTEXT_ARCHIVE_PATH = CACHE / "fulltext" / "bodies_archive.parquet"
FULLTEXT_TOPUP_BUDGET_S = 240   # body-fetch budget inside the daily top-up

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
