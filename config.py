"""
config.py
---------
Central configuration for the Financial Contradiction Tracker.

Edit COMPANIES and QUARTERS here to control what gets ingested.
All other modules import from this file — no magic strings elsewhere.
"""

from pathlib import Path

# ── Project root ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

# ── Data directories ───────────────────────────────────────────────
DATA_DIR        = ROOT / "data"
TRANSCRIPT_DIR  = DATA_DIR / "transcripts"
DB_PATH         = DATA_DIR / "tracker.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────
# Target companies (Week 1: 5 companies)
# Format: { display_name, bse_code, nse_ticker, sector }
# ──────────────────────────────────────────────────────────────────
COMPANIES = [
    {
        "name":       "Reliance Industries",
        "bse_code":   "500325",
        "nse_ticker": "RELIANCE",
        "sector":     "Conglomerate",
    },
    {
        "name":       "Infosys",
        "bse_code":   "500209",
        "nse_ticker": "INFY",
        "sector":     "IT Services",
    },
    {
        "name":       "HDFC Bank",
        "bse_code":   "500180",
        "nse_ticker": "HDFCBANK",
        "sector":     "Banking",
    },
    {
        "name":       "Tata Consultancy Services",
        "bse_code":   "532540",
        "nse_ticker": "TCS",
        "sector":     "IT Services",
    },
    {
        "name":       "Wipro",
        "bse_code":   "507685",
        "nse_ticker": "WIPRO",
        "sector":     "IT Services",
    },
]

# ──────────────────────────────────────────────────────────────────
# Target quarters (Week 1: 8 quarters = FY23 Q1–Q4 + FY24 Q1–Q4)
# Used as date range for BSE scraper.
# ──────────────────────────────────────────────────────────────────
SCRAPE_FROM_DATE = "20220401"   # YYYYMMDD — start of FY23
SCRAPE_TO_DATE   = "20250331"   # end of FY25 (collect extra, filter later)

TARGET_QUARTERS = [
    "Q1FY23", "Q2FY23", "Q3FY23", "Q4FY23",
    "Q1FY24", "Q2FY24", "Q3FY24", "Q4FY24",
]

# ──────────────────────────────────────────────────────────────────
# Model settings (used from Week 2 onward)
# ──────────────────────────────────────────────────────────────────
FINBERT_MODEL   = "ProsusAI/finbert"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
NLI_MODEL       = "cross-encoder/nli-deberta-v3-base"
SPACY_MODEL     = "en_core_web_sm"

# ──────────────────────────────────────────────────────────────────
# Contradiction thresholds (Week 3-4 tuning)
# ──────────────────────────────────────────────────────────────────
HARD_CONTRADICTION_THRESHOLD = 0.5   # NLI contradiction prob
SOFT_CONTRADICTION_THRESHOLD = 0.6   # composite soft score
TOPIC_SIMILARITY_THRESHOLD   = 0.6   # cosine similarity for same-topic check
OMISSION_MIN_PRIOR_QUARTERS  = 3     # topic must appear in N consecutive quarters

# ──────────────────────────────────────────────────────────────────
# Hedge scale (Week 4)
# ──────────────────────────────────────────────────────────────────
HEDGE_SCALE = {
    "confident":   1.0,
    "strong":      0.9,
    "robust":      0.9,
    "optimistic":  0.8,
    "positive":    0.75,
    "steady":      0.65,
    "stable":      0.60,
    "cautious":    0.4,
    "monitor":     0.35,
    "watchful":    0.30,
    "headwind":    0.2,
    "challenge":   0.15,
    "difficult":   0.1,
    "concern":     0.1,
    "uncertain":   0.05,
    "pressure":    0.15,
    "subdued":     0.2,
}

# ──────────────────────────────────────────────────────────────────
# Guidance keywords (Week 2 classifier)
# ──────────────────────────────────────────────────────────────────
GUIDANCE_KEYWORDS = [
    "expect", "guidance", "outlook", "anticipate", "forecast",
    "project", "target", "aim", "plan to", "confident",
    "optimistic", "cautious", "headwind", "tailwind",
    "momentum", "pipeline", "growth", "margin", "revenue",
]

# ──────────────────────────────────────────────────────────────────
# Executive role identifiers
# ──────────────────────────────────────────────────────────────────
EXECUTIVE_ROLES = [
    "CFO", "CEO", "MD", "Managing Director",
    "Chief Financial", "Chief Executive",
    "Executive Director", "Whole-time Director",
]

# ──────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────
LOG_DIR   = ROOT / "logs"
LOG_FILE  = LOG_DIR / "tracker.log"
LOG_DIR.mkdir(exist_ok=True)
