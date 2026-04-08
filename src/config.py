import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUTS_DIR   = os.path.join(BASE_DIR, "data", "outputs")

PAYSIM_PATH       = os.path.join(RAW_DIR, "paysim.csv")
LENDING_CLUB_PATH = os.path.join(RAW_DIR, "lending_club.csv.gz")

# ── Database ───────────────────────────────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "5432")
DB_NAME     = os.getenv("DB_NAME", "banking_risk")
DB_USER     = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password")
DB_URL      = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Temporal split ─────────────────────────────────────────────────────────────
TRAIN_CUTOFF_YEAR = 2015
TEST_START_YEAR   = 2016

# ── XGBoost ────────────────────────────────────────────────────────────────────
XGB_N_ESTIMATORS      = 1500
XGB_MAX_DEPTH         = 6
XGB_LEARNING_RATE     = 0.02
XGB_EARLY_STOPPING    = 50

# ── SARIMAX ────────────────────────────────────────────────────────────────────
SARIMAX_M        = 12
FORECAST_HORIZON = 6

# ── Risk tiers ─────────────────────────────────────────────────────────────────
RISK_TIER_THRESHOLDS = {
    "Low":       (0.00, 0.25),
    "Medium":    (0.25, 0.50),
    "High":      (0.50, 0.75),
    "Very High": (0.75, 1.00)
}

# ── Anomaly detection ──────────────────────────────────────────────────────────
ANOMALY_STD_THRESHOLD = 2.0