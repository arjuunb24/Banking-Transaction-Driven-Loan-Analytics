"""
SQL LAYER — DATA LOADER
Loads all Python outputs into PostgreSQL banking_risk database.
Run: python sql/load_to_postgres.py

Loads in order:
1. Dimension tables
2. Fact tables from parquet outputs
3. Verifies row counts
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

from src.config import PROCESSED_DIR, OUTPUTS_DIR

# ── Database connection ────────────────────────────────────────
DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
engine = create_engine(DB_URL)


def log(msg):
    print(f"  {msg}")


def load_table(df, table_name, if_exists="replace"):
    """Load dataframe to PostgreSQL table."""
    df.to_sql(
        table_name,
        engine,
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=10_000
    )
    log(f"Loaded {len(df):>10,} rows → {table_name}")


# ══════════════════════════════════════════════════════════════
# STEP 1 — RUN DDL
# ══════════════════════════════════════════════════════════════

def run_ddl():
    print("\n" + "=" * 55)
    print("STEP 1 — CREATING TABLES")
    print("=" * 55)

    ddl_path = os.path.join(
        os.path.dirname(__file__), "create_tables.sql"
    )
    with open(ddl_path, "r") as f:
        ddl = f.read()

    with engine.connect() as conn:
        # Remove COMMENT statements entirely — they cause
        # parsing issues with semicolons inside strings
        # Table comments are documented in code instead
        lines = ddl.split("\n")
        cleaned_lines = []
        skip = False
        for line in lines:
            if line.strip().startswith("COMMENT ON"):
                skip = True
            if skip:
                if line.strip().endswith(";"):
                    skip = False
                continue
            cleaned_lines.append(line)

        cleaned_ddl = "\n".join(cleaned_lines)

        # Split on semicolons at end of line only
        statements = [
            s.strip() for s in cleaned_ddl.split(";\n")
            if s.strip()
            and not s.strip().startswith("--")
            and len(s.strip()) > 10
        ]

        for stmt in statements:
            try:
                conn.execute(text(stmt))
            except Exception as e:
                print(f"  Warning skipping statement: {e}")
                continue

        conn.commit()

    log("All tables created successfully")


# ══════════════════════════════════════════════════════════════
# STEP 2 — DIMENSION TABLES
# ══════════════════════════════════════════════════════════════

def load_dimensions(loan_df):
    print("\n" + "=" * 55)
    print("STEP 2 — LOADING DIMENSION TABLES")
    print("=" * 55)

    # ── dim_customer ─────────────────────────────────────────
    dim_cust = loan_df[[
        "customer_id", "income_tier", "risk_tier",
        "paysim_customer_id", "cohort", "addr_state"
    ]].copy()
    dim_cust = dim_cust.rename(columns={
        "income_tier": "income_band",
        "risk_tier":   "risk_tier_cohort"
    })

    # Assign region from addr_state
    northeast = ["CT","ME","MA","NH","RI","VT","NY","NJ","PA"]
    south     = ["DE","MD","DC","VA","WV","NC","SC","GA","FL",
                 "KY","TN","AL","MS","AR","LA","OK","TX"]
    midwest   = ["OH","MI","IN","IL","WI","MN","IA","MO","ND",
                 "SD","NE","KS"]
    west      = ["MT","ID","WY","CO","NM","AZ","UT","NV","WA",
                 "OR","CA","AK","HI"]

    def map_region(s):
        if s in northeast: return "Northeast"
        if s in south:     return "South"
        if s in midwest:   return "Midwest"
        if s in west:      return "West"
        return "Other"

    dim_cust["region"] = dim_cust["addr_state"].apply(map_region)
    dim_cust = dim_cust.drop_duplicates("customer_id")
    load_table(dim_cust, "dim_customer")

    # ── dim_date ─────────────────────────────────────────────
    # Generate monthly date spine 2007-01 to 2019-12
    dates = pd.date_range(start="2007-01-01", end="2019-12-01", freq="MS")
    dim_date = pd.DataFrame({
        "month_key":      dates,
        "year":           dates.year,
        "month":          dates.month,
        "quarter":        dates.quarter,
        "is_month_end":   [True] * len(dates),
        "is_quarter_end": [m in [3, 6, 9, 12] for m in dates.month],
        "month_label":    dates.strftime("%b-%Y")
    })
    load_table(dim_date, "dim_date")

    # ── dim_segment ──────────────────────────────────────────
    segments = [
        f"Grade_{g}_{r}"
        for g in ["AB", "CD", "EG"]
        for r in ["Midwest", "Northeast", "South", "West"]
    ]
    dim_seg = pd.DataFrame({
        "segment_id":   segments,
        "grade_bucket": [s.split("_")[1] for s in segments],
        "region":       [s.split("_")[2] for s in segments]
    })
    load_table(dim_seg, "dim_segment")


# ══════════════════════════════════════════════════════════════
# STEP 3 — FACT TABLES
# ══════════════════════════════════════════════════════════════

def load_facts():
    print("\n" + "=" * 55)
    print("STEP 3 — LOADING FACT TABLES")
    print("=" * 55)

    # ── fact_loans ────────────────────────────────────────────
    loan_df = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "loan_features.parquet")
    )
    loan_cols = [
        "customer_id", "loan_amnt", "term_months", "int_rate",
        "installment", "grade", "grade_encoded", "sub_grade",
        "annual_inc", "monthly_income", "loan_status", "default_flag",
        "dti", "issue_date", "issue_year", "issue_month", "addr_state",
        "fico_mid", "open_acc", "pub_rec", "has_pub_rec",
        "revol_util", "revol_bal", "high_revol_util_flag",
        "loan_to_income", "interest_burden"
    ]
    load_table(loan_df[loan_cols], "fact_loans")

    # ── fact_features ─────────────────────────────────────────
    txn_df = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "txn_features.parquet")
    )
    txn_cols = [
        "customer_id", "month_key", "txn_count", "mean_amount",
        "std_amount", "cv_amount", "total_outflow", "total_inflow",
        "cashout_amount", "closing_balance", "mean_balance", "std_balance",
        "withdrawal_deposit_ratio", "cashout_proportion",
        "max_withdrawal_to_balance", "overdraft_flag",
        "balance_vol_3m", "balance_slope_3m", "txn_count_lag1",
        "mean_amount_mom_change", "distress_duration_6m"
    ]

    # Add behavioral_risk_score
    def minmax(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    txn_df["behavioral_risk_score"] = (
        0.30 * minmax(txn_df["distress_duration_6m"])    +
        0.25 * minmax(txn_df["withdrawal_deposit_ratio"])+
        0.20 * minmax(txn_df["balance_vol_3m"])          +
        0.15 * minmax(txn_df["overdraft_flag"])          +
        0.10 * minmax(txn_df["cashout_proportion"])
    )

    load_table(txn_df[txn_cols + ["behavioral_risk_score"]], "fact_features")

    # ── fact_predictions ──────────────────────────────────────
    pred_df = pd.read_parquet(
        os.path.join(OUTPUTS_DIR, "xgb_predictions.parquet")
    )
    # Ensure prediction_month is date type
    pred_df["prediction_month"] = pd.to_datetime(
        pred_df["prediction_month"]
    ).dt.date
    load_table(pred_df, "fact_predictions")

    # ── fact_forecasts ────────────────────────────────────────
    fc_df = pd.read_parquet(
        os.path.join(OUTPUTS_DIR, "sarimax_forecasts.parquet")
    )
    fc_df["month_key"] = pd.to_datetime(fc_df["month_key"]).dt.date
    fc_df["n_validation_steps"] = fc_df["n_validation_steps"].astype(
        "Int64"
    )
    load_table(fc_df, "fact_forecasts")

    # ── fact_decile_analysis ──────────────────────────────────
    decile_df = pd.read_csv(
        os.path.join(OUTPUTS_DIR, "xgb_decile_analysis.csv")
    )
    decile_df = decile_df.rename(
        columns={"cum_defaults_pct": "cum_defaults_pct"}
    )
    load_table(decile_df, "fact_decile_analysis")

    # ── fact_shap_importance ──────────────────────────────────
    shap_df = pd.read_csv(
        os.path.join(OUTPUTS_DIR, "xgb_shap_importance.csv")
    )
    gain_df = pd.read_csv(
        os.path.join(OUTPUTS_DIR, "xgb_feature_importance.csv")
    )
    shap_merged = shap_df.merge(
        gain_df.rename(columns={"importance": "gain_importance"}),
        on="feature",
        how="left"
    )
    load_table(shap_merged, "fact_shap_importance")

    return loan_df


# ══════════════════════════════════════════════════════════════
# STEP 4 — VERIFICATION
# ══════════════════════════════════════════════════════════════

def verify_counts():
    print("\n" + "=" * 55)
    print("STEP 4 — VERIFICATION")
    print("=" * 55)

    tables = [
        "dim_customer", "dim_date", "dim_segment",
        "fact_loans", "fact_features", "fact_predictions",
        "fact_forecasts", "fact_decile_analysis",
        "fact_shap_importance"
    ]

    with engine.connect() as conn:
        print(f"\n  {'Table':<30} {'Rows':>12}")
        print(f"  {'─'*29} {'─'*12}")
        for table in tables:
            result = conn.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            )
            count = result.fetchone()[0]
            print(f"  {table:<30} {count:>12,}")

    print(f"\n  All tables verified.")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("SQL DATA MART LOADER")
    print("Banking Risk Analytics")
    print("=" * 55)

    run_ddl()

    # Load loan features for dimension building
    loan_df = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "loan_features.parquet")
    )
    load_dimensions(loan_df)
    load_facts()
    verify_counts()

    print("\n" + "=" * 55)
    print("DATA MART LOADED SUCCESSFULLY")
    print("Next: python sql/create_views.py")
    print("=" * 55)