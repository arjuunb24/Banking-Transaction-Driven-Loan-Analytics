"""
STAGE 2 — FEATURE ENGINEERING + SYNTHETIC COHORT ASSIGNMENT
Run independently: python src/feature_engineering.py
Inputs:  data/processed/paysim_clean.parquet
         data/processed/lending_club_clean.parquet
Outputs: data/processed/txn_features.parquet
         data/processed/loan_features.parquet
         data/processed/master_features.parquet
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.config import PROCESSED_DIR, RANDOM_SEED


# ══════════════════════════════════════════════════════════════════════════════
# TRANSACTION FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 55)
    print("TRANSACTION FEATURE ENGINEERING")
    print("=" * 55)

    df = df.copy()

    outflow_types = {"CASH_OUT", "TRANSFER", "DEBIT", "PAYMENT"}
    inflow_types  = {"CASH_IN"}

    df["is_outflow"] = df["transaction_type"].isin(outflow_types).astype(np.int8)
    df["is_inflow"]  = df["transaction_type"].isin(inflow_types).astype(np.int8)
    df["is_cashout"] = (df["transaction_type"] == "CASH_OUT").astype(np.int8)
    df["outflow_amt"] = df["amount"] * df["is_outflow"]
    df["inflow_amt"]  = df["amount"] * df["is_inflow"]
    df["cashout_amt"] = df["amount"] * df["is_cashout"]

    print("  Step 1/4: Monthly aggregation...")
    monthly = df.groupby(
        ["customer_id", "month_key"], sort=True
    ).agg(
        total_amount    = ("amount",         "sum"),
        mean_amount     = ("amount",         "mean"),
        std_amount      = ("amount",         "std"),
        txn_count       = ("amount",         "count"),
        total_outflow   = ("outflow_amt",    "sum"),
        total_inflow    = ("inflow_amt",     "sum"),
        cashout_amount  = ("cashout_amt",    "sum"),
        closing_balance = ("balance_after",  "last"),
        mean_balance    = ("balance_after",  "mean"),
        std_balance     = ("balance_after",  "std"),
    ).reset_index()

    print(f"  Monthly rows: {len(monthly):,}")

    print("  Step 2/4: Ratios...")
    monthly["std_amount"]  = monthly["std_amount"].fillna(0)
    monthly["std_balance"] = monthly["std_balance"].fillna(0)

    monthly["cv_amount"] = (
        monthly["std_amount"] / (monthly["mean_amount"].abs() + 1e-9)
    )
    monthly["withdrawal_deposit_ratio"] = (
        monthly["total_outflow"] / (monthly["total_inflow"] + 1e-9)
    )
    monthly["cashout_proportion"] = (
        monthly["cashout_amount"] / (monthly["total_amount"].abs() + 1e-9)
    )
    monthly["max_withdrawal_to_balance"] = (
        monthly["total_outflow"] / (monthly["mean_balance"].abs() + 1e-9)
    )
    monthly["overdraft_flag"] = (
        monthly["closing_balance"] < 0.10 * monthly["mean_balance"]
    ).astype(np.int8)
    monthly["distress_flag"] = (
        monthly["total_outflow"] > monthly["total_inflow"]
    ).astype(np.int8)

    for col in ["cv_amount", "withdrawal_deposit_ratio",
                "cashout_proportion", "max_withdrawal_to_balance"]:
        monthly[col] = monthly[col].clip(-100, 100)

    print("  Step 3/4: Rolling features (vectorized)...")
    monthly = monthly.sort_values(
        ["customer_id", "month_key"]
    ).reset_index(drop=True)

    cust_ids = monthly["customer_id"].values
    new_cust = np.concatenate([[True], cust_ids[1:] != cust_ids[:-1]])

    def safe_shift(arr, n):
        shifted = np.roll(arr.astype(np.float32), n)
        mask = new_cust.copy()
        for i in range(1, n):
            mask = mask | np.roll(new_cust, i)
        shifted[mask] = 0
        return shifted

    close_bal = monthly["closing_balance"].values.astype(np.float32)
    std_bal   = monthly["std_balance"].values.astype(np.float32)
    txn_cnt   = monthly["txn_count"].values.astype(np.float32)
    mean_amt  = monthly["mean_amount"].values.astype(np.float32)
    distress  = monthly["distress_flag"].values.astype(np.float32)

    sb1 = safe_shift(std_bal, 1)
    sb2 = safe_shift(std_bal, 2)
    monthly["balance_vol_3m"] = (std_bal + sb1 + sb2) / 3

    cb2 = safe_shift(close_bal, 2)
    monthly["balance_slope_3m"] = (close_bal - cb2) / 2

    monthly["txn_count_lag1"] = safe_shift(txn_cnt, 1)

    ma1 = safe_shift(mean_amt, 1)
    monthly["mean_amount_mom_change"] = np.where(
        ma1 == 0, 0,
        np.clip((mean_amt - ma1) / (np.abs(ma1) + 1e-9), -10, 10)
    )

    d1 = safe_shift(distress, 1)
    d2 = safe_shift(distress, 2)
    d3 = safe_shift(distress, 3)
    d4 = safe_shift(distress, 4)
    d5 = safe_shift(distress, 5)
    monthly["distress_duration_6m"] = distress + d1 + d2 + d3 + d4 + d5

    print("  Step 4/4: Cleanup...")
    monthly = monthly.drop(columns=["distress_flag"])
    monthly = monthly.fillna(0)

    print(f"  Transaction features shape: {monthly.shape}")
    return monthly


# ══════════════════════════════════════════════════════════════════════════════
# LOAN FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_loan_features(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 55)
    print("LOAN FEATURE ENGINEERING")
    print("=" * 55)

    df = df.copy()

    df["monthly_income"]     = df["annual_inc"] / 12
    df["loan_to_income"]     = df["loan_amnt"] / (df["annual_inc"] + 1e-9)
    df["interest_burden"]    = df["installment"] / (df["monthly_income"] + 1e-9)
    df["fico_mid"]           = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["term_months"]        = df["term"].str.extract(r"(\d+)").astype(int)
    df["revol_util"]         = df["revol_util"].fillna(df["revol_util"].median())
    df["high_revol_util_flag"] = (df["revol_util"] > 75).astype(int)
    df["has_pub_rec"]        = (df["pub_rec"] > 0).astype(int)

    keep = [
        "customer_id", "loan_amnt", "term_months", "int_rate",
        "installment", "grade", "grade_encoded", "sub_grade",
        "annual_inc", "monthly_income", "loan_status", "default_flag",
        "dti", "issue_date", "issue_year", "issue_month", "addr_state",
        "fico_mid", "open_acc", "pub_rec", "has_pub_rec",
        "revol_util", "revol_bal", "high_revol_util_flag",
        "loan_to_income", "interest_burden"
    ]
    df = df[keep]

    print(f"  Loan features shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC COHORT ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def synthetic_cohort_assignment(
    loan_df: pd.DataFrame,
    txn_features: pd.DataFrame
) -> pd.DataFrame:
    print("=" * 55)
    print("SYNTHETIC COHORT ASSIGNMENT")
    print("=" * 55)

    np.random.seed(RANDOM_SEED)
    loan_df = loan_df.copy()

    # Lending Club cohorts
    loan_df["income_tier"] = pd.qcut(
        loan_df["annual_inc"].clip(
            upper=loan_df["annual_inc"].quantile(0.99)
        ),
        q=4, labels=["Q1","Q2","Q3","Q4"],
        duplicates="drop"
    ).astype(str)

    loan_df["risk_tier"] = pd.cut(
        loan_df["grade_encoded"].fillna(4),
        bins=[0,2,4,7],
        labels=["Low","Medium","High"]
    ).astype(str)

    loan_df["cohort"] = loan_df["income_tier"] + "_" + loan_df["risk_tier"]

    # PaySim cohorts
    txn_cust = txn_features.groupby(
        "customer_id", sort=False
    ).agg(
        mean_balance  = ("mean_balance",         "mean"),
        total_txn     = ("txn_count",            "sum"),
        mean_distress = ("distress_duration_6m", "mean")
    ).reset_index()

    bal = txn_cust["mean_balance"]
    pos_bal = bal[bal > 0]
    if len(pos_bal) >= 4:
        q25, q50, q75 = pos_bal.quantile([0.25, 0.50, 0.75])
    else:
        q25, q50, q75 = 1, 2, 3

    bal_bins = sorted(set([-np.inf, q25, q50, q75, np.inf]))
    while len(bal_bins) < 5:
        bal_bins.append(bal_bins[-1] + 1)

    n_labels = len(bal_bins) - 1
    tier_labels = ["Q1","Q2","Q3","Q4"][:n_labels]

    txn_cust["income_tier"] = pd.cut(
        bal, bins=bal_bins,
        labels=tier_labels,
        include_lowest=True
    ).astype(str)

    txn_cust["risk_tier"] = pd.cut(
        txn_cust["mean_distress"].fillna(0),
        bins=[-0.1, 1, 3, 100],
        labels=["Low","Medium","High"]
    ).astype(str)

    txn_cust["cohort"] = (
        txn_cust["income_tier"] + "_" + txn_cust["risk_tier"]
    )

    # Within-cohort random assignment
    all_paysim_ids = txn_cust["customer_id"].values
    cohort_map     = {}

    for cohort in loan_df["cohort"].unique():
        lc_ids  = loan_df.loc[
            loan_df["cohort"] == cohort, "customer_id"
        ].values
        txn_ids = txn_cust.loc[
            txn_cust["cohort"] == cohort, "customer_id"
        ].values

        if len(txn_ids) == 0:
            txn_ids = all_paysim_ids

        assigned = np.random.choice(txn_ids, size=len(lc_ids), replace=True)
        for lc_id, pid in zip(lc_ids, assigned):
            cohort_map[lc_id] = pid

    loan_df["paysim_customer_id"] = loan_df["customer_id"].map(cohort_map)
    print(f"  Unique cohorts: {loan_df['cohort'].nunique()}")
    return loan_df


# ══════════════════════════════════════════════════════════════════════════════
# MASTER TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_master_feature_table(
    loan_df: pd.DataFrame,
    txn_features: pd.DataFrame
) -> pd.DataFrame:
    print("=" * 55)
    print("BUILDING MASTER FEATURE TABLE")
    print("=" * 55)

    txn_renamed = txn_features.rename(
        columns={"customer_id": "paysim_customer_id"}
    )

    master = loan_df.merge(
        txn_renamed, on="paysim_customer_id", how="left"
    )

    txn_cols = [
        c for c in txn_renamed.columns
        if c != "paysim_customer_id"
    ]
    master[txn_cols] = master[txn_cols].fillna(0)

    # Interaction features — computed here so both models can use them
    master["grade_x_distress"]   = (
        master["grade_encoded"] * master["distress_duration_6m"]
    )
    master["grade_x_volatility"] = (
        master["grade_encoded"] * master["balance_vol_3m"]
    )
    master["grade_x_withdrawal"] = (
        master["grade_encoded"] * master["withdrawal_deposit_ratio"]
    )
    master["dti_x_distress"]     = (
        master["dti"] * master["distress_duration_6m"]
    )
    master["burden_x_overdraft"] = (
        master["interest_burden"] * master["overdraft_flag"]
    )
    master["fico_x_volatility"]  = (
        master["fico_mid"] * master["balance_vol_3m"]
    )
    master["dti_x_withdrawal"]   = (
        master["dti"] * master["withdrawal_deposit_ratio"]
    )
    master["income_x_cashout"]   = (
        master["annual_inc"] * master["cashout_proportion"]
    )

    def minmax(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    master["behavioral_risk_score"] = (
        0.30 * minmax(master["distress_duration_6m"])    +
        0.25 * minmax(master["withdrawal_deposit_ratio"])+
        0.20 * minmax(master["balance_vol_3m"])          +
        0.15 * minmax(master["overdraft_flag"])          +
        0.10 * minmax(master["cashout_proportion"])
    )

    print(f"  Master table shape: {master.shape}")
    return master


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load from stage 1 outputs
    print("Loading stage 1 outputs...")
    paysim = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "paysim_clean.parquet")
    )
    lc_raw = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "lending_club_clean.parquet")
    )
    print(f"  PaySim:       {paysim.shape}")
    print(f"  LendingClub:  {lc_raw.shape}")

    txn_features  = build_transaction_features(paysim)
    loan_features = build_loan_features(lc_raw)
    loan_features = synthetic_cohort_assignment(loan_features, txn_features)
    master        = build_master_feature_table(loan_features, txn_features)

    txn_features.to_parquet(
        os.path.join(PROCESSED_DIR, "txn_features.parquet"), index=False
    )
    loan_features.to_parquet(
        os.path.join(PROCESSED_DIR, "loan_features.parquet"), index=False
    )
    master.to_parquet(
        os.path.join(PROCESSED_DIR, "master_features.parquet"), index=False
    )

    print("\nAll feature files saved.")
    print(f"  txn_features:    {txn_features.shape}")
    print(f"  loan_features:   {loan_features.shape}")
    print(f"  master_features: {master.shape}")
    print("\nStage 2 complete. Next: python src/model_xgb.py")