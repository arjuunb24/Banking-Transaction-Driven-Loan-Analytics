"""
STAGE 1 — DATA LOADING
Run independently: python src/data_loader.py
Inputs:  data/raw/paysim.csv
         data/raw/lending_club.csv.gz
Outputs: data/processed/paysim_clean.parquet
         data/processed/lending_club_clean.parquet
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.config import (
    PAYSIM_PATH, LENDING_CLUB_PATH,
    PROCESSED_DIR, RANDOM_SEED
)


def load_paysim() -> pd.DataFrame:
    print("=" * 55)
    print("LOADING PAYSIM")
    print("=" * 55)

    df = pd.read_csv(PAYSIM_PATH)
    print(f"  Raw rows: {len(df):,}")

    # Drop fraudulent/cancelled transactions
    # Balance columns are zeroed out for fraud rows — unreliable for features
    df = df[df["isFraud"] == 0].copy()
    print(f"  After dropping fraud: {len(df):,}")

    # Keep only customer originating accounts
    df = df[df["nameOrig"].str.startswith("C")].copy()
    print(f"  After filtering to customers: {len(df):,}")

    # Simulate dates — step=1 is hour 1 of Jan 2019
    base_date = pd.Timestamp("2019-01-01")
    df["transaction_date"] = base_date + pd.to_timedelta(df["step"] - 1, unit="h")
    df["month_key"]        = df["transaction_date"].dt.to_period("M").astype(str)

    df = df.rename(columns={
        "nameOrig":       "customer_id",
        "amount":         "amount",
        "oldbalanceOrg":  "balance_before",
        "newbalanceOrig": "balance_after",
        "type":           "transaction_type",
        "isFraud":        "fraud_flag"
    })

    df = df[[
        "customer_id", "transaction_date", "month_key",
        "transaction_type", "amount",
        "balance_before", "balance_after", "fraud_flag"
    ]]

    print(f"  Final shape: {df.shape}")
    return df


def load_lending_club() -> pd.DataFrame:
    print("=" * 55)
    print("LOADING LENDING CLUB (chunked)")
    print("=" * 55)

    usecols = [
        "loan_amnt", "term", "int_rate", "installment",
        "grade", "sub_grade", "annual_inc", "loan_status",
        "dti", "issue_d", "addr_state",
        "fico_range_low", "fico_range_high",
        "open_acc", "pub_rec", "revol_util",
        "revol_bal", "total_acc"
    ]

    resolved_statuses = [
        "Fully Paid", "Charged Off", "Default",
        "Does not meet the credit policy. Status:Fully Paid",
        "Does not meet the credit policy. Status:Charged Off"
    ]

    chunks = []
    for i, chunk in enumerate(pd.read_csv(
        LENDING_CLUB_PATH,
        usecols=usecols,
        low_memory=False,
        chunksize=50_000,
        on_bad_lines="skip"
    )):
        chunk = chunk[chunk["loan_status"].isin(resolved_statuses)]
        if not chunk.empty:
            chunks.append(chunk)
        if i % 20 == 0:
            kept = sum(len(c) for c in chunks)
            print(f"  Chunk {i:>4} — kept rows so far: {kept:,}")

    df = pd.concat(chunks, ignore_index=True)
    print(f"  After resolved filter: {len(df):,}")

    # Binary target
    default_statuses = [
        "Charged Off", "Default",
        "Does not meet the credit policy. Status:Charged Off"
    ]
    df["default_flag"] = df["loan_status"].isin(default_statuses).astype(int)

    # Parse dates
    df["issue_date"]  = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["issue_year"]  = df["issue_date"].dt.year
    df["issue_month"] = df["issue_date"].dt.month

    # Clean pct columns
    for col in ["int_rate", "revol_util"]:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].str.replace("%", "", regex=False), errors="coerce"
            )

    # Ordinal grade
    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    df["grade_encoded"] = df["grade"].map(grade_map)

    # Synthetic customer IDs — reproducible
    np.random.seed(RANDOM_SEED)
    df = df.reset_index(drop=True)
    df["customer_id"] = [
        "C" + str(i).zfill(8)
        for i in np.random.permutation(len(df))
    ]

    df = df.dropna(subset=["loan_amnt", "annual_inc", "grade", "issue_date"])
    df = df.reset_index(drop=True)

    print(f"  Final shape: {df.shape}")
    print(f"  Default rate: {df['default_flag'].mean():.2%}")
    return df


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    paysim = load_paysim()
    paysim.to_parquet(
        os.path.join(PROCESSED_DIR, "paysim_clean.parquet"), index=False
    )
    print(f"\nSaved: data/processed/paysim_clean.parquet")

    lc = load_lending_club()
    lc.to_parquet(
        os.path.join(PROCESSED_DIR, "lending_club_clean.parquet"), index=False
    )
    print(f"Saved: data/processed/lending_club_clean.parquet")

    print("\nStage 1 complete. Next: python src/feature_engineering.py")