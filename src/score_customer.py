"""
INTERACTIVE CUSTOMER SCORING
Run: python src/score_customer.py

Loads the trained XGBoost model and scores a single customer
based on manually provided loan and transaction inputs.
Outputs: default probability, risk tier, SHAP reason codes.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
import shap
from src.config import OUTPUTS_DIR, RISK_TIER_THRESHOLDS


# ── Load model once ────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(OUTPUTS_DIR, "xgb_model.json")

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model not found. Run python src/model_xgb.py first.")
    sys.exit(1)

booster = xgb.Booster()
booster.load_model(MODEL_PATH)
explainer = shap.TreeExplainer(booster)

ALL_FEATURES = [
    "txn_count", "mean_amount", "std_amount", "cv_amount",
    "withdrawal_deposit_ratio", "cashout_proportion",
    "max_withdrawal_to_balance", "overdraft_flag",
    "balance_vol_3m", "balance_slope_3m", "txn_count_lag1",
    "mean_amount_mom_change", "distress_duration_6m",
    "mean_balance", "closing_balance",
    "loan_amnt", "term_months", "int_rate", "installment",
    "grade_encoded", "annual_inc", "dti", "fico_mid",
    "open_acc", "revol_util", "revol_bal",
    "loan_to_income", "interest_burden",
    "high_revol_util_flag", "has_pub_rec",
    "grade_x_distress", "grade_x_volatility", "grade_x_withdrawal",
    "dti_x_distress", "burden_x_overdraft", "fico_x_volatility",
    "dti_x_withdrawal", "income_x_cashout", "behavioral_risk_score"
]

GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}


# ══════════════════════════════════════════════════════════════════════════════
# INPUT COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

def get_float(prompt, min_val=None, max_val=None, default=None):
    """Get a validated float input from the user."""
    while True:
        try:
            raw = input(prompt).strip()
            if raw == "" and default is not None:
                return float(default)
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f"  Value must be >= {min_val}. Try again.")
                continue
            if max_val is not None and val > max_val:
                print(f"  Value must be <= {max_val}. Try again.")
                continue
            return val
        except ValueError:
            print("  Invalid input. Enter a number.")


def get_int(prompt, choices=None, default=None):
    """Get a validated integer input from the user."""
    while True:
        try:
            raw = input(prompt).strip()
            if raw == "" and default is not None:
                return int(default)
            val = int(raw)
            if choices and val not in choices:
                print(f"  Choose from: {choices}")
                continue
            return val
        except ValueError:
            print("  Invalid input. Enter an integer.")


def get_grade(prompt):
    """Get a validated loan grade."""
    while True:
        raw = input(prompt).strip().upper()
        if raw in GRADE_MAP:
            return raw
        print("  Invalid grade. Enter A, B, C, D, E, F, or G.")


def collect_loan_inputs():
    """Collect loan-level inputs from the user."""
    print("\n" + "─" * 50)
    print("  LOAN DETAILS")
    print("─" * 50)

    loan_amnt  = get_float("  Loan amount ($):                   ", min_val=500, max_val=40000)
    term_months = get_int("  Loan term (36 or 60 months):       ", choices=[36, 60])
    int_rate   = get_float("  Interest rate (%):                 ", min_val=5, max_val=31)
    installment = get_float("  Monthly installment ($):           ", min_val=10, max_val=1500)
    grade      = get_grade("  Loan grade (A/B/C/D/E/F/G):       ")
    annual_inc = get_float("  Annual income ($):                 ", min_val=10000)
    dti        = get_float("  Debt-to-income ratio (%):          ", min_val=0, max_val=100)
    fico_low   = get_float("  FICO score low end:                ", min_val=300, max_val=850)
    fico_high  = get_float("  FICO score high end:               ", min_val=300, max_val=850)
    open_acc   = get_int("  Number of open credit lines:       ", default=8)
    revol_util = get_float("  Revolving utilization (%):         ", min_val=0, max_val=100)
    revol_bal  = get_float("  Revolving balance ($):             ", min_val=0)
    pub_rec    = get_int("  Number of public derogatory records:", default=0)

    return {
        "loan_amnt":   loan_amnt,
        "term_months": term_months,
        "int_rate":    int_rate,
        "installment": installment,
        "grade":       grade,
        "annual_inc":  annual_inc,
        "dti":         dti,
        "fico_low":    fico_low,
        "fico_high":   fico_high,
        "open_acc":    open_acc,
        "revol_util":  revol_util,
        "revol_bal":   revol_bal,
        "pub_rec":     pub_rec
    }


def collect_transaction_inputs():
    """Collect transaction behavioral inputs from the user."""
    print("\n" + "─" * 50)
    print("  TRANSACTION BEHAVIOR (last month)")
    print("─" * 50)
    print("  (Press Enter to use default values)\n")

    txn_count    = get_float("  Number of transactions last month: ", min_val=0, default=15)
    mean_amount  = get_float("  Average transaction amount ($):    ", min_val=0, default=500)
    std_amount   = get_float("  Std dev of transaction amounts ($):", min_val=0, default=200)
    total_outflow = get_float("  Total outflows last month ($):     ", min_val=0, default=2000)
    total_inflow  = get_float("  Total inflows last month ($):      ", min_val=0, default=2500)
    cashout_amt  = get_float("  Cash-out amount last month ($):    ", min_val=0, default=300)
    closing_bal  = get_float("  Closing balance last month ($):    ", default=5000)
    mean_balance = get_float("  Average balance last month ($):    ", min_val=0, default=5500)
    std_balance  = get_float("  Std dev of balance ($):            ", min_val=0, default=800)

    print("\n  ── Rolling / historical behavior ──")
    distress_6m  = get_float("  Months in distress last 6 months (0-6): ", min_val=0, max_val=6, default=1)
    balance_slope = get_float("  Balance slope last 3 months (+/- $):    ", default=100)
    txn_lag1     = get_float("  Transactions count prior month:          ", min_val=0, default=14)
    mom_change   = get_float("  Month-over-month change in avg amount (%)", default=2)

    return {
        "txn_count":            txn_count,
        "mean_amount":          mean_amount,
        "std_amount":           std_amount,
        "total_outflow":        total_outflow,
        "total_inflow":         total_inflow,
        "cashout_amount":       cashout_amt,
        "closing_balance":      closing_bal,
        "mean_balance":         mean_balance,
        "std_balance":          std_balance,
        "distress_duration_6m": distress_6m,
        "balance_slope_3m":     balance_slope,
        "txn_count_lag1":       txn_lag1,
        "mean_amount_mom_change": mom_change / 100
    }


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_features(loan: dict, txn: dict) -> pd.DataFrame:
    """
    Transform raw inputs into the 39 model features.
    Mirrors exactly what feature_engineering.py computes
    so the scoring is consistent with training.
    """

    # ── Loan derived features ─────────────────────────────────────────────────
    grade_encoded    = GRADE_MAP[loan["grade"]]
    monthly_income   = loan["annual_inc"] / 12
    loan_to_income   = loan["loan_amnt"] / (loan["annual_inc"] + 1e-9)
    interest_burden  = loan["installment"] / (monthly_income + 1e-9)
    fico_mid         = (loan["fico_low"] + loan["fico_high"]) / 2
    has_pub_rec      = int(loan["pub_rec"] > 0)
    high_revol_util  = int(loan["revol_util"] > 75)

    # ── Transaction derived features ─────────────────────────────────────────
    mean_bal  = txn["mean_balance"]
    close_bal = txn["closing_balance"]

    cv_amount = (
        txn["std_amount"] / (txn["mean_amount"] + 1e-9)
    )
    withdrawal_deposit_ratio = (
        txn["total_outflow"] / (txn["total_inflow"] + 1e-9)
    )
    cashout_proportion = (
        txn["cashout_amount"] /
        (txn["total_outflow"] + txn["total_inflow"] + 1e-9)
    )
    max_withdrawal_to_balance = (
        txn["total_outflow"] / (mean_bal + 1e-9)
    )
    overdraft_flag = int(close_bal < 0.10 * mean_bal)
    balance_vol_3m = txn["std_balance"]  # current month std as proxy

    # ── Interaction features ──────────────────────────────────────────────────
    grade_x_distress   = grade_encoded * txn["distress_duration_6m"]
    grade_x_volatility = grade_encoded * balance_vol_3m
    grade_x_withdrawal = grade_encoded * withdrawal_deposit_ratio
    dti_x_distress     = loan["dti"]      * txn["distress_duration_6m"]
    burden_x_overdraft = interest_burden  * overdraft_flag
    fico_x_volatility  = fico_mid         * balance_vol_3m
    dti_x_withdrawal   = loan["dti"]      * withdrawal_deposit_ratio
    income_x_cashout   = loan["annual_inc"] * cashout_proportion

    # ── Behavioral risk score ─────────────────────────────────────────────────
    # Uses raw values — normalization against training population
    # is approximated here with reasonable scale assumptions
    behavioral_risk_score = (
        0.30 * min(txn["distress_duration_6m"] / 6, 1.0) +
        0.25 * min(withdrawal_deposit_ratio / 3, 1.0) +
        0.20 * min(balance_vol_3m / 5000, 1.0) +
        0.15 * overdraft_flag +
        0.10 * min(cashout_proportion, 1.0)
    )

    # ── Assemble feature vector ───────────────────────────────────────────────
    features = {
        "txn_count":                txn["txn_count"],
        "mean_amount":              txn["mean_amount"],
        "std_amount":               txn["std_amount"],
        "cv_amount":                cv_amount,
        "withdrawal_deposit_ratio": withdrawal_deposit_ratio,
        "cashout_proportion":       cashout_proportion,
        "max_withdrawal_to_balance":max_withdrawal_to_balance,
        "overdraft_flag":           overdraft_flag,
        "balance_vol_3m":           balance_vol_3m,
        "balance_slope_3m":         txn["balance_slope_3m"],
        "txn_count_lag1":           txn["txn_count_lag1"],
        "mean_amount_mom_change":   txn["mean_amount_mom_change"],
        "distress_duration_6m":     txn["distress_duration_6m"],
        "mean_balance":             mean_bal,
        "closing_balance":          close_bal,
        "loan_amnt":                loan["loan_amnt"],
        "term_months":              loan["term_months"],
        "int_rate":                 loan["int_rate"],
        "installment":              loan["installment"],
        "grade_encoded":            grade_encoded,
        "annual_inc":               loan["annual_inc"],
        "dti":                      loan["dti"],
        "fico_mid":                 fico_mid,
        "open_acc":                 loan["open_acc"],
        "revol_util":               loan["revol_util"],
        "revol_bal":                loan["revol_bal"],
        "loan_to_income":           loan_to_income,
        "interest_burden":          interest_burden,
        "high_revol_util_flag":     high_revol_util,
        "has_pub_rec":              has_pub_rec,
        "grade_x_distress":         grade_x_distress,
        "grade_x_volatility":       grade_x_volatility,
        "grade_x_withdrawal":       grade_x_withdrawal,
        "dti_x_distress":           dti_x_distress,
        "burden_x_overdraft":       burden_x_overdraft,
        "fico_x_volatility":        fico_x_volatility,
        "dti_x_withdrawal":         dti_x_withdrawal,
        "income_x_cashout":         income_x_cashout,
        "behavioral_risk_score":    behavioral_risk_score,
    }

    return pd.DataFrame([features])[ALL_FEATURES].astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SCORING AND OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def assign_risk_tier(prob: float) -> str:
    for tier, (lo, hi) in RISK_TIER_THRESHOLDS.items():
        if lo <= prob < hi:
            return tier
    return "Very High"


def score(feature_df: pd.DataFrame):
    """Run prediction and SHAP explanation."""
    dmatrix = xgb.DMatrix(feature_df, feature_names=ALL_FEATURES)
    prob    = float(booster.predict(dmatrix)[0])
    tier    = assign_risk_tier(prob)

    # SHAP reason codes
    shap_vals = explainer.shap_values(feature_df)[0]
    top3_idx  = np.argsort(np.abs(shap_vals))[::-1][:5]

    reason_codes = [
        {
            "rank":    i + 1,
            "feature": ALL_FEATURES[idx],
            "value":   float(feature_df.iloc[0][ALL_FEATURES[idx]]),
            "shap":    float(shap_vals[idx]),
            "direction": "↑ increases risk" if shap_vals[idx] > 0
                         else "↓ decreases risk"
        }
        for i, idx in enumerate(top3_idx)
    ]

    return prob, tier, reason_codes


def print_result(prob, tier, reason_codes, loan, txn):
    """Print formatted scoring result."""

    # Risk tier color coding via text
    tier_display = {
        "Low":       "LOW RISK",
        "Medium":    "MEDIUM RISK",
        "High":      "HIGH RISK",
        "Very High": "VERY HIGH RISK"
    }

    print("\n" + "═" * 55)
    print("  SCORING RESULT")
    print("═" * 55)
    print(f"  Default Probability:  {prob:.4f}  ({prob*100:.1f}%)")
    print(f"  Risk Tier:            {tier_display.get(tier, tier)}")
    print(f"  Loan Grade:           {loan['grade']}  "
          f"(encoded: {GRADE_MAP[loan['grade']]})")

    # Risk tier context
    tier_context = {
        "Low":       "Model expects this borrower to repay. "
                     "Standard monitoring applies.",
        "Medium":    "Elevated risk. Recommend enhanced monitoring "
                     "and review at next payment cycle.",
        "High":      "High default risk. Flag for early intervention "
                     "and consider credit limit review.",
        "Very High": "Critical default risk. Immediate review "
                     "recommended. Escalate to credit committee."
    }
    print(f"\n  Assessment: {tier_context[tier]}")

    print(f"\n{'─'*55}")
    print("  TOP 5 RISK DRIVERS  (SHAP Reason Codes)")
    print(f"{'─'*55}")
    print(f"  {'Rank':<5} {'Feature':<30} {'Value':>10} "
          f"{'SHAP':>8}  Direction")
    print(f"  {'─'*4} {'─'*29} {'─'*10} "
          f"{'─'*8}  {'─'*18}")

    for r in reason_codes:
        print(f"  {r['rank']:<5} {r['feature']:<30} "
              f"{r['value']:>10.3f} "
              f"{r['shap']:>8.4f}  {r['direction']}")

    print(f"\n{'─'*55}")
    print("  KEY RATIOS")
    print(f"{'─'*55}")

    monthly_income = loan["annual_inc"] / 12
    wd_ratio       = txn["total_outflow"] / (txn["total_inflow"] + 1e-9)
    burden         = loan["installment"] / (monthly_income + 1e-9) * 100
    lti            = loan["loan_amnt"] / loan["annual_inc"]

    print(f"  Withdrawal/Deposit Ratio:  {wd_ratio:.2f}x  "
          f"{'⚠ Above 1 — spending > earning' if wd_ratio > 1 else '✓ Within normal range'}")
    print(f"  Interest Burden:           {burden:.1f}%  "
          f"{'⚠ Above 40% threshold' if burden > 40 else '✓ Within affordability range'}")
    print(f"  Loan-to-Income:            {lti:.2f}x  "
          f"{'⚠ High leverage' if lti > 0.5 else '✓ Reasonable leverage'}")
    print(f"  Distress Duration (6m):    "
          f"{int(txn['distress_duration_6m'])} months  "
          f"{'⚠ Persistent distress' if txn['distress_duration_6m'] >= 3 else '✓ Low distress'}")

    print("═" * 55)


# ══════════════════════════════════════════════════════════════════════════════
# PRESET SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "1": {
        "name": "Healthy Borrower — Grade A, stable income",
        "loan": {
            "loan_amnt": 10000, "term_months": 36,
            "int_rate": 7.5,    "installment": 312,
            "grade": "A",       "annual_inc": 90000,
            "dti": 10,          "fico_low": 750,
            "fico_high": 754,   "open_acc": 8,
            "revol_util": 15,   "revol_bal": 5000,
            "pub_rec": 0
        },
        "txn": {
            "txn_count": 20,        "mean_amount": 800,
            "std_amount": 150,      "total_outflow": 3000,
            "total_inflow": 5000,   "cashout_amount": 200,
            "closing_balance": 12000, "mean_balance": 13000,
            "std_balance": 500,     "distress_duration_6m": 0,
            "balance_slope_3m": 300, "txn_count_lag1": 18,
            "mean_amount_mom_change": 0.02
        }
    },
    "2": {
        "name": "Stressed Borrower — Grade D, high withdrawal ratio",
        "loan": {
            "loan_amnt": 25000, "term_months": 60,
            "int_rate": 19.5,   "installment": 650,
            "grade": "D",       "annual_inc": 48000,
            "dti": 28,          "fico_low": 660,
            "fico_high": 664,   "open_acc": 12,
            "revol_util": 78,   "revol_bal": 18000,
            "pub_rec": 1
        },
        "txn": {
            "txn_count": 35,        "mean_amount": 400,
            "std_amount": 350,      "total_outflow": 4500,
            "total_inflow": 3800,   "cashout_amount": 1200,
            "closing_balance": 800,  "mean_balance": 1500,
            "std_balance": 1200,    "distress_duration_6m": 4,
            "balance_slope_3m": -300, "txn_count_lag1": 32,
            "mean_amount_mom_change": -0.05
        }
    },
    "3": {
        "name": "Borderline Case — Grade C, mixed signals",
        "loan": {
            "loan_amnt": 15000, "term_months": 36,
            "int_rate": 13.5,   "installment": 510,
            "grade": "C",       "annual_inc": 62000,
            "dti": 18,          "fico_low": 700,
            "fico_high": 704,   "open_acc": 9,
            "revol_util": 45,   "revol_bal": 9000,
            "pub_rec": 0
        },
        "txn": {
            "txn_count": 22,        "mean_amount": 550,
            "std_amount": 280,      "total_outflow": 3200,
            "total_inflow": 3400,   "cashout_amount": 500,
            "closing_balance": 4500, "mean_balance": 5000,
            "std_balance": 900,     "distress_duration_6m": 2,
            "balance_slope_3m": 50, "txn_count_lag1": 20,
            "mean_amount_mom_change": 0.01
        }
    }
}


def run_preset(scenario_key: str):
    """Run a preset scenario without manual input."""
    s = SCENARIOS[scenario_key]
    print(f"\n  Running preset: {s['name']}")
    features = compute_features(s["loan"], s["txn"])
    prob, tier, reason_codes = score(features)
    print_result(prob, tier, reason_codes, s["loan"], s["txn"])
    return prob, tier


def run_manual():
    """Run full manual input collection."""
    loan = collect_loan_inputs()
    txn  = collect_transaction_inputs()
    features = compute_features(loan, txn)
    prob, tier, reason_codes = score(features)
    print_result(prob, tier, reason_codes, loan, txn)
    return prob, tier


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 55)
    print("  BANKING RISK ANALYTICS — CUSTOMER SCORER")
    print("  XGBoost Default Probability Engine v1.0")
    print("═" * 55)

    while True:
        print("\n  Choose an option:")
        print("  [1] Score manually entered customer")
        print("  [2] Run preset: Healthy Borrower (Grade A)")
        print("  [3] Run preset: Stressed Borrower (Grade D)")
        print("  [4] Run preset: Borderline Case (Grade C)")
        print("  [5] Run all three presets for comparison")
        print("  [Q] Quit")

        choice = input("\n  Your choice: ").strip().upper()

        if choice == "1":
            run_manual()

        elif choice == "2":
            run_preset("1")

        elif choice == "3":
            run_preset("2")

        elif choice == "4":
            run_preset("3")

        elif choice == "5":
            print("\n" + "═" * 55)
            print("  RUNNING ALL THREE PRESETS")
            print("═" * 55)
            results = []
            for key in ["1", "2", "3"]:
                prob, tier = run_preset(key)
                results.append({
                    "scenario": SCENARIOS[key]["name"],
                    "probability": prob,
                    "tier": tier
                })

            print("\n" + "═" * 55)
            print("  COMPARISON SUMMARY")
            print("═" * 55)
            for r in results:
                print(f"  {r['probability']*100:5.1f}%  "
                      f"{r['tier']:<12}  {r['scenario']}")
            print("═" * 55)

        elif choice == "Q":
            print("\n  Exiting scorer.\n")
            break

        else:
            print("  Invalid choice. Enter 1, 2, 3, 4, 5, or Q.")

        cont = input("\n  Score another customer? (Y/N): ").strip().upper()
        if cont != "Y":
            print("\n  Exiting scorer.\n")
            break


if __name__ == "__main__":
    main()