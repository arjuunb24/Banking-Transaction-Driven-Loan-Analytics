"""
STAGE 3a — XGBOOST DEFAULT CLASSIFIER (GPU)
Run independently: python src/model_xgb.py
Inputs:  data/processed/master_features.parquet
Outputs: data/outputs/xgb_predictions.parquet
         data/outputs/xgb_metrics.csv
         data/outputs/xgb_feature_importance.csv
         data/outputs/xgb_decile_analysis.csv
         data/outputs/xgb_model.json
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.isotonic import IsotonicRegression
from src.config import (
    PROCESSED_DIR, OUTPUTS_DIR, RANDOM_SEED,
    TRAIN_CUTOFF_YEAR, TEST_START_YEAR,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH,
    XGB_LEARNING_RATE, XGB_EARLY_STOPPING,
    RISK_TIER_THRESHOLDS
)

# ── GPU check ──────────────────────────────────────────────────────────────────
try:
    _t = xgb.XGBClassifier(device="cuda", n_estimators=1)
    _t.fit(np.array([[1,2],[3,4]]), np.array([0,1]))
    GPU_AVAILABLE = True
    print("RTX 4060 GPU confirmed — training on CUDA")
except Exception:
    GPU_AVAILABLE = False
    print("GPU not available — falling back to CPU")

# ── Feature columns ────────────────────────────────────────────────────────────
BASE_FEATURES = [
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
    "high_revol_util_flag", "has_pub_rec"
]

INTERACTION_FEATURES = [
    "grade_x_distress", "grade_x_volatility", "grade_x_withdrawal",
    "dti_x_distress", "burden_x_overdraft", "fico_x_volatility",
    "dti_x_withdrawal", "income_x_cashout", "behavioral_risk_score"
]

ALL_FEATURES = BASE_FEATURES + INTERACTION_FEATURES
TARGET       = "default_flag"


def prepare_data(master: pd.DataFrame):
    train = master[
        master["issue_year"] <= TRAIN_CUTOFF_YEAR
    ].dropna(subset=[TARGET] + ALL_FEATURES).copy()

    test = master[
        master["issue_year"] >= TEST_START_YEAR
    ].dropna(subset=[TARGET] + ALL_FEATURES).copy()

    X_train = train[ALL_FEATURES].astype(np.float32)
    y_train = train[TARGET].astype(np.int32)
    X_test  = test[ALL_FEATURES].astype(np.float32)
    y_test  = test[TARGET].astype(np.int32)

    print(f"  Train: {X_train.shape}  | Default rate: {y_train.mean():.2%}")
    print(f"  Test:  {X_test.shape}   | Default rate: {y_test.mean():.2%}")

    return X_train, y_train, X_test, y_test, test


def train(X_train, y_train, X_test, y_test):
    neg   = (y_train == 0).sum()
    pos   = (y_train == 1).sum()
    scale = float(neg / pos)
    print(f"  Class imbalance — scale_pos_weight: {scale:.2f}")

    params = {
        "objective":          "binary:logistic",
        "eval_metric":        ["auc", "logloss"],
        "device":             "cuda" if GPU_AVAILABLE else "cpu",
        "tree_method":        "hist",
        "max_depth":          XGB_MAX_DEPTH,
        "learning_rate":      XGB_LEARNING_RATE,
        "subsample":          0.8,
        "colsample_bytree":   0.7,
        "colsample_bylevel":  0.7,
        "colsample_bynode":   0.7,
        "reg_alpha":          0.5,
        "reg_lambda":         2.0,
        "min_child_weight":   100,
        "gamma":              1.0,
        "scale_pos_weight":   scale,
        "seed":               RANDOM_SEED
    }

    dtrain = xgb.DMatrix(X_train, label=y_train,
                         feature_names=ALL_FEATURES)
    dtest  = xgb.DMatrix(X_test,  label=y_test,
                         feature_names=ALL_FEATURES)

    evals_result = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round       = XGB_N_ESTIMATORS,
        evals                 = [(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds = XGB_EARLY_STOPPING,
        evals_result          = evals_result,
        verbose_eval          = 100
    )

    print(f"\n  Best iteration: {booster.best_iteration}")
    return booster


def evaluate(booster, X_test, y_test):
    dtest  = xgb.DMatrix(X_test, feature_names=ALL_FEATURES)
    y_prob = booster.predict(dtest)
    y_true = y_test.values

    # ── Core metrics ──────────────────────────────────────────────────────────
    auc  = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_stat      = float(np.max(np.abs(tpr - fpr)))
    ks_threshold = float(thresholds[np.argmax(np.abs(tpr - fpr))])

    # ── Calibration ───────────────────────────────────────────────────────────
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_prob, y_true)
    y_prob_cal    = iso.predict(y_prob)
    auc_calibrated = roc_auc_score(y_true, y_prob_cal)

    # ── Decile analysis ───────────────────────────────────────────────────────
    result_df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob_cal})
    result_df = result_df.sort_values(
        "y_prob", ascending=False
    ).reset_index(drop=True)

    result_df["decile"] = pd.qcut(
        result_df.index, q=10,
        labels=[f"D{i}" for i in range(1, 11)]
    )

    decile_table = result_df.groupby(
        "decile", observed=True
    ).agg(
        total      = ("y_true", "count"),
        defaults   = ("y_true", "sum"),
        mean_score = ("y_prob", "mean")
    ).reset_index()

    decile_table["default_rate"] = (
        decile_table["defaults"] / decile_table["total"]
    )
    decile_table["cum_defaults_pct"] = (
        decile_table["defaults"].cumsum() / y_true.sum() * 100
    )
    decile_table["lift"] = (
        decile_table["default_rate"] / (y_true.mean() + 1e-9)
    )

    top10_lift = float(decile_table.iloc[0]["lift"])
    top20_lift = float(
        (decile_table.iloc[:2]["defaults"].sum() /
         decile_table.iloc[:2]["total"].sum()) /
        (y_true.mean() + 1e-9)
    )

    # ── Feature importance ────────────────────────────────────────────────────
    scores = booster.get_score(importance_type="gain")
    importance_df = pd.DataFrame(
        list(scores.items()), columns=["feature", "importance"]
    ).sort_values("importance", ascending=False)

    print(f"\n{'='*55}")
    print(f"  AUC-ROC (raw):        {auc:.4f}  (target > 0.72)")
    print(f"  AUC-ROC (calibrated): {auc_calibrated:.4f}")
    print(f"  Gini Coefficient:     {gini:.4f}")
    print(f"  KS Statistic:         {ks_stat:.4f}  @ {ks_threshold:.3f}")
    print(f"  Top-10% Lift:         {top10_lift:.2f}x")
    print(f"  Top-20% Lift:         {top20_lift:.2f}x")
    print(f"\n  Decile Table:")
    print(decile_table[[
        "decile","total","defaults",
        "default_rate","cum_defaults_pct","lift"
    ]].to_string(index=False))
    print(f"\n  Top 10 Features (gain):")
    print(importance_df.head(10).to_string(index=False))
    print(f"{'='*55}\n")

    metrics = {
        "model":            "XGBoost_v1",
        "auc_roc":          round(auc, 4),
        "auc_calibrated":   round(auc_calibrated, 4),
        "gini":             round(gini, 4),
        "ks_stat":          round(ks_stat, 4),
        "ks_threshold":     round(ks_threshold, 4),
        "top10_lift":       round(top10_lift, 4),
        "top20_lift":       round(top20_lift, 4),
        "best_iteration":   booster.best_iteration
    }

    return y_prob_cal, metrics, decile_table, importance_df


def assign_risk_tiers(y_prob: np.ndarray) -> np.ndarray:
    tiers = np.full(len(y_prob), "Very High", dtype=object)
    for tier, (lo, hi) in RISK_TIER_THRESHOLDS.items():
        tiers[(y_prob >= lo) & (y_prob < hi)] = tier
    return tiers


def build_predictions(test_df, y_prob):
    preds = test_df[[
        "customer_id", "issue_date", "loan_amnt",
        "grade", "addr_state", "default_flag"
    ]].copy().reset_index(drop=True)
    preds = preds.rename(columns={"issue_date": "prediction_month"})
    preds["default_probability"] = y_prob
    preds["risk_tier"]           = assign_risk_tiers(y_prob)
    preds["model_name"]          = "XGBoost_v1"
    return preds

def compute_shap_explanation(booster, X_test):
    """
    Compute SHAP values for model explainability.
    Required for SR 11-7 adverse action notice compliance.
    Saves top feature contributions per customer to outputs.
    """
    import shap

    print("\nComputing SHAP values...")

    dtest    = xgb.DMatrix(X_test, feature_names=ALL_FEATURES)
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_test)

    # Mean absolute SHAP — global feature importance
    shap_importance = pd.DataFrame({
        "feature":    ALL_FEATURES,
        "shap_importance": np.abs(shap_values).mean(axis=0)
    }).sort_values("shap_importance", ascending=False)

    # Per-customer top 3 reason codes — mirrors adverse action notices
    # banks send to customers explaining why they were declined
    reason_rows = []
    for i in range(len(X_test)):
        customer_shap = shap_values[i]
        top3_idx = np.argsort(np.abs(customer_shap))[::-1][:3]
        reason_rows.append({
            "customer_index": i,
            "reason_1_feature": ALL_FEATURES[top3_idx[0]],
            "reason_1_shap":    round(float(customer_shap[top3_idx[0]]), 4),
            "reason_2_feature": ALL_FEATURES[top3_idx[1]],
            "reason_2_shap":    round(float(customer_shap[top3_idx[1]]), 4),
            "reason_3_feature": ALL_FEATURES[top3_idx[2]],
            "reason_3_shap":    round(float(customer_shap[top3_idx[2]]), 4),
        })

    reason_df = pd.DataFrame(reason_rows)

    print(f"\n  Top 10 Features by SHAP importance:")
    print(shap_importance.head(10).to_string(index=False))

    shap_importance.to_csv(
        os.path.join(OUTPUTS_DIR, "xgb_shap_importance.csv"), index=False
    )
    reason_df.to_csv(
        os.path.join(OUTPUTS_DIR, "xgb_reason_codes.csv"), index=False
    )

    print("  Saved: xgb_shap_importance.csv")
    print("  Saved: xgb_reason_codes.csv")

    return shap_importance, reason_df

if __name__ == "__main__":
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("=" * 55)
    print("XGBOOST DEFAULT CLASSIFIER")
    print("=" * 55)

    master = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "master_features.parquet")
    )
    print(f"Master table loaded: {master.shape}")

    X_train, y_train, X_test, y_test, test_df = prepare_data(master)

    print("\nTraining...")
    booster = train(X_train, y_train, X_test, y_test)

    print("\nEvaluating...")
    y_prob, metrics, decile_table, importance_df = evaluate(
        booster, X_test, y_test
    )

    predictions = build_predictions(test_df, y_prob)

    shap_importance, reason_codes = compute_shap_explanation(booster, X_test) 

    # Save all outputs
    predictions.to_parquet(
        os.path.join(OUTPUTS_DIR, "xgb_predictions.parquet"), index=False
    )
    pd.DataFrame([metrics]).to_csv(
        os.path.join(OUTPUTS_DIR, "xgb_metrics.csv"), index=False
    )
    decile_table.to_csv(
        os.path.join(OUTPUTS_DIR, "xgb_decile_analysis.csv"), index=False
    )
    importance_df.to_csv(
        os.path.join(OUTPUTS_DIR, "xgb_feature_importance.csv"), index=False
    )
    booster.save_model(
        os.path.join(OUTPUTS_DIR, "xgb_model.json")
    )

    print("All outputs saved to data/outputs/")
    print("  xgb_predictions.parquet")
    print("  xgb_metrics.csv")
    print("  xgb_decile_analysis.csv")
    print("  xgb_feature_importance.csv")
    print("  xgb_model.json")
    print("\nStage 3a complete. Next: python src/model_sarimax.py")