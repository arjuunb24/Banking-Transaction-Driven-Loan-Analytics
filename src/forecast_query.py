"""
INTERACTIVE SARIMAX FORECAST QUERY
Run: python src/forecast_query.py

Loads saved SARIMAX forecasts and provides interactive
queries for segment-level cash flow analysis and
what-if scenario simulation.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.config import OUTPUTS_DIR, PROCESSED_DIR, SARIMAX_M, FORECAST_HORIZON


# ── Load saved forecasts ───────────────────────────────────────────────────────
FORECAST_PATH = os.path.join(OUTPUTS_DIR, "sarimax_forecasts.parquet")
METRICS_PATH  = os.path.join(OUTPUTS_DIR, "sarimax_metrics.csv")

if not os.path.exists(FORECAST_PATH):
    print("ERROR: Forecasts not found. Run python src/model_sarimax.py first.")
    sys.exit(1)

forecasts_df = pd.read_parquet(FORECAST_PATH)
metrics_df   = pd.read_csv(METRICS_PATH)

AVAILABLE_SEGMENTS = sorted(forecasts_df["segment_id"].unique().tolist())


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def format_currency(val):
    """Format large numbers as readable currency strings."""
    if pd.isna(val):
        return "N/A"
    if abs(val) >= 1_000_000:
        return f"${val/1_000_000:.2f}M"
    if abs(val) >= 1_000:
        return f"${val/1_000:.1f}K"
    return f"${val:.0f}"


def print_divider(char="═", width=60):
    print(char * width)


def select_segment():
    """Let user select a segment from available options."""
    print("\n  Available segments:")
    for i, seg in enumerate(AVAILABLE_SEGMENTS, 1):
        print(f"  [{i:>2}] {seg}")

    while True:
        try:
            choice = int(input("\n  Select segment number: ").strip())
            if 1 <= choice <= len(AVAILABLE_SEGMENTS):
                return AVAILABLE_SEGMENTS[choice - 1]
            print(f"  Enter a number between 1 and {len(AVAILABLE_SEGMENTS)}")
        except ValueError:
            print("  Invalid input. Enter a number.")


# ══════════════════════════════════════════════════════════════════════════════
# QUERY 1 — SEGMENT FORECAST
# ══════════════════════════════════════════════════════════════════════════════

def query_segment_forecast():
    """
    Show the 6-month forward forecast for a selected segment
    with confidence intervals and model performance metrics.
    """
    print_divider()
    print("  SEGMENT CASH FLOW FORECAST")
    print_divider()

    segment_id = select_segment()

    seg_data = forecasts_df[
        forecasts_df["segment_id"] == segment_id
    ].copy()
    seg_data["month_key"] = pd.to_datetime(seg_data["month_key"])
    seg_data = seg_data.sort_values("month_key")

    # Split historical and forecast
    historical = seg_data[~seg_data["is_forecast"]].copy()
    forecast   = seg_data[seg_data["is_forecast"]].copy()

    # Model performance metrics
    seg_metrics = metrics_df[
        metrics_df["segment_id"] == segment_id
    ].iloc[0]

    print(f"\n  Segment:         {segment_id}")
    print(f"  Model Order:     {seg_metrics['best_order']} x "
          f"{seg_metrics['best_seasonal_order']}")
    print(f"  SARIMAX MAPE:    {seg_metrics['sarimax_mape']:.3f}%")
    print(f"  HW Benchmark:    {seg_metrics['hw_mape']:.2f}%")
    print(f"  Model Lift:      {seg_metrics['sarimax_lift_pp']:.2f}pp "
          f"over benchmark")
    print(f"  Validation Steps:{int(seg_metrics['n_validation_steps'])}")

    # Last 6 months of history
    print(f"\n{'─'*60}")
    print(f"  RECENT HISTORY (last 6 months)")
    print(f"{'─'*60}")
    print(f"  {'Month':<12} {'Actual':>14} {'Fitted':>14} "
          f"{'Error':>10}")
    print(f"  {'─'*11} {'─'*14} {'─'*14} {'─'*10}")

    recent = historical.tail(6)
    for _, row in recent.iterrows():
        actual    = row["actual_cash_flow"]
        fitted    = row["forecasted_cash_flow"]
        error_pct = ((actual - fitted) / (abs(actual) + 1e-9)) * 100
        print(f"  {str(row['month_key'].date()):<12} "
              f"{format_currency(actual):>14} "
              f"{format_currency(fitted):>14} "
              f"{error_pct:>9.2f}%")

    # 6-month forward forecast
    print(f"\n{'─'*60}")
    print(f"  6-MONTH FORWARD FORECAST")
    print(f"{'─'*60}")
    print(f"  {'Month':<12} {'Forecast':>14} {'Lower (80%)':>14} "
          f"{'Upper (80%)':>14}")
    print(f"  {'─'*11} {'─'*14} {'─'*14} {'─'*14}")

    total_forecast = 0
    for _, row in forecast.iterrows():
        fc    = row["forecasted_cash_flow"]
        lower = row["lower_bound"]
        upper = row["upper_bound"]
        total_forecast += fc if not pd.isna(fc) else 0

        print(f"  {str(row['month_key'].date()):<12} "
              f"{format_currency(fc):>14} "
              f"{format_currency(lower):>14} "
              f"{format_currency(upper):>14}")

    print(f"\n  Total 6-month forecast:  {format_currency(total_forecast)}")

    # Cash flow interpretation
    if total_forecast > 0:
        print(f"\n  ✓ Portfolio generating surplus — repayments exceed "
              f"disbursements over forecast horizon.")
    else:
        print(f"\n  ⚠ Portfolio in net outflow — disbursements exceed "
              f"repayments. Monitor liquidity position.")

    print_divider()


# ══════════════════════════════════════════════════════════════════════════════
# QUERY 2 — MODEL FIT QUALITY
# ══════════════════════════════════════════════════════════════════════════════

def query_fit_quality():
    """
    Show full historical fitted vs actual values
    with period-by-period error analysis.
    """
    print_divider()
    print("  MODEL FIT QUALITY ANALYSIS")
    print_divider()

    segment_id = select_segment()

    seg_data = forecasts_df[
        forecasts_df["segment_id"] == segment_id
    ].copy()
    seg_data["month_key"] = pd.to_datetime(seg_data["month_key"])
    seg_data = seg_data.sort_values("month_key")
    historical = seg_data[~seg_data["is_forecast"]].copy()

    print(f"\n  Segment: {segment_id}")
    print(f"  Months:  {len(historical)}")

    # Period error analysis
    historical["error"]     = (
        historical["actual_cash_flow"] -
        historical["forecasted_cash_flow"]
    )
    historical["error_pct"] = (
        historical["error"] /
        (historical["actual_cash_flow"].abs() + 1e-9) * 100
    )
    historical["direction"] = np.where(
        historical["error"] > 0, "Under", "Over"
    )

    # Summary statistics
    mae  = historical["error"].abs().mean()
    mape = historical["error_pct"].abs().mean()
    bias = historical["error"].mean()
    over_count  = (historical["direction"] == "Over").sum()
    under_count = (historical["direction"] == "Under").sum()

    print(f"\n  {'─'*40}")
    print(f"  In-sample fit statistics:")
    print(f"  MAE:              {format_currency(mae)}")
    print(f"  MAPE:             {mape:.3f}%")
    print(f"  Bias:             {format_currency(bias)}")
    print(f"  Over-predicted:   {over_count} months")
    print(f"  Under-predicted:  {under_count} months")
    print(f"  {'─'*40}")

    # Worst 5 fit months
    worst = historical.nlargest(5, "error_pct".__class__,
                                keep="first")
    worst = historical.reindex(
        historical["error_pct"].abs().nlargest(5).index
    )

    print(f"\n  Largest forecast errors (worst 5 months):")
    print(f"  {'Month':<12} {'Actual':>14} {'Fitted':>14} "
          f"{'Error %':>10} {'Direction':>10}")
    print(f"  {'─'*11} {'─'*14} {'─'*14} "
          f"{'─'*10} {'─'*10}")

    for _, row in worst.iterrows():
        print(f"  {str(row['month_key'].date()):<12} "
              f"{format_currency(row['actual_cash_flow']):>14} "
              f"{format_currency(row['forecasted_cash_flow']):>14} "
              f"{row['error_pct']:>9.2f}% "
              f"{row['direction']:>10}")

    print_divider()


# ══════════════════════════════════════════════════════════════════════════════
# QUERY 3 — WHAT-IF: TRANSACTION VELOCITY SHOCK
# ══════════════════════════════════════════════════════════════════════════════

def query_velocity_shock():
    """
    Simulate impact of a transaction velocity shock on forecast.
    Models economic slowdown — customers transact less frequently.
    Refits SARIMAX with modified exogenous inputs.
    """
    print_divider()
    print("  WHAT-IF: TRANSACTION VELOCITY SHOCK")
    print_divider()
    print("  This simulates an economic slowdown where customers")
    print("  reduce transaction activity. The SARIMAX exogenous")
    print("  regressor (avg_txn_velocity) is reduced by your")
    print("  chosen percentage and the model is re-forecast.\n")

    segment_id = select_segment()

    while True:
        try:
            shock_pct = float(input(
                "\n  Transaction velocity reduction (% e.g. 20): "
            ).strip())
            if 0 < shock_pct < 100:
                break
            print("  Enter a value between 1 and 99.")
        except ValueError:
            print("  Invalid input.")

    print(f"\n  Applying {shock_pct:.0f}% velocity reduction "
          f"to segment: {segment_id}...")

    # Load master features to refit SARIMAX
    master_path = os.path.join(PROCESSED_DIR, "master_features.parquet")
    if not os.path.exists(master_path):
        print("  ERROR: master_features.parquet not found.")
        return

    master = pd.read_parquet(master_path)

    # Rebuild segment cash flow — same logic as model_sarimax.py
    from src.model_sarimax import build_segment_cashflow, prepare_series

    segment_cf = build_segment_cashflow(master)
    seg_df = segment_cf[
        segment_cf["segment_id"] == segment_id
    ].reset_index(drop=True)

    series, exog = prepare_series(seg_df)

    # Get best order from saved metrics
    seg_metrics = metrics_df[
        metrics_df["segment_id"] == segment_id
    ].iloc[0]

    import ast
    order    = ast.literal_eval(seg_metrics["best_order"])
    seasonal = ast.literal_eval(seg_metrics["best_seasonal_order"])

    # Fit final model on full history
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final = SARIMAX(
            series, exog=exog,
            order=order, seasonal_order=seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False, maxiter=300, method="lbfgs")

    # Baseline forecast
    last_exog    = exog.iloc[-1:].values.copy()
    future_exog_base = pd.DataFrame(
        np.tile(last_exog, (FORECAST_HORIZON, 1)),
        columns=exog.columns
    )

    # Shocked forecast — reduce velocity
    future_exog_shock = future_exog_base.copy()
    future_exog_shock["avg_txn_velocity"] *= (1 - shock_pct / 100)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc_base  = final.get_forecast(
            steps=FORECAST_HORIZON,
            exog=future_exog_base
        ).predicted_mean

        fc_shock = final.get_forecast(
            steps=FORECAST_HORIZON,
            exog=future_exog_shock
        ).predicted_mean

    future_idx = pd.date_range(
        start   = series.index[-1] + pd.DateOffset(months=1),
        periods = FORECAST_HORIZON,
        freq    = "MS"
    )

    print(f"\n{'─'*65}")
    print(f"  {'Month':<12} {'Baseline':>14} "
          f"{'Shocked ({:.0f}% drop)'.format(shock_pct):>20} "
          f"{'Impact':>12}")
    print(f"  {'─'*11} {'─'*14} {'─'*20} {'─'*12}")

    total_base  = 0
    total_shock = 0

    for i, month in enumerate(future_idx):
        base  = float(fc_base.iloc[i])
        shock = float(fc_shock.iloc[i])
        impact = shock - base
        total_base  += base
        total_shock += shock
        print(f"  {str(month.date()):<12} "
              f"{format_currency(base):>14} "
              f"{format_currency(shock):>20} "
              f"{format_currency(impact):>12}")

    total_impact = total_shock - total_base
    impact_pct   = (total_impact / (abs(total_base) + 1e-9)) * 100

    print(f"\n  {'─'*65}")
    print(f"  {'6-month total':<12} "
          f"{format_currency(total_base):>14} "
          f"{format_currency(total_shock):>20} "
          f"{format_currency(total_impact):>12}")
    print(f"\n  Net 6-month cash flow impact of {shock_pct:.0f}% "
          f"velocity shock: {format_currency(total_impact)} "
          f"({impact_pct:.1f}%)")

    if total_impact < 0:
        print(f"  ⚠ Negative impact — reduced transaction activity "
              f"is associated with lower repayments.")
    else:
        print(f"  ✓ Positive impact — reduced disbursement activity "
              f"improves net cash flow position.")

    print_divider()


# ══════════════════════════════════════════════════════════════════════════════
# QUERY 4 — WHAT-IF: DISBURSEMENT SURGE
# ══════════════════════════════════════════════════════════════════════════════

def query_disbursement_surge():
    """
    Simulate impact of increased loan disbursements on forecast.
    Models a scenario where new loan issuance accelerates.
    """
    print_divider()
    print("  WHAT-IF: DISBURSEMENT SURGE SCENARIO")
    print_divider()
    print("  This simulates a period of accelerated loan issuance.")
    print("  The SARIMAX exogenous regressor (total_disbursements)")
    print("  is increased by your chosen percentage.\n")

    segment_id = select_segment()

    while True:
        try:
            surge_pct = float(input(
                "\n  Disbursement increase (% e.g. 30): "
            ).strip())
            if 0 < surge_pct < 500:
                break
            print("  Enter a value between 1 and 499.")
        except ValueError:
            print("  Invalid input.")

    print(f"\n  Applying {surge_pct:.0f}% disbursement surge "
          f"to segment: {segment_id}...")

    master_path = os.path.join(PROCESSED_DIR, "master_features.parquet")
    master = pd.read_parquet(master_path)

    from src.model_sarimax import build_segment_cashflow, prepare_series
    segment_cf = build_segment_cashflow(master)
    seg_df = segment_cf[
        segment_cf["segment_id"] == segment_id
    ].reset_index(drop=True)

    series, exog = prepare_series(seg_df)

    seg_metrics = metrics_df[
        metrics_df["segment_id"] == segment_id
    ].iloc[0]

    import ast
    order    = ast.literal_eval(seg_metrics["best_order"])
    seasonal = ast.literal_eval(seg_metrics["best_seasonal_order"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final = SARIMAX(
            series, exog=exog,
            order=order, seasonal_order=seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False, maxiter=300, method="lbfgs")

    last_exog    = exog.iloc[-1:].values.copy()
    future_exog_base = pd.DataFrame(
        np.tile(last_exog, (FORECAST_HORIZON, 1)),
        columns=exog.columns
    )

    future_exog_surge = future_exog_base.copy()
    future_exog_surge["total_disbursements"] *= (1 + surge_pct / 100)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc_base  = final.get_forecast(
            steps=FORECAST_HORIZON,
            exog=future_exog_base
        ).predicted_mean

        fc_surge = final.get_forecast(
            steps=FORECAST_HORIZON,
            exog=future_exog_surge
        ).predicted_mean

    future_idx = pd.date_range(
        start   = series.index[-1] + pd.DateOffset(months=1),
        periods = FORECAST_HORIZON,
        freq    = "MS"
    )

    print(f"\n{'─'*65}")
    print(f"  {'Month':<12} {'Baseline':>14} "
          f"{'Surge (+{:.0f}%)'.format(surge_pct):>20} "
          f"{'Impact':>12}")
    print(f"  {'─'*11} {'─'*14} {'─'*20} {'─'*12}")

    total_base  = 0
    total_surge = 0

    for i, month in enumerate(future_idx):
        base  = float(fc_base.iloc[i])
        surge = float(fc_surge.iloc[i])
        impact = surge - base
        total_base  += base
        total_surge += surge
        print(f"  {str(month.date()):<12} "
              f"{format_currency(base):>14} "
              f"{format_currency(surge):>20} "
              f"{format_currency(impact):>12}")

    total_impact = total_surge - total_base
    impact_pct   = (total_impact / (abs(total_base) + 1e-9)) * 100

    print(f"\n  {'─'*65}")
    print(f"  {'6-month total':<12} "
          f"{format_currency(total_base):>14} "
          f"{format_currency(total_surge):>20} "
          f"{format_currency(total_impact):>12}")
    print(f"\n  Net 6-month cash flow impact of {surge_pct:.0f}% "
          f"disbursement surge: {format_currency(total_impact)} "
          f"({impact_pct:.1f}%)")

    if total_impact < 0:
        print(f"  ⚠ Negative impact — higher disbursements increase "
              f"net outflows. Monitor liquidity.")
    else:
        print(f"  ✓ Positive impact — disbursement surge associated "
              f"with higher future repayments.")

    print_divider()


# ══════════════════════════════════════════════════════════════════════════════
# QUERY 5 — PORTFOLIO SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def query_portfolio_summary():
    """
    Show forecast summary across all 12 segments simultaneously.
    Answers: which segments are generating surplus vs deficit?
    """
    print_divider()
    print("  PORTFOLIO-WIDE FORECAST SUMMARY")
    print_divider()

    forecast_only = forecasts_df[
        forecasts_df["is_forecast"]
    ].copy()

    summary = forecast_only.groupby("segment_id").agg(
        total_forecast  = ("forecasted_cash_flow", "sum"),
        avg_monthly     = ("forecasted_cash_flow", "mean"),
        min_monthly     = ("forecasted_cash_flow", "min"),
        max_monthly     = ("forecasted_cash_flow", "max")
    ).reset_index()

    summary = summary.sort_values(
        "total_forecast", ascending=False
    )

    # Merge in model metrics
    summary = summary.merge(
        metrics_df[["segment_id","sarimax_mape","hw_mape"]],
        on="segment_id"
    )

    print(f"\n  6-month forward forecast by segment:\n")
    print(f"  {'Segment':<22} {'6M Total':>14} "
          f"{'Avg/Month':>12} {'MAPE':>8} {'Status':>12}")
    print(f"  {'─'*21} {'─'*14} "
          f"{'─'*12} {'─'*8} {'─'*12}")

    total_portfolio = 0
    surplus_count   = 0
    deficit_count   = 0

    for _, row in summary.iterrows():
        total    = row["total_forecast"]
        avg      = row["avg_monthly"]
        mape     = row["sarimax_mape"]
        status   = "✓ Surplus" if total > 0 else "⚠ Deficit"
        total_portfolio += total

        if total > 0:
            surplus_count += 1
        else:
            deficit_count += 1

        print(f"  {row['segment_id']:<22} "
              f"{format_currency(total):>14} "
              f"{format_currency(avg):>12} "
              f"{mape:>7.3f}% "
              f"{status:>12}")

    print(f"\n  {'─'*60}")
    print(f"  Portfolio total (6M):     {format_currency(total_portfolio)}")
    print(f"  Segments in surplus:      {surplus_count}")
    print(f"  Segments in deficit:      {deficit_count}")

    # Rank segments by risk
    worst_segments = summary.nsmallest(3, "total_forecast")
    print(f"\n  Bottom 3 segments by forecast cash flow:")
    for _, row in worst_segments.iterrows():
        print(f"    {row['segment_id']:<25} "
              f"{format_currency(row['total_forecast'])}")

    best_segments = summary.nlargest(3, "total_forecast")
    print(f"\n  Top 3 segments by forecast cash flow:")
    for _, row in best_segments.iterrows():
        print(f"    {row['segment_id']:<25} "
              f"{format_currency(row['total_forecast'])}")

    print_divider()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 60)
    print("  BANKING RISK ANALYTICS — SARIMAX FORECAST QUERY")
    print("  Portfolio Cash Flow Forecasting Engine v1.0")
    print("═" * 60)
    print(f"  Loaded: {len(forecasts_df):,} forecast rows")
    print(f"  Segments: {len(AVAILABLE_SEGMENTS)}")

    while True:
        print("\n  Choose a query:")
        print("  [1] Segment forecast — 6-month forward view")
        print("  [2] Model fit quality — historical accuracy")
        print("  [3] What-if: transaction velocity shock")
        print("  [4] What-if: disbursement surge scenario")
        print("  [5] Portfolio summary — all segments")
        print("  [Q] Quit")

        choice = input("\n  Your choice: ").strip().upper()

        if choice == "1":
            query_segment_forecast()
        elif choice == "2":
            query_fit_quality()
        elif choice == "3":
            query_velocity_shock()
        elif choice == "4":
            query_disbursement_surge()
        elif choice == "5":
            query_portfolio_summary()
        elif choice == "Q":
            print("\n  Exiting forecast query tool.\n")
            break
        else:
            print("  Invalid choice. Enter 1, 2, 3, 4, 5, or Q.")

        cont = input(
            "\n  Run another query? (Y/N): "
        ).strip().upper()
        if cont != "Y":
            print("\n  Exiting forecast query tool.\n")
            break


if __name__ == "__main__":
    main()