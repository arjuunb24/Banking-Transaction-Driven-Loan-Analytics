"""
STAGE 3b — SARIMAX PORTFOLIO CASH FLOW FORECASTING
Run independently: python src/model_sarimax.py
Inputs:  data/processed/master_features.parquet
Outputs: data/outputs/sarimax_forecasts.parquet
         data/outputs/sarimax_metrics.csv
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("statsmodels").setLevel(logging.ERROR)

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.config import (
    PROCESSED_DIR, OUTPUTS_DIR,
    SARIMAX_M, FORECAST_HORIZON
)


def build_segment_cashflow(master: pd.DataFrame) -> pd.DataFrame:
    print("Building segment cash flow series...")

    df = master.copy()

    df["grade_bucket"] = df["grade"].map({
        "A": "Grade_AB", "B": "Grade_AB",
        "C": "Grade_CD", "D": "Grade_CD",
        "E": "Grade_EG", "F": "Grade_EG", "G": "Grade_EG"
    }).fillna("Grade_EG")

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

    df["region"]     = df["addr_state"].apply(map_region)
    df["segment_id"] = df["grade_bucket"] + "_" + df["region"]
    df["month_key"]  = pd.to_datetime(
        df["issue_date"]
    ).dt.to_period("M")

    seg = df.groupby(["segment_id", "month_key"]).agg(
        total_repayments    = ("installment",  "sum"),
        total_disbursements = ("loan_amnt",    "sum"),
        avg_txn_velocity    = ("txn_count",    "mean"),
        loan_count          = ("loan_amnt",    "count")
    ).reset_index()

    seg["net_cash_flow"] = (
        seg["total_repayments"] - seg["total_disbursements"]
    )
    seg["month_key"] = seg["month_key"].dt.to_timestamp()
    seg = seg.sort_values(["segment_id", "month_key"])

    print(f"  Segments: {seg['segment_id'].nunique()}")
    print(f"  Rows: {len(seg):,}")
    return seg


def prepare_series(segment_df: pd.DataFrame):
    """Build regular monthly series with no gaps."""
    df = segment_df.copy()
    df["month_key"] = pd.to_datetime(df["month_key"])
    df = df.set_index("month_key").sort_index()

    full_idx = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="MS"
    )
    df = df.reindex(full_idx)
    df["net_cash_flow"]       = df["net_cash_flow"].fillna(0)
    df["avg_txn_velocity"]    = df["avg_txn_velocity"].ffill().fillna(0)
    df["total_disbursements"] = df["total_disbursements"].ffill().fillna(0)
    df.index.freq = "MS"

    series = df["net_cash_flow"]
    exog   = df[["avg_txn_velocity", "total_disbursements"]]
    return series, exog


def select_order(series, exog):
    candidates = [
        ((1,1,1), (1,1,0,12)),
        ((1,1,1), (0,1,1,12)),
        ((1,1,1), (1,1,1,12)),
        ((0,1,1), (0,1,1,12)),
        ((1,1,0), (1,1,0,12)),
        ((2,1,1), (1,1,0,12)),
        ((1,1,2), (0,1,1,12)),
    ]
    best_aic, best_order, best_seasonal = np.inf, (1,1,1), (1,1,0,12)

    for order, seasonal in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(
                    series, exog=exog,
                    order=order, seasonal_order=seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False, maxiter=200, method="lbfgs")

                if np.isfinite(res.aic) and res.aic < best_aic:
                    best_aic      = res.aic
                    best_order    = order
                    best_seasonal = seasonal
        except Exception:
            continue

    return best_order, best_seasonal, best_aic


def walk_forward(series, exog, order, seasonal, min_train=36):
    """True out-of-sample walk-forward validation."""
    n = len(series)
    actuals, forecasts = [], []

    for t in range(min_train, n):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(
                    series.iloc[:t], exog=exog.iloc[:t],
                    order=order, seasonal_order=seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False, maxiter=200, method="lbfgs")

                fc = res.forecast(
                    steps=1, exog=exog.iloc[[t]]
                ).values[0]

                if np.isfinite(fc):
                    actuals.append(series.iloc[t])
                    forecasts.append(fc)
        except Exception:
            continue

    if len(actuals) < 5:
        return {"mape": np.nan, "mae": np.nan,
                "bias": np.nan, "n_steps": len(actuals)}

    a = np.array(actuals)
    f = np.array(forecasts)

    raw_pct = np.abs((a - f) / (np.abs(a) + 1e-9)) * 100
    mape    = float(np.mean(
        np.clip(raw_pct, 0, np.percentile(raw_pct, 95))
    ))
    mae     = float(np.mean(np.abs(a - f)))
    bias    = float(np.mean(f - a))

    return {"mape": mape, "mae": mae,
            "bias": bias, "n_steps": len(actuals)}


def holtwinters_benchmark(series):
    """Holt-Winters walk-forward benchmark."""
    n         = len(series)
    min_train = min(36, int(n * 0.7))
    actuals, forecasts = [], []

    for t in range(min_train, n):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ExponentialSmoothing(
                    series.iloc[:t],
                    trend="add", seasonal="add",
                    seasonal_periods=12
                ).fit(optimized=True)

                fc = res.forecast(1).values[0]
                if np.isfinite(fc):
                    actuals.append(series.iloc[t])
                    forecasts.append(fc)
        except Exception:
            continue

    if len(actuals) < 5:
        return {"hw_mape": np.nan, "hw_mae": np.nan}

    a = np.array(actuals)
    f = np.array(forecasts)
    raw_pct = np.abs((a - f) / (np.abs(a) + 1e-9)) * 100

    return {
        "hw_mape": float(np.mean(
            np.clip(raw_pct, 0, np.percentile(raw_pct, 95))
        )),
        "hw_mae": float(np.mean(np.abs(a - f)))
    }


def fit_and_forecast(segment_df, segment_id):
    print(f"  [{segment_id}]")

    series, exog = prepare_series(segment_df)

    if len(series) < 36:
        print(f"    Skipping — {len(series)} months only")
        return pd.DataFrame()

    order, seasonal, aic = select_order(series, exog)
    val = walk_forward(series, exog, order, seasonal)
    hw  = holtwinters_benchmark(series)

    mape_s = f"{val['mape']:.2f}%" if np.isfinite(val['mape']) else "NaN"
    mape_h = f"{hw['hw_mape']:.2f}%" if np.isfinite(hw['hw_mape']) else "NaN"
    lift   = (hw['hw_mape'] - val['mape']
              if np.isfinite(val['mape']) and np.isfinite(hw['hw_mape'])
              else np.nan)

    print(f"    Order: {order}x{seasonal}  AIC: {aic:.0f}  "
          f"SARIMAX: {mape_s}  HW: {mape_h}  "
          f"Lift: {lift:.1f}pp" if np.isfinite(lift)
          else f"    Order: {order}x{seasonal}  AIC: {aic:.0f}  "
               f"SARIMAX: {mape_s}  HW: {mape_h}")

    # Final model on full history
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final = SARIMAX(
            series, exog=exog,
            order=order, seasonal_order=seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False, maxiter=300, method="lbfgs")

    # 6-month forward forecast
    last_exog   = exog.iloc[-1:].values
    future_exog = pd.DataFrame(
        np.tile(last_exog, (FORECAST_HORIZON, 1)),
        columns=exog.columns
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc_res = final.get_forecast(
            steps=FORECAST_HORIZON, exog=future_exog
        )

    fc_mean = fc_res.predicted_mean
    fc_ci   = fc_res.conf_int(alpha=0.20).clip(-1e12, 1e12)

    future_idx = pd.date_range(
        start   = series.index[-1] + pd.DateOffset(months=1),
        periods = FORECAST_HORIZON,
        freq    = "MS"
    )

    shared = {
        "segment_id":          segment_id,
        "sarimax_mape":        val["mape"],
        "sarimax_mae":         val["mae"],
        "sarimax_bias":        val["bias"],
        "hw_mape":             hw["hw_mape"],
        "sarimax_lift_pp":     lift,
        "best_order":          str(order),
        "best_seasonal_order": str(seasonal),
        "n_validation_steps":  val["n_steps"]
    }

    hist = pd.DataFrame({
        **shared,
        "month_key":           series.index,
        "actual_cash_flow":    series.values,
        "forecasted_cash_flow":final.fittedvalues.values,
        "upper_bound":         np.nan,
        "lower_bound":         np.nan,
        "is_forecast":         False
    })

    fcast = pd.DataFrame({
        **shared,
        "month_key":           future_idx,
        "actual_cash_flow":    np.nan,
        "forecasted_cash_flow":fc_mean.values,
        "upper_bound":         fc_ci.iloc[:, 1].values,
        "lower_bound":         fc_ci.iloc[:, 0].values,
        "is_forecast":         True
    })

    return pd.concat([hist, fcast], ignore_index=True)


if __name__ == "__main__":
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("=" * 55)
    print("SARIMAX PORTFOLIO CASH FLOW FORECASTING")
    print("=" * 55)

    master = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "master_features.parquet")
    )
    print(f"Master table loaded: {master.shape}")

    segment_cf  = build_segment_cashflow(master)
    all_results = []

    for seg_id in segment_cf["segment_id"].unique():
        seg_df = segment_cf[
            segment_cf["segment_id"] == seg_id
        ].reset_index(drop=True)
        result = fit_and_forecast(seg_df, seg_id)
        if not result.empty:
            all_results.append(result)

    forecasts_df = pd.concat(all_results, ignore_index=True)

    forecasts_df.to_parquet(
        os.path.join(OUTPUTS_DIR, "sarimax_forecasts.parquet"), index=False
    )

    summary = forecasts_df.drop_duplicates("segment_id")[[
        "segment_id", "sarimax_mape", "hw_mape",
        "sarimax_lift_pp", "best_order",
        "best_seasonal_order", "n_validation_steps"
    ]]
    summary.to_csv(
        os.path.join(OUTPUTS_DIR, "sarimax_metrics.csv"), index=False
    )

    print("\nSARIMAX metrics summary:")
    print(summary.to_string(index=False))
    print(f"\nForecasts shape: {forecasts_df.shape}")
    print("\nAll outputs saved to data/outputs/")
    print("  sarimax_forecasts.parquet")
    print("  sarimax_metrics.csv")
    print("\nStage 3b complete. Next: SQL layer")