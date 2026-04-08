"""
FULL PIPELINE RUNNER
Runs all stages end to end in order.
For individual stage runs use:
  python src/data_loader.py
  python src/feature_engineering.py
  python src/model_xgb.py
  python src/model_sarimax.py
"""

import os
import subprocess
import sys

stages = [
    ("Stage 1 — Data Loading",         "src/data_loader.py"),
    ("Stage 2 — Feature Engineering",   "src/feature_engineering.py"),
    ("Stage 3a — XGBoost Classifier",   "src/model_xgb.py"),
    ("Stage 3b — SARIMAX Forecasting",  "src/model_sarimax.py"),
]

for name, script in stages:
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, script],
        check=True
    )

print("\n" + "=" * 60)
print("PIPELINE COMPLETE — all outputs in data/outputs/")
print("=" * 60)