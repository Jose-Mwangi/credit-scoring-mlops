import pandas as pd
import numpy as np
import os
import json
import mlflow.sklearn
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.metrics import DriftedColumnsCount, ValueDrift
from features import load_data, engineer_features, split_features_target

def run_monitoring():
    os.makedirs("outputs/monitoring", exist_ok=True)
    df = load_data()
    df = engineer_features(df)
    split = int(len(df) * 0.7)
    reference_df = df.iloc[:split].copy()
    current_df = df.iloc[split:].copy()
    print(f"Reference data : {reference_df.shape}")
    print(f"Current data   : {current_df.shape}")
    model = mlflow.sklearn.load_model("models:/agri-credit-scorer/1")
    reference_df["prediction"] = model.predict(reference_df.drop(columns=["default"]))
    current_df["prediction"] = model.predict(current_df.drop(columns=["default"]))
    report = Report(metrics=[
        DriftedColumnsCount(),
        ValueDrift(column="monthly_income_ksh"),
        ValueDrift(column="loan_amount_requested"),
        ValueDrift(column="agri_risk_score"),
        DataDriftPreset(),
    ])
    report.run(
        reference_data=reference_df.drop(columns=["default", "prediction"]),
        current_data=current_df.drop(columns=["default", "prediction"]),
    )
    results = []
    for m in report.metrics:
        metric_name = type(m).__name__
        metric_json = json.loads(m.json())
        results.append({"metric": metric_name, "data": metric_json})
        print(f"  {metric_name}: {metric_json}")
    with open("outputs/monitoring/drift_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== Drift Monitoring Summary ===")
    print(f"  {len(results)} metrics computed")
    print(f"  Saved: outputs/monitoring/drift_report.json")

if __name__ == "__main__":
    run_monitoring()
