import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay,
    f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
import os

from features import load_data, engineer_features, split_features_target

def evaluate():
    # ── Load data ──────────────────────────────────────────
    df    = load_data()
    df    = engineer_features(df)
    X, y  = split_features_target(df)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs("outputs", exist_ok=True)

    # ── Load best model from MLflow ────────────────────────
    client   = mlflow.tracking.MlflowClient()
    exp      = client.get_experiment_by_name("agri-credit-scoring")
    runs     = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.roc_auc DESC"]
    )

    best_run = runs[0]
    run_id   = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    print(f"Best run       : {best_run.data.tags.get('mlflow.runName')}")
    print(f"Best ROC-AUC   : {best_run.data.metrics['roc_auc']}")
    print(f"Best F1        : {best_run.data.metrics['f1']}")
    print(f"Model URI      : {model_uri}")

    model   = mlflow.sklearn.load_model(model_uri)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ── Confusion matrix ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["No Default", "Default"],
        ax=axes[0], colorbar=False
    )
    axes[0].set_title("Confusion Matrix")

    # ── ROC curve ──────────────────────────────────────────
    RocCurveDisplay.from_predictions(
        y_test, y_proba, ax=axes[1], name="Best Model"
    )
    axes[1].set_title("ROC Curve")

    plt.tight_layout()
    plt.savefig("outputs/evaluation_plots.png", dpi=150)
    plt.close()
    print("Saved: outputs/evaluation_plots.png")

    # ── Log evaluation artifacts back to same run ──────────
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact("outputs/evaluation_plots.png")
    print("Evaluation plots logged to MLflow run")

    # ── Register best model in Model Registry ──────────────
    reg = mlflow.register_model(model_uri, "agri-credit-scorer")
    print(f"\nModel registered: agri-credit-scorer v{reg.version}")
    print("Next step: promote to Staging then Production in MLflow UI")

if __name__ == "__main__":
    evaluate()