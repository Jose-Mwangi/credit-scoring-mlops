import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score,
    precision_score, recall_score,
    classification_report
)
import shap
import matplotlib.pyplot as plt
import os

from features import load_data, engineer_features, split_features_target

def train():
    # ── Load and prepare data ──────────────────────────────
    df = load_data()
    df = engineer_features(df)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale = neg / pos  # handles class imbalance

    # ── MLflow setup ───────────────────────────────────────
    mlflow.set_experiment("agri-credit-scoring")

    # ── Run 1: XGBoost ─────────────────────────────────────
    with mlflow.start_run(run_name="xgboost-baseline"):

        params = {
            "n_estimators":     300,
            "max_depth":        5,
            "learning_rate":    0.05,
            "scale_pos_weight": round(scale, 2),
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "random_state":     42,
        }

        model = xgb.XGBClassifier(**params, eval_metric="auc")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "f1":        round(f1_score(y_test, y_pred), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall":    round(recall_score(y_test, y_pred), 4),
        }

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # SHAP feature importance plot
        os.makedirs("outputs", exist_ok=True)
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        plt.figure()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig("outputs/shap_xgboost.png")
        plt.close()
        mlflow.log_artifact("outputs/shap_xgboost.png")

        print("\n=== XGBoost Results ===")
        for k, v in metrics.items():
            print(f"  {k:12}: {v}")
        print(classification_report(y_test, y_pred))

    # ── Run 2: LightGBM ────────────────────────────────────
    with mlflow.start_run(run_name="lightgbm-baseline"):

        lgb_params = {
            "n_estimators":  300,
            "max_depth":     5,
            "learning_rate": 0.05,
            "scale_pos_weight": round(scale, 2),
            "subsample":     0.8,
            "random_state":  42,
        }

        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train, y_train)

        y_pred_lgb  = lgb_model.predict(X_test)
        y_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]

        lgb_metrics = {
            "f1":        round(f1_score(y_test, y_pred_lgb), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_proba_lgb), 4),
            "precision": round(precision_score(y_test, y_pred_lgb), 4),
            "recall":    round(recall_score(y_test, y_pred_lgb), 4),
        }

        mlflow.log_params(lgb_params)
        mlflow.log_metrics(lgb_metrics)
        mlflow.sklearn.log_model(lgb_model, artifact_path="model")

        print("\n=== LightGBM Results ===")
        for k, v in lgb_metrics.items():
            print(f"  {k:12}: {v}")
        print(classification_report(y_test, y_pred_lgb))

    print("\n=== MLflow Tracking ===")
    print("Run: mlflow ui --port 5000  to view all experiments")
    print("Both runs logged to experiment: agri-credit-scoring")

if __name__ == "__main__":
    train()