import mlflow
import mlflow.sklearn
import mlflow.tracking
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── App setup ──────────────────────────────────────────────
app = FastAPI(
    title="Agri Credit Scoring API",
    description="Predicts loan default risk for smallholder farmers — MkulimaScore pipeline",
    version="1.0.0"
)

# ── Load model once at startup ─────────────────────────────
try:
    model = mlflow.sklearn.load_model("models:/agri-credit-scorer/1")
    print("Model loaded: agri-credit-scorer v1")
except Exception as e:
    print(f"Model load failed: {e}")
    model = None

# ── Request schema ─────────────────────────────────────────
class LoanApplication(BaseModel):
    age:                     int
    farm_size_acres:         float
    years_farming:           int
    num_dependents:          int
    monthly_income_ksh:      float
    monthly_expenses_ksh:    float
    existing_loans:          int
    loan_amount_requested:   float
    repayment_period_months: int
    times_30_days_late:      int
    times_90_days_late:      int
    crop_type:               str    # maize, tea, coffee, horticulture, dairy
    irrigation:              int    # 0 or 1
    crop_insurance:          int    # 0 or 1
    mobile_money_user:       int    # 0 or 1
    sacco_member:            int    # 0 or 1
    distance_to_market_km:   float
    rainfall_reliability:    str    # low, medium, high
    soil_quality_score:      int    # 1-10

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "farm_size_acres": 2.5,
                "years_farming": 10,
                "num_dependents": 3,
                "monthly_income_ksh": 18000,
                "monthly_expenses_ksh": 9000,
                "existing_loans": 1,
                "loan_amount_requested": 50000,
                "repayment_period_months": 12,
                "times_30_days_late": 0,
                "times_90_days_late": 0,
                "crop_type": "maize",
                "irrigation": 1,
                "crop_insurance": 1,
                "mobile_money_user": 1,
                "sacco_member": 1,
                "distance_to_market_km": 5.0,
                "rainfall_reliability": "medium",
                "soil_quality_score": 7
            }
        }

# ── Feature engineering (mirrors src/features.py) ──────────
def prepare_features(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Derived financial features
    df["debt_to_income"]       = df["monthly_expenses_ksh"] / (df["monthly_income_ksh"] + 1)
    df["loan_to_income"]       = df["loan_amount_requested"] / (df["monthly_income_ksh"] + 1)
    df["total_late_payments"]  = df["times_30_days_late"] + df["times_90_days_late"]
    df["income_per_dependent"] = df["monthly_income_ksh"] / (df["num_dependents"] + 1)

    # Agri risk score
    df["agri_risk_score"] = (
        df["irrigation"].map({1: 0, 0: 1}) +
        df["crop_insurance"].map({1: 0, 0: 1}) +
        df["sacco_member"].map({1: 0, 0: 1}) +
        df["rainfall_reliability"].map({"high": 0, "medium": 1, "low": 2})
    )

    # Encode categoricals — must match training encoding
    crop_map = {"coffee": 0, "dairy": 1, "horticulture": 2, "maize": 3, "tea": 4}
    rain_map = {"high": 0, "low": 1, "medium": 2}
    df["crop_type_enc"]            = df["crop_type"].map(crop_map).fillna(0).astype(int)
    df["rainfall_reliability_enc"] = df["rainfall_reliability"].map(rain_map).fillna(0).astype(int)

    # Drop original categoricals
    df = df.drop(columns=["crop_type", "rainfall_reliability"])

    return df

# ── Routes ──────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":        "Agri Credit Scoring API",
        "version":     "1.0.0",
        "status":      "running",
        "description": "Smallholder farmer loan default prediction",
        "endpoints": {
            "health":     "/health",
            "predict":    "/predict",
            "model_info": "/model-info",
            "docs":       "/docs"
        }
    }

@app.get("/health")
def health():
    return {
        "status":       "healthy",
        "model_loaded": model is not None,
        "model":        "agri-credit-scorer v1"
    }

@app.post("/predict")
def predict(application: LoanApplication):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data     = application.dict()
    features = prepare_features(data)

    probability = model.predict_proba(features)[0][1]
    prediction  = int(model.predict(features)[0])

    # Risk tier
    if probability < 0.3:
        risk_tier      = "LOW"
        recommendation = "APPROVE"
    elif probability < 0.6:
        risk_tier      = "MEDIUM"
        recommendation = "REVIEW"
    else:
        risk_tier      = "HIGH"
        recommendation = "DECLINE"

    return {
        "default_prediction":  prediction,
        "default_probability": round(float(probability), 4),
        "risk_tier":           risk_tier,
        "recommendation":      recommendation,
        "input_summary": {
            "farmer_age":         data["age"],
            "loan_amount_ksh":    data["loan_amount_requested"],
            "monthly_income_ksh": data["monthly_income_ksh"],
            "crop_type":          data["crop_type"],
            "sacco_member":       bool(data["sacco_member"]),
            "crop_insurance":     bool(data["crop_insurance"]),
        }
    }

@app.get("/model-info")
def model_info():
    try:
        client   = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("agri-credit-scorer")
        latest   = versions[0]
        run      = client.get_run(latest.run_id)
        metrics  = run.data.metrics
        params   = run.data.params

        return {
            "model_name":    "agri-credit-scorer",
            "version":       latest.version,
            "stage":         latest.current_stage,
            "run_id":        latest.run_id,
            "framework":     "XGBoost",
            "training_data": "10,000 agri-fintech loan records",
            "metrics": {
                "roc_auc":   round(metrics.get("roc_auc", 0), 4),
                "f1":        round(metrics.get("f1", 0), 4),
                "precision": round(metrics.get("precision", 0), 4),
                "recall":    round(metrics.get("recall", 0), 4),
            },
            "params": {
                "n_estimators":  params.get("n_estimators"),
                "max_depth":     params.get("max_depth"),
                "learning_rate": params.get("learning_rate"),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch model info: {str(e)}")