# Agri Credit Scoring MLOps Pipeline

An end-to-end production ML pipeline for predicting loan default risk among
smallholder farmers, built with a full MLOps stack.

Inspired by real-world agri-fintech credit assessment work at MkulimaScore.

---

## Architecture
```
Data Generation → Feature Engineering → Model Training → Model Registry
      ↓                                       ↓                ↓
  Prefect Pipeline              MLflow Experiment Tracking   FastAPI REST API
                                                                    ↓
                                                          Evidently AI Monitoring
```

## Results

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| XGBoost | 0.9365 | 0.6621 | 0.5647 | 0.80 |
| LightGBM | 0.9344 | 0.6612 | 0.5602 | 0.8067 |

**XGBoost selected as production model** based on ROC-AUC.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data generation | Python, NumPy, Pandas |
| Feature engineering | Scikit-learn, domain features (agri-risk score, debt-to-income) |
| Model training | XGBoost, LightGBM, SHAP explainability |
| Experiment tracking | MLflow (params, metrics, artifacts, model registry) |
| Pipeline orchestration | Prefect (4-step flow: ingest → features → train → evaluate) |
| REST API | FastAPI + Uvicorn |
| Containerisation | Docker |
| Drift monitoring | Evidently AI (5 metrics: dataset drift, feature drift) |

---

## Project Structure
```
credit-scoring-mlops/
├── app/
│   └── main.py              # FastAPI REST API
├── src/
│   ├── ingest.py            # Synthetic agri dataset generation
│   ├── features.py          # Feature engineering pipeline
│   ├── train.py             # XGBoost + LightGBM training with MLflow
│   ├── evaluate.py          # Model evaluation + registry
│   ├── pipeline.py          # Prefect orchestration pipeline
│   └── monitor.py           # Evidently AI drift monitoring
├── data/
│   ├── raw/                 # Raw generated dataset (10,000 rows)
│   └── processed/           # Engineered features (25 features)
├── outputs/
│   ├── shap_xgboost.png     # SHAP feature importance plot
│   ├── evaluation_plots.png # Confusion matrix + ROC curve
│   └── monitoring/          # Drift monitoring reports
├── Dockerfile
└── requirements.txt
```

---

## Dataset

Synthetic agri-fintech dataset of 10,000 loan applications with features
designed from domain knowledge of smallholder farmer credit assessment:

**Financial features:** monthly income, expenses, existing loans, loan amount,
repayment history (30/90 day late payments)

**Agri-specific features:** crop type, irrigation access, crop insurance,
SACCO membership, distance to market, rainfall reliability, soil quality score

**Engineered features:** debt-to-income ratio, loan-to-income ratio,
agri-risk score, income per dependent

**Target:** `default` — binary (15% default rate, realistic for agri lending)

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API info and available endpoints |
| `/health` | GET | Health check + model status |
| `/predict` | POST | Score a loan application |
| `/model-info` | GET | Live model metrics from MLflow |
| `/docs` | GET | Interactive Swagger UI |

### Sample Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Sample Response
```json
{
  "default_prediction": 0,
  "default_probability": 0.0001,
  "risk_tier": "LOW",
  "recommendation": "APPROVE",
  "input_summary": {
    "farmer_age": 35,
    "loan_amount_ksh": 50000,
    "monthly_income_ksh": 18000,
    "crop_type": "maize",
    "sacco_member": true,
    "crop_insurance": true
  }
}
```

---

## Running Locally
```bash
# 1. Clone
git clone https://github.com/Jose-Mwangi/credit-scoring-mlops
cd credit-scoring-mlops

# 2. Install
pip install -r requirements.txt

# 3. Generate data + train
python src/ingest.py
python src/features.py
python src/train.py
python src/evaluate.py

# 4. Or run full pipeline
python src/pipeline.py

# 5. View experiments
mlflow ui --port 5000

# 6. Start API
uvicorn app.main:app --reload --port 8000

# 7. Run drift monitoring
python src/monitor.py
```

## Docker
```bash
docker build -t agri-credit-api .
docker run -p 8000:8000 agri-credit-api
```

---

## Key Design Decisions

**Synthetic data:** Financial data from smallholder farmers is sensitive and
rarely public. Generating synthetic data from domain knowledge is standard
practice in fintech ML — it demonstrates feature design skills rather than
hiding them.

**XGBoost over LightGBM:** Marginally higher ROC-AUC (0.9365 vs 0.9344).
Both models were tracked in MLflow and the registry promotes the best run
automatically via `evaluate.py`.

**SHAP explainability:** Critical for credit decisions — regulators and loan
officers need to understand why a farmer was declined. SHAP plots are logged
as MLflow artifacts on every training run.

**Agri-risk score:** A composite feature combining irrigation access, crop
insurance, SACCO membership, and rainfall reliability — designed from
MkulimaScore domain knowledge. It's the top driver of default risk in the
SHAP plot.