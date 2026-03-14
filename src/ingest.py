import pandas as pd
import numpy as np

np.random.seed(42)
n = 10000

df = pd.DataFrame({
    "age":                     np.random.randint(20, 70, n),
    "farm_size_acres":         np.round(np.random.exponential(3, n), 2),
    "years_farming":           np.random.randint(1, 40, n),
    "num_dependents":          np.random.randint(0, 8, n),
    "monthly_income_ksh":      np.round(np.random.exponential(15000, n), 0),
    "monthly_expenses_ksh":    np.round(np.random.exponential(8000, n), 0),
    "existing_loans":          np.random.randint(0, 4, n),
    "loan_amount_requested":   np.round(np.random.uniform(5000, 200000, n), 0),
    "repayment_period_months": np.random.choice([3, 6, 12, 18, 24], n),
    "times_30_days_late":      np.random.randint(0, 6, n),
    "times_90_days_late":      np.random.randint(0, 3, n),
    "crop_type":               np.random.choice(["maize","tea","coffee","horticulture","dairy"], n),
    "irrigation":              np.random.choice([0, 1], n, p=[0.65, 0.35]),
    "crop_insurance":          np.random.choice([0, 1], n, p=[0.7, 0.3]),
    "mobile_money_user":       np.random.choice([0, 1], n, p=[0.3, 0.7]),
    "sacco_member":            np.random.choice([0, 1], n, p=[0.5, 0.5]),
    "distance_to_market_km":   np.round(np.random.exponential(10, n), 1),
    "rainfall_reliability":    np.random.choice(["low","medium","high"], n),
    "soil_quality_score":      np.random.randint(1, 10, n),
})

default_score = (
    - 0.00003 * df["monthly_income_ksh"]
    + 0.4     * df["times_90_days_late"]
    + 0.2     * df["times_30_days_late"]
    + 0.00001 * df["loan_amount_requested"]
    + 0.3     * df["existing_loans"]
    - 0.3     * df["crop_insurance"]
    - 0.2     * df["sacco_member"]
    - 0.2     * df["irrigation"]
    + np.random.normal(0, 0.5, n)
)

threshold = np.percentile(default_score, 85)
df["default"] = (default_score > threshold).astype(int)

df.to_csv("data/raw/agri_credit.csv", index=False)

print("=== Dataset Generated ===")
print(f"Shape      : {df.shape}")
print(f"Default %  : {df['default'].value_counts(normalize=True).mul(100).round(2).to_dict()}")
print(f"Features   : {list(df.columns)}")
print("Saved to   : data/raw/agri_credit.csv")
