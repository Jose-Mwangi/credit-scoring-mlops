import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(path="data/raw/agri_credit.csv"):
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape}")
    return df

def engineer_features(df):
    df = df.copy()

    # Derived financial features
    df["debt_to_income"]        = df["monthly_expenses_ksh"] / (df["monthly_income_ksh"] + 1)
    df["loan_to_income"]        = df["loan_amount_requested"] / (df["monthly_income_ksh"] + 1)
    df["total_late_payments"]   = df["times_30_days_late"] + df["times_90_days_late"]
    df["income_per_dependent"]  = df["monthly_income_ksh"] / (df["num_dependents"] + 1)

    # Agri risk score (domain knowledge from MkulimaScore)
    df["agri_risk_score"] = (
        df["irrigation"].map({1: 0, 0: 1}) +
        df["crop_insurance"].map({1: 0, 0: 1}) +
        df["sacco_member"].map({1: 0, 0: 1}) +
        df["rainfall_reliability"].map({"high": 0, "medium": 1, "low": 2})
    )

    # Encode categoricals
    le = LabelEncoder()
    df["crop_type_enc"]          = le.fit_transform(df["crop_type"])
    df["rainfall_reliability_enc"] = le.fit_transform(df["rainfall_reliability"])

    # Drop original categoricals
    df = df.drop(columns=["crop_type", "rainfall_reliability"])

    return df

def split_features_target(df, target="default"):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

if __name__ == "__main__":
    df = load_data()
    df = engineer_features(df)
    X, y = split_features_target(df)
    df.to_csv("data/processed/agri_credit_features.csv", index=False)
    print(f"Features shape : {X.shape}")
    print(f"Target balance : {y.value_counts(normalize=True).mul(100).round(2).to_dict()}")
    print(f"New features   : debt_to_income, loan_to_income, total_late_payments, agri_risk_score")
    print("Saved: data/processed/agri_credit_features.csv")