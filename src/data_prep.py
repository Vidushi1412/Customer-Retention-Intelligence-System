"""
data_prep.py  —  Telco Customer Churn
======================================
Loads the IBM Telco Customer Churn dataset, engineers features,
handles class imbalance with SMOTE, and saves train/test splits.

Usage:
    python src/data_prep.py
"""

import json, os, zipfile, joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TELCO_FILE = RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


# ─────────────────────────────────────────────────────────────────────────────
def download_dataset():
    if TELCO_FILE.exists():
        print(f"  Dataset already at {TELCO_FILE}")
        return
    print("Downloading Telco Churn dataset from Kaggle...")
    os.system(
        "kaggle datasets download -d blastchar/telco-customer-churn "
        f"-p {RAW_DIR} --unzip --quiet"
    )
    if not TELCO_FILE.exists():
        # Try alternate filename
        for f in RAW_DIR.glob("*.csv"):
            f.rename(TELCO_FILE)
            break
    if not TELCO_FILE.exists():
        raise FileNotFoundError(
            "Dataset not found. Download manually from:\n"
            "https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            f"and place the CSV in {RAW_DIR}/"
        )


# ─────────────────────────────────────────────────────────────────────────────
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(TELCO_FILE)
    print(f"Raw shape: {df.shape}")
    # TotalCharges has spaces for some new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    print(f"Churn rate: {(df['Churn']=='Yes').mean():.1%}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["customerID"])
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    # Binary yes/no → 0/1
    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling"
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = (df[col] == "Yes").astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-informed features for telecom churn prediction.
    Each feature addresses a known driver in telecom customer behaviour.
    """
    df = df.copy()

    # ── Tenure group (strong non-linear effect) ───────────────────────────────
    # New customers are most vulnerable; retention improves sharply after 12m.
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 3, 12, 36, 999],
        labels=[0, 1, 2, 3],   # new / early / established / loyal
    ).astype(int)

    # ── Monthly-to-total charges ratio ────────────────────────────────────────
    # High ratio = new customer or recently upgraded — elevated risk.
    df["monthly_to_total_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    # ── Service count ─────────────────────────────────────────────────────────
    # Each additional service increases switching cost — reduces churn.
    service_cols = [
        "PhoneService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    existing_services = [c for c in service_cols if c in df.columns]
    # Convert Yes/No to binary first for service cols
    for col in existing_services:
        if df[col].dtype == object:
            df[col] = (df[col] == "Yes").astype(int)
    df["service_count"] = df[existing_services].sum(axis=1)

    # ── Contract risk score ───────────────────────────────────────────────────
    contract_map = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    df["contract_risk_score"] = df["Contract"].map(contract_map)

    # ── Support gap ───────────────────────────────────────────────────────────
    # Internet without tech support = most common high-churn combo.
    has_internet = df["InternetService"].isin(["DSL", "Fiber optic"])
    no_support   = df.get("TechSupport", pd.Series("No", index=df.index)) != "Yes"
    df["support_gap"] = (has_internet & no_support).astype(int)

    # ── Fiber optic flag (highest churn internet type) ────────────────────────
    df["fiber_optic"] = (df["InternetService"] == "Fiber optic").astype(int)

    # ── Digital payment risk ──────────────────────────────────────────────────
    elec_check    = df["PaymentMethod"] == "Electronic check"
    paperless     = df["PaperlessBilling"] == 1
    df["digital_payment_risk"] = (elec_check & paperless).astype(int)

    # ── Charge per service ────────────────────────────────────────────────────
    df["charge_per_service"] = df["MonthlyCharges"] / (df["service_count"] + 1)

    # ── Senior + no support ───────────────────────────────────────────────────
    df["senior_no_support"] = (
        (df.get("SeniorCitizen", 0) == 1) & (df["support_gap"] == 1)
    ).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode remaining object columns."""
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
def split_and_balance(df, target="Churn", test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Train churn: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")

    print("Applying SMOTE...")
    smote = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — 0: {(y_train_bal==0).sum():,} | 1: {(y_train_bal==1).sum():,}")
    return X_train_bal, X_test, y_train_bal, y_test, X_train.columns.tolist()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("Telco Churn — Data Preparation")
    print("=" * 55)

    download_dataset()

    print("\nLoading raw data...")
    df = load_raw()

    print("\nCleaning...")
    df = clean(df)

    print("\nEngineering features...")
    df = engineer_features(df)

    print("\nEncoding categoricals...")
    df = encode_categoricals(df)

    print(f"\nFinal features: {df.shape[1] - 1}")

    print("\nSplitting + balancing...")
    X_train, X_test, y_train, y_test, feature_names = split_and_balance(df)

    # Scale for Logistic Regression
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),  columns=X_test.columns)

    # Save
    X_train.to_parquet(PROCESSED_DIR / "X_train.parquet", index=False)
    X_test.to_parquet(PROCESSED_DIR  / "X_test.parquet",  index=False)
    X_train_sc.to_parquet(PROCESSED_DIR / "X_train_scaled.parquet", index=False)
    X_test_sc.to_parquet(PROCESSED_DIR  / "X_test_scaled.parquet",  index=False)
    y_train.reset_index(drop=True).to_frame().to_parquet(PROCESSED_DIR / "y_train.parquet", index=False)
    y_test.reset_index(drop=True).to_frame().to_parquet(PROCESSED_DIR  / "y_test.parquet",  index=False)

    # Also save clean df for dashboard
    df.to_parquet(PROCESSED_DIR / "clean_full.parquet", index=False)

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"\n✓ Data preparation complete.")
    print(f"  Saved to {PROCESSED_DIR}/")
    print("  Next: python src/train.py")
