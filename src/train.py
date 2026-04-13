"""
train.py  —  Telco Customer Churn
===================================
Trains Logistic Regression, Random Forest, XGBoost, LightGBM.
Logs all experiments to MLflow. Optional Optuna tuning.

Usage:
    python src/train.py
    python src/train.py --tune --n-trials 40
"""

import argparse, json, warnings, joblib
import mlflow, mlflow.sklearn, mlflow.xgboost, mlflow.lightgbm
import numpy as np, pandas as pd, optuna
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, classification_report,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
RANDOM_STATE  = 42
EXP_NAME      = "telco_churn_prediction"


def load_data():
    X_train    = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test     = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    X_train_sc = pd.read_parquet(PROCESSED_DIR / "X_train_scaled.parquet")
    X_test_sc  = pd.read_parquet(PROCESSED_DIR / "X_test_scaled.parquet")
    y_train    = pd.read_parquet(PROCESSED_DIR / "y_train.parquet").squeeze()
    y_test     = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").squeeze()
    return X_train, X_test, X_train_sc, X_test_sc, y_train, y_test


def metrics(y_true, y_proba, thresh=0.5):
    yp = (y_proba >= thresh).astype(int)
    return {
        "auc_roc":   round(roc_auc_score(y_true, y_proba), 4),
        "avg_prec":  round(average_precision_score(y_true, y_proba), 4),
        "f1":        round(f1_score(y_true, yp, zero_division=0), 4),
        "precision": round(precision_score(y_true, yp, zero_division=0), 4),
        "recall":    round(recall_score(y_true, yp, zero_division=0), 4),
    }


def train_log(name, model, Xtr, Xte, ytr, yte):
    print(f"  Training {name}...")
    with mlflow.start_run(run_name=name):
        mlflow.log_params(model.get_params())
        model.fit(Xtr, ytr)
        yp = model.predict_proba(Xte)[:, 1]
        m  = metrics(yte, yp)
        mlflow.log_metrics(m)
        if "xgboost" in name.lower():
            mlflow.xgboost.log_model(model, "model")
        elif "lightgbm" in name.lower():
            mlflow.lightgbm.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
        print(f"    AUC={m['auc_roc']} F1={m['f1']} Prec={m['precision']} Rec={m['recall']}")
    return model, m


def tune_xgboost(Xtr, Xte, ytr, yte, n_trials=40):
    print(f"\n  Optuna tuning XGBoost ({n_trials} trials)...")

    def objective(trial):
        p = dict(
            n_estimators    = trial.suggest_int("n_estimators", 100, 500),
            max_depth       = trial.suggest_int("max_depth", 3, 8),
            learning_rate   = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample       = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree= trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha       = trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            reg_lambda      = trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            scale_pos_weight= trial.suggest_float("scale_pos_weight", 1, 5),
        )
        m = XGBClassifier(**p, random_state=RANDOM_STATE, verbosity=0, use_label_encoder=False, eval_metric="auc")
        m.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Best AUC: {study.best_value:.4f}")

    best = XGBClassifier(**study.best_params, random_state=RANDOM_STATE, verbosity=0, use_label_encoder=False, eval_metric="auc")
    best.fit(Xtr, ytr)

    with mlflow.start_run(run_name="xgboost_optuna_tuned"):
        mlflow.log_params(study.best_params)
        mlflow.log_metrics(metrics(yte, best.predict_proba(Xte)[:, 1]))
        mlflow.xgboost.log_model(best, "model")

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune",     action="store_true")
    parser.add_argument("--n-trials", type=int, default=40)
    args = parser.parse_args()

    print("=" * 55)
    print("Telco Churn — Model Training")
    print("=" * 55)

    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test = load_data()
    mlflow.set_experiment(EXP_NAME)

    base_models = {
        "logistic_regression": (LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"), "scaled"),
        "random_forest":       (RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1), "raw"),
        "xgboost":             (XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=3, random_state=RANDOM_STATE, verbosity=0, use_label_encoder=False, eval_metric="auc"), "raw"),
        "lightgbm":            (LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, class_weight="balanced", random_state=RANDOM_STATE, verbose=-1), "raw"),
    }

    all_metrics, trained = {}, {}
    for name, (model, fs) in base_models.items():
        Xtr = X_train_sc if fs == "scaled" else X_train
        Xte = X_test_sc  if fs == "scaled" else X_test
        m, met = train_log(name, model, Xtr, Xte, y_train, y_test)
        all_metrics[name] = met
        trained[name]     = m

    print("\n" + "=" * 55)
    print("RESULTS")
    print("=" * 55)
    results = pd.DataFrame(all_metrics).T.sort_values("auc_roc", ascending=False)
    print(results.to_string())

    best_model = trained["xgboost"]
    if args.tune:
        best_model = tune_xgboost(X_train, X_test, y_train, y_test, n_trials=args.n_trials)

    joblib.dump(best_model, MODELS_DIR / "xgboost_best.pkl")
    print(f"\n✓ Best model saved → {MODELS_DIR}/xgboost_best.pkl")
    print("  Next: python src/shap_analysis.py")
