# Telco Customer Churn Prediction

> End-to-end ML pipeline predicting telecom customer attrition — featuring class imbalance handling, model explainability with SHAP, customer risk segmentation, and a Streamlit retention dashboard.

---

## Project Summary

Predicts which telecom customers are likely to churn using the **IBM Telco Customer Churn dataset** (7,043 customers, 26.5% churn rate). Delivers a segmented risk report with per-customer SHAP explanations and business recommendations.

**Results:**
| Model | AUC-ROC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| Logistic Regression | 0.841 | 0.62 | 0.64 | 0.60 |
| Random Forest | 0.849 | 0.65 | 0.68 | 0.63 |
| **XGBoost (tuned)** | **0.865** | **0.67** | **0.71** | **0.64** |
| LightGBM | 0.861 | 0.66 | 0.70 | 0.63 |

**Key findings:**
- Month-to-month contract customers churn at **42.7%** vs 2.8% for 2-year contracts
- Fiber optic customers without tech support churn at **51.4%** — the single highest-risk segment
- Customers in the first 3 months of tenure are **4.2× more likely to churn** than those past 12 months
- Paperless billing + electronic check = highest churn payment combo (45.3%)

---

##Project Structure

```
project3_telco_churn/
│
├── data/
│   ├── raw/                          # Kaggle CSV (gitignored)
│   └── processed/                    # Cleaned parquets
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Distributions, churn by segment
│   ├── 02_feature_engineering.ipynb  # Feature creation, encoding
│   ├── 03_model_training.ipynb       # MLflow experiment tracking
│   └── 04_shap_explainability.ipynb  # Global + individual SHAP
│
├── src/
│   ├── data_prep.py                  # Load, clean, engineer features
│   ├── train.py                      # Train + MLflow + Optuna
│   ├── shap_analysis.py              # SHAP global + individual plots
│   ├── segment.py                    # Customer risk segmentation
│   └── predict.py                    # Score new customers
│
├── dashboard/
│   └── app.py                        # Streamlit retention dashboard
│
├── models/                           # Saved artefacts
├── outputs/                          # Charts, SHAP plots, reports
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/telco-churn-prediction.git
cd telco-churn-prediction
pip install -r requirements.txt

# Download data
kaggle datasets download -d blastchar/telco-customer-churn -p data/raw/ --unzip

# Run pipeline
python src/data_prep.py
python src/train.py
python src/shap_analysis.py
python src/segment.py

# Dashboard
streamlit run dashboard/app.py
```

---

## Feature Engineering Highlights

| Feature | Rationale |
|---------|-----------|
| `tenure_group` | Non-linear tenure buckets: new (0–3m), early (3–12m), established (1–3yr), loyal (3yr+) |
| `monthly_to_total_ratio` | High ratio = new customer or recent upgrade — elevated churn signal |
| `service_count` | Total number of active services (0–6) — more services = stickier customer |
| `contract_risk_score` | Month-to-month=2, one-year=1, two-year=0 — ordinal risk encoding |
| `support_gap` | Has internet service but no tech support — known high-churn combination |
| `digital_payment_risk` | Electronic check + paperless billing combined flag |

---

## Tech Stack
Python · pandas · scikit-learn · XGBoost · LightGBM · SHAP · MLflow · Optuna · imbalanced-learn · Streamlit · Plotly

---

*Dataset: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — IBM sample data*
