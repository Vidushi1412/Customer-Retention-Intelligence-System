"""
shap_analysis.py  —  Telco Customer Churn
==========================================
Generates global and individual SHAP plots for the best XGBoost model.

Usage:
    python src/shap_analysis.py
"""

import json, joblib
import shap, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
OUTPUTS_DIR   = Path("outputs")
FORCE_DIR     = OUTPUTS_DIR / "shap_force_plots"
OUTPUTS_DIR.mkdir(exist_ok=True)
FORCE_DIR.mkdir(exist_ok=True)


def load_artefacts():
    model  = joblib.load(MODELS_DIR / "xgboost_best.pkl")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    X_train= pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").squeeze()
    with open(MODELS_DIR / "feature_names.json") as f:
        feat = json.load(f)
    X_test  = X_test[feat].astype(float).astype(float)
    X_train = X_train[feat].astype(float)
    return model, X_train, X_test, y_test


def compute_shap(model, X_train, X_test, n_bg=150):
    print("Computing SHAP values...")
    bg = X_train.sample(n=min(n_bg, len(X_train)), random_state=42)
    explainer   = shap.TreeExplainer(model, bg)
    shap_values = explainer(X_test)
    print(f"  Shape: {shap_values.values.shape}")
    return explainer, shap_values


def plot_summary(shap_values, X_test, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values.values, X_test, plot_type="dot",
                      max_display=18, show=False, alpha=0.7)
    plt.title("SHAP Summary — Telco Customer Churn\nTop 18 Feature Drivers",
              fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    return fig


def plot_bar(shap_values, X_test, save_path=None):
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    df = pd.DataFrame({"feature": X_test.columns, "importance": mean_abs})
    df = df.sort_values("importance", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df)))
    bars = ax.barh(df["feature"], df["importance"], color=colors, height=0.7, edgecolor="none")
    for bar, val in zip(bars, df["importance"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9, color="#333")
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title("Global Feature Importance (SHAP) — Telco Churn", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    return fig


def plot_segment_shap(shap_values, X_test, y_test, model, save_path=None):
    """Bar chart comparing mean SHAP by churn segment."""
    y_proba = model.predict_proba(X_test)[:, 1]
    segments = pd.cut(y_proba, bins=[0, 0.35, 0.65, 1.0],
                      labels=["Low risk", "Medium risk", "High risk"])

    top_feat_idx = np.argsort(np.abs(shap_values.values).mean(axis=0))[::-1][:6]
    top_feats    = [X_test.columns[i] for i in top_feat_idx]

    seg_shap = pd.DataFrame(shap_values.values[:, top_feat_idx],
                             columns=top_feats)
    seg_shap["segment"] = segments.to_numpy()

    seg_mean = seg_shap.groupby("segment")[top_feats].mean()

    fig, ax = plt.subplots(figsize=(11, 5))
    x     = np.arange(len(top_feats))
    width = 0.25
    colors = ["#3b82f6", "#f59e0b", "#ef4444"]
    for i, (seg, row) in enumerate(seg_mean.iterrows()):
        ax.bar(x + i*width, row.values, width, label=seg, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(top_feats, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean SHAP value", fontsize=10)
    ax.set_title("Mean SHAP by Risk Segment — Top 6 Features", fontsize=12, fontweight="bold")
    ax.legend()
    ax.axhline(0, color="#ccc", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    return fig


def plot_waterfalls(shap_values, X_test, model, save_path_dir=None):
    """Waterfall plots for high/medium/low risk customers."""
    y_proba = model.predict_proba(X_test)[:, 1]
    cases = {
        "high_risk":   np.where(y_proba > 0.75)[0],
        "medium_risk": np.where((y_proba > 0.4) & (y_proba < 0.6))[0],
        "low_risk":    np.where(y_proba < 0.15)[0],
    }
    for label, idxs in cases.items():
        if len(idxs) == 0:
            continue
        idx = idxs[0]
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(shap_values[idx], max_display=12, show=False)
        plt.title(f"SHAP Waterfall — {label.replace('_',' ').title()} Customer\n"
                  f"Predicted churn probability: {y_proba[idx]:.1%}",
                  fontsize=12, fontweight="bold")
        plt.tight_layout()
        if save_path_dir:
            p = save_path_dir / f"waterfall_{label}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            print(f"  Saved → {p}")
        plt.close(fig)


def build_feature_table(shap_values, X_test):
    return pd.DataFrame({
        "feature":       X_test.columns,
        "mean_shap":     shap_values.values.mean(axis=0).round(4),
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0).round(4),
        "pct_positive":  (shap_values.values > 0).mean(axis=0).round(3) * 100,
        "direction":     ["risk_increasing" if v > 0 else "risk_reducing"
                          for v in shap_values.values.mean(axis=0)],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print("=" * 55)
    print("Telco Churn — SHAP Explainability")
    print("=" * 55)

    model, X_train, X_test, y_test = load_artefacts()
    explainer, shap_values = compute_shap(model, X_train, X_test)

    print("\nGenerating global plots...")
    plot_summary(shap_values, X_test, OUTPUTS_DIR / "shap_summary.png")
    plot_bar(shap_values, X_test, OUTPUTS_DIR / "shap_bar.png")
    plot_segment_shap(shap_values, X_test, y_test, model, OUTPUTS_DIR / "shap_by_segment.png")

    print("\nGenerating individual waterfall plots...")
    plot_waterfalls(shap_values, X_test, model, FORCE_DIR)

    table = build_feature_table(shap_values, X_test)
    table.to_csv(OUTPUTS_DIR / "shap_feature_importance.csv", index=False)
    print("\nTop 10 features:")
    print(table.head(10).to_string(index=False))

    print(f"\n✓ SHAP analysis complete. Outputs in {OUTPUTS_DIR}/")
