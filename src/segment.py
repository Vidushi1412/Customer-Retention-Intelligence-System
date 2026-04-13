"""
segment.py  —  Telco Customer Churn
=====================================
Segments customers into risk tiers and generates a business retention report.
Produces segment profiles, revenue-at-risk estimates, and recommended actions.

Usage:
    python src/segment.py
"""

import json, joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
OUTPUTS_DIR   = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)


def load_artefacts():
    model  = joblib.load(MODELS_DIR / "xgboost_best.pkl")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").squeeze()
    with open(MODELS_DIR / "feature_names.json") as f:
        feat = json.load(f)
    X_test = X_test[feat]
    return model, X_test, y_test


def segment_customers(model, X_test, y_test) -> pd.DataFrame:
    """
    Assign each customer a risk tier and compute segment-level stats.

    Risk tiers:
        GREEN  — P(churn) < 0.30  — retain passively
        AMBER  — 0.30 ≤ P < 0.60  — proactive outreach
        RED    — P(churn) ≥ 0.60  — urgent intervention required
    """
    y_proba = model.predict_proba(X_test)[:, 1]

    seg = pd.DataFrame(X_test.copy())
    seg["churn_probability"] = y_proba
    seg["actual_churn"]      = y_test.values
    seg["risk_tier"] = pd.cut(
        y_proba,
        bins=[-0.001, 0.30, 0.60, 1.001],
        labels=["Low", "Medium", "High"],
    )

    # Monthly charges is in the feature set
    monthly_col = "MonthlyCharges" if "MonthlyCharges" in seg.columns else None

    if monthly_col:
        seg["monthly_revenue"] = seg[monthly_col]
        seg["annual_revenue_at_risk"] = seg["monthly_revenue"] * seg["churn_probability"] * 12
    else:
        seg["monthly_revenue"]          = 65   # default average
        seg["annual_revenue_at_risk"]   = 65 * seg["churn_probability"] * 12

    return seg


def segment_summary(seg: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stats per risk tier."""
    summary = (
        seg.groupby("risk_tier", observed=True)
        .agg(
            n_customers=("churn_probability", "count"),
            avg_churn_prob=("churn_probability", "mean"),
            actual_churn_rate=("actual_churn", "mean"),
            avg_monthly_rev=("monthly_revenue", "mean"),
            total_annual_risk=("annual_revenue_at_risk", "sum"),
        )
        .reset_index()
    )
    summary["avg_churn_prob"]     = summary["avg_churn_prob"].round(3)
    summary["actual_churn_rate"]  = summary["actual_churn_rate"].round(3)
    summary["avg_monthly_rev"]    = summary["avg_monthly_rev"].round(2)
    summary["total_annual_risk"]  = summary["total_annual_risk"].round(0).astype(int)
    return summary


def plot_risk_distribution(seg: pd.DataFrame, save_path=None):
    """Stacked bar: risk tier × actual churn outcome."""
    ct = pd.crosstab(seg["risk_tier"], seg["actual_churn"], normalize="index") * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Count by tier
    colors = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}
    tier_counts = seg["risk_tier"].value_counts().reindex(["Low", "Medium", "High"])
    axes[0].bar(tier_counts.index, tier_counts.values,
                color=[colors[t] for t in tier_counts.index], edgecolor="none", width=0.5)
    for i, (tier, val) in enumerate(tier_counts.items()):
        axes[0].text(i, val + 5, f"{val:,}", ha="center", fontsize=11, fontweight="bold")
    axes[0].set_title("Customer Count by Risk Tier", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Number of Customers")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Right: Actual churn rate by tier
    actual_rates = seg.groupby("risk_tier", observed=True)["actual_churn"].mean() * 100
    actual_rates = actual_rates.reindex(["Low", "Medium", "High"])
    bars = axes[1].bar(actual_rates.index, actual_rates.values,
                       color=[colors[t] for t in actual_rates.index], edgecolor="none", width=0.5)
    for bar, val in zip(bars, actual_rates.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.5,
                     f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    axes[1].set_title("Actual Churn Rate by Risk Tier", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Churn Rate (%)")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.suptitle("Customer Risk Segmentation — Telco Churn", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    return fig


def plot_revenue_at_risk(seg: pd.DataFrame, save_path=None):
    """Revenue-at-risk breakdown by tier."""
    rev = seg.groupby("risk_tier", observed=True)["annual_revenue_at_risk"].sum()
    rev = rev.reindex(["Low", "Medium", "High"])
    colors = ["#22c55e", "#f59e0b", "#ef4444"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(rev.index, rev.values / 1000, color=colors, edgecolor="none", width=0.5)
    for bar, val in zip(bars, rev.values):
        ax.text(bar.get_x() + bar.get_width()/2, val/1000 + 0.5,
                f"${val/1000:,.0f}k", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Estimated Annual Revenue at Risk by Tier", fontsize=12, fontweight="bold")
    ax.set_ylabel("Revenue at Risk ($k)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    return fig


RETENTION_ACTIONS = {
    "Low":    "Quarterly check-in email. No discount needed. Upsell opportunity.",
    "Medium": "Proactive outreach call within 7 days. Offer loyalty discount (10–15%). Review plan fit.",
    "High":   "Immediate retention call within 48 hrs. Offer contract upgrade incentive. Escalate to senior agent.",
}


def print_retention_report(seg: pd.DataFrame, summary: pd.DataFrame):
    print("\n" + "=" * 65)
    print("CUSTOMER RISK SEGMENTATION REPORT — Telco Churn")
    print("=" * 65)
    total_risk = seg["annual_revenue_at_risk"].sum()
    print(f"\n  Total customers scored  : {len(seg):,}")
    print(f"  Total annual revenue @risk: ${total_risk:,.0f}")

    for _, row in summary.iterrows():
        tier   = row["risk_tier"]
        action = RETENTION_ACTIONS.get(tier, "")
        print(f"\n  ── {tier.upper()} RISK ──")
        print(f"  Customers         : {row['n_customers']:,}")
        print(f"  Avg churn prob    : {row['avg_churn_prob']:.1%}")
        print(f"  Actual churn rate : {row['actual_churn_rate']:.1%}")
        print(f"  Avg monthly rev   : ${row['avg_monthly_rev']:.2f}")
        print(f"  Annual rev @ risk : ${row['total_annual_risk']:,}")
        print(f"  Recommended action: {action}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    print("=" * 55)
    print("Telco Churn — Customer Segmentation")
    print("=" * 55)

    model, X_test, y_test = load_artefacts()

    print("\nSegmenting customers...")
    seg     = segment_customers(model, X_test, y_test)
    summary = segment_summary(seg)

    print_retention_report(seg, summary)

    # Save
    seg.to_csv(OUTPUTS_DIR / "customer_segments.csv", index=False)
    summary.to_csv(OUTPUTS_DIR / "segment_summary.csv", index=False)
    print(f"\n  Saved segments → {OUTPUTS_DIR}/customer_segments.csv")

    print("\nGenerating plots...")
    plot_risk_distribution(seg, OUTPUTS_DIR / "risk_distribution.png")
    plot_revenue_at_risk(seg,   OUTPUTS_DIR / "revenue_at_risk.png")

    print(f"\n✓ Segmentation complete.")
