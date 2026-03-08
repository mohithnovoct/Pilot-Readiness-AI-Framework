"""
Visualization Plots
====================
Static and interactive plots for analysis, model evaluation,
and feature importance visualization.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict
import os


def setup_style():
    """Configure matplotlib style for publication-quality plots."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 120,
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a confusion matrix heatmap."""
    setup_style()
    if labels is None:
        labels = ["Baseline", "Stress"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_roc_curve(
    false_alarm_rates: np.ndarray,
    detection_rates: np.ndarray,
    auc_score: Optional[float] = None,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot ROC curve."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    label = f"Model (AUC = {auc_score:.3f})" if auc_score else "Model"
    ax.plot(false_alarm_rates, detection_rates, "b-", linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="Random")

    ax.set_xlabel("False Alarm Rate (α)")
    ax.set_ylabel("Detection Rate (1 - β)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect("equal")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_k: int = 15,
    title: str = "Feature Importance (SHAP)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot horizontal bar chart of feature importance."""
    setup_style()
    df = importance_df.head(top_k).sort_values(
        importance_df.columns[1], ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.4)))

    importance_col = df.columns[1]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
    ax.barh(df.iloc[:, 0], df[importance_col], color=colors)
    ax.set_xlabel(importance_col.replace("_", " ").title())
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_risk_timeline(
    timestamps: np.ndarray,
    risk_scores: np.ndarray,
    threshold: float = 0.5,
    labels: Optional[np.ndarray] = None,
    title: str = "Readiness Risk Score Timeline",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot risk score over time with threshold line.

    Parameters
    ----------
    timestamps : np.ndarray
        Time points (seconds or indices).
    risk_scores : np.ndarray
        Risk scores at each time point.
    threshold : float
        Current alert threshold.
    labels : np.ndarray, optional
        True condition labels (1=baseline, 2=stress) for shading.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    # Color-code by risk level
    colors = []
    for r in risk_scores:
        if r < 0.25:
            colors.append("#22c55e")  # Green
        elif r < 0.5:
            colors.append("#eab308")  # Yellow
        elif r < 0.75:
            colors.append("#f97316")  # Orange
        else:
            colors.append("#ef4444")  # Red

    ax.scatter(timestamps, risk_scores, c=colors, s=15, alpha=0.7, zorder=3)
    ax.plot(timestamps, risk_scores, "k-", alpha=0.3, linewidth=0.5)

    # Threshold line
    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=2,
               label=f"Threshold (γ = {threshold:.2f})")

    # Shade stress regions if labels provided
    if labels is not None:
        stress_mask = labels == 2
        if np.any(stress_mask):
            for start_idx in np.where(np.diff(stress_mask.astype(int)) == 1)[0]:
                end_candidates = np.where(
                    np.diff(stress_mask.astype(int)) == -1
                )[0]
                end_idx = end_candidates[end_candidates > start_idx]
                end_idx = end_idx[0] if len(end_idx) > 0 else len(labels) - 1
                ax.axvspan(timestamps[start_idx], timestamps[min(end_idx, len(timestamps)-1)],
                          alpha=0.15, color="red", label="Stress Period" if start_idx == np.where(np.diff(stress_mask.astype(int)) == 1)[0][0] else "")

    ax.set_xlabel("Time")
    ax.set_ylabel("Readiness Risk Score")
    ax.set_ylim([-0.05, 1.05])
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_hrv_comparison(
    features_df: pd.DataFrame,
    feature_cols: List[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot box-and-whisker comparison of HRV features across conditions.
    """
    setup_style()
    if feature_cols is None:
        feature_cols = ["SDNN", "RMSSD", "pNN50", "LF_HF_ratio", "SampEn"]

    available = [c for c in feature_cols if c in features_df.columns]
    n_features = len(available)

    if n_features == 0:
        return None

    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 5))
    if n_features == 1:
        axes = [axes]

    for ax, feat in zip(axes, available):
        data = features_df[["label_name", feat]].dropna()
        sns.boxplot(data=data, x="label_name", y=feat, ax=ax,
                    palette={"baseline": "#22c55e", "stress": "#ef4444"})
        ax.set_title(feat)
        ax.set_xlabel("")

    plt.suptitle("HRV Feature Distribution: Baseline vs. Stress", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_loso_results(
    fold_results: List[Dict],
    metric: str = "f1",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot per-subject LOSO CV results."""
    setup_style()
    df = pd.DataFrame(fold_results)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(df)), df[metric], color=plt.cm.viridis(0.6))
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["subject"], rotation=45)
    ax.set_xlabel("Left-Out Subject")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Leave-One-Subject-Out CV: {metric.upper()} Score")

    # Add mean line
    mean_val = df[metric].mean()
    ax.axhline(y=mean_val, color="r", linestyle="--", linewidth=2,
               label=f"Mean = {mean_val:.3f}")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def save_all_plots(output_dir: str, plots: Dict[str, plt.Figure]):
    """Save multiple plot figures."""
    os.makedirs(output_dir, exist_ok=True)
    for name, fig in plots.items():
        if fig is not None:
            path = os.path.join(output_dir, f"{name}.png")
            fig.savefig(path, bbox_inches="tight", dpi=150)
            plt.close(fig)
    print(f"Saved {len(plots)} plots to {output_dir}")
