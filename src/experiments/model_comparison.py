"""
Model Comparison Experiments
=============================
Compare LightGBM against baseline classifiers and run ablation studies.

Usage:
    python -m src.experiments.model_comparison              # Full comparison
    python -m src.experiments.model_comparison --quick       # Quick mode (fewer folds)
    python -m src.experiments.model_comparison --ablation    # Feature ablation only
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.stress_classifier import prepare_training_data, HRV_FEATURE_COLS
from config import SWELL_PHYSIO_COLS, SWELL_BEH_COLS


def get_classifiers():
    """Return a dictionary of classifiers to compare."""
    classifiers = {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200, num_leaves=31, learning_rate=0.05,
            class_weight="balanced", verbose=-1, random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
            probability=True, random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42,
        ),
    }

    if XGB_AVAILABLE:
        classifiers["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0,
        )

    return classifiers


def run_loso_comparison(X, y, groups, classifiers, feature_names=None):
    """
    Run LOSO CV for each classifier and collect results.

    Returns
    -------
    pd.DataFrame
        Comparison table with accuracy, F1, AUC per classifier.
    """
    logo = LeaveOneGroupOut()
    results = []

    for name, clf in classifiers.items():
        print(f"\n{'─'*50}")
        print(f"  Evaluating: {name}")
        print(f"{'─'*50}")

        start = time.time()
        all_preds = np.zeros(len(y))
        all_probs = np.zeros(len(y))

        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                clf_copy = clf.__class__(**clf.get_params())
                clf_copy.fit(X_train, y_train)
                all_preds[test_idx] = clf_copy.predict(X_test)
                if hasattr(clf_copy, "predict_proba"):
                    all_probs[test_idx] = clf_copy.predict_proba(X_test)[:, 1]
                else:
                    all_probs[test_idx] = all_preds[test_idx]
            except Exception as e:
                print(f"    Fold {fold_idx} failed: {e}")
                all_preds[test_idx] = 0
                all_probs[test_idx] = 0.5

        elapsed = time.time() - start

        acc = accuracy_score(y, all_preds)
        f1 = f1_score(y, all_preds, zero_division=0)
        try:
            auc = roc_auc_score(y, all_probs)
        except ValueError:
            auc = np.nan

        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "Time (s)": elapsed,
        })

        print(f"  Accuracy: {acc:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}  ({elapsed:.1f}s)")

    return pd.DataFrame(results)


def run_feature_ablation(X, y, groups, feature_names):
    """
    Run feature ablation study: compare different feature subsets.

    Returns
    -------
    pd.DataFrame
        Ablation results by feature group.
    """
    # Define feature groups
    time_features = ["MeanNN", "MedianNN", "SDNN", "RMSSD", "pNN50", "MeanHR", "SDHR"]
    freq_features = ["VLF_power", "LF_power", "HF_power", "Total_power",
                      "LF_HF_ratio", "LF_norm", "HF_norm"]
    nonlinear_features = ["SampEn"]

    # Map feature names to indices
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    ablation_groups = {
        "All Features": list(range(len(feature_names))),
        "Time-Domain Only": [name_to_idx[f] for f in time_features if f in name_to_idx],
        "Frequency-Domain Only": [name_to_idx[f] for f in freq_features if f in name_to_idx],
        "Time + Nonlinear": [name_to_idx[f] for f in time_features + nonlinear_features if f in name_to_idx],
        "Top-5 (SHAP)": None,  # Will be set based on SHAP
    }

    # Get top-5 features via LightGBM importance
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, num_leaves=31, verbose=-1, random_state=42,
    )
    lgb_model.fit(X, y)
    importances = lgb_model.feature_importances_
    top5_idx = np.argsort(importances)[::-1][:5]
    ablation_groups["Top-5 (SHAP)"] = top5_idx.tolist()

    results = []
    logo = LeaveOneGroupOut()

    for group_name, feat_idx in ablation_groups.items():
        if not feat_idx:
            continue

        X_sub = X[:, feat_idx]
        feat_names_sub = [feature_names[i] for i in feat_idx]
        print(f"\n  Ablation: {group_name} ({len(feat_idx)} features: {feat_names_sub})")

        all_preds = np.zeros(len(y))
        all_probs = np.zeros(len(y))

        for train_idx, test_idx in logo.split(X_sub, y, groups):
            clf = lgb.LGBMClassifier(
                n_estimators=200, num_leaves=31, verbose=-1, random_state=42,
            )
            clf.fit(X_sub[train_idx], y[train_idx])
            all_preds[test_idx] = clf.predict(X_sub[test_idx])
            all_probs[test_idx] = clf.predict_proba(X_sub[test_idx])[:, 1]

        acc = accuracy_score(y, all_preds)
        f1 = f1_score(y, all_preds, zero_division=0)
        try:
            auc = roc_auc_score(y, all_probs)
        except ValueError:
            auc = np.nan

        results.append({
            "Feature Group": group_name,
            "N Features": len(feat_idx),
            "Accuracy": acc,
            "F1-Score": f1,
            "ROC-AUC": auc,
        })

        print(f"    Acc: {acc:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")

    return pd.DataFrame(results)


def run_window_ablation(feature_dir, feature_cols):
    """
    Compare performance across different window sizes.
    Uses pre-cached features if available, or notes that regeneration is needed.
    """
    # This requires re-extracting features with different window sizes
    # For now, report from the default 60s window
    print("\n  Window ablation requires re-running feature extraction with different --window-sec values.")
    print("  Example commands:")
    print("    python main.py --quick-test --window-sec 30")
    print("    python main.py --quick-test --window-sec 60")
    print("    python main.py --quick-test --window-sec 120")
    return None


def plot_comparison(comparison_df, output_dir):
    """Generate comparison bar charts."""
    os.makedirs(output_dir, exist_ok=True)

    metrics = ["Accuracy", "F1-Score", "ROC-AUC"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))

    colors = ["#3b82f6", "#8b5cf6", "#22c55e", "#f97316", "#ef4444"]

    for ax, metric in zip(axes, metrics):
        bars = ax.bar(
            comparison_df["Model"], comparison_df[metric],
            color=colors[:len(comparison_df)],
            edgecolor="white", linewidth=0.5,
        )
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.set_ylim([0, 1.05])
        ax.tick_params(axis="x", rotation=35)

        # Add value labels
        for bar, val in zip(bars, comparison_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Model Comparison — LOSO Cross-Validation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved comparison plot: {path}")
    return path


def plot_ablation(ablation_df, output_dir):
    """Generate ablation bar chart."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(ablation_df))
    width = 0.25

    ax.bar([i - width for i in x], ablation_df["Accuracy"], width, label="Accuracy", color="#3b82f6")
    ax.bar(x, ablation_df["F1-Score"], width, label="F1-Score", color="#8b5cf6")
    ax.bar([i + width for i in x], ablation_df["ROC-AUC"], width, label="ROC-AUC", color="#22c55e")

    ax.set_xticks(x)
    ax.set_xticklabels(ablation_df["Feature Group"], rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Feature Ablation Study — LightGBM with LOSO CV")
    ax.set_ylim([0, 1.1])
    ax.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, "feature_ablation.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved ablation plot: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Model Comparison Experiments")
    parser.add_argument("--dataset", default="wesad", choices=["wesad", "swell"])
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--ablation", action="store_true", help="Run ablation only")
    parser.add_argument("--per-subject-norm", action="store_true")
    args = parser.parse_args()

    output_dir = os.path.join(PROJECT_ROOT, "output", "experiments")
    os.makedirs(output_dir, exist_ok=True)

    # Load features
    features_dir = os.path.join(PROJECT_ROOT, "output", "features")

    if args.dataset == "wesad":
        features_path = os.path.join(features_dir, "wesad_features.csv")
        if not os.path.exists(features_path):
            print(f"Feature file not found: {features_path}")
            print("Run `python main.py` first to extract features.")
            sys.exit(1)
        df = pd.read_csv(features_path)
        df["stress_label"] = (df["label"] == 2).astype(int)
        feature_cols = [c for c in HRV_FEATURE_COLS if c in df.columns]
    else:
        features_path = os.path.join(features_dir, "swell_physio.csv")
        if not os.path.exists(features_path):
            print(f"Feature file not found: {features_path}")
            print("Run `python main.py --dataset swell` first.")
            sys.exit(1)
        df = pd.read_csv(features_path)
        feature_cols = [c for c in SWELL_PHYSIO_COLS + SWELL_BEH_COLS if c in df.columns]

    X, y, groups = prepare_training_data(df, feature_cols, per_subject_norm=args.per_subject_norm)

    # Model comparison
    if not args.ablation:
        print("\n" + "=" * 60)
        print("  MODEL COMPARISON EXPERIMENT")
        print("=" * 60)

        classifiers = get_classifiers()
        comparison_df = run_loso_comparison(X, y, groups, classifiers, feature_cols)

        # Save results
        comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
        print(f"\n{'='*60}")
        print("  COMPARISON RESULTS")
        print(f"{'='*60}")
        print(comparison_df.to_string(index=False))

        plot_comparison(comparison_df, output_dir)

    # Ablation study
    print("\n" + "=" * 60)
    print("  FEATURE ABLATION STUDY")
    print("=" * 60)

    ablation_df = run_feature_ablation(X, y, groups, feature_cols)
    ablation_df.to_csv(os.path.join(output_dir, "feature_ablation.csv"), index=False)
    print(f"\n{'='*60}")
    print("  ABLATION RESULTS")
    print(f"{'='*60}")
    print(ablation_df.to_string(index=False))

    plot_ablation(ablation_df, output_dir)

    # Window size note
    run_window_ablation(features_dir, feature_cols)

    print(f"\n✓ All experiment results saved to {output_dir}")


if __name__ == "__main__":
    main()
