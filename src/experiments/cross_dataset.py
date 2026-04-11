"""
Cross-Dataset Validation
=========================
Train on one dataset, test on another to evaluate generalizability.

Approach:
  - Identify overlapping features between WESAD and SWELL (e.g., RMSSD)
  - Train stress classifier on one dataset, evaluate on the other
  - Report transfer performance metrics

Usage:
    python -m src.experiments.cross_dataset
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import lightgbm as lgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import HRV_FEATURE_COLS, SWELL_PHYSIO_COLS, SWELL_BEH_COLS


def identify_overlapping_features():
    """
    Identify semantically overlapping features between WESAD and SWELL.

    WESAD has: MeanNN, SDNN, RMSSD, pNN50, MeanHR, SDHR, LF/HF, SampEn, etc.
    SWELL has: HR, RMSSD, SCL + behavioral features

    Overlap: RMSSD (direct), HR ↔ MeanHR (equivalent)
    """
    # Direct feature overlap (same name, same meaning)
    direct_overlap = ["RMSSD"]

    # Semantic mapping (different names, same concept)
    semantic_map = {
        "MeanHR": "HR",  # WESAD MeanHR ≈ SWELL HR
    }

    return direct_overlap, semantic_map


def prepare_wesad_data(features_path):
    """Load and prepare WESAD HRV features."""
    df = pd.read_csv(features_path)
    df["stress_label"] = (df["label"] == 2).astype(int)
    return df


def prepare_swell_data(features_path):
    """Load and prepare SWELL physiological features."""
    df = pd.read_csv(features_path)
    # Ensure stress_label column exists
    if "stress_label" not in df.columns and "label" in df.columns:
        df["stress_label"] = df["label"].astype(int)
    return df


def align_features(wesad_df, swell_df):
    """
    Align features between datasets for cross-validation.

    Returns standardized DataFrames with only overlapping features.
    """
    direct_overlap, semantic_map = identify_overlapping_features()

    # Build unified feature set
    # For WESAD: use MeanHR as HR equivalent
    wesad_aligned = pd.DataFrame()
    swell_aligned = pd.DataFrame()

    shared_features = []

    # Direct overlaps
    for feat in direct_overlap:
        if feat in wesad_df.columns and feat in swell_df.columns:
            wesad_aligned[feat] = wesad_df[feat]
            swell_aligned[feat] = swell_df[feat]
            shared_features.append(feat)

    # Semantic mappings
    for wesad_feat, swell_feat in semantic_map.items():
        if wesad_feat in wesad_df.columns and swell_feat in swell_df.columns:
            # Use swell feature name as canonical
            wesad_aligned[swell_feat] = wesad_df[wesad_feat]
            swell_aligned[swell_feat] = swell_df[swell_feat]
            shared_features.append(swell_feat)

    wesad_aligned["stress_label"] = wesad_df["stress_label"].values
    swell_aligned["stress_label"] = swell_df["stress_label"].values

    if "subject_id" in wesad_df.columns:
        wesad_aligned["subject_id"] = wesad_df["subject_id"].values
    if "subject_id" in swell_df.columns:
        swell_aligned["subject_id"] = swell_df["subject_id"].values

    print(f"  Shared features ({len(shared_features)}): {shared_features}")
    return wesad_aligned, swell_aligned, shared_features


def train_and_evaluate(X_train, y_train, X_test, y_test, model_name="LightGBM"):
    """Train on one dataset, evaluate on another."""
    # Standardize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = lgb.LGBMClassifier(
        n_estimators=200, num_leaves=31, learning_rate=0.05,
        class_weight="balanced", verbose=-1, random_state=42,
    )
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = np.nan

    return {
        "Accuracy": acc,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def plot_cross_dataset_results(results, output_dir):
    """Plot cross-dataset validation results."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    scenarios = list(results.keys())
    metrics = ["Accuracy", "F1-Score", "ROC-AUC"]
    colors = ["#3b82f6", "#8b5cf6", "#22c55e", "#f97316"]

    for ax, metric in zip(axes, metrics):
        values = [results[s][metric] for s in scenarios]
        bars = ax.bar(scenarios, values, color=colors[:len(scenarios)], edgecolor="white")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.set_ylim([0, 1.05])
        ax.tick_params(axis="x", rotation=20)

        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    plt.suptitle("Cross-Dataset Generalization Study", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "cross_dataset_validation.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved cross-dataset plot: {path}")
    return path


def main():
    output_dir = os.path.join(PROJECT_ROOT, "output", "experiments")
    features_dir = os.path.join(PROJECT_ROOT, "output", "features")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  CROSS-DATASET VALIDATION EXPERIMENT")
    print("=" * 60)

    # Load datasets
    wesad_path = os.path.join(features_dir, "wesad_features.csv")
    swell_path = os.path.join(features_dir, "swell_physio.csv")

    if not os.path.exists(wesad_path):
        print(f"WESAD features not found: {wesad_path}")
        print("Run `python main.py` first.")
        sys.exit(1)

    if not os.path.exists(swell_path):
        print(f"SWELL features not found: {swell_path}")
        print("Run `python main.py --dataset swell` first.")
        sys.exit(1)

    wesad_df = prepare_wesad_data(wesad_path)
    swell_df = prepare_swell_data(swell_path)

    print(f"\nWESAD: {len(wesad_df)} samples, {wesad_df['subject_id'].nunique()} subjects")
    print(f"SWELL: {len(swell_df)} samples, {swell_df['subject_id'].nunique()} subjects")

    # Align features
    print("\nAligning features across datasets...")
    wesad_aligned, swell_aligned, shared_features = align_features(wesad_df, swell_df)

    if len(shared_features) == 0:
        print("ERROR: No overlapping features found between WESAD and SWELL.")
        sys.exit(1)

    # Prepare data
    X_wesad = wesad_aligned[shared_features].values
    y_wesad = wesad_aligned["stress_label"].values
    X_swell = swell_aligned[shared_features].values
    y_swell = swell_aligned["stress_label"].values

    # Drop NaN rows
    mask_w = ~np.any(np.isnan(X_wesad), axis=1)
    mask_s = ~np.any(np.isnan(X_swell), axis=1)
    X_wesad, y_wesad = X_wesad[mask_w], y_wesad[mask_w]
    X_swell, y_swell = X_swell[mask_s], y_swell[mask_s]

    results = {}

    # Scenario 1: Train WESAD → Test SWELL
    print("\n" + "─" * 50)
    print("  Train: WESAD → Test: SWELL")
    print("─" * 50)
    r1 = train_and_evaluate(X_wesad, y_wesad, X_swell, y_swell)
    results["WESAD → SWELL"] = r1
    print(f"  Accuracy: {r1['Accuracy']:.4f}  F1: {r1['F1-Score']:.4f}  AUC: {r1['ROC-AUC']:.4f}")
    print(classification_report(y_swell, r1["y_pred"], target_names=["Baseline", "Stress"]))

    # Scenario 2: Train SWELL → Test WESAD
    print("─" * 50)
    print("  Train: SWELL → Test: WESAD")
    print("─" * 50)
    r2 = train_and_evaluate(X_swell, y_swell, X_wesad, y_wesad)
    results["SWELL → WESAD"] = r2
    print(f"  Accuracy: {r2['Accuracy']:.4f}  F1: {r2['F1-Score']:.4f}  AUC: {r2['ROC-AUC']:.4f}")
    print(classification_report(y_wesad, r2["y_pred"], target_names=["Baseline", "Stress"]))

    # Scenario 3: Combined training, leave-dataset-out
    print("─" * 50)
    print("  Combined: Train on both (mixed)")
    print("─" * 50)
    X_combined = np.vstack([X_wesad, X_swell])
    y_combined = np.concatenate([y_wesad, y_swell])
    # Use 5-fold stratified CV on combined data
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    clf = lgb.LGBMClassifier(
        n_estimators=200, num_leaves=31, learning_rate=0.05,
        class_weight="balanced", verbose=-1, random_state=42,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_combined_s = scaler.fit_transform(X_combined)
    y_pred_combined = cross_val_predict(clf, X_combined_s, y_combined, cv=skf)

    acc3 = accuracy_score(y_combined, y_pred_combined)
    f1_3 = f1_score(y_combined, y_pred_combined, zero_division=0)
    results["Combined (5-fold)"] = {"Accuracy": acc3, "F1-Score": f1_3, "ROC-AUC": np.nan}
    print(f"  Accuracy: {acc3:.4f}  F1: {f1_3:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("  CROSS-DATASET SUMMARY")
    print("=" * 60)
    summary_df = pd.DataFrame([
        {"Scenario": k, **{m: v[m] for m in ["Accuracy", "F1-Score", "ROC-AUC"]}}
        for k, v in results.items()
    ])
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(output_dir, "cross_dataset_results.csv"), index=False)

    # Plot
    plot_cross_dataset_results(results, output_dir)

    print(f"\n✓ Cross-dataset results saved to {output_dir}")


if __name__ == "__main__":
    main()
