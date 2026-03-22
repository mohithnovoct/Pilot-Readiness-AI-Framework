"""
Stress Classifier (LightGBM)
=============================
Trains a LightGBM classifier on WESAD HRV features to predict stress state.

Key design decisions:
  - Leave-One-Subject-Out (LOSO) cross-validation for generalization
  - Grid search hyperparameter tuning
  - SHAP-based feature importance and selection
  - Output: P(stress) ∈ [0, 1] for fusion with performance model

Based on research plan Section 6.1.1:
  - LightGBM ~88% accuracy with 12 HRV features (published benchmarks)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from typing import Dict, Optional, Tuple, List
import pickle
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# Feature columns used for training (HRV features)
from config import HRV_FEATURE_COLS


def prepare_training_data(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare feature matrix, labels, and group IDs for LOSO CV.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with 'stress_label' and 'subject_id' columns.
    feature_cols : list of str, optional
        Which columns to use as features.

    Returns
    -------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Binary labels (0=baseline, 1=stress).
    groups : np.ndarray
        Subject IDs for LOSO grouping.
    """
    if feature_cols is None:
        feature_cols = [c for c in HRV_FEATURE_COLS if c in df.columns]

    # Drop rows with missing features
    clean = df.dropna(subset=feature_cols + ["stress_label"])

    X = clean[feature_cols].values.copy()
    y = clean["stress_label"].values.astype(int)
    groups = clean["subject_id"].values

    # Replace any remaining NaN with column median
    col_medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_medians[j]

    print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Class distribution: baseline={np.sum(y == 0)}, stress={np.sum(y == 1)}")
    print(f"  Subjects: {len(np.unique(groups))}")

    return X, y, groups


def train_loso_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    params: Optional[dict] = None,
) -> Dict:
    """
    Train LightGBM with Leave-One-Subject-Out cross-validation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    groups : np.ndarray
        Subject IDs.
    params : dict, optional
        LightGBM parameters.

    Returns
    -------
    dict
        Results including per-fold metrics, aggregated metrics,
        and out-of-fold predictions.
    """
    if params is None:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "max_depth": -1,
            "min_child_samples": 10,
            "class_weight": "balanced",
            "verbose": -1,
            "random_state": 42,
        }

    logo = LeaveOneGroupOut()
    fold_results = []
    all_preds = np.zeros(len(y))
    all_probs = np.zeros(len(y))

    unique_subjects = np.unique(groups)
    print(f"\nRunning LOSO CV across {len(unique_subjects)} subjects...")

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_subject = groups[test_idx[0]]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train LightGBM
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(20, verbose=False),
                       lgb.log_evaluation(period=0)],
        )

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        all_preds[test_idx] = y_pred
        all_probs[test_idx] = y_prob

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = np.nan

        fold_results.append({
            "subject": test_subject,
            "accuracy": acc,
            "f1": f1,
            "auc": auc,
            "n_test": len(y_test),
        })

        print(f"  Fold {fold_idx + 1:2d} ({test_subject}): "
              f"Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}  "
              f"(n={len(y_test)})")

    # Aggregate metrics
    overall_acc = accuracy_score(y, all_preds)
    overall_f1 = f1_score(y, all_preds, zero_division=0)
    try:
        overall_auc = roc_auc_score(y, all_probs)
    except ValueError:
        overall_auc = np.nan

    print(f"\n{'='*50}")
    print(f"LOSO CV Results:")
    print(f"  Overall Accuracy: {overall_acc:.4f}")
    print(f"  Overall F1-Score: {overall_f1:.4f}")
    print(f"  Overall ROC-AUC:  {overall_auc:.4f}")
    print(f"{'='*50}")
    print(f"\nClassification Report:")
    print(classification_report(y, all_preds, target_names=["Baseline", "Stress"]))

    return {
        "fold_results": fold_results,
        "overall_accuracy": overall_acc,
        "overall_f1": overall_f1,
        "overall_auc": overall_auc,
        "predictions": all_preds,
        "probabilities": all_probs,
        "confusion_matrix": confusion_matrix(y, all_preds),
    }


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    params: Optional[dict] = None,
    do_grid_search: bool = True,
) -> lgb.LGBMClassifier:
    """
    Train the final LightGBM model on all data.

    Parameters
    ----------
    X, y : arrays
        Full training data.
    params : dict, optional
        Base LightGBM parameters.
    do_grid_search : bool
        Whether to run grid search for hyperparameter tuning.

    Returns
    -------
    lgb.LGBMClassifier
        Trained model.
    """
    if params is None:
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "class_weight": "balanced",
            "verbose": -1,
            "random_state": 42,
        }

    if do_grid_search:
        print("\nRunning Grid Search for hyperparameter tuning...")
        param_grid = {
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 200, 300],
            "max_depth": [-1, 5, 10],
        }

        base_model = lgb.LGBMClassifier(**params)
        grid = GridSearchCV(
            base_model, param_grid,
            cv=5, scoring="f1", n_jobs=-1, verbose=0,
        )
        grid.fit(X, y)

        print(f"  Best params: {grid.best_params_}")
        print(f"  Best F1 (CV): {grid.best_score_:.4f}")

        model = grid.best_estimator_
    else:
        model = lgb.LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            max_depth=-1,
            **params,
        )
        model.fit(X, y)

    return model


def compute_shap_importance(
    model: lgb.LGBMClassifier,
    X: np.ndarray,
    feature_names: List[str],
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Compute SHAP feature importance.

    Returns
    -------
    pd.DataFrame
        Feature importance ranking with SHAP values.
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For binary classification, shap_values might be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class 1 (stress)

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)

        print(f"\nTop {top_k} features by SHAP importance:")
        for _, row in importance_df.head(top_k).iterrows():
            print(f"  {row['feature']:20s}: {row['mean_abs_shap']:.6f}")

        return importance_df

    except ImportError:
        warnings.warn("SHAP not available. Using LightGBM built-in importance.")
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)
        return importance_df


def save_model(model, filepath: str):
    """Save trained model to pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath: str):
    """Load trained model from pickle."""
    with open(filepath, "rb") as f:
        return pickle.load(f)
