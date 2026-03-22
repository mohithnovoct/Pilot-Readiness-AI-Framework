"""
Performance Model (LightGBM Regressor)
=======================================
Trains a LightGBM regressor on MATB-II simulated performance data
to predict normalized performance deviation scores.

The model maps behavioral features (CVRT, Inceptor Entropy, RMSD, etc.)
to a continuous workload/degradation score needed for the fusion formula.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Tuple, List
import pickle
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# Performance feature columns
from config import PERF_FEATURE_COLS


def prepare_performance_data(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    normalize_target: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[MinMaxScaler]]:
    """
    Prepare performance data for regression.

    The target is the workload_level (0=low, 1=medium, 2=high),
    normalized to [0, 1] for fusion compatibility.

    Parameters
    ----------
    df : pd.DataFrame
        Simulated performance data.
    feature_cols : list of str, optional
        Which columns to use as features.
    normalize_target : bool
        Whether to normalize target to [0, 1].

    Returns
    -------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Normalized workload scores.
    scaler : MinMaxScaler or None
        Target scaler (for inverse transform).
    """
    if feature_cols is None:
        feature_cols = [c for c in PERF_FEATURE_COLS if c in df.columns]

    clean = df.dropna(subset=feature_cols)
    X = clean[feature_cols].values
    y = clean["workload_level"].values.astype(float)

    scaler = None
    if normalize_target:
        scaler = MinMaxScaler()
        y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # Replace NaN features with column median
    col_medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_medians[j]

    print(f"Performance data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")

    return X, y, scaler


def train_performance_model(
    X: np.ndarray,
    y: np.ndarray,
    do_grid_search: bool = True,
) -> lgb.LGBMRegressor:
    """
    Train a LightGBM regressor to predict performance deviation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Normalized target values.
    do_grid_search : bool
        Whether to run grid search.

    Returns
    -------
    lgb.LGBMRegressor
        Trained regression model.
    """
    base_params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "verbose": -1,
        "random_state": 42,
    }

    if do_grid_search:
        print("\nGrid Search for performance model...")
        param_grid = {
            "num_leaves": [15, 31],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 200],
            "max_depth": [-1, 5],
        }

        model = lgb.LGBMRegressor(**base_params)
        grid = GridSearchCV(
            model, param_grid,
            cv=5, scoring="neg_mean_squared_error",
            n_jobs=-1, verbose=0,
        )
        grid.fit(X, y)

        print(f"  Best params: {grid.best_params_}")
        print(f"  Best MSE (CV): {-grid.best_score_:.6f}")

        best_model = grid.best_estimator_
    else:
        best_model = lgb.LGBMRegressor(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            **base_params,
        )
        best_model.fit(X, y)

    # Evaluate on training set
    y_pred = best_model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print(f"\nTraining set performance:")
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  MAE:   {mae:.4f}")
    print(f"  R²:    {r2:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(
        best_model, X, y, cv=5,
        scoring="neg_mean_squared_error",
    )
    print(f"  CV RMSE: {np.sqrt(-cv_scores.mean()):.4f} "
          f"± {np.sqrt(-cv_scores).std():.4f}")

    return best_model


def predict_performance_score(
    model: lgb.LGBMRegressor,
    features: np.ndarray,
) -> np.ndarray:
    """
    Predict normalized performance deviation score.

    Returns
    -------
    np.ndarray
        P_perf ∈ [0, 1], where higher values indicate more degradation.
    """
    preds = model.predict(features)
    return np.clip(preds, 0.0, 1.0)


def save_model(model, filepath: str):
    """Save trained model to pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"Performance model saved to {filepath}")


def load_model(filepath: str):
    """Load trained model from pickle."""
    with open(filepath, "rb") as f:
        return pickle.load(f)
