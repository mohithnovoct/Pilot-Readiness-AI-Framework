"""
LightGBM Model Plugins
========================
Wraps the existing LightGBM stress classifier and performance
regressor as ``BaseModel`` implementations.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.base import BaseModel

logger = logging.getLogger("pilot_readiness.models.lightgbm")


class LightGBMStressModel(BaseModel):
    """
    LightGBM-based binary stress classifier.

    Wraps ``src.models.stress_classifier`` — supports training with
    LOSO cross-validation, SHAP importance, and grid search.
    """

    name = "lightgbm_stress"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self._model = None
        self._feature_names: List[str] = []
        self._params = params or {}

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        from src.models.stress_classifier import train_final_model, compute_shap_importance

        do_grid_search = kwargs.get("do_grid_search", False)
        feature_names = kwargs.get("feature_names", [])

        self._model = train_final_model(X, y, do_grid_search=do_grid_search)
        self._feature_names = feature_names

        # Compute basic training metrics
        y_pred = self._model.predict(X)
        from sklearn.metrics import accuracy_score, f1_score
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
        }

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        X = np.nan_to_num(X, nan=0.0)
        return self._model.predict_proba(X)[:, 1]

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info("Saved LightGBM stress model to %s", path)
        return path

    def load(self, path: str):
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        logger.info("Loaded LightGBM stress model from %s", path)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if self._model is None or not hasattr(self._model, "feature_importances_"):
            return None
        importances = self._model.feature_importances_
        names = self._feature_names or [f"f{i}" for i in range(len(importances))]
        return dict(zip(names[:len(importances)], importances.tolist()))


class LightGBMPerfModel(BaseModel):
    """
    LightGBM-based performance regression model.

    Predicts a performance degradation score ∈ [0, 1].
    """

    name = "lightgbm_perf"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self._model = None
        self._feature_names: List[str] = []
        self._params = params or {}

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        from src.models.performance_model import train_performance_model

        do_grid_search = kwargs.get("do_grid_search", False)
        feature_names = kwargs.get("feature_names", [])

        self._model = train_performance_model(X, y, do_grid_search=do_grid_search)
        self._feature_names = feature_names

        # Compute basic training metrics
        y_pred = self._model.predict(X)
        from sklearn.metrics import mean_squared_error, r2_score
        metrics = {
            "mse": float(mean_squared_error(y, y_pred)),
            "r2": float(r2_score(y, y_pred)),
            "n_samples": int(len(y)),
        }

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        X = np.nan_to_num(X, nan=0.0)
        preds = self._model.predict(X)
        return np.clip(preds, 0.0, 1.0)

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info("Saved LightGBM performance model to %s", path)
        return path

    def load(self, path: str):
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        logger.info("Loaded LightGBM performance model from %s", path)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if self._model is None or not hasattr(self._model, "feature_importances_"):
            return None
        importances = self._model.feature_importances_
        names = self._feature_names or [f"f{i}" for i in range(len(importances))]
        return dict(zip(names[:len(importances)], importances.tolist()))
