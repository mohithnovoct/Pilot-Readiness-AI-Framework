"""
Abstract Base Classes
======================
Defines the extension points of the Pilot Readiness AI Framework.

Every component in the pipeline is represented by an ABC.  Users extend
these to plug in custom sensors, feature extractors, models, fusion
strategies, threshold strategies, and alert handlers.

Example
-------
>>> from src.core.base import BaseModel
>>> class MyTransformerModel(BaseModel):
...     name = "transformer_stress"
...     def train(self, X, y, **kw):
...         ...
...     def predict(self, X):
...         return probabilities
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Data Containers ──────────────────────────────────────────────────

@dataclass
class SensorData:
    """Container for raw sensor data returned by a SensorAdapter."""
    signals: Dict[str, np.ndarray]       # channel_name → 1-D signal array
    sampling_rates: Dict[str, float]     # channel_name → Hz
    labels: Optional[np.ndarray] = None  # per-sample ground-truth labels
    metadata: Dict[str, Any] = field(default_factory=dict)
    subject_id: str = "unknown"


@dataclass
class FeatureSet:
    """Container for extracted features."""
    dataframe: pd.DataFrame              # rows = windows, cols = features
    feature_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Container for a single prediction output."""
    risk_score: float
    stress_probability: float
    performance_score: float
    alert_level: str = "NOMINAL"
    decision: str = "READY"
    confidence: float = 1.0
    top_features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertLevel(Enum):
    """Multi-level alert enumeration."""
    NOMINAL = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3


# ── Abstract Base Classes ────────────────────────────────────────────

class BaseSensorAdapter(abc.ABC):
    """
    Abstract adapter for loading or streaming sensor data.

    Built-in implementations: WESADSensor, SWELLSensor, MATBSensor.
    """

    name: str = "base_sensor"

    @abc.abstractmethod
    def load(self, **kwargs) -> SensorData:
        """Load a batch of sensor data (e.g. from a file).

        Returns
        -------
        SensorData
            Raw signals, sampling rates, optional labels.
        """

    def stream(self, callback, **kwargs):
        """
        Stream data in real time, calling *callback* with each new chunk.

        Override this for live sensor adapters (BLE, serial, LSL, …).
        The default raises NotImplementedError so batch-only adapters
        don't need to implement it.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support real-time streaming."
        )

    def validate(self, data: SensorData) -> bool:
        """Optional hook — return False to reject bad data."""
        return True


class BaseFeatureExtractor(abc.ABC):
    """
    Abstract feature extractor.

    Built-in implementations: HRVExtractor, PerformanceExtractor,
    FatigueExtractor.
    """

    name: str = "base_extractor"
    feature_names: List[str] = []

    @abc.abstractmethod
    def extract(self, data: SensorData, **kwargs) -> FeatureSet:
        """Extract features from raw sensor data.

        Parameters
        ----------
        data : SensorData
            Raw signals to process.

        Returns
        -------
        FeatureSet
            Extracted feature matrix.
        """

    def get_feature_names(self) -> List[str]:
        """Return the ordered list of feature names this extractor produces."""
        return list(self.feature_names)


class BaseModel(abc.ABC):
    """
    Abstract model for classification or regression.

    Built-in implementations: LightGBMStressModel, LightGBMPerfModel.
    """

    name: str = "base_model"

    @abc.abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train the model.

        Returns
        -------
        dict
            Training metrics (accuracy, f1, etc.).
        """

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities or scores (shape ``(n,)``)."""

    def save(self, path: str) -> str:
        """Persist model to disk.  Returns the saved path."""
        raise NotImplementedError

    def load(self, path: str):
        """Load model from disk."""
        raise NotImplementedError

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Return feature importances if available."""
        return None


class BaseFusionEngine(abc.ABC):
    """
    Abstract multi-modal score fusion.

    Built-in implementations: WeightedLinearFusion, BayesianFusion.
    """

    name: str = "base_fusion"

    @abc.abstractmethod
    def fuse(
        self,
        scores: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> float:
        """Fuse multiple channel scores into one risk score ∈ [0, 1].

        Parameters
        ----------
        scores : dict
            Channel name → score, e.g. ``{"stress": 0.7, "performance": 0.4}``.
        weights : dict, optional
            Channel name → weight override.

        Returns
        -------
        float
            Fused risk score.
        """

    def fuse_batch(
        self,
        scores: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Vectorised batch fusion (default: loop over ``fuse``).  Override for speed."""
        n = len(next(iter(scores.values())))
        out = np.empty(n)
        for i in range(n):
            row = {k: float(v[i]) for k, v in scores.items()}
            out[i] = self.fuse(row, weights, **kwargs)
        return out


class BaseThresholdStrategy(abc.ABC):
    """
    Abstract decision-threshold strategy.

    Built-in implementations: NeymanPearsonThreshold, AdaptiveThreshold.
    """

    name: str = "base_threshold"

    @abc.abstractmethod
    def compute_threshold(self, baseline_scores: np.ndarray, **kwargs) -> float:
        """Compute the decision threshold from baseline data.

        Returns
        -------
        float
            Threshold γ — alert when risk_score > γ.
        """

    @abc.abstractmethod
    def decide(self, risk_score: float) -> Tuple[AlertLevel, str]:
        """Apply threshold to a single risk score.

        Returns
        -------
        alert_level : AlertLevel
        decision : str
            Human-readable label, e.g. ``"READY"`` / ``"ALERT"``.
        """


class BaseAlertHandler(abc.ABC):
    """
    Abstract alert action handler.

    Subclass to send emails, trigger cockpit audio, log to a SIEM, etc.
    """

    name: str = "base_alert_handler"

    @abc.abstractmethod
    def handle(self, alert_level: AlertLevel, prediction: PredictionResult, **kwargs):
        """Execute the alert action."""

    def should_fire(self, alert_level: AlertLevel) -> bool:
        """Optional filter — return False to skip this handler for a given level."""
        return True
