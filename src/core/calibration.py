"""
Calibration System
===================
Per-pilot baseline calibration workflow.

Collects baseline data, computes personalised feature statistics
and decision thresholds, and persists pilot profiles to JSON
for later retrieval.

Usage
-----
>>> session = CalibrationSession(pilot_id="CPT_SMITH")
>>> profile = session.calibrate(
...     features=baseline_features,
...     models={"stress": stress_model},
...     fusion=fusion_engine,
...     threshold_strategy=threshold,
... )
>>> session.save()
>>> # Later ...
>>> loaded = CalibrationSession.load_profile("CPT_SMITH")
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.base import BaseFusionEngine, BaseModel, BaseThresholdStrategy

logger = logging.getLogger("pilot_readiness.calibration")


@dataclass
class PilotProfile:
    """Stores per-pilot calibration parameters."""
    pilot_id: str
    calibration_timestamp: str = ""
    n_baseline_windows: int = 0
    feature_means: Dict[str, float] = field(default_factory=dict)
    feature_stds: Dict[str, float] = field(default_factory=dict)
    threshold: float = 0.5
    baseline_risk_mean: float = 0.0
    baseline_risk_std: float = 0.0
    alpha: float = 0.05
    metadata: Dict[str, Any] = field(default_factory=dict)


class CalibrationSession:
    """
    Manages per-pilot calibration workflow.

    Steps
    -----
    1. Ingest baseline feature vectors (from a calm/rested session).
    2. Compute personalized feature means and standard deviations.
    3. Run models to get baseline risk score distribution.
    4. Compute a personalized decision threshold.
    5. Save the profile to disk as JSON.
    """

    def __init__(
        self,
        pilot_id: str,
        profile_dir: str = "output/profiles",
    ):
        self.pilot_id = pilot_id
        self.profile_dir = profile_dir
        self.profile: Optional[PilotProfile] = None

    def calibrate(
        self,
        features: np.ndarray,
        models: Dict[str, BaseModel],
        fusion: Optional[BaseFusionEngine] = None,
        threshold_strategy: Optional[BaseThresholdStrategy] = None,
        feature_names: Optional[List[str]] = None,
        fusion_weights: Optional[Dict[str, float]] = None,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Run the calibration workflow.

        Parameters
        ----------
        features : np.ndarray
            Baseline feature matrix (n_windows × n_features).
        models : dict
            Name → trained BaseModel instances.
        fusion : BaseFusionEngine, optional
        threshold_strategy : BaseThresholdStrategy, optional
        feature_names : list of str, optional
        fusion_weights : dict, optional
        alpha : float
            False alarm rate for threshold computation.

        Returns
        -------
        dict
            Calibration profile as a dictionary.
        """
        import datetime

        logger.info("Calibrating pilot %s with %d baseline windows", self.pilot_id, len(features))

        features = np.nan_to_num(features, nan=0.0)
        f_names = feature_names or [f"f{i}" for i in range(features.shape[1])]

        # Compute feature statistics
        feature_means = {name: float(np.mean(features[:, i]))
                         for i, name in enumerate(f_names) if i < features.shape[1]}
        feature_stds = {name: float(np.std(features[:, i]))
                        for i, name in enumerate(f_names) if i < features.shape[1]}

        # Run models to get baseline risk distribution
        channel_scores: Dict[str, np.ndarray] = {}
        for name, model in models.items():
            try:
                preds = model.predict(features)
                channel = "stress" if "stress" in name else "performance"
                channel_scores[channel] = preds
            except Exception as exc:
                logger.warning("Model %s failed during calibration: %s", name, exc)

        if "performance" not in channel_scores:
            stress = channel_scores.get("stress", np.full(len(features), 0.3))
            channel_scores["performance"] = np.clip(
                stress * 0.6 + np.random.normal(0, 0.1, len(features)), 0, 1
            )

        # Compute risk scores
        weights = fusion_weights or {"stress": 0.6, "performance": 0.4}
        if fusion is not None:
            risk_scores = fusion.fuse_batch(channel_scores, weights=weights)
        else:
            # Simple weighted average fallback
            risk_scores = np.zeros(len(features))
            w_total = sum(weights.values())
            for ch, w in weights.items():
                if ch in channel_scores:
                    risk_scores += (w / w_total) * channel_scores[ch]

        baseline_risk_mean = float(np.mean(risk_scores))
        baseline_risk_std = float(np.std(risk_scores))

        # Compute personalised threshold
        threshold = float(np.quantile(risk_scores, 1.0 - alpha))

        if threshold_strategy is not None:
            threshold = threshold_strategy.compute_threshold(risk_scores, alpha=alpha)

        # Build profile
        self.profile = PilotProfile(
            pilot_id=self.pilot_id,
            calibration_timestamp=datetime.datetime.now().isoformat(),
            n_baseline_windows=len(features),
            feature_means=feature_means,
            feature_stds=feature_stds,
            threshold=threshold,
            baseline_risk_mean=baseline_risk_mean,
            baseline_risk_std=baseline_risk_std,
            alpha=alpha,
        )

        logger.info(
            "Calibration complete: threshold=%.4f, baseline_risk=%.4f±%.4f",
            threshold, baseline_risk_mean, baseline_risk_std,
        )

        return asdict(self.profile)

    def save(self, path: Optional[str] = None) -> str:
        """Save the profile to a JSON file."""
        if self.profile is None:
            raise RuntimeError("No calibration profile to save. Call calibrate() first.")

        if path is None:
            os.makedirs(self.profile_dir, exist_ok=True)
            path = os.path.join(self.profile_dir, f"{self.pilot_id}_profile.json")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self.profile), f, indent=2)

        logger.info("Saved pilot profile to %s", path)
        return path

    @classmethod
    def load_profile(
        cls,
        pilot_id: str,
        profile_dir: str = "output/profiles",
    ) -> PilotProfile:
        """Load a previously saved pilot profile."""
        path = os.path.join(profile_dir, f"{pilot_id}_profile.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No profile found for pilot {pilot_id} at {path}")

        with open(path, "r") as f:
            data = json.load(f)

        return PilotProfile(**data)

    @classmethod
    def list_profiles(cls, profile_dir: str = "output/profiles") -> List[str]:
        """List all available pilot profiles."""
        if not os.path.exists(profile_dir):
            return []
        return [
            f.replace("_profile.json", "")
            for f in os.listdir(profile_dir)
            if f.endswith("_profile.json")
        ]

    def normalize_features(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Z-score normalise features using the pilot's baseline statistics.

        Returns
        -------
        np.ndarray
            Normalised features (same shape as input).
        """
        if self.profile is None:
            raise RuntimeError("No calibration profile loaded.")

        f_names = feature_names or [f"f{i}" for i in range(features.shape[1])]
        normalised = features.copy().astype(float)

        for i, name in enumerate(f_names):
            if i >= features.shape[1]:
                break
            mean = self.profile.feature_means.get(name, 0.0)
            std = self.profile.feature_stds.get(name, 1.0)
            if std > 1e-8:
                normalised[:, i] = (normalised[:, i] - mean) / std

        return normalised
