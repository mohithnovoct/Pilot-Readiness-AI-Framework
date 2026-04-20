"""
Framework Orchestrator
=======================
The main ``PilotReadinessFramework`` class — the single entry point for
using the framework programmatically.

Usage
-----
>>> from src.core.framework import PilotReadinessFramework
>>> fw = PilotReadinessFramework()
>>> fw.configure({"dataset": "wesad", "fusion": {"weights": {"stress": 0.7}}})
>>> results = fw.fit(data_dir="Data/WESAD")
>>> prediction = fw.predict(features)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.core.base import (
    AlertLevel,
    BaseAlertHandler,
    BaseFeatureExtractor,
    BaseFusionEngine,
    BaseModel,
    BaseSensorAdapter,
    BaseThresholdStrategy,
    FeatureSet,
    PredictionResult,
    SensorData,
)
from src.core.config_schema import FrameworkConfig
from src.core.registry import ComponentRegistry, registry

logger = logging.getLogger("pilot_readiness.framework")


class PilotReadinessFramework:
    """
    High-level orchestrator for the Pilot Readiness AI pipeline.

    Coordinates sensors → feature extraction → model inference →
    risk fusion → threshold decision → alerting.
    """

    def __init__(self, config: Optional[Union[str, dict, FrameworkConfig]] = None):
        self._config: FrameworkConfig = FrameworkConfig()
        self._registry: ComponentRegistry = registry
        self._fitted = False

        # Active component instances
        self._sensors: List[BaseSensorAdapter] = []
        self._extractors: List[BaseFeatureExtractor] = []
        self._models: Dict[str, BaseModel] = {}
        self._fusion: Optional[BaseFusionEngine] = None
        self._threshold: Optional[BaseThresholdStrategy] = None
        self._alert_handlers: List[BaseAlertHandler] = []

        # Cached training artefacts
        self._baseline_scores: Optional[np.ndarray] = None
        self._training_metrics: Dict[str, Any] = {}

        # Auto-discover built-in plugins
        self._registry.auto_discover()

        # Apply config if provided
        if config is not None:
            self.configure(config)

    # ── Configuration ────────────────────────────────────────────

    def configure(self, config: Union[str, dict, FrameworkConfig]) -> "PilotReadinessFramework":
        """
        Configure the framework.

        Parameters
        ----------
        config : str | dict | FrameworkConfig
            A YAML file path, a dict, or a FrameworkConfig object.

        Returns
        -------
        self
            For fluent chaining.
        """
        if isinstance(config, str):
            self._config = FrameworkConfig.from_yaml(config)
        elif isinstance(config, dict):
            self._config = FrameworkConfig.from_dict(config)
        elif isinstance(config, FrameworkConfig):
            self._config = config
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        self._resolve_components()
        logger.info("Framework configured: dataset=%s", self._config.dataset)
        return self

    def _resolve_components(self):
        """Instantiate components from the current config."""
        # Sensors
        self._sensors = []
        for sc in self._config.sensors:
            if self._registry.has("sensor", sc.name):
                sensor = self._registry.get("sensor", sc.name)
                self._sensors.append(sensor)

        # Extractors
        self._extractors = []
        for ec in self._config.extractors:
            if self._registry.has("extractor", ec.name):
                extractor = self._registry.get("extractor", ec.name)
                self._extractors.append(extractor)

        # Models
        self._models = {}
        for mc in self._config.models:
            if self._registry.has("model", mc.name):
                model = self._registry.get("model", mc.name)
                self._models[mc.name] = model

        # Fusion
        if self._registry.has("fusion", self._config.fusion.name):
            self._fusion = self._registry.get("fusion", self._config.fusion.name)

        # Threshold
        if self._registry.has("threshold", self._config.threshold.name):
            self._threshold = self._registry.get("threshold", self._config.threshold.name)

    # ── Registration (programmatic) ──────────────────────────────

    def register_sensor(self, sensor: BaseSensorAdapter) -> "PilotReadinessFramework":
        """Register a custom sensor adapter."""
        self._registry.register("sensor", sensor)
        self._sensors.append(sensor)
        return self

    def register_extractor(self, extractor: BaseFeatureExtractor) -> "PilotReadinessFramework":
        """Register a custom feature extractor."""
        self._registry.register("extractor", extractor)
        self._extractors.append(extractor)
        return self

    def register_model(self, model: BaseModel) -> "PilotReadinessFramework":
        """Register a custom model."""
        self._registry.register("model", model)
        self._models[model.name] = model
        return self

    def register_fusion(self, fusion: BaseFusionEngine) -> "PilotReadinessFramework":
        """Register a custom fusion engine."""
        self._registry.register("fusion", fusion)
        self._fusion = fusion
        return self

    def register_threshold(self, threshold: BaseThresholdStrategy) -> "PilotReadinessFramework":
        """Register a custom threshold strategy."""
        self._registry.register("threshold", threshold)
        self._threshold = threshold
        return self

    def register_alert_handler(self, handler: BaseAlertHandler) -> "PilotReadinessFramework":
        """Register a custom alert handler."""
        self._registry.register("alert_handler", handler)
        self._alert_handlers.append(handler)
        return self

    # ── Training ─────────────────────────────────────────────────

    def fit(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        features_df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train all registered models.

        Can accept either raw arrays (X, y) or a DataFrame with
        feature column names.

        Returns
        -------
        dict
            Training metrics per model.
        """
        start = time.time()
        results: Dict[str, Any] = {}

        if features_df is not None:
            # Train from DataFrame
            if feature_cols is None:
                feature_cols = [c for c in features_df.columns
                                if c not in ("subject_id", "label", "label_name",
                                             "window_idx", "stress_label", "Condition",
                                             "timestamp")]
            X = features_df[feature_cols].values.astype(float)
            X = np.nan_to_num(X, nan=0.0)

            if "stress_label" in features_df.columns:
                y = features_df["stress_label"].values
            elif "label" in features_df.columns:
                y = (features_df["label"] == 2).astype(int).values

        if X is None or y is None:
            raise ValueError("Provide either (X, y) or (features_df, feature_cols).")

        for name, model in self._models.items():
            logger.info("Training model: %s", name)
            try:
                metrics = model.train(X, y, **kwargs)
                results[name] = metrics
                logger.info("  %s metrics: %s", name, metrics)
            except Exception as exc:
                logger.error("  Failed to train %s: %s", name, exc)
                results[name] = {"error": str(exc)}

        # Compute baseline scores for threshold calibration
        if self._fusion and self._threshold:
            self._calibrate_threshold(X, y)

        self._training_metrics = results
        self._fitted = True
        elapsed = time.time() - start
        logger.info("Training complete in %.1fs", elapsed)
        return results

    def _calibrate_threshold(self, X: np.ndarray, y: np.ndarray):
        """Compute baseline risk scores and calibrate the threshold."""
        try:
            scores: Dict[str, np.ndarray] = {}
            for name, model in self._models.items():
                preds = model.predict(X)
                channel = "stress" if "stress" in name else "performance"
                scores[channel] = preds

            if "performance" not in scores:
                # Synthesise a correlated performance channel
                stress = scores.get("stress", np.zeros(len(X)))
                scores["performance"] = np.clip(
                    stress * 0.6 + np.random.normal(0, 0.1, len(X)), 0, 1
                )

            risk_scores = self._fusion.fuse_batch(
                scores, weights=self._config.fusion.weights
            )
            baseline_mask = y == 0
            baseline_risks = risk_scores[baseline_mask]
            if len(baseline_risks) > 0:
                self._baseline_scores = baseline_risks
                self._threshold.compute_threshold(
                    baseline_risks, alpha=self._config.threshold.alpha
                )
                logger.info(
                    "Threshold calibrated: γ=%.4f (α=%.3f)",
                    self._threshold._threshold,
                    self._config.threshold.alpha,
                )
        except Exception as exc:
            logger.warning("Threshold calibration failed: %s", exc)

    # ── Inference ────────────────────────────────────────────────

    def predict(
        self,
        X: Optional[np.ndarray] = None,
        features_df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> List[PredictionResult]:
        """
        Run batch inference and return a list of PredictionResult objects.
        """
        if features_df is not None:
            if feature_cols is None:
                feature_cols = [c for c in features_df.columns
                                if c not in ("subject_id", "label", "label_name",
                                             "window_idx", "stress_label", "Condition",
                                             "timestamp")]
            X = features_df[feature_cols].values.astype(float)
            X = np.nan_to_num(X, nan=0.0)

        if X is None:
            raise ValueError("Provide either X or features_df.")

        # Get predictions from all models
        channel_scores: Dict[str, np.ndarray] = {}
        for name, model in self._models.items():
            channel = "stress" if "stress" in name else "performance"
            try:
                channel_scores[channel] = model.predict(X)
            except Exception as exc:
                logger.warning("Model %s failed: %s", name, exc)
                channel_scores[channel] = np.full(len(X), 0.5)

        if "performance" not in channel_scores:
            stress = channel_scores.get("stress", np.full(len(X), 0.5))
            channel_scores["performance"] = np.clip(
                stress * 0.6 + np.random.normal(0, 0.1, len(X)), 0, 1
            )

        # Fuse
        if self._fusion:
            risk_scores = self._fusion.fuse_batch(
                channel_scores, weights=self._config.fusion.weights
            )
        else:
            risk_scores = channel_scores.get("stress", np.full(len(X), 0.5))

        # Package results
        results = []
        for i in range(len(X)):
            stress_p = float(channel_scores.get("stress", np.zeros(1))[min(i, len(channel_scores.get("stress", [0]))-1)])
            perf_p = float(channel_scores.get("performance", np.zeros(1))[min(i, len(channel_scores.get("performance", [0]))-1)])

            alert_level = AlertLevel.NOMINAL
            decision = "READY"

            if self._threshold:
                alert_level, decision = self._threshold.decide(float(risk_scores[i]))

            results.append(PredictionResult(
                risk_score=float(risk_scores[i]),
                stress_probability=stress_p,
                performance_score=perf_p,
                alert_level=alert_level.name,
                decision=decision,
            ))

        return results

    def predict_single(self, x: np.ndarray) -> PredictionResult:
        """Convenience method for single-sample inference."""
        results = self.predict(X=x.reshape(1, -1))
        return results[0]

    # ── Calibration ──────────────────────────────────────────────

    def calibrate(
        self,
        pilot_id: str,
        baseline_features: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run a calibration session for a specific pilot.

        Computes per-pilot baseline statistics and adjusts the
        decision threshold.
        """
        from src.core.calibration import CalibrationSession

        session = CalibrationSession(
            pilot_id=pilot_id,
            profile_dir=self._config.calibration.profile_dir,
        )

        profile = session.calibrate(
            features=baseline_features,
            models=self._models,
            fusion=self._fusion,
            threshold_strategy=self._threshold,
            fusion_weights=self._config.fusion.weights,
            alpha=self._config.threshold.alpha,
        )

        logger.info("Calibrated pilot %s: threshold=%.4f", pilot_id, profile["threshold"])
        return profile

    # ── Edge Export ───────────────────────────────────────────────

    def export(self, format: str = "all", output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Export models for edge deployment.

        Parameters
        ----------
        format : str
            ``"c"``, ``"python"``, ``"onnx"``, or ``"all"``.
        output_dir : str, optional
            Output directory.  Defaults to ``config.edge.output_dir``.

        Returns
        -------
        dict
            Mapping format → exported file path.
        """
        out = output_dir or self._config.edge.output_dir
        os.makedirs(out, exist_ok=True)
        exported: Dict[str, str] = {}

        for name, model in self._models.items():
            if format in ("c", "all") and self._config.edge.export_c:
                try:
                    from src.edge.export_model import export_to_c_code
                    path = os.path.join(out, f"{name}.c")
                    export_to_c_code(model._model, path)
                    exported[f"{name}_c"] = path
                except Exception as exc:
                    logger.warning("C export for %s failed: %s", name, exc)

            if format in ("python", "all") and self._config.edge.export_python:
                try:
                    from src.edge.export_model import export_to_python_code
                    path = os.path.join(out, f"{name}_pure.py")
                    export_to_python_code(model._model, path)
                    exported[f"{name}_python"] = path
                except Exception as exc:
                    logger.warning("Python export for %s failed: %s", name, exc)

            if format in ("onnx", "all") and self._config.edge.export_onnx:
                try:
                    from src.edge.onnx_export import export_to_onnx
                    path = os.path.join(out, f"{name}.onnx")
                    export_to_onnx(model._model, path)
                    exported[f"{name}_onnx"] = path
                except Exception as exc:
                    logger.warning("ONNX export for %s failed: %s", name, exc)

        return exported

    # ── Utilities ────────────────────────────────────────────────

    @property
    def config(self) -> FrameworkConfig:
        return self._config

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def training_metrics(self) -> Dict[str, Any]:
        return self._training_metrics

    def list_components(self) -> Dict[str, List[str]]:
        """List all registered components."""
        return self._registry.list_all()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        n_models = len(self._models)
        return (
            f"PilotReadinessFramework("
            f"dataset={self._config.dataset!r}, "
            f"models={n_models}, "
            f"status={status})"
        )
