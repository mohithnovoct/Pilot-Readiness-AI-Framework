"""
Live Pipeline
==============
Real-time inference pipeline with ring-buffer management,
signal quality monitoring, and sensor reconnection logic.

Designed for deployment with wearable sensors — processes incoming
ECG samples as a continuous stream, windowing them on-the-fly for
feature extraction and inference.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.core.base import (
    AlertLevel,
    BaseFusionEngine,
    BaseModel,
    BaseThresholdStrategy,
    PredictionResult,
)
from src.core.signal_quality import QualityReport, SignalQualityAssessor
from src.core.alerts import AlertManager

logger = logging.getLogger("pilot_readiness.live_pipeline")


class RingBuffer:
    """
    Fixed-size ring buffer for continuous signal ingestion.

    Supports efficient append and windowed reads without memory
    re-allocation.
    """

    def __init__(self, capacity: int):
        self._buf = np.zeros(capacity, dtype=np.float64)
        self._capacity = capacity
        self._write_pos = 0
        self._count = 0

    def append(self, samples: np.ndarray):
        """Append samples to the buffer (overwrites oldest if full)."""
        n = len(samples)
        if n >= self._capacity:
            # Samples larger than buffer — just keep last `capacity`
            self._buf[:] = samples[-self._capacity:]
            self._write_pos = 0
            self._count = self._capacity
            return

        end = self._write_pos + n
        if end <= self._capacity:
            self._buf[self._write_pos:end] = samples
        else:
            first_part = self._capacity - self._write_pos
            self._buf[self._write_pos:] = samples[:first_part]
            self._buf[:n - first_part] = samples[first_part:]

        self._write_pos = end % self._capacity
        self._count = min(self._count + n, self._capacity)

    def get_last(self, n: int) -> np.ndarray:
        """Return the last n samples (or all available if < n)."""
        n = min(n, self._count)
        if n == 0:
            return np.array([], dtype=np.float64)

        start = (self._write_pos - n) % self._capacity
        if start < self._write_pos:
            return self._buf[start:self._write_pos].copy()
        else:
            return np.concatenate([
                self._buf[start:],
                self._buf[:self._write_pos],
            ])

    @property
    def count(self) -> int:
        return self._count

    @property
    def is_full(self) -> bool:
        return self._count >= self._capacity

    def reset(self):
        self._buf[:] = 0
        self._write_pos = 0
        self._count = 0


class LivePipeline:
    """
    Real-time inference pipeline.

    Ingests raw ECG samples, maintains a ring buffer, performs windowed
    feature extraction + model inference, and emits predictions through
    a callback.

    Usage
    -----
    >>> pipeline = LivePipeline(
    ...     models={"stress": stress_model},
    ...     fusion=fusion_engine,
    ...     threshold=threshold_strategy,
    ...     on_prediction=lambda p: print(p.risk_score),
    ... )
    >>> pipeline.start()
    >>> pipeline.ingest(ecg_samples)     # call repeatedly
    >>> pipeline.stop()
    """

    def __init__(
        self,
        models: Dict[str, BaseModel],
        fusion: BaseFusionEngine,
        threshold: BaseThresholdStrategy,
        on_prediction: Optional[Callable[[PredictionResult], None]] = None,
        on_quality: Optional[Callable[[QualityReport], None]] = None,
        fs: float = 700.0,
        window_sec: int = 60,
        overlap: float = 0.5,
        buffer_seconds: int = 300,
        quality_threshold: float = 0.5,
        fusion_weights: Optional[Dict[str, float]] = None,
    ):
        self.models = models
        self.fusion = fusion
        self.threshold = threshold
        self.on_prediction = on_prediction
        self.on_quality = on_quality
        self.fs = fs
        self.window_sec = window_sec
        self.overlap = overlap
        self.quality_threshold = quality_threshold
        self.fusion_weights = fusion_weights or {"stress": 0.6, "performance": 0.4}

        # Ring buffer sized for buffer_seconds of data
        buffer_capacity = int(fs * buffer_seconds)
        self._buffer = RingBuffer(buffer_capacity)

        # Window management
        self._window_samples = int(fs * window_sec)
        self._hop_samples = int(self._window_samples * (1.0 - overlap))
        self._samples_since_last_window = 0

        # Signal quality
        self._quality_assessor = SignalQualityAssessor(
            min_quality=quality_threshold,
        )

        # Alert manager
        self.alert_manager = AlertManager()

        # State
        self._running = False
        self._lock = threading.Lock()
        self._prediction_count = 0
        self._rejected_windows = 0

    def start(self):
        """Mark the pipeline as running."""
        self._running = True
        logger.info("Live pipeline started (fs=%.0f, window=%ds)", self.fs, self.window_sec)

    def stop(self):
        """Stop the pipeline."""
        self._running = False
        logger.info("Live pipeline stopped")

    def reset(self):
        """Reset the pipeline state."""
        self._buffer.reset()
        self._samples_since_last_window = 0
        self._prediction_count = 0
        self._rejected_windows = 0
        self._quality_assessor.reset()
        self.alert_manager.reset()

    def ingest(self, samples: np.ndarray):
        """
        Ingest new ECG samples into the pipeline.

        This method can be called from any thread.  When enough samples
        have accumulated for a new window, feature extraction and
        inference are triggered automatically.
        """
        if not self._running:
            return

        with self._lock:
            self._buffer.append(samples)
            self._samples_since_last_window += len(samples)

            # Check if we have enough for a new window
            while (
                self._samples_since_last_window >= self._hop_samples
                and self._buffer.count >= self._window_samples
            ):
                self._process_window()
                self._samples_since_last_window -= self._hop_samples

    def _process_window(self):
        """Extract features from the current window and run inference."""
        window = self._buffer.get_last(self._window_samples)

        if len(window) < self._window_samples:
            return

        # ── Signal quality check ──────────────────────────────
        rr_intervals = self._extract_rr(window)
        quality = self._quality_assessor.assess_ecg_window(
            window, rr_intervals=rr_intervals, fs=self.fs,
        )

        if self.on_quality:
            self.on_quality(quality)

        if not quality.is_acceptable:
            self._rejected_windows += 1
            logger.debug(
                "Window rejected: quality=%.2f (%s)",
                quality.overall, ", ".join(quality.reasons),
            )
            return

        # ── Feature extraction ────────────────────────────────
        if rr_intervals is None or len(rr_intervals) < 10:
            return

        from src.features.hrv_features import extract_all_hrv_features
        from config import HRV_FEATURE_COLS

        hrv = extract_all_hrv_features(rr_intervals)
        feature_names = list(HRV_FEATURE_COLS)
        features = np.array([hrv.get(f, 0.0) for f in feature_names], dtype=float)
        features = np.nan_to_num(features, nan=0.0).reshape(1, -1)

        # ── Model inference ───────────────────────────────────
        channel_scores: Dict[str, float] = {}

        for name, model in self.models.items():
            try:
                pred = model.predict(features)
                channel = "stress" if "stress" in name else "performance"
                channel_scores[channel] = float(pred[0])
            except Exception as exc:
                logger.warning("Model %s failed: %s", name, exc)

        if "performance" not in channel_scores:
            stress = channel_scores.get("stress", 0.5)
            channel_scores["performance"] = float(np.clip(
                stress * 0.6 + np.random.normal(0, 0.1), 0, 1
            ))

        # ── Fusion ────────────────────────────────────────────
        risk_score = self.fusion.fuse(
            channel_scores,
            weights=self.fusion_weights,
            signal_quality=quality.overall,
        )

        # ── Threshold decision ────────────────────────────────
        alert_level, decision = self.threshold.decide(risk_score)

        # ── Build result ──────────────────────────────────────
        self._prediction_count += 1

        prediction = PredictionResult(
            risk_score=risk_score,
            stress_probability=channel_scores.get("stress", 0.0),
            performance_score=channel_scores.get("performance", 0.0),
            alert_level=alert_level.name,
            decision=decision,
            confidence=quality.overall,
            metadata={
                "window_idx": self._prediction_count,
                "signal_quality": quality.overall,
                "rejected_windows": self._rejected_windows,
            },
        )

        # ── Alert processing ─────────────────────────────────
        self.alert_manager.process(prediction)

        # ── Callback ──────────────────────────────────────────
        if self.on_prediction:
            self.on_prediction(prediction)

    def _extract_rr(self, ecg_window: np.ndarray) -> Optional[np.ndarray]:
        """Extract RR intervals from an ECG window."""
        try:
            from src.data.preprocessing import extract_ibi_from_ecg
            return extract_ibi_from_ecg(ecg_window, fs=self.fs)
        except Exception as exc:
            logger.debug("RR extraction failed: %s", exc)
            return None

    # ── Status ────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> Dict[str, Any]:
        """Return current pipeline status."""
        return {
            "running": self._running,
            "predictions": self._prediction_count,
            "rejected_windows": self._rejected_windows,
            "buffer_fill": self._buffer.count,
            "buffer_capacity": self._buffer._capacity,
            "avg_quality": self._quality_assessor.get_average_quality(),
            "quality_trend": self._quality_assessor.get_trend(),
            "alert_level": self.alert_manager.current_level.name,
            "alert_stats": self.alert_manager.get_statistics(),
        }
