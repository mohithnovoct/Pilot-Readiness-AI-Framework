"""
Neyman-Pearson Threshold Plugin
================================
Wraps the existing hypothesis-testing threshold engine as a
``BaseThresholdStrategy``.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from src.core.base import AlertLevel, BaseThresholdStrategy

logger = logging.getLogger("pilot_readiness.threshold.np")


class NeymanPearsonThreshold(BaseThresholdStrategy):
    """
    Neyman-Pearson based threshold strategy.

    Selects threshold γ such that:
        P(alert | pilot is Ready) ≤ α

    Supports multi-level alerting via configurable risk-tier boundaries.
    """

    name = "neyman_pearson"

    def __init__(
        self,
        alpha: float = 0.05,
        caution_threshold: float = 0.40,
        warning_threshold: float = 0.60,
        critical_threshold: float = 0.80,
    ):
        self.alpha = alpha
        self._threshold = 0.5  # default until calibrated
        self.caution_threshold = caution_threshold
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def compute_threshold(self, baseline_scores: np.ndarray, **kwargs) -> float:
        """
        Compute γ from the (1-α) quantile of baseline risk scores.
        """
        alpha = kwargs.get("alpha", self.alpha)

        if len(baseline_scores) == 0:
            self._threshold = 0.5
            return self._threshold

        self._threshold = float(np.quantile(baseline_scores, 1.0 - alpha))
        logger.info("Computed NP threshold: γ=%.4f (α=%.3f)", self._threshold, alpha)
        return self._threshold

    def decide(self, risk_score: float) -> Tuple[AlertLevel, str]:
        """
        Multi-level decision:
        - NOMINAL : risk ≤ caution_threshold
        - CAUTION : caution_threshold < risk ≤ warning_threshold
        - WARNING : warning_threshold < risk ≤ critical_threshold
        - CRITICAL: risk > critical_threshold

        Additionally, the binary READY/ALERT boundary is at ``self._threshold``.
        """
        if risk_score > self.critical_threshold:
            return AlertLevel.CRITICAL, "CRITICAL"
        elif risk_score > self.warning_threshold:
            return AlertLevel.WARNING, "WARNING"
        elif risk_score > self.caution_threshold:
            return AlertLevel.CAUTION, "CAUTION"
        else:
            return AlertLevel.NOMINAL, "READY"

    def is_alert(self, risk_score: float) -> bool:
        """Binary alert check against the calibrated NP threshold."""
        return risk_score > self._threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    def set_threshold(self, value: float):
        """Manually override the threshold (e.g. from a UI slider)."""
        self._threshold = float(np.clip(value, 0.01, 0.99))

    def evaluate_performance(
        self,
        baseline_scores: np.ndarray,
        stress_scores: np.ndarray,
    ) -> Dict[str, float]:
        """Compute FAR, DR, precision, F1 at the current threshold."""
        fa = float(np.mean(baseline_scores > self._threshold))
        dr = float(np.mean(stress_scores > self._threshold)) if len(stress_scores) > 0 else 0.0

        n_fa = int(np.sum(baseline_scores > self._threshold))
        n_dr = int(np.sum(stress_scores > self._threshold))
        total_alerts = n_fa + n_dr
        precision = n_dr / total_alerts if total_alerts > 0 else 0.0

        f1 = 2 * dr * precision / (dr + precision) if (dr + precision) > 0 else 0.0

        return {
            "false_alarm_rate": fa,
            "detection_rate": dr,
            "precision": precision,
            "f1": f1,
            "threshold": self._threshold,
        }
