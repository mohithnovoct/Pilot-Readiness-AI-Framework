"""
Neyman-Pearson Tunable Threshold Engine
========================================
Implements the hypothesis-testing framework for adaptive alarm thresholds.

Based on research plan Section 7.2:
  - H0: Pilot is "Ready" (null hypothesis)
  - H1: Pilot is "Not Ready" (alternative)
  - α: Maximum acceptable false alarm rate (user-defined)
  - The system selects threshold γ such that P(alarm | ready) ≤ α

Scenario Presets:
  - Training:    α = 10%  (catch every potential stressor)
  - Operational: α = 5%   (balanced sensitivity)
  - Critical:    α = 1%   (only definitive risks)
"""

import numpy as np
from typing import Tuple, Optional, Dict


# Predefined scenario presets
SCENARIO_PRESETS = {
    "training": {"alpha": 0.10, "description": "High sensitivity — training missions"},
    "operational": {"alpha": 0.05, "description": "Balanced — standard operations"},
    "critical": {"alpha": 0.01, "description": "Low false-alarm — critical missions"},
}


def compute_threshold(
    risk_scores_ready: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """
    Compute the optimal detection threshold using the Neyman-Pearson principle.

    The threshold γ is chosen such that:
        P(Risk > γ | pilot is Ready) ≤ α

    This corresponds to finding the (1 - α) quantile of the "ready" distribution.

    Parameters
    ----------
    risk_scores_ready : np.ndarray
        Historical risk scores from known "Ready" (baseline) conditions.
    alpha : float
        Maximum acceptable false alarm rate ∈ (0, 1).

    Returns
    -------
    float
        Threshold γ — alert when risk_score > γ.
    """
    if len(risk_scores_ready) == 0:
        return 0.5  # Default fallback

    # Threshold = (1 - α) quantile of the "ready" risk distribution
    gamma = np.quantile(risk_scores_ready, 1.0 - alpha)

    return float(gamma)


def apply_threshold(
    risk_score: float,
    threshold: float,
) -> Tuple[bool, str]:
    """
    Apply the threshold to a risk score.

    Parameters
    ----------
    risk_score : float
        Current risk score ∈ [0, 1].
    threshold : float
        Decision threshold γ.

    Returns
    -------
    alert : bool
        True if risk_score exceeds threshold (potential unreadiness).
    decision : str
        "READY" or "ALERT".
    """
    if risk_score > threshold:
        return True, "ALERT"
    return False, "READY"


def evaluate_threshold_performance(
    risk_scores_ready: np.ndarray,
    risk_scores_not_ready: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Evaluate the detection performance at a given threshold.

    Parameters
    ----------
    risk_scores_ready : np.ndarray
        Risk scores from "Ready" conditions (baseline).
    risk_scores_not_ready : np.ndarray
        Risk scores from "Not Ready" conditions (stress/fatigue).
    threshold : float
        Decision threshold γ.

    Returns
    -------
    dict
        - false_alarm_rate: P(alert | Ready) — should be ≤ α
        - detection_rate: P(alert | Not Ready) — want this high
        - precision: P(Not Ready | alert)
        - f1: Harmonic mean of detection_rate and precision
        - threshold: The threshold value
    """
    fa = np.mean(risk_scores_ready > threshold)
    dr = np.mean(risk_scores_not_ready > threshold) if len(risk_scores_not_ready) > 0 else 0.0

    # Precision
    n_alerts_ready = np.sum(risk_scores_ready > threshold)
    n_alerts_not_ready = np.sum(risk_scores_not_ready > threshold)
    total_alerts = n_alerts_ready + n_alerts_not_ready
    precision = n_alerts_not_ready / total_alerts if total_alerts > 0 else 0.0

    # F1
    if (dr + precision) > 0:
        f1 = 2 * (dr * precision) / (dr + precision)
    else:
        f1 = 0.0

    return {
        "false_alarm_rate": float(fa),
        "detection_rate": float(dr),
        "precision": float(precision),
        "f1": float(f1),
        "threshold": float(threshold),
    }


def compute_roc_curve(
    risk_scores_ready: np.ndarray,
    risk_scores_not_ready: np.ndarray,
    n_thresholds: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve by sweeping thresholds.

    Returns
    -------
    false_alarm_rates : np.ndarray
        False alarm rates at each threshold.
    detection_rates : np.ndarray
        Detection (true positive) rates at each threshold.
    thresholds : np.ndarray
        Threshold values.
    """
    all_scores = np.concatenate([risk_scores_ready, risk_scores_not_ready])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), n_thresholds)

    fars = np.array([np.mean(risk_scores_ready > t) for t in thresholds])
    drs = np.array([np.mean(risk_scores_not_ready > t) for t in thresholds])

    return fars, drs, thresholds


class ThresholdManager:
    """
    Manages tunable thresholds with scenario presets and a sensitivity slider.
    """

    def __init__(
        self,
        baseline_scores: Optional[np.ndarray] = None,
        alpha: float = 0.05,
    ):
        self.baseline_scores = baseline_scores
        self.alpha = alpha
        self.threshold = 0.5

        if baseline_scores is not None and len(baseline_scores) > 0:
            self.threshold = compute_threshold(baseline_scores, alpha)

    def set_alpha(self, alpha: float):
        """Update the false-alarm constraint and recompute threshold."""
        self.alpha = np.clip(alpha, 0.001, 0.50)
        if self.baseline_scores is not None:
            self.threshold = compute_threshold(self.baseline_scores, self.alpha)
        return self.threshold

    def set_scenario(self, scenario: str) -> float:
        """Set a predefined scenario preset."""
        if scenario not in SCENARIO_PRESETS:
            raise ValueError(f"Unknown scenario: {scenario}. "
                           f"Choose from {list(SCENARIO_PRESETS.keys())}")
        return self.set_alpha(SCENARIO_PRESETS[scenario]["alpha"])

    def evaluate(self, risk_score: float) -> Tuple[bool, str, float]:
        """
        Evaluate a risk score against the current threshold.

        Returns
        -------
        alert : bool
        decision : str
        threshold : float
        """
        alert, decision = apply_threshold(risk_score, self.threshold)
        return alert, decision, self.threshold

    def get_sensitivity_range(self) -> Dict:
        """Get the range of alpha values and their corresponding thresholds."""
        if self.baseline_scores is None:
            return {}

        alphas = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
        return {
            a: compute_threshold(self.baseline_scores, a)
            for a in alphas
        }
