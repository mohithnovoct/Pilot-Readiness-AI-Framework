"""
Risk Fusion Engine
==================
Fuses physiological (HRV-based stress) and behavioral (performance degradation)
model outputs into a unified Continuous Readiness Risk Score.

Weighted Fusion Equation (Research Plan Section 7.1):
    R = w_phys × P_stress + w_perf × P_perf

Where:
    P_stress : Probability of stress from the HRV model (0-1)
    P_perf   : Normalized performance deviation score (0-1)
    w_phys, w_perf : Dynamic weights based on signal quality
"""

import numpy as np
from typing import Optional, Tuple


def compute_risk_score(
    p_stress: float,
    p_perf: float,
    w_phys: float = 0.6,
    w_perf: float = 0.4,
    signal_quality: Optional[float] = None,
) -> float:
    """
    Compute the fused Continuous Readiness Risk Score.

    Parameters
    ----------
    p_stress : float
        Stress probability from the physiological model ∈ [0, 1].
    p_perf : float
        Performance deviation score ∈ [0, 1].
    w_phys : float
        Base weight for physiological channel.
    w_perf : float
        Base weight for performance channel.
    signal_quality : float, optional
        ECG signal quality metric ∈ [0, 1]. If low, shifts weight
        from physiological to behavioral channel.

    Returns
    -------
    float
        Risk score ∈ [0, 1]. Higher = more at risk.
    """
    # Dynamic weighting based on signal quality
    if signal_quality is not None:
        # Low signal quality → reduce phys weight
        sq = np.clip(signal_quality, 0.0, 1.0)
        w_phys_adj = w_phys * sq
        w_perf_adj = w_perf + w_phys * (1 - sq) * 0.5
    else:
        w_phys_adj = w_phys
        w_perf_adj = w_perf

    # Normalize weights
    w_total = w_phys_adj + w_perf_adj
    if w_total > 0:
        w_phys_adj /= w_total
        w_perf_adj /= w_total

    # Fuse
    risk = w_phys_adj * np.clip(p_stress, 0, 1) + w_perf_adj * np.clip(p_perf, 0, 1)

    return float(np.clip(risk, 0.0, 1.0))


def compute_risk_batch(
    p_stress_arr: np.ndarray,
    p_perf_arr: np.ndarray,
    w_phys: float = 0.6,
    w_perf: float = 0.4,
    signal_quality_arr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Batch compute risk scores for arrays of predictions.

    Parameters
    ----------
    p_stress_arr : np.ndarray
        Array of stress probabilities.
    p_perf_arr : np.ndarray
        Array of performance deviation scores.
    w_phys, w_perf : float
        Base weights.
    signal_quality_arr : np.ndarray, optional
        Signal quality per sample.

    Returns
    -------
    np.ndarray
        Risk scores ∈ [0, 1].
    """
    n = len(p_stress_arr)
    risks = np.zeros(n)

    for i in range(n):
        sq = signal_quality_arr[i] if signal_quality_arr is not None else None
        risks[i] = compute_risk_score(
            p_stress_arr[i], p_perf_arr[i], w_phys, w_perf, sq
        )

    return risks


def estimate_signal_quality(rr_intervals: np.ndarray) -> float:
    """
    Estimate ECG signal quality from RR interval characteristics.

    A simple heuristic based on:
      - Proportion of physiologically plausible intervals
      - Consistency (low std relative to mean)

    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals in seconds.

    Returns
    -------
    float
        Signal quality score ∈ [0, 1].
    """
    if len(rr_intervals) < 5:
        return 0.0

    # Check plausibility range (0.3-2.0 seconds = 30-200 bpm)
    in_range = np.mean((rr_intervals >= 0.3) & (rr_intervals <= 2.0))

    # Check consistency (CV should be reasonable: 0.02-0.30)
    cv = np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 1.0
    cv_quality = 1.0 if 0.02 < cv < 0.30 else max(0.0, 1.0 - abs(cv - 0.15) / 0.5)

    # Combined quality
    quality = 0.6 * in_range + 0.4 * cv_quality

    return float(np.clip(quality, 0.0, 1.0))


def get_risk_category(risk_score: float) -> Tuple[str, str]:
    """
    Categorize a risk score into operational risk levels.

    Returns
    -------
    category : str
        'LOW', 'MODERATE', 'ELEVATED', or 'HIGH'
    color : str
        Suggested display color.
    """
    if risk_score < 0.25:
        return "LOW", "#22c55e"          # Green
    elif risk_score < 0.50:
        return "MODERATE", "#eab308"     # Yellow
    elif risk_score < 0.75:
        return "ELEVATED", "#f97316"     # Orange
    else:
        return "HIGH", "#ef4444"         # Red
