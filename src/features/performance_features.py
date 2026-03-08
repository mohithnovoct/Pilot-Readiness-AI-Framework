"""
Performance-Proxy Feature Extraction
=====================================
Extracts behavioral metrics from MATB-II task performance data.

Based on the research plan (Section 5.2):
  - CVRT: Coefficient of Variation of Reaction Time
  - Lag-1 Autocorrelation: dependency between consecutive RTs
  - Inceptor Entropy: Shannon Entropy of control inputs
  - RMSD_Track: Root mean square tracking deviation
  - RT distribution shape features (skewness, kurtosis)
"""

import numpy as np
from scipy.stats import skew, kurtosis, entropy as sp_entropy
from typing import Dict, Optional


def compute_rt_features(reaction_times: np.ndarray) -> Dict[str, float]:
    """
    Compute reaction-time-based features.

    Parameters
    ----------
    reaction_times : np.ndarray
        Array of reaction times in seconds.

    Returns
    -------
    dict
        - MeanRT: Mean reaction time (s)
        - MedianRT: Median reaction time (s)
        - StdRT: Std dev of reaction times (s)
        - CVRT: Coefficient of Variation (StdRT / MeanRT)
        - Lag1_Autocorr: Lag-1 autocorrelation
        - RT_Skewness: Skewness of RT distribution
        - RT_Kurtosis: Excess kurtosis of RT distribution
        - Lapse_Rate: Fraction of RTs > 5 seconds
    """
    if len(reaction_times) < 3:
        return _empty_rt_features()

    rt = reaction_times
    mean_rt = np.mean(rt)
    median_rt = np.median(rt)
    std_rt = np.std(rt, ddof=1)

    # CVRT – normalizes variability against speed
    cvrt = std_rt / mean_rt if mean_rt > 1e-10 else np.nan

    # Lag-1 autocorrelation – detects "clumpy" performance degradation
    if len(rt) > 2:
        lag1 = np.corrcoef(rt[:-1], rt[1:])[0, 1]
        if np.isnan(lag1):
            lag1 = 0.0
    else:
        lag1 = 0.0

    # Distribution shape
    rt_skew = skew(rt) if len(rt) > 3 else 0.0
    rt_kurt = kurtosis(rt) if len(rt) > 3 else 0.0

    # Lapse rate (fraction of very slow responses)
    lapse_rate = np.mean(rt > 5.0)

    return {
        "MeanRT": mean_rt,
        "MedianRT": median_rt,
        "StdRT": std_rt,
        "CVRT": cvrt,
        "Lag1_Autocorr": lag1,
        "RT_Skewness": rt_skew,
        "RT_Kurtosis": rt_kurt,
        "Lapse_Rate": lapse_rate,
    }


def _empty_rt_features() -> Dict[str, float]:
    """Return NaN-filled RT features."""
    return {k: np.nan for k in
            ["MeanRT", "MedianRT", "StdRT", "CVRT", "Lag1_Autocorr",
             "RT_Skewness", "RT_Kurtosis", "Lapse_Rate"]}


def compute_tracking_features(rmsd_values: np.ndarray) -> Dict[str, float]:
    """
    Compute tracking task features from RMSD values.

    Parameters
    ----------
    rmsd_values : np.ndarray
        Array of RMSD tracking error values.

    Returns
    -------
    dict
        - MeanRMSD: Mean tracking RMSD
        - StdRMSD: Std dev of tracking RMSD
        - MaxRMSD: Maximum RMSD (worst episode)
        - RMSD_Trend: Slope of RMSD over time (degradation indicator)
        - RMSD_CV: Coefficient of variation of RMSD
    """
    if len(rmsd_values) < 2:
        return _empty_tracking_features()

    mean_rmsd = np.mean(rmsd_values)
    std_rmsd = np.std(rmsd_values, ddof=1)
    max_rmsd = np.max(rmsd_values)

    # Linear trend (positive = degrading)
    x = np.arange(len(rmsd_values), dtype=float)
    if len(rmsd_values) > 2:
        slope = np.polyfit(x, rmsd_values, 1)[0]
    else:
        slope = 0.0

    rmsd_cv = std_rmsd / mean_rmsd if mean_rmsd > 1e-10 else np.nan

    return {
        "MeanRMSD": mean_rmsd,
        "StdRMSD": std_rmsd,
        "MaxRMSD": max_rmsd,
        "RMSD_Trend": slope,
        "RMSD_CV": rmsd_cv,
    }


def _empty_tracking_features() -> Dict[str, float]:
    return {k: np.nan for k in
            ["MeanRMSD", "StdRMSD", "MaxRMSD", "RMSD_Trend", "RMSD_CV"]}


def compute_inceptor_entropy(
    control_inputs: np.ndarray,
    n_bins: int = 20,
) -> float:
    """
    Compute Inceptor Entropy — Shannon Entropy of control input distribution.

    Smooth, predictive control → low entropy.
    Erratic, fatigue-induced corrections → high entropy.

    Parameters
    ----------
    control_inputs : np.ndarray
        Array of joystick positions or tracking errors.
    n_bins : int
        Number of histogram bins for discretization.

    Returns
    -------
    float
        Shannon entropy value (nats).
    """
    if len(control_inputs) < 5:
        return np.nan

    # Histogram of positions
    hist, _ = np.histogram(control_inputs, bins=n_bins, density=True)

    # Remove zeros before computing entropy
    hist = hist[hist > 0]

    if len(hist) == 0:
        return 0.0

    return sp_entropy(hist)


def compute_comm_features(
    response_times: np.ndarray,
    correct: np.ndarray,
    timeout_val: float = 30.0,
) -> Dict[str, float]:
    """
    Compute Communication task features.

    Parameters
    ----------
    response_times : np.ndarray
        Response times in seconds.
    correct : np.ndarray
        Boolean array of correct responses.
    timeout_val : float
        Value indicating a timeout (missed event).

    Returns
    -------
    dict
        - MeanCommRT: Mean comm response time
        - CommAccuracy: Fraction of correct responses
        - TimeoutRate: Fraction of timed-out events
    """
    if len(response_times) == 0:
        return {"MeanCommRT": np.nan, "CommAccuracy": np.nan, "TimeoutRate": np.nan}

    return {
        "MeanCommRT": np.mean(response_times),
        "CommAccuracy": np.mean(correct),
        "TimeoutRate": np.mean(response_times >= timeout_val),
    }


def extract_all_performance_features(
    reaction_times: Optional[np.ndarray] = None,
    tracking_rmsd: Optional[np.ndarray] = None,
    control_inputs: Optional[np.ndarray] = None,
    comm_rts: Optional[np.ndarray] = None,
    comm_correct: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Extract all performance-proxy features from available data.

    Parameters
    ----------
    reaction_times : np.ndarray, optional
        SYSMON reaction times.
    tracking_rmsd : np.ndarray, optional
        TRACK task RMSD values.
    control_inputs : np.ndarray, optional
        Joystick positions (for entropy).
    comm_rts : np.ndarray, optional
        COMM response times.
    comm_correct : np.ndarray, optional
        COMM correctness array.

    Returns
    -------
    dict
        Combined performance feature dictionary.
    """
    features = {}

    if reaction_times is not None and len(reaction_times) > 0:
        features.update(compute_rt_features(reaction_times))

    if tracking_rmsd is not None and len(tracking_rmsd) > 0:
        features.update(compute_tracking_features(tracking_rmsd))

        # Use tracking RMSD as proxy for control inputs if not provided
        inputs = control_inputs if control_inputs is not None else tracking_rmsd
        features["Inceptor_Entropy"] = compute_inceptor_entropy(inputs)

    if comm_rts is not None and comm_correct is not None:
        features.update(compute_comm_features(comm_rts, comm_correct))

    return features
