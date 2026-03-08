"""
Signal Preprocessing Pipeline
==============================
Bandpass filtering, artifact removal, IBI extraction, and normalization
for ECG and physiological signals from WESAD.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d
from typing import Optional, Tuple, List
import warnings


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to a signal.

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal.
    fs : float
        Sampling frequency in Hz.
    lowcut : float
        Low cutoff frequency in Hz.
    highcut : float
        High cutoff frequency in Hz.
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    # Ensure valid range
    low = max(low, 1e-5)
    high = min(high, 1.0 - 1e-5)

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal, padlen=min(3 * max(len(b), len(a)), len(signal) - 1))


def detect_r_peaks(
    ecg: np.ndarray,
    fs: float = 700.0,
    height_percentile: float = 70.0,
    min_distance_sec: float = 0.3,
) -> np.ndarray:
    """
    Detect R-peaks in an ECG signal using a simple peak detection approach.

    Parameters
    ----------
    ecg : np.ndarray
        Filtered ECG signal.
    fs : float
        Sampling rate (Hz).
    height_percentile : float
        Percentile of signal amplitude to use as minimum peak height.
    min_distance_sec : float
        Minimum distance between peaks in seconds.

    Returns
    -------
    np.ndarray
        Indices of detected R-peaks.
    """
    min_distance = int(min_distance_sec * fs)
    height = np.percentile(np.abs(ecg), height_percentile)

    peaks, _ = find_peaks(ecg, height=height, distance=min_distance)

    return peaks


def compute_rr_intervals(
    r_peaks: np.ndarray,
    fs: float = 700.0,
) -> np.ndarray:
    """
    Compute R-R intervals (inter-beat intervals) from R-peak indices.

    Parameters
    ----------
    r_peaks : np.ndarray
        Array of R-peak sample indices.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray
        R-R intervals in seconds.
    """
    if len(r_peaks) < 2:
        return np.array([])

    rr = np.diff(r_peaks) / fs
    return rr


def remove_rr_artifacts(
    rr_intervals: np.ndarray,
    min_rr: float = 0.3,
    max_rr: float = 2.0,
    max_deviation: float = 0.20,
) -> np.ndarray:
    """
    Remove physiologically implausible RR intervals.

    Applies two filters:
      1. Absolute bounds (min_rr, max_rr) in seconds
      2. Relative deviation from running median (> max_deviation fraction)

    Parameters
    ----------
    rr_intervals : np.ndarray
        Raw RR intervals in seconds.
    min_rr : float
        Minimum plausible RR interval (seconds). ~200 bpm ceiling.
    max_rr : float
        Maximum plausible RR interval (seconds). ~30 bpm floor.
    max_deviation : float
        Maximum relative deviation from local median (0-1).

    Returns
    -------
    np.ndarray
        Cleaned RR intervals.
    """
    if len(rr_intervals) == 0:
        return rr_intervals

    # Step 1: absolute bounds
    mask = (rr_intervals >= min_rr) & (rr_intervals <= max_rr)

    # Step 2: relative deviation from rolling median
    if np.sum(mask) > 5:
        filtered = rr_intervals[mask]
        median_rr = np.median(filtered)
        relative_dev = np.abs(filtered - median_rr) / median_rr
        mask2 = relative_dev <= max_deviation
        filtered = filtered[mask2]
        return filtered

    return rr_intervals[mask]


def extract_ibi_from_ecg(
    ecg: np.ndarray,
    fs: float = 700.0,
    filter_signal: bool = True,
) -> np.ndarray:
    """
    Full pipeline: ECG → filter → R-peaks → clean RR intervals.

    Parameters
    ----------
    ecg : np.ndarray
        Raw ECG signal.
    fs : float
        Sampling rate (Hz).
    filter_signal : bool
        Whether to apply bandpass filter first.

    Returns
    -------
    np.ndarray
        Cleaned RR intervals in seconds.
    """
    if filter_signal:
        ecg_filtered = bandpass_filter(ecg, fs, lowcut=0.5, highcut=40.0)
    else:
        ecg_filtered = ecg

    r_peaks = detect_r_peaks(ecg_filtered, fs)

    if len(r_peaks) < 3:
        warnings.warn("Too few R-peaks detected. Check signal quality.")
        return np.array([])

    rr = compute_rr_intervals(r_peaks, fs)
    rr_clean = remove_rr_artifacts(rr)

    return rr_clean


def normalize_zscore(data: np.ndarray) -> np.ndarray:
    """Z-score normalize a feature array."""
    mean = np.mean(data)
    std = np.std(data)
    if std < 1e-10:
        return np.zeros_like(data)
    return (data - mean) / std


def normalize_minmax(data: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    dmin = np.min(data)
    dmax = np.max(data)
    if dmax - dmin < 1e-10:
        return np.zeros_like(data)
    return (data - dmin) / (dmax - dmin)


def segment_rr_intervals(
    rr_intervals: np.ndarray,
    window_n: int = 60,
    overlap_frac: float = 0.5,
) -> List[np.ndarray]:
    """
    Segment a sequence of RR intervals into overlapping windows.

    Parameters
    ----------
    rr_intervals : np.ndarray
        Cleaned RR intervals.
    window_n : int
        Number of RR intervals per window (~60 beats ≈ 1 minute at 60 bpm).
    overlap_frac : float
        Overlap fraction between windows.

    Returns
    -------
    list of np.ndarray
        Windowed RR interval segments.
    """
    step = max(1, int(window_n * (1 - overlap_frac)))
    segments = []

    for start in range(0, len(rr_intervals) - window_n + 1, step):
        segments.append(rr_intervals[start:start + window_n])

    return segments
