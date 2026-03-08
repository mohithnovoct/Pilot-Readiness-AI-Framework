"""
HRV Feature Extraction
=======================
Extracts time-domain, frequency-domain, and non-linear features from
RR interval (IBI) sequences.

Based on the research plan (Section 5.1):
  - Time-Domain: SDNN, RMSSD, pNN50, MeanNN, MedianNN
  - Frequency-Domain: LF, HF, LF/HF ratio (Welch's method)
  - Non-Linear: Sample Entropy

References:
  - Task Force of ESA/NASPE (1996) HRV Standards
  - Richman & Moorman (2000) Sample Entropy
"""

import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid
from typing import Dict, Optional
import warnings


# ======================================================================
#                       TIME-DOMAIN FEATURES
# ======================================================================

def compute_time_domain(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Compute time-domain HRV features from RR intervals.

    Parameters
    ----------
    rr_intervals : np.ndarray
        Cleaned RR intervals in seconds.

    Returns
    -------
    dict
        Feature name → value mapping:
        - MeanNN: Mean NN interval (seconds)
        - MedianNN: Median NN interval (seconds)
        - SDNN: Standard deviation of NN intervals (seconds)
        - RMSSD: Root mean square of successive differences (seconds)
        - pNN50: Percentage of successive intervals > 50ms difference (%)
        - MeanHR: Mean heart rate (bpm)
        - SDHR: Std dev of instantaneous HR (bpm)
    """
    if len(rr_intervals) < 2:
        return _empty_time_domain()

    rr = rr_intervals

    mean_nn = np.mean(rr)
    median_nn = np.median(rr)
    sdnn = np.std(rr, ddof=1)

    # Successive differences
    diffs = np.diff(rr)
    rmssd = np.sqrt(np.mean(diffs ** 2))

    # pNN50: percentage of |successive diffs| > 50ms
    pnn50 = 100.0 * np.sum(np.abs(diffs) > 0.050) / len(diffs)

    # Heart rate statistics
    hr = 60.0 / rr
    mean_hr = np.mean(hr)
    sd_hr = np.std(hr, ddof=1)

    return {
        "MeanNN": mean_nn,
        "MedianNN": median_nn,
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "pNN50": pnn50,
        "MeanHR": mean_hr,
        "SDHR": sd_hr,
    }


def _empty_time_domain() -> Dict[str, float]:
    """Return NaN-filled time-domain features."""
    return {k: np.nan for k in
            ["MeanNN", "MedianNN", "SDNN", "RMSSD", "pNN50", "MeanHR", "SDHR"]}


# ======================================================================
#                     FREQUENCY-DOMAIN FEATURES
# ======================================================================

def compute_frequency_domain(
    rr_intervals: np.ndarray,
    fs_resample: float = 4.0,
    vlf_band: tuple = (0.003, 0.04),
    lf_band: tuple = (0.04, 0.15),
    hf_band: tuple = (0.15, 0.4),
) -> Dict[str, float]:
    """
    Compute frequency-domain HRV features using Welch's periodogram.

    The RR interval series is first resampled to a uniform rate (4 Hz)
    using cubic interpolation, then detrended.

    Parameters
    ----------
    rr_intervals : np.ndarray
        Cleaned RR intervals in seconds.
    fs_resample : float
        Resampling frequency for the uniformly sampled RR tachogram.
    vlf_band : tuple
        Very Low Frequency band limits (Hz).
    lf_band : tuple
        Low Frequency band limits (Hz).
    hf_band : tuple
        High Frequency band limits (Hz).

    Returns
    -------
    dict
        - VLF_power: Power in VLF band (s²)
        - LF_power: Power in LF band (s²)
        - HF_power: Power in HF band (s²)
        - Total_power: Total spectral power (s²)
        - LF_HF_ratio: LF/HF ratio (sympathovagal balance)
        - LF_norm: LF in normalized units (%)
        - HF_norm: HF in normalized units (%)
    """
    if len(rr_intervals) < 10:
        return _empty_freq_domain()

    # Create cumulative time axis
    t_rr = np.cumsum(rr_intervals) - rr_intervals[0]

    # Resample to uniform time grid
    duration = t_rr[-1]
    t_uniform = np.arange(0, duration, 1.0 / fs_resample)

    if len(t_uniform) < 16:
        return _empty_freq_domain()

    try:
        from scipy.interpolate import interp1d
        interp_func = interp1d(t_rr, rr_intervals, kind="cubic",
                               fill_value="extrapolate")
        rr_resampled = interp_func(t_uniform)

        # Detrend (remove mean)
        rr_resampled = rr_resampled - np.mean(rr_resampled)

        # Welch's periodogram
        nperseg = min(256, len(rr_resampled))
        freqs, psd = welch(rr_resampled, fs=fs_resample, nperseg=nperseg,
                           noverlap=nperseg // 2, detrend="linear")

        # Integrate power in each band
        def band_power(f_low, f_high):
            idx = (freqs >= f_low) & (freqs <= f_high)
            if not np.any(idx):
                return 0.0
            return trapezoid(psd[idx], freqs[idx])

        vlf = band_power(*vlf_band)
        lf = band_power(*lf_band)
        hf = band_power(*hf_band)
        total = band_power(vlf_band[0], hf_band[1])

        lf_hf = lf / hf if hf > 1e-10 else np.nan
        lf_norm = 100.0 * lf / (lf + hf) if (lf + hf) > 1e-10 else np.nan
        hf_norm = 100.0 * hf / (lf + hf) if (lf + hf) > 1e-10 else np.nan

        return {
            "VLF_power": vlf,
            "LF_power": lf,
            "HF_power": hf,
            "Total_power": total,
            "LF_HF_ratio": lf_hf,
            "LF_norm": lf_norm,
            "HF_norm": hf_norm,
        }

    except Exception as e:
        warnings.warn(f"Frequency-domain computation failed: {e}")
        return _empty_freq_domain()


def _empty_freq_domain() -> Dict[str, float]:
    """Return NaN-filled frequency-domain features."""
    return {k: np.nan for k in
            ["VLF_power", "LF_power", "HF_power", "Total_power",
             "LF_HF_ratio", "LF_norm", "HF_norm"]}


# ======================================================================
#                       NON-LINEAR FEATURES
# ======================================================================

def sample_entropy(
    data: np.ndarray,
    m: int = 2,
    r_factor: float = 0.2,
) -> float:
    """
    Compute Sample Entropy of a time series.

    Parameters
    ----------
    data : np.ndarray
        Input time series (RR intervals).
    m : int
        Embedding dimension.
    r_factor : float
        Tolerance factor (proportion of std dev).

    Returns
    -------
    float
        Sample Entropy value. Higher = more complex/irregular.
    """
    if len(data) < m + 2:
        return np.nan

    r = r_factor * np.std(data)
    if r < 1e-10:
        return np.nan

    N = len(data)

    def _count_templates(m_val):
        """Count matching template pairs for embedding dimension m_val."""
        templates = np.array([data[i:i + m_val] for i in range(N - m_val)])
        count = 0
        for i in range(len(templates)):
            # Compare with all templates j > i
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    count += 1
        return count

    A = _count_templates(m + 1)
    B = _count_templates(m)

    if B == 0:
        return np.nan

    return -np.log(A / B) if A > 0 else np.nan


def compute_nonlinear(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Compute non-linear HRV features.

    Returns
    -------
    dict
        - SampEn: Sample Entropy (m=2, r=0.2*std)
    """
    if len(rr_intervals) < 10:
        return {"SampEn": np.nan}

    try:
        # Try using antropy for faster computation if available
        import antropy
        sampen = antropy.sample_entropy(rr_intervals, order=2,
                                         metric="chebyshev")
        # antropy returns an array; take last value
        if hasattr(sampen, "__len__"):
            sampen = sampen[-1] if len(sampen) > 0 else np.nan
    except ImportError:
        # Fallback to our implementation
        sampen = sample_entropy(rr_intervals, m=2, r_factor=0.2)

    return {"SampEn": sampen if np.isfinite(sampen) else np.nan}


# ======================================================================
#                       COMBINED EXTRACTION
# ======================================================================

def extract_all_hrv_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Extract all HRV features (time, frequency, non-linear) from RR intervals.

    Parameters
    ----------
    rr_intervals : np.ndarray
        Cleaned RR intervals in seconds.

    Returns
    -------
    dict
        Combined dictionary of all HRV feature values.
    """
    features = {}
    features.update(compute_time_domain(rr_intervals))
    features.update(compute_frequency_domain(rr_intervals))
    features.update(compute_nonlinear(rr_intervals))
    return features
