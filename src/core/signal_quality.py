"""
Signal Quality Assessment
==========================
Provides signal-quality metrics for ECG and other physiological
channels.  Used by the live pipeline to reject bad windows and
dynamically adjust fusion weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger("pilot_readiness.signal_quality")


@dataclass
class QualityReport:
    """Per-window signal quality assessment."""
    overall: float                    # 0-1 composite quality
    snr_estimate: float = 0.0        # signal-to-noise ratio estimate
    rr_plausibility: float = 0.0     # fraction of plausible RR intervals
    rr_consistency: float = 0.0      # CV-based consistency
    n_peaks: int = 0                 # number of R-peaks detected
    is_acceptable: bool = True       # whether the window passes QC
    reasons: List[str] = field(default_factory=list)


class SignalQualityAssessor:
    """
    Assess ECG signal quality for each analysis window.

    Used by ``LivePipeline`` to:
    - Reject windows with quality below ``min_quality``
    - Log degradation trends
    - Feed quality into the fusion engine for weight adjustment
    """

    def __init__(
        self,
        min_quality: float = 0.5,
        min_peaks: int = 10,
        history_size: int = 20,
    ):
        self.min_quality = min_quality
        self.min_peaks = min_peaks
        self.history: List[float] = []
        self.history_size = history_size

    def assess_ecg_window(
        self,
        ecg_window: np.ndarray,
        rr_intervals: Optional[np.ndarray] = None,
        fs: float = 700.0,
    ) -> QualityReport:
        """
        Assess the quality of a single ECG window.

        Parameters
        ----------
        ecg_window : np.ndarray
            Raw ECG samples for the window.
        rr_intervals : np.ndarray, optional
            Pre-computed RR intervals in seconds.
        fs : float
            Sampling rate in Hz.

        Returns
        -------
        QualityReport
        """
        reasons = []

        # ── 1. Check basic signal statistics ──────────────────
        if len(ecg_window) < int(fs * 5):
            return QualityReport(
                overall=0.0, is_acceptable=False,
                reasons=["Window too short"],
            )

        # Check for flat-line (all zeros or constant)
        signal_range = float(np.ptp(ecg_window))
        if signal_range < 1e-6:
            return QualityReport(
                overall=0.0, is_acceptable=False,
                reasons=["Flat-line detected"],
            )

        # ── 2. SNR estimate (signal power / noise power) ──────
        # Simple approach: power in typical ECG band vs. high-freq noise
        from scipy.signal import welch
        try:
            freqs, psd = welch(ecg_window, fs=fs, nperseg=min(1024, len(ecg_window)))
            # Signal power: 0.5-40 Hz (ECG band)
            signal_mask = (freqs >= 0.5) & (freqs <= 40.0)
            noise_mask = freqs > 40.0
            signal_power = float(np.trapz(psd[signal_mask], freqs[signal_mask])) if signal_mask.any() else 0
            noise_power = float(np.trapz(psd[noise_mask], freqs[noise_mask])) if noise_mask.any() else 1e-10
            snr = signal_power / max(noise_power, 1e-10)
            snr_quality = float(np.clip(snr / 100.0, 0, 1))  # normalize: SNR>100 → quality=1
        except Exception:
            snr = 0.0
            snr_quality = 0.0
            reasons.append("SNR computation failed")

        # ── 3. RR interval plausibility ───────────────────────
        rr_plausibility = 0.0
        rr_consistency = 0.0
        n_peaks = 0

        if rr_intervals is not None and len(rr_intervals) > 0:
            n_peaks = len(rr_intervals) + 1

            # Plausibility: RR between 0.3s (200 bpm) and 2.0s (30 bpm)
            in_range = (rr_intervals >= 0.3) & (rr_intervals <= 2.0)
            rr_plausibility = float(np.mean(in_range))

            if rr_plausibility < 0.7:
                reasons.append(f"Low RR plausibility: {rr_plausibility:.2f}")

            # Consistency: CV should be reasonable (0.02-0.30)
            mean_rr = float(np.mean(rr_intervals))
            if mean_rr > 0:
                cv = float(np.std(rr_intervals) / mean_rr)
                rr_consistency = 1.0 if 0.02 < cv < 0.30 else max(0.0, 1.0 - abs(cv - 0.15) / 0.5)
            else:
                rr_consistency = 0.0
                reasons.append("Mean RR is zero")
        else:
            reasons.append("No RR intervals available")

        # ── 4. Peak count check ───────────────────────────────
        if n_peaks < self.min_peaks:
            reasons.append(f"Too few R-peaks: {n_peaks}")

        # ── 5. Composite quality score ────────────────────────
        overall = 0.3 * snr_quality + 0.4 * rr_plausibility + 0.3 * rr_consistency
        overall = float(np.clip(overall, 0.0, 1.0))

        is_acceptable = overall >= self.min_quality and n_peaks >= self.min_peaks

        # Track history
        self.history.append(overall)
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]

        return QualityReport(
            overall=overall,
            snr_estimate=snr,
            rr_plausibility=rr_plausibility,
            rr_consistency=rr_consistency,
            n_peaks=n_peaks,
            is_acceptable=is_acceptable,
            reasons=reasons,
        )

    def get_trend(self) -> Optional[float]:
        """
        Return the quality trend over recent windows.

        Positive = improving, negative = degrading.
        Returns None if not enough history.
        """
        if len(self.history) < 3:
            return None
        recent = np.array(self.history[-5:])
        return float(np.polyfit(range(len(recent)), recent, 1)[0])

    def get_average_quality(self) -> float:
        """Return the average quality over recent windows."""
        if not self.history:
            return 1.0
        return float(np.mean(self.history))

    def reset(self):
        """Reset the quality history."""
        self.history.clear()
