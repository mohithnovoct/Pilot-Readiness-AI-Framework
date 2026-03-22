"""
Unit Tests: Signal Preprocessing
=================================
Tests for bandpass filtering, R-peak detection, RR artifact removal.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.data.preprocessing import (
    bandpass_filter, detect_r_peaks, compute_rr_intervals,
    remove_rr_artifacts, extract_ibi_from_ecg,
    normalize_zscore, normalize_minmax, segment_rr_intervals,
)


class TestBandpassFilter:
    """Tests for bandpass_filter()."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        signal = np.random.randn(5000)
        filtered = bandpass_filter(signal, fs=700.0)
        assert filtered.shape == signal.shape

    def test_removes_dc_offset(self):
        """DC component (0 Hz) should be attenuated by the highpass."""
        t = np.arange(0, 5, 1/700)
        signal = np.ones_like(t) * 10 + np.sin(2 * np.pi * 10 * t)
        filtered = bandpass_filter(signal, fs=700.0, lowcut=0.5, highcut=40.0)
        assert abs(np.mean(filtered)) < 1.0  # DC should be mostly gone

    def test_preserves_passband(self):
        """Signal within passband should be preserved."""
        fs = 700
        t = np.arange(0, 5, 1/fs)
        freq = 10.0  # within [0.5, 40] Hz
        signal = np.sin(2 * np.pi * freq * t)
        filtered = bandpass_filter(signal, fs=fs)
        # Power should be mostly preserved
        ratio = np.std(filtered) / np.std(signal)
        assert ratio > 0.5


class TestRPeakDetection:
    """Tests for detect_r_peaks()."""

    def test_synthetic_ecg(self):
        """Should detect peaks in a synthetic periodic signal."""
        fs = 700
        duration = 10  # seconds
        heart_rate = 75  # bpm
        t = np.arange(0, duration, 1/fs)
        # Simulate ECG-like signal with Gaussian-shaped R-peaks
        ecg = np.zeros_like(t)
        peak_interval = fs * 60 / heart_rate
        for i in range(int(duration * heart_rate / 60)):
            center = int(i * peak_interval)
            if center < len(ecg):
                # Gaussian pulse centered at peak
                for k in range(-20, 21):
                    idx = center + k
                    if 0 <= idx < len(ecg):
                        ecg[idx] += np.exp(-0.5 * (k / 5) ** 2)
        peaks = detect_r_peaks(ecg, fs=fs)
        expected_peaks = int(duration * heart_rate / 60)
        # Allow some tolerance — peak detection is approximate
        assert len(peaks) > 0
        assert abs(len(peaks) - expected_peaks) <= expected_peaks * 0.5

    def test_empty_signal(self):
        """Empty signal should raise an error."""
        with pytest.raises((IndexError, ValueError)):
            detect_r_peaks(np.array([]), fs=700)


class TestRRIntervals:
    """Tests for compute_rr_intervals()."""

    def test_basic_computation(self):
        """RR intervals should be correct."""
        r_peaks = np.array([0, 700, 1400, 2100])  # every second at 700 Hz
        rr = compute_rr_intervals(r_peaks, fs=700)
        np.testing.assert_allclose(rr, [1.0, 1.0, 1.0], atol=1e-10)

    def test_single_peak(self):
        """Single peak should return empty RR array."""
        rr = compute_rr_intervals(np.array([100]), fs=700)
        assert len(rr) == 0

    def test_no_peaks(self):
        """No peaks should return empty RR array."""
        rr = compute_rr_intervals(np.array([]), fs=700)
        assert len(rr) == 0


class TestRRArtifactRemoval:
    """Tests for remove_rr_artifacts()."""

    def test_removes_implausible(self):
        """Should remove RR intervals outside physiological range."""
        rr = np.array([0.8, 0.85, 0.1, 0.82, 3.0, 0.79])
        clean = remove_rr_artifacts(rr, min_rr=0.3, max_rr=2.0)
        assert 0.1 not in clean
        assert 3.0 not in clean

    def test_preserves_normal(self):
        """Normal RR intervals should be preserved."""
        rr = np.array([0.8, 0.82, 0.79, 0.81, 0.83, 0.80])
        clean = remove_rr_artifacts(rr)
        assert len(clean) >= 4  # Most should be preserved

    def test_empty_input(self):
        """Empty input should return empty."""
        clean = remove_rr_artifacts(np.array([]))
        assert len(clean) == 0


class TestNormalization:
    """Tests for normalization functions."""

    def test_zscore_mean_std(self):
        """Z-scored data should have mean≈0 and std≈1."""
        data = np.random.randn(100) * 5 + 10
        normed = normalize_zscore(data)
        assert abs(np.mean(normed)) < 1e-10
        assert abs(np.std(normed) - 1.0) < 0.01

    def test_minmax_range(self):
        """Min-max scaled data should be in [0, 1]."""
        data = np.random.randn(100) * 5 + 10
        normed = normalize_minmax(data)
        assert np.min(normed) >= 0.0
        assert np.max(normed) <= 1.0

    def test_constant_signal_zscore(self):
        """Constant signal should produce zeros (avoids division by zero)."""
        data = np.ones(50) * 5
        normed = normalize_zscore(data)
        np.testing.assert_array_equal(normed, np.zeros(50))


class TestSegmentation:
    """Tests for segment_rr_intervals()."""

    def test_output_count(self):
        """Number of segments should be approximately correct."""
        rr = np.random.uniform(0.7, 0.9, 200)
        segments = segment_rr_intervals(rr, window_n=60, overlap_frac=0.5)
        assert len(segments) > 0
        for seg in segments:
            assert len(seg) == 60

    def test_short_input(self):
        """Input shorter than window should return no segments."""
        rr = np.random.uniform(0.7, 0.9, 10)
        segments = segment_rr_intervals(rr, window_n=60)
        assert len(segments) == 0
