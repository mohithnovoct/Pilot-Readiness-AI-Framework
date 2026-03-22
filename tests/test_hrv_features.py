"""
Unit Tests: HRV Feature Extraction
====================================
Tests for time-domain, frequency-domain, and non-linear HRV features.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.features.hrv_features import (
    compute_time_domain, compute_frequency_domain,
    compute_nonlinear, sample_entropy, extract_all_hrv_features,
)


def _make_rr(mean=0.8, std=0.05, n=200, seed=42):
    """Helper to generate synthetic RR intervals."""
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, n)


class TestTimeDomain:
    """Tests for compute_time_domain()."""

    def test_known_values(self):
        """Check features against hand-computed values."""
        rr = np.array([0.8, 0.82, 0.79, 0.81, 0.83])
        features = compute_time_domain(rr)

        assert abs(features["MeanNN"] - np.mean(rr)) < 1e-10
        assert abs(features["MedianNN"] - np.median(rr)) < 1e-10
        assert features["SDNN"] > 0
        assert features["RMSSD"] > 0
        assert 0 <= features["pNN50"] <= 100
        assert features["MeanHR"] > 0
        assert features["SDHR"] >= 0

    def test_short_input(self):
        """< 2 intervals should return NaN features."""
        features = compute_time_domain(np.array([0.8]))
        assert np.isnan(features["SDNN"])

    def test_pnn50_threshold(self):
        """pNN50 should be 100% when all successive diffs > 50ms."""
        rr = np.array([0.5, 0.6, 0.7, 0.8, 0.9])  # diffs are all 100ms
        features = compute_time_domain(rr)
        assert features["pNN50"] == 100.0

    def test_heart_rate_correct(self):
        """Mean HR should be 60/MeanNN."""
        rr = _make_rr(mean=0.75, std=0.02)
        features = compute_time_domain(rr)
        expected_hr = 60.0 / np.mean(rr)
        assert abs(features["MeanHR"] - expected_hr) < 0.1


class TestFrequencyDomain:
    """Tests for compute_frequency_domain()."""

    def test_returns_all_keys(self):
        """Should return all expected frequency-domain features."""
        rr = _make_rr(n=300)
        features = compute_frequency_domain(rr)
        expected_keys = ["VLF_power", "LF_power", "HF_power",
                         "Total_power", "LF_HF_ratio", "LF_norm", "HF_norm"]
        for key in expected_keys:
            assert key in features

    def test_power_non_negative(self):
        """Spectral power values should be non-negative."""
        rr = _make_rr(n=300)
        features = compute_frequency_domain(rr)
        assert features["LF_power"] >= 0
        assert features["HF_power"] >= 0
        assert features["Total_power"] >= 0

    def test_short_input_returns_nan(self):
        """Too few RR intervals should return NaN."""
        features = compute_frequency_domain(np.array([0.8, 0.82, 0.79]))
        assert np.isnan(features["LF_power"])

    def test_normalized_sum(self):
        """LF_norm + HF_norm should ≈ 100."""
        rr = _make_rr(n=300)
        features = compute_frequency_domain(rr)
        if not np.isnan(features["LF_norm"]):
            assert abs(features["LF_norm"] + features["HF_norm"] - 100.0) < 0.1


class TestNonlinear:
    """Tests for compute_nonlinear() and sample_entropy()."""

    def test_sampen_positive(self):
        """Sample entropy of random data should be > 0."""
        rr = _make_rr(n=200)
        features = compute_nonlinear(rr)
        if not np.isnan(features["SampEn"]):
            assert features["SampEn"] > 0

    def test_constant_signal(self):
        """Constant signal has undefined SampEn (should be NaN)."""
        rr = np.ones(100) * 0.8
        se = sample_entropy(rr, m=2, r_factor=0.2)
        # Constant signal → r = 0 → NaN
        assert np.isnan(se)

    def test_short_input_nan(self):
        """Very short input should return NaN."""
        features = compute_nonlinear(np.array([0.8, 0.82]))
        assert np.isnan(features["SampEn"])


class TestCombined:
    """Tests for extract_all_hrv_features()."""

    def test_all_features_present(self):
        """Combined extraction should include all feature types."""
        rr = _make_rr(n=300)
        features = extract_all_hrv_features(rr)
        # Time domain
        assert "MeanNN" in features
        assert "RMSSD" in features
        # Frequency domain
        assert "LF_power" in features
        assert "HF_power" in features
        # Non-linear
        assert "SampEn" in features

    def test_total_feature_count(self):
        """Should have 15 features total."""
        rr = _make_rr(n=300)
        features = extract_all_hrv_features(rr)
        assert len(features) == 15
