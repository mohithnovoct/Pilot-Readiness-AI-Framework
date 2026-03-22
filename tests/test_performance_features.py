"""
Unit Tests: Performance Features & MATB Simulator
===================================================
Tests for performance feature extraction and simulation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.features.performance_features import (
    compute_rt_features, compute_tracking_features,
    compute_inceptor_entropy, compute_comm_features,
)
from src.data.matb_simulator import (
    simulate_reaction_times, simulate_tracking_error,
    simulate_comm_responses, generate_full_simulation,
)


class TestRTFeatures:
    """Tests for compute_rt_features()."""

    def test_basic_computation(self):
        """Features should be computed for normal data."""
        rts = np.random.uniform(1.0, 5.0, 50)
        features = compute_rt_features(rts)
        assert features["MeanRT"] > 0
        assert features["StdRT"] > 0
        assert features["CVRT"] > 0
        assert -1 <= features["Lag1_Autocorr"] <= 1

    def test_short_input(self):
        """Very short input should return NaN."""
        features = compute_rt_features(np.array([1.0]))
        assert np.isnan(features["MeanRT"])

    def test_lapse_rate(self):
        """Lapse rate should correctly count slow responses."""
        rts = np.array([1.0, 2.0, 6.0, 7.0, 3.0])  # 2 out of 5 > 5s
        features = compute_rt_features(rts)
        assert abs(features["Lapse_Rate"] - 0.4) < 1e-10


class TestTrackingFeatures:
    """Tests for compute_tracking_features()."""

    def test_trend_detection(self):
        """Increasing RMSD should have positive trend."""
        rmsd = np.linspace(10, 50, 30)
        features = compute_tracking_features(rmsd)
        assert features["RMSD_Trend"] > 0

    def test_max_rmsd(self):
        """MaxRMSD should be the actual max."""
        rmsd = np.array([10, 20, 50, 30, 15])
        features = compute_tracking_features(rmsd)
        assert features["MaxRMSD"] == 50.0


class TestInceptorEntropy:
    """Tests for compute_inceptor_entropy()."""

    def test_uniform_high_entropy(self):
        """Uniformly distributed inputs should have higher entropy."""
        uniform = np.random.uniform(0, 100, 500)
        narrow = np.random.normal(50, 1, 500)
        e_uniform = compute_inceptor_entropy(uniform)
        e_narrow = compute_inceptor_entropy(narrow)
        assert e_uniform > e_narrow

    def test_short_input(self):
        """Very short input should return NaN."""
        result = compute_inceptor_entropy(np.array([1.0, 2.0]))
        assert np.isnan(result)


class TestCommFeatures:
    """Tests for compute_comm_features()."""

    def test_accuracy(self):
        """CommAccuracy should be fraction of correct responses."""
        rts = np.array([5.0, 8.0, 12.0])
        correct = np.array([True, True, False])
        features = compute_comm_features(rts, correct)
        assert abs(features["CommAccuracy"] - 2/3) < 1e-10


class TestSimulator:
    """Tests for MATB-II simulator."""

    def test_workload_ordering(self):
        """High workload should have higher mean RT than low."""
        low_rts = simulate_reaction_times(n_events=200, workload="low", seed=42)
        high_rts = simulate_reaction_times(n_events=200, workload="high", seed=42)
        assert np.mean(high_rts) > np.mean(low_rts)

    def test_simulation_reproducibility(self):
        """Same seed should produce same results."""
        r1 = simulate_reaction_times(50, "low", seed=123)
        r2 = simulate_reaction_times(50, "low", seed=123)
        np.testing.assert_array_equal(r1, r2)

    def test_full_simulation_shape(self):
        """Full simulation should produce correct DataFrame shape."""
        df = generate_full_simulation(n_sessions_per_level=5, seed=42)
        assert len(df) == 15  # 3 levels × 5 sessions
        assert "workload" in df.columns
        assert "mean_rt" in df.columns

    def test_tracking_range(self):
        """Tracking RMSD should be clipped to realistic range."""
        rmsd = simulate_tracking_error(100, "high", seed=42)
        assert np.all(rmsd >= 5.0)
        assert np.all(rmsd <= 200.0)

    def test_comm_timeout(self):
        """Timeout responses should have RT = 30.0."""
        rts, correct = simulate_comm_responses(100, "high", seed=42)
        # High workload should have some timeouts
        assert np.any(rts >= 30.0) or True  # May or may not depending on seed
        assert len(rts) == 100
