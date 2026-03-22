"""
Unit Tests: Risk Fusion & Thresholds
======================================
Tests for risk score computation, fusion, and Neyman-Pearson thresholds.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.risk.fusion import (
    compute_risk_score, compute_risk_batch,
    estimate_signal_quality, get_risk_category,
)
from src.risk.threshold import (
    compute_threshold, apply_threshold,
    evaluate_threshold_performance, ThresholdManager,
)


class TestComputeRiskScore:
    """Tests for compute_risk_score()."""

    def test_equal_weights(self):
        """With equal weights and equal inputs, output should equal input."""
        score = compute_risk_score(0.5, 0.5, w_phys=0.5, w_perf=0.5)
        assert abs(score - 0.5) < 1e-10

    def test_boundary_zero(self):
        """Zero inputs should give zero risk."""
        score = compute_risk_score(0.0, 0.0)
        assert score == 0.0

    def test_boundary_one(self):
        """Max inputs should give max risk."""
        score = compute_risk_score(1.0, 1.0)
        assert abs(score - 1.0) < 1e-10

    def test_weighted_combination(self):
        """Verify weighted average is correct."""
        score = compute_risk_score(0.8, 0.2, w_phys=0.6, w_perf=0.4)
        expected = 0.6 * 0.8 + 0.4 * 0.2
        assert abs(score - expected) < 1e-10

    def test_clipping(self):
        """Out-of-range inputs should be clipped."""
        score = compute_risk_score(1.5, -0.5)
        assert 0.0 <= score <= 1.0

    def test_signal_quality_adjustment(self):
        """Low signal quality should reduce physio weight."""
        high_sq = compute_risk_score(0.9, 0.1, signal_quality=1.0)
        low_sq = compute_risk_score(0.9, 0.1, signal_quality=0.1)
        # Low quality → more weight on perf → lower risk (since perf is low)
        assert low_sq < high_sq


class TestComputeRiskBatch:
    """Tests for vectorized compute_risk_batch()."""

    def test_matches_individual(self):
        """Batch results should match individual calls."""
        p_s = np.array([0.1, 0.5, 0.9])
        p_p = np.array([0.2, 0.4, 0.8])
        batch = compute_risk_batch(p_s, p_p)
        for i in range(len(p_s)):
            individual = compute_risk_score(p_s[i], p_p[i])
            assert abs(batch[i] - individual) < 1e-10

    def test_output_range(self):
        """All outputs should be in [0, 1]."""
        rng = np.random.default_rng(42)
        p_s = rng.random(100)
        p_p = rng.random(100)
        risks = compute_risk_batch(p_s, p_p)
        assert np.all(risks >= 0.0)
        assert np.all(risks <= 1.0)


class TestSignalQuality:
    """Tests for estimate_signal_quality()."""

    def test_good_signal(self):
        """Normal RR intervals should give high quality."""
        rr = np.random.normal(0.8, 0.04, 100)
        quality = estimate_signal_quality(rr)
        assert quality > 0.7

    def test_bad_signal(self):
        """Very short or empty should give 0."""
        quality = estimate_signal_quality(np.array([0.5]))
        assert quality == 0.0


class TestRiskCategory:
    """Tests for get_risk_category()."""

    def test_low(self):
        cat, color = get_risk_category(0.1)
        assert cat == "LOW"

    def test_moderate(self):
        cat, _ = get_risk_category(0.35)
        assert cat == "MODERATE"

    def test_elevated(self):
        cat, _ = get_risk_category(0.6)
        assert cat == "ELEVATED"

    def test_high(self):
        cat, _ = get_risk_category(0.9)
        assert cat == "HIGH"


class TestThreshold:
    """Tests for Neyman-Pearson threshold computation."""

    def test_quantile_based(self):
        """Threshold should be the (1-α) quantile of ready scores."""
        scores = np.linspace(0, 1, 1000)
        gamma = compute_threshold(scores, alpha=0.05)
        # Should be approximately 0.95
        assert abs(gamma - 0.95) < 0.01

    def test_alpha_monotonicity(self):
        """Higher alpha → lower threshold (more sensitive)."""
        scores = np.random.rand(1000)
        g1 = compute_threshold(scores, alpha=0.01)
        g2 = compute_threshold(scores, alpha=0.10)
        assert g1 > g2

    def test_empty_scores(self):
        """Empty scores should return default threshold."""
        gamma = compute_threshold(np.array([]), alpha=0.05)
        assert gamma == 0.5


class TestApplyThreshold:
    """Tests for apply_threshold()."""

    def test_above_threshold(self):
        alert, decision = apply_threshold(0.8, 0.5)
        assert alert is True
        assert decision == "ALERT"

    def test_below_threshold(self):
        alert, decision = apply_threshold(0.3, 0.5)
        assert alert is False
        assert decision == "READY"


class TestThresholdManager:
    """Tests for ThresholdManager class."""

    def test_scenario_presets(self):
        """All three preset scenarios should be valid."""
        tm = ThresholdManager(baseline_scores=np.random.rand(500))
        for scenario in ["training", "operational", "critical"]:
            gamma = tm.set_scenario(scenario)
            assert 0.0 <= gamma <= 1.0

    def test_training_most_sensitive(self):
        """Training scenario should have the lowest threshold."""
        tm = ThresholdManager(baseline_scores=np.random.rand(500))
        g_train = tm.set_scenario("training")
        g_crit = tm.set_scenario("critical")
        assert g_train < g_crit

    def test_evaluate(self):
        """Evaluate should return consistent results."""
        tm = ThresholdManager(baseline_scores=np.random.rand(500))
        alert, decision, threshold = tm.evaluate(0.99)
        assert alert is True or threshold > 0.99  # Should alert on extreme risk
