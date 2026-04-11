"""
Unit Tests: Edge Model Export
================================
Tests for model profiling, benchmarking, and code export.
Uses a lightweight dummy LightGBM model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import lightgbm as lgb

from src.edge.export_model import (
    profile_model, benchmark_inference,
    export_to_c_code, export_to_python_code,
    generate_edge_report,
)


@pytest.fixture
def dummy_model():
    """Create a small LightGBM model for testing."""
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    model = lgb.LGBMClassifier(
        n_estimators=10, num_leaves=8, verbose=-1, random_state=42,
    )
    model.fit(X, y)
    return model, X


class TestProfileModel:
    """Tests for profile_model()."""

    def test_returns_dict(self, dummy_model):
        model, _ = dummy_model
        profile = profile_model(model)
        assert isinstance(profile, dict)

    def test_contains_size_info(self, dummy_model):
        model, _ = dummy_model
        profile = profile_model(model)
        assert "model_size_bytes" in profile
        assert "model_size_kb" in profile
        assert profile["model_size_bytes"] > 0

    def test_estimator_count(self, dummy_model):
        model, _ = dummy_model
        profile = profile_model(model)
        assert profile.get("n_estimators", 10) == 10


class TestBenchmarkInference:
    """Tests for benchmark_inference()."""

    def test_returns_latency_stats(self, dummy_model):
        model, X = dummy_model
        result = benchmark_inference(model, X, n_iterations=50)

        assert "mean_latency_ms" in result
        assert "median_latency_ms" in result
        assert "p95_latency_ms" in result
        assert "throughput_samples_per_sec" in result

    def test_latency_is_positive(self, dummy_model):
        model, X = dummy_model
        result = benchmark_inference(model, X, n_iterations=50)
        assert result["mean_latency_ms"] > 0
        assert result["throughput_samples_per_sec"] > 0

    def test_percentiles_ordered(self, dummy_model):
        model, X = dummy_model
        result = benchmark_inference(model, X, n_iterations=100)
        assert result["min_latency_ms"] <= result["median_latency_ms"]
        assert result["median_latency_ms"] <= result["p99_latency_ms"]


class TestExportCode:
    """Tests for C/Python code export."""

    def test_c_export(self, dummy_model, tmp_path):
        model, _ = dummy_model
        c_path = str(tmp_path / "test_model.c")
        result = export_to_c_code(model, c_path)

        assert os.path.exists(c_path)
        assert os.path.getsize(c_path) > 0
        with open(c_path) as f:
            code = f.read()
        assert "double" in code  # C code should have double types

    def test_python_export(self, dummy_model, tmp_path):
        model, _ = dummy_model
        py_path = str(tmp_path / "test_model.py")
        result = export_to_python_code(model, py_path)

        assert os.path.exists(py_path)
        assert os.path.getsize(py_path) > 0
        with open(py_path) as f:
            code = f.read()
        assert "def score" in code  # m2cgen generates a score function


class TestGenerateEdgeReport:
    """Tests for generate_edge_report()."""

    def test_full_report(self, dummy_model, tmp_path):
        model, X = dummy_model
        report = generate_edge_report(
            model, X, output_dir=str(tmp_path), model_name="test_model"
        )

        assert "profile" in report
        assert "benchmark" in report
        assert report["model_name"] == "test_model"

    def test_report_files_created(self, dummy_model, tmp_path):
        model, X = dummy_model
        generate_edge_report(
            model, X, output_dir=str(tmp_path), model_name="test_model"
        )

        assert os.path.exists(tmp_path / "test_model.c")
        assert os.path.exists(tmp_path / "test_model_pure.py")
