"""
Integration Tests
==================
End-to-end smoke tests that verify the complete pipeline works
with synthetic data. No real WESAD/SWELL data needed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb


class TestStressClassifierPipeline:
    """Test the stress classifier end-to-end with synthetic data."""

    def _make_synthetic_features(self, n=200):
        """Create synthetic HRV features mimicking WESAD format."""
        np.random.seed(42)
        subjects = [f"S{i}" for i in range(5) for _ in range(n // 5)]
        labels = [1] * (n // 2) + [2] * (n // 2)

        df = pd.DataFrame({
            "subject_id": subjects,
            "label": labels,
            "stress_label": [0 if l == 1 else 1 for l in labels],
            "MeanNN": np.random.normal(800, 100, n),
            "SDNN": np.random.normal(50, 20, n),
            "RMSSD": np.random.normal(40, 15, n),
            "pNN50": np.random.uniform(0, 50, n),
            "MeanHR": np.random.normal(75, 10, n),
            "SDHR": np.random.normal(5, 2, n),
            "LF_power": np.random.uniform(100, 2000, n),
            "HF_power": np.random.uniform(100, 2000, n),
            "LF_HF_ratio": np.random.uniform(0.5, 5, n),
        })
        return df

    def test_prepare_training_data(self):
        from src.models.stress_classifier import prepare_training_data
        df = self._make_synthetic_features()
        feature_cols = ["MeanNN", "SDNN", "RMSSD", "pNN50", "MeanHR"]
        X, y, groups = prepare_training_data(df, feature_cols)

        assert X.shape[0] == 200
        assert X.shape[1] == 5
        assert len(y) == 200
        assert set(y) == {0, 1}
        assert len(np.unique(groups)) == 5

    def test_train_final_model(self):
        from src.models.stress_classifier import prepare_training_data, train_final_model
        df = self._make_synthetic_features()
        feature_cols = ["MeanNN", "SDNN", "RMSSD"]
        X, y, _ = prepare_training_data(df, feature_cols)

        model = train_final_model(X, y, do_grid_search=False)
        assert hasattr(model, "predict_proba")

        probs = model.predict_proba(X[:5])
        assert probs.shape == (5, 2)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_bootstrap_ci(self):
        from src.models.stress_classifier import compute_bootstrap_ci
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.6, 0.8, 0.4, 0.9, 0.15, 0.85, 0.1, 0.7])

        ci = compute_bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=100)

        assert "accuracy_ci" in ci
        assert "f1_ci" in ci
        assert "auc_ci" in ci
        assert ci["accuracy_ci"][0] <= ci["accuracy_ci"][1]
        assert ci["f1_ci"][0] <= ci["f1_ci"][1]


class TestPerformanceModelPipeline:
    """Test the performance model end-to-end."""

    def _make_synthetic_perf_data(self, n=100):
        np.random.seed(42)
        df = pd.DataFrame({
            "workload_level": np.random.choice([0, 1, 2], n),
            "mean_rt": np.random.uniform(200, 600, n),
            "std_rt": np.random.uniform(20, 100, n),
            "cv_rt": np.random.uniform(0.1, 0.5, n),
        })
        return df

    def test_prepare_data(self):
        from src.models.performance_model import prepare_performance_data
        df = self._make_synthetic_perf_data()
        feature_cols = ["mean_rt", "std_rt", "cv_rt"]
        X, y, scaler = prepare_performance_data(df, feature_cols)

        assert X.shape == (100, 3)
        assert y.min() >= 0
        assert y.max() <= 1
        assert scaler is not None

    def test_train_model(self):
        from src.models.performance_model import (
            prepare_performance_data, train_performance_model
        )
        df = self._make_synthetic_perf_data()
        feature_cols = ["mean_rt", "std_rt", "cv_rt"]
        X, y, _ = prepare_performance_data(df, feature_cols)

        model = train_performance_model(X, y, do_grid_search=False)
        preds = model.predict(X[:5])
        assert len(preds) == 5


class TestRiskFusionPipeline:
    """Test the full risk scoring pipeline."""

    def test_full_risk_pipeline(self):
        from src.risk.fusion import compute_risk_batch, get_risk_category
        from src.risk.threshold import ThresholdManager

        np.random.seed(42)
        stress_probs = np.random.uniform(0, 1, 100)
        perf_scores = np.random.uniform(0, 1, 100)

        # Fuse
        risks = compute_risk_batch(stress_probs, perf_scores, w_phys=0.6, w_perf=0.4)
        assert len(risks) == 100
        assert np.all(risks >= 0) and np.all(risks <= 1)

        # Threshold
        baseline_risks = risks[:50]
        tm = ThresholdManager(baseline_scores=baseline_risks, alpha=0.05)
        gamma = tm.set_scenario("operational")
        assert 0 < gamma < 1

        # Categorize
        cat, color = get_risk_category(0.3)
        assert cat in ["LOW", "MODERATE", "ELEVATED", "HIGH"]
        assert color.startswith("#")


class TestEdgeExportPipeline:
    """Test edge export with a real model."""

    def test_export_roundtrip(self, tmp_path):
        from src.edge.export_model import generate_edge_report

        np.random.seed(42)
        X = np.random.rand(50, 3)
        y = (X[:, 0] > 0.5).astype(int)

        model = lgb.LGBMClassifier(n_estimators=5, verbose=-1, random_state=42)
        model.fit(X, y)

        report = generate_edge_report(
            model, X, output_dir=str(tmp_path), model_name="integration_test"
        )

        assert os.path.exists(tmp_path / "integration_test.c")
        assert os.path.exists(tmp_path / "integration_test_pure.py")
        assert report["benchmark"]["mean_latency_ms"] < 100  # Should be fast


class TestVisualizationPipeline:
    """Test that visualization functions don't crash."""

    def test_dashboard_generation(self, tmp_path):
        from src.visualization.dashboard import generate_dashboard

        np.random.seed(42)
        risk_scores = np.random.uniform(0, 1, 50)
        timestamps = np.arange(50)
        threshold = 0.5

        output_path = str(tmp_path / "test_dashboard.html")
        result = generate_dashboard(
            risk_scores=risk_scores,
            timestamps=timestamps,
            threshold=threshold,
            output_path=output_path,
        )

        assert os.path.exists(output_path)
        with open(output_path) as f:
            html = f.read()
        assert "Pilot Readiness" in html
        assert "plotly" in html.lower()

    def test_static_plots(self):
        from src.visualization.plots import (
            plot_confusion_matrix, plot_feature_importance
        )

        cm = np.array([[45, 5], [8, 42]])
        fig = plot_confusion_matrix(cm)
        assert fig is not None

        imp_df = pd.DataFrame({
            "feature": ["RMSSD", "SDNN", "pNN50"],
            "importance": [0.3, 0.25, 0.15],
        })
        fig2 = plot_feature_importance(imp_df)
        assert fig2 is not None
