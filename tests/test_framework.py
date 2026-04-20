"""
Framework Integration Tests
=============================
Tests the core framework components — ABCs, registry, config,
orchestrator, alerts, signal quality, live pipeline, and plugins.
"""

import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ── Core Base Classes ────────────────────────────────────────

class TestDataContainers:
    def test_sensor_data(self):
        from src.core.base import SensorData
        sd = SensorData(
            signals={"ECG": np.array([1.0, 2.0, 3.0])},
            sampling_rates={"ECG": 700.0},
            subject_id="S2",
        )
        assert sd.subject_id == "S2"
        assert "ECG" in sd.signals
        assert sd.sampling_rates["ECG"] == 700.0

    def test_prediction_result(self):
        from src.core.base import PredictionResult
        pr = PredictionResult(
            risk_score=0.75,
            stress_probability=0.8,
            performance_score=0.6,
            alert_level="WARNING",
            decision="WARNING",
        )
        assert pr.risk_score == 0.75
        assert pr.alert_level == "WARNING"

    def test_alert_level_enum(self):
        from src.core.base import AlertLevel
        assert AlertLevel.NOMINAL.value == 0
        assert AlertLevel.CRITICAL.value == 3
        assert AlertLevel.WARNING.value > AlertLevel.CAUTION.value


# ── Registry ─────────────────────────────────────────────────

class TestRegistry:
    def test_register_and_get(self):
        from src.core.registry import ComponentRegistry
        from src.plugins.fusion.weighted_fusion import WeightedLinearFusion

        reg = ComponentRegistry()
        fusion = WeightedLinearFusion()
        reg.register("fusion", fusion, name="test_fusion")
        retrieved = reg.get("fusion", "test_fusion")
        assert retrieved is fusion

    def test_register_class_lazy(self):
        from src.core.registry import ComponentRegistry
        from src.plugins.fusion.weighted_fusion import WeightedLinearFusion

        reg = ComponentRegistry()
        reg.register_class("fusion", WeightedLinearFusion, name="lazy_fusion")
        assert reg.has("fusion", "lazy_fusion")
        instance = reg.get("fusion", "lazy_fusion")
        assert isinstance(instance, WeightedLinearFusion)

    def test_list_components(self):
        from src.core.registry import ComponentRegistry
        from src.plugins.fusion.weighted_fusion import WeightedLinearFusion

        reg = ComponentRegistry()
        reg.register("fusion", WeightedLinearFusion())
        names = reg.list("fusion")
        assert "weighted_linear" in names

    def test_invalid_category(self):
        from src.core.registry import ComponentRegistry
        reg = ComponentRegistry()
        with pytest.raises(ValueError, match="Invalid category"):
            reg.register("invalid", object())

    def test_auto_discover(self):
        from src.core.registry import ComponentRegistry
        reg = ComponentRegistry()
        reg.auto_discover()
        all_components = reg.list_all()
        # Should find at least the built-in plugins
        assert len(reg.list("fusion")) >= 1
        assert len(reg.list("threshold")) >= 1
        assert len(reg.list("model")) >= 1


# ── Config ───────────────────────────────────────────────────

class TestConfig:
    def test_defaults(self):
        from src.core.config_schema import FrameworkConfig
        cfg = FrameworkConfig()
        assert cfg.dataset == "wesad"
        assert cfg.threshold.alpha == 0.05
        assert cfg.fusion.weights["stress"] == 0.6

    def test_from_dict(self):
        from src.core.config_schema import FrameworkConfig
        cfg = FrameworkConfig.from_dict({
            "dataset": "swell",
            "fusion": {"name": "bayesian", "weights": {"stress": 0.7, "performance": 0.3}},
            "threshold": {"alpha": 0.01},
        })
        assert cfg.dataset == "swell"
        assert cfg.fusion.name == "bayesian"
        assert cfg.fusion.weights["stress"] == 0.7
        assert cfg.threshold.alpha == 0.01

    def test_from_yaml(self, tmp_path):
        from src.core.config_schema import FrameworkConfig
        yaml_path = tmp_path / "test_config.yaml"
        cfg = FrameworkConfig()
        cfg.dataset = "swell"
        cfg.save_yaml(str(yaml_path))

        loaded = FrameworkConfig.from_yaml(str(yaml_path))
        assert loaded.dataset == "swell"

    def test_to_dict_roundtrip(self):
        from src.core.config_schema import FrameworkConfig
        cfg = FrameworkConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["dataset"] == "wesad"


# ── Fusion Plugins ───────────────────────────────────────────

class TestWeightedFusion:
    def test_basic_fusion(self):
        from src.plugins.fusion.weighted_fusion import WeightedLinearFusion
        fusion = WeightedLinearFusion()
        score = fusion.fuse({"stress": 0.8, "performance": 0.4})
        assert 0 <= score <= 1
        # With default weights (0.6, 0.4): 0.6*0.8 + 0.4*0.4 = 0.64
        assert abs(score - 0.64) < 0.01

    def test_batch_fusion(self):
        from src.plugins.fusion.weighted_fusion import WeightedLinearFusion
        fusion = WeightedLinearFusion()
        scores = fusion.fuse_batch({
            "stress": np.array([0.2, 0.5, 0.9]),
            "performance": np.array([0.1, 0.4, 0.8]),
        })
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_signal_quality_adjustment(self):
        from src.plugins.fusion.weighted_fusion import WeightedLinearFusion
        fusion = WeightedLinearFusion()
        # Low quality should reduce stress weight
        high_q = fusion.fuse({"stress": 0.8, "performance": 0.4}, signal_quality=1.0)
        low_q = fusion.fuse({"stress": 0.8, "performance": 0.4}, signal_quality=0.2)
        # With low quality, stress contribution drops, so score should differ
        assert high_q != low_q


class TestBayesianFusion:
    def test_basic_fusion(self):
        from src.plugins.fusion.bayesian_fusion import BayesianFusion
        fusion = BayesianFusion()
        score = fusion.fuse({"stress": 0.8, "performance": 0.6})
        assert 0 <= score <= 1

    def test_temporal_context(self):
        from src.plugins.fusion.bayesian_fusion import BayesianFusion
        fusion = BayesianFusion(temporal_decay=0.9)
        # Feed consistently high scores
        for _ in range(10):
            fusion.fuse({"stress": 0.9, "performance": 0.8})
        high_score = fusion.get_posterior_mean()

        # Reset and feed low scores
        fusion.reset()
        for _ in range(10):
            fusion.fuse({"stress": 0.1, "performance": 0.2})
        low_score = fusion.get_posterior_mean()

        assert high_score > low_score


# ── Threshold ────────────────────────────────────────────────

class TestNeymanPearsonThreshold:
    def test_compute_threshold(self):
        from src.plugins.threshold.neyman_pearson import NeymanPearsonThreshold
        np_thresh = NeymanPearsonThreshold(alpha=0.05)
        baseline = np.random.uniform(0.1, 0.4, 100)
        gamma = np_thresh.compute_threshold(baseline)
        assert 0 < gamma < 1

    def test_multi_level_decide(self):
        from src.plugins.threshold.neyman_pearson import NeymanPearsonThreshold
        from src.core.base import AlertLevel
        np_thresh = NeymanPearsonThreshold()

        level, decision = np_thresh.decide(0.2)
        assert level == AlertLevel.NOMINAL

        level, decision = np_thresh.decide(0.5)
        assert level == AlertLevel.CAUTION

        level, decision = np_thresh.decide(0.7)
        assert level == AlertLevel.WARNING

        level, decision = np_thresh.decide(0.9)
        assert level == AlertLevel.CRITICAL


# ── Alert System ─────────────────────────────────────────────

class TestAlertManager:
    def test_process_prediction(self):
        from src.core.alerts import AlertManager
        from src.core.base import PredictionResult
        mgr = AlertManager(escalation_window=2, cooldown_seconds=0)

        pred = PredictionResult(
            risk_score=0.9,
            stress_probability=0.85,
            performance_score=0.7,
            alert_level="CRITICAL",
            decision="CRITICAL",
        )
        event = mgr.process(pred)
        assert event.level is not None
        assert event.risk_score == 0.9

    def test_statistics(self):
        from src.core.alerts import AlertManager
        from src.core.base import PredictionResult
        mgr = AlertManager(escalation_window=1, cooldown_seconds=0)

        for _ in range(5):
            mgr.process(PredictionResult(
                risk_score=0.3, stress_probability=0.2,
                performance_score=0.1, alert_level="NOMINAL", decision="READY",
            ))
        stats = mgr.get_statistics()
        assert stats["total_events"] == 5


# ── Signal Quality ───────────────────────────────────────────

class TestSignalQuality:
    def test_good_signal(self):
        from src.core.signal_quality import SignalQualityAssessor
        assessor = SignalQualityAssessor()

        # Simulate plausible RR intervals (normal sinus rhythm)
        rr = np.random.normal(0.8, 0.05, 60)
        # Generate a fake ECG window
        ecg = np.random.randn(700 * 60) * 0.5  # 60 seconds
        report = assessor.assess_ecg_window(ecg, rr_intervals=rr, fs=700.0)
        assert report.rr_plausibility > 0.5
        assert report.n_peaks == 61

    def test_flat_line_rejected(self):
        from src.core.signal_quality import SignalQualityAssessor
        assessor = SignalQualityAssessor()
        ecg = np.zeros(700 * 60)
        report = assessor.assess_ecg_window(ecg, fs=700.0)
        assert not report.is_acceptable
        assert "Flat-line" in report.reasons[0]


# ── Live Pipeline ─────────────────────────────────────────────

class TestRingBuffer:
    def test_basic_operations(self):
        from src.core.live_pipeline import RingBuffer
        buf = RingBuffer(100)
        assert buf.count == 0

        buf.append(np.arange(50, dtype=float))
        assert buf.count == 50

        last_10 = buf.get_last(10)
        np.testing.assert_array_equal(last_10, np.arange(40, 50, dtype=float))

    def test_overflow(self):
        from src.core.live_pipeline import RingBuffer
        buf = RingBuffer(20)
        buf.append(np.arange(30, dtype=float))
        assert buf.count == 20

        last_5 = buf.get_last(5)
        np.testing.assert_array_equal(last_5, np.arange(25, 30, dtype=float))


# ── Framework Orchestrator ───────────────────────────────────

class TestFramework:
    def test_instantiation(self):
        from src.core.framework import PilotReadinessFramework
        fw = PilotReadinessFramework()
        assert not fw.is_fitted
        assert fw.config.dataset == "wesad"

    def test_configure_dict(self):
        from src.core.framework import PilotReadinessFramework
        fw = PilotReadinessFramework()
        fw.configure({"dataset": "swell", "threshold": {"alpha": 0.01}})
        assert fw.config.dataset == "swell"
        assert fw.config.threshold.alpha == 0.01

    def test_list_components(self):
        from src.core.framework import PilotReadinessFramework
        fw = PilotReadinessFramework()
        components = fw.list_components()
        assert isinstance(components, dict)
        assert "fusion" in components
        assert "threshold" in components

    def test_repr(self):
        from src.core.framework import PilotReadinessFramework
        fw = PilotReadinessFramework()
        r = repr(fw)
        assert "PilotReadinessFramework" in r

    def test_fit_and_predict(self):
        from src.core.framework import PilotReadinessFramework
        fw = PilotReadinessFramework()
        fw.configure({"models": [{"name": "lightgbm_stress"}]})

        # Create synthetic data
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 15)
        y = np.random.randint(0, 2, n)

        metrics = fw.fit(X=X, y=y)
        assert fw.is_fitted
        assert "lightgbm_stress" in metrics

        results = fw.predict(X=X[:5])
        assert len(results) == 5
        assert all(0 <= r.risk_score <= 1 for r in results)
        assert all(r.alert_level in ("NOMINAL", "CAUTION", "WARNING", "CRITICAL") for r in results)


# ── Calibration ──────────────────────────────────────────────

class TestCalibration:
    def test_calibrate_and_save(self, tmp_path):
        from src.core.calibration import CalibrationSession
        from src.plugins.models.lightgbm_model import LightGBMStressModel
        from src.plugins.fusion.weighted_fusion import WeightedLinearFusion
        from src.plugins.threshold.neyman_pearson import NeymanPearsonThreshold

        # Train a quick model
        np.random.seed(42)
        model = LightGBMStressModel()
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        model.train(X, y)

        session = CalibrationSession(
            pilot_id="TEST_PILOT",
            profile_dir=str(tmp_path),
        )
        profile = session.calibrate(
            features=X[:20],
            models={"lightgbm_stress": model},
            fusion=WeightedLinearFusion(),
            threshold_strategy=NeymanPearsonThreshold(),
        )

        assert profile["pilot_id"] == "TEST_PILOT"
        assert "threshold" in profile
        assert profile["n_baseline_windows"] == 20

        # Save and reload
        path = session.save()
        loaded = CalibrationSession.load_profile("TEST_PILOT", str(tmp_path))
        assert loaded.pilot_id == "TEST_PILOT"
        assert abs(loaded.threshold - profile["threshold"]) < 1e-6
