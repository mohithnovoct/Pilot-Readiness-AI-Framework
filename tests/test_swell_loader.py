"""
Unit Tests: SWELL Data Loader
================================
Tests for SWELL dataset loading and processing.
Uses synthetic data to avoid requiring real SWELL files.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


class TestConditionMapping:
    """Tests for SWELL condition → label mapping."""

    def test_condition_map_values(self):
        from src.data.swell_loader import CONDITION_MAP
        assert CONDITION_MAP["R"] == 0  # Relax → Baseline
        assert CONDITION_MAP["N"] == 0  # Neutral → Baseline
        assert CONDITION_MAP["T"] == 1  # Time pressure → Stress
        assert CONDITION_MAP["I"] == 1  # Interruption → Stress

    def test_all_conditions_covered(self):
        from src.data.swell_loader import CONDITION_MAP
        assert set(CONDITION_MAP.keys()) == {"R", "N", "T", "I"}


class TestLoadSwellPhysio:
    """Tests for load_swell_physio with synthetic data."""

    def _make_synthetic_physio_csv(self, tmp_path):
        """Create a synthetic SWELL physiology CSV."""
        n = 50
        df = pd.DataFrame({
            "PP": [f"P{i//10+1}" for i in range(n)],
            "Condition": (["R"] * 10 + ["N"] * 10 + ["T"] * 15 + ["I"] * 15),
            "HR": np.random.normal(75, 10, n),
            "RMSSD": np.random.normal(40, 15, n),
            "SCL": np.random.normal(5, 2, n),
        })

        physio_dir = tmp_path / "3 - Feature dataset" / "per sensor"
        physio_dir.mkdir(parents=True, exist_ok=True)
        csv_path = physio_dir / "D - Physiology features (HR_HRV_SCL - final).csv"
        df.to_csv(csv_path, index=False)
        return tmp_path

    def test_loads_correctly(self, tmp_path):
        data_dir = self._make_synthetic_physio_csv(tmp_path)
        from src.data.swell_loader import load_swell_physio
        df = load_swell_physio(data_dir=str(data_dir))

        assert "subject_id" in df.columns
        assert "stress_label" in df.columns
        assert "HR" in df.columns
        assert "RMSSD" in df.columns
        assert len(df) == 50

    def test_label_mapping(self, tmp_path):
        data_dir = self._make_synthetic_physio_csv(tmp_path)
        from src.data.swell_loader import load_swell_physio
        df = load_swell_physio(data_dir=str(data_dir))

        # R and N → 0, T and I → 1
        baseline_count = (df["stress_label"] == 0).sum()
        stress_count = (df["stress_label"] == 1).sum()
        assert baseline_count == 20  # 10 R + 10 N
        assert stress_count == 30   # 15 T + 15 I

    def test_numeric_conversion(self, tmp_path):
        data_dir = self._make_synthetic_physio_csv(tmp_path)
        from src.data.swell_loader import load_swell_physio
        df = load_swell_physio(data_dir=str(data_dir))

        assert df["HR"].dtype in [np.float64, np.float32]
        assert df["RMSSD"].dtype in [np.float64, np.float32]

    def test_missing_file_raises(self):
        from src.data.swell_loader import load_swell_physio
        with pytest.raises(FileNotFoundError):
            load_swell_physio(data_dir="/nonexistent/path")


class TestLoadSwellBehavioral:
    """Tests for load_swell_behavioral with synthetic data."""

    def _make_synthetic_behavioral_csv(self, tmp_path):
        """Create a synthetic SWELL behavioral CSV."""
        n = 40
        df = pd.DataFrame({
            "PP": [f"P{i//10+1}" for i in range(n)],
            "Condition": (["R"] * 10 + ["N"] * 10 + ["T"] * 10 + ["I"] * 10),
            "SnMouseAct": np.random.randint(0, 100, n),
            "SnLeftClicked": np.random.randint(0, 50, n),
            "SnRightClicked": np.random.randint(0, 10, n),
            "SnDoubleClicked": np.random.randint(0, 5, n),
            "SnWheel": np.random.randint(0, 30, n),
            "SnDragged": np.random.randint(0, 10, n),
            "SnMouseDistance": np.random.uniform(0, 10000, n),
            "SnKeyStrokes": np.random.randint(0, 200, n),
            "SnChars": np.random.randint(0, 150, n),
            "SnSpecialKeys": np.random.randint(0, 30, n),
            "SnDirectionKeys": np.random.randint(0, 20, n),
            "SnErrorKeys": np.random.randint(0, 10, n),
            "SnShortcutKeys": np.random.randint(0, 15, n),
            "SnSpaces": np.random.randint(0, 50, n),
            "SnAppChange": np.random.randint(0, 5, n),
            "SnTabfocusChange": np.random.randint(0, 5, n),
        })

        beh_dir = tmp_path / "3 - Feature dataset" / "per sensor"
        beh_dir.mkdir(parents=True, exist_ok=True)
        csv_path = beh_dir / "A - Computer interaction features.csv"
        df.to_csv(csv_path, index=False)
        return tmp_path

    def test_loads_correctly(self, tmp_path):
        data_dir = self._make_synthetic_behavioral_csv(tmp_path)
        from src.data.swell_loader import load_swell_behavioral
        df = load_swell_behavioral(data_dir=str(data_dir))

        assert "subject_id" in df.columns
        assert "workload_level" in df.columns
        assert len(df) == 40

    def test_workload_mapping(self, tmp_path):
        data_dir = self._make_synthetic_behavioral_csv(tmp_path)
        from src.data.swell_loader import load_swell_behavioral
        df = load_swell_behavioral(data_dir=str(data_dir))

        assert set(df["workload_level"].unique()) == {0, 1}

    def test_missing_file_raises(self):
        from src.data.swell_loader import load_swell_behavioral
        with pytest.raises(FileNotFoundError):
            load_swell_behavioral(data_dir="/nonexistent/path")
