"""
Performance Feature Extractor Plugin
======================================
Wraps the existing performance feature extraction (real MATB-II
or simulated) as a ``BaseFeatureExtractor``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.core.base import BaseFeatureExtractor, FeatureSet, SensorData
from config import PERF_FEATURE_COLS, SWELL_BEH_COLS

logger = logging.getLogger("pilot_readiness.extractors.performance")


class PerformanceExtractor(BaseFeatureExtractor):
    """
    Extract performance / behavioral features.

    Supports:
    - SWELL behavioral data (keyboard, mouse metrics)
    - Simulated MATB-II data (reaction time, tracking, comms)
    - Real MATB-II log data (via matb_parser)
    """

    name = "performance"
    feature_names = list(PERF_FEATURE_COLS)

    def __init__(self, mode: str = "auto"):
        """
        Parameters
        ----------
        mode : str
            ``"swell"`` — extract from SWELL behavioral data.
            ``"matb_sim"`` — generate simulated MATB-II data.
            ``"matb_real"`` — parse real MATB-II log files.
            ``"auto"`` — detect from sensor metadata.
        """
        self.mode = mode

    def extract(self, data: SensorData, **kwargs) -> FeatureSet:
        mode = self.mode

        # Auto-detect mode from sensor metadata
        if mode == "auto":
            dataset = data.metadata.get("dataset", "")
            if "SWELL" in dataset:
                mode = "swell"
            elif "MATB" in dataset:
                mode = "matb_real"
            else:
                mode = "matb_sim"

        if mode == "swell":
            return self._extract_swell(data, **kwargs)
        elif mode == "matb_real":
            return self._extract_matb_real(data, **kwargs)
        else:
            return self._extract_matb_simulated(**kwargs)

    def _extract_swell(self, data: SensorData, **kwargs) -> FeatureSet:
        """Extract performance features from SWELL behavioral data."""
        beh_df = data.metadata.get("behavioral_df")

        if beh_df is None:
            logger.warning("No behavioral data in SWELL sensor data")
            return FeatureSet(dataframe=pd.DataFrame(), feature_names=list(SWELL_BEH_COLS))

        available_cols = [c for c in SWELL_BEH_COLS if c in beh_df.columns]
        self.feature_names = available_cols

        return FeatureSet(
            dataframe=beh_df,
            feature_names=available_cols,
            metadata={"mode": "swell"},
        )

    def _extract_matb_simulated(self, **kwargs) -> FeatureSet:
        """Generate simulated MATB-II performance data."""
        from src.data.matb_simulator import generate_full_simulation

        n_sessions = kwargs.get("n_sessions", 30)
        seed = kwargs.get("seed", 42)

        logger.info("Generating %d simulated MATB-II sessions per level", n_sessions)
        df = generate_full_simulation(n_sessions_per_level=n_sessions, seed=seed)

        return FeatureSet(
            dataframe=df,
            feature_names=list(PERF_FEATURE_COLS),
            metadata={"mode": "matb_simulated", "n_sessions": n_sessions},
        )

    def _extract_matb_real(self, data: SensorData, **kwargs) -> FeatureSet:
        """Parse real MATB-II log files."""
        from src.data.matb_parser import parse_matb_session

        log_dir = kwargs.get("log_dir") or data.metadata.get("matb_log_dir", "")

        if not log_dir:
            logger.warning("No MATB-II log directory specified, falling back to simulation")
            return self._extract_matb_simulated(**kwargs)

        import os
        import glob

        rows = []
        log_files = sorted(glob.glob(os.path.join(log_dir, "*.log")))

        for log_file in log_files:
            try:
                session = parse_matb_session(log_file)
                rows.append(session)
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", log_file, exc)

        if not rows:
            logger.warning("No valid MATB-II logs found, falling back to simulation")
            return self._extract_matb_simulated(**kwargs)

        df = pd.DataFrame(rows)

        return FeatureSet(
            dataframe=df,
            feature_names=[c for c in PERF_FEATURE_COLS if c in df.columns],
            metadata={"mode": "matb_real", "n_sessions": len(rows)},
        )
