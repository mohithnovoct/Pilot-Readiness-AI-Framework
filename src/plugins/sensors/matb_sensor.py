"""
MATB-II Sensor Adapter
=======================
Integrates the existing ``matb_parser.py`` as a proper
``BaseSensorAdapter`` for loading real MATB-II log data.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from src.core.base import BaseSensorAdapter, SensorData

logger = logging.getLogger("pilot_readiness.sensors.matb")


class MATBSensor(BaseSensorAdapter):
    """
    Load performance data from real NASA MATB-II log files.

    Parses SYSMON (system monitoring), TRACK (tracking), and COMM
    (communications) sub-task logs into unified performance metrics.
    """

    name = "matb"

    def load(self, log_dir: Optional[str] = None, **kwargs) -> SensorData:
        """
        Load and parse MATB-II log files from a directory.

        Parameters
        ----------
        log_dir : str
            Directory containing MATB-II .log or .csv files.

        Returns
        -------
        SensorData
            Performance metrics as signal arrays.
        """
        from config import MATB_DATA_DIR
        log_dir = log_dir or MATB_DATA_DIR

        if not os.path.exists(log_dir):
            logger.warning("MATB-II log directory not found: %s", log_dir)
            logger.info("Falling back to simulated MATB-II data")
            return self._load_simulated(**kwargs)

        logger.info("Parsing MATB-II logs from %s", log_dir)

        # Try to parse with the existing matb_parser
        try:
            from src.data.matb_parser import parse_matb_session

            log_files = sorted(
                glob.glob(os.path.join(log_dir, "**", "*.log"), recursive=True)
                + glob.glob(os.path.join(log_dir, "**", "*.csv"), recursive=True)
            )

            if not log_files:
                logger.warning("No log files found in %s", log_dir)
                return self._load_simulated(**kwargs)

            sessions = []
            for log_file in log_files:
                try:
                    session = parse_matb_session(log_file)
                    sessions.append(session)
                except Exception as exc:
                    logger.debug("Failed to parse %s: %s", log_file, exc)

            if not sessions:
                logger.warning("No valid sessions parsed, falling back to simulation")
                return self._load_simulated(**kwargs)

            df = pd.DataFrame(sessions)

            signals = {}
            sampling_rates = {}

            # Extract key performance columns as signal arrays
            for col in df.select_dtypes(include=[np.number]).columns:
                signals[col] = df[col].values.astype(float)
                sampling_rates[col] = 1.0  # 1 sample per session

            labels = None
            if "workload_level" in df.columns:
                labels = df["workload_level"].values
            elif "condition" in df.columns:
                labels = df["condition"].values

            return SensorData(
                signals=signals,
                sampling_rates=sampling_rates,
                labels=labels,
                subject_id="MATB",
                metadata={
                    "dataset": "MATB-II",
                    "n_sessions": len(sessions),
                    "log_dir": log_dir,
                    "performance_df": df,
                },
            )

        except ImportError:
            logger.warning("matb_parser not available, using simulation")
            return self._load_simulated(**kwargs)

    def _load_simulated(self, **kwargs) -> SensorData:
        """Fall back to simulated MATB-II data."""
        from src.data.matb_simulator import generate_full_simulation

        n_sessions = kwargs.get("n_sessions", 30)
        seed = kwargs.get("seed", 42)

        df = generate_full_simulation(n_sessions_per_level=n_sessions, seed=seed)

        signals = {}
        sampling_rates = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ("workload_level",):
                signals[col] = df[col].values.astype(float)
                sampling_rates[col] = 1.0

        labels = df["workload_level"].values if "workload_level" in df.columns else None

        return SensorData(
            signals=signals,
            sampling_rates=sampling_rates,
            labels=labels,
            subject_id="MATB_SIM",
            metadata={
                "dataset": "MATB-II-Simulated",
                "n_sessions": n_sessions,
                "performance_df": df,
            },
        )
