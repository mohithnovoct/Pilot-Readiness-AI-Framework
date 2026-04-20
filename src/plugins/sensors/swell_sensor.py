"""
SWELL Sensor Adapter
=====================
Wraps the existing ``swell_loader`` as a ``BaseSensorAdapter``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.core.base import BaseSensorAdapter, SensorData

logger = logging.getLogger("pilot_readiness.sensors.swell")


class SWELLSensor(BaseSensorAdapter):
    """
    Load physiological and behavioral data from the SWELL-KW dataset.
    """

    name = "swell"

    def load(self, data_dir: Optional[str] = None, **kwargs) -> SensorData:
        from src.data.swell_loader import load_swell_physio, load_swell_behavioral

        logger.info("Loading SWELL dataset...")

        physio_df = load_swell_physio(data_dir=data_dir)
        beh_df = load_swell_behavioral(data_dir=data_dir)

        # Convert to signal arrays for the standard interface
        signals = {}
        sampling_rates = {}

        if "HR" in physio_df.columns:
            signals["HR"] = physio_df["HR"].values.astype(float)
            sampling_rates["HR"] = 1.0  # 1 sample per timestamp row
        if "RMSSD" in physio_df.columns:
            signals["RMSSD"] = physio_df["RMSSD"].values.astype(float)
            sampling_rates["RMSSD"] = 1.0
        if "SCL" in physio_df.columns:
            signals["SCL"] = physio_df["SCL"].values.astype(float)
            sampling_rates["SCL"] = 1.0

        labels = None
        if "label" in physio_df.columns:
            labels = physio_df["label"].values
        elif "Condition" in physio_df.columns:
            # Map conditions to numeric labels
            cond_map = {"No Stress": 0, "Time Pressure": 1, "Interruption": 1}
            labels = physio_df["Condition"].map(cond_map).fillna(0).values.astype(int)

        subject_id = "SWELL_multi"
        if "subject_id" in physio_df.columns:
            subject_id = str(physio_df["subject_id"].iloc[0])

        return SensorData(
            signals=signals,
            sampling_rates=sampling_rates,
            labels=labels,
            subject_id=subject_id,
            metadata={
                "dataset": "SWELL-KW",
                "n_subjects": physio_df["subject_id"].nunique() if "subject_id" in physio_df.columns else 1,
                "physio_df": physio_df,
                "behavioral_df": beh_df,
            },
        )
