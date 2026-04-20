"""
WESAD Sensor Adapter
=====================
Wraps the existing ``wesad_loader`` as a ``BaseSensorAdapter``.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from src.core.base import BaseSensorAdapter, SensorData

logger = logging.getLogger("pilot_readiness.sensors.wesad")


class WESADSensor(BaseSensorAdapter):
    """
    Load physiological data from the WESAD dataset .pkl files.

    Parameters (via ``load()``)
    ---------------------------
    subject_id : str
        Subject identifier, e.g. ``"S2"``.
    data_dir : str, optional
        Path to the WESAD data directory.
    """

    name = "wesad"

    def load(self, subject_id: str = "S2", data_dir: Optional[str] = None, **kwargs) -> SensorData:
        from src.data.wesad_loader import (
            load_subject_pkl, extract_chest_signals, extract_labels, WESAD_DIR,
        )

        ddir = data_dir or WESAD_DIR
        logger.info("Loading WESAD subject %s from %s", subject_id, ddir)

        raw = load_subject_pkl(subject_id, ddir)
        chest = extract_chest_signals(raw)
        labels = extract_labels(raw)

        signals = {}
        sampling_rates = {}

        if "ECG" in chest:
            signals["ECG"] = chest["ECG"]
            sampling_rates["ECG"] = 700.0
        if "EDA" in chest:
            signals["EDA"] = chest["EDA"]
            sampling_rates["EDA"] = 700.0
        if "EMG" in chest:
            signals["EMG"] = chest["EMG"]
            sampling_rates["EMG"] = 700.0
        if "Resp" in chest:
            signals["Resp"] = chest["Resp"]
            sampling_rates["Resp"] = 700.0

        return SensorData(
            signals=signals,
            sampling_rates=sampling_rates,
            labels=labels,
            subject_id=subject_id,
            metadata={"dataset": "WESAD", "device": "RespiBAN"},
        )
