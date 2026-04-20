"""
HRV Feature Extractor Plugin
==============================
Wraps the existing HRV feature extraction pipeline as a
``BaseFeatureExtractor``.
"""

from __future__ import annotations

import logging
from typing import Optional, List

import numpy as np
import pandas as pd

from src.core.base import BaseFeatureExtractor, FeatureSet, SensorData
from config import HRV_FEATURE_COLS

logger = logging.getLogger("pilot_readiness.extractors.hrv")


class HRVExtractor(BaseFeatureExtractor):
    """
    Extract Heart Rate Variability features from ECG signals.

    Performs: ECG → bandpass filter → R-peak detection → RR intervals
    → 15 HRV features (time-domain, frequency-domain, non-linear).
    """

    name = "hrv"
    feature_names = list(HRV_FEATURE_COLS)

    def __init__(self, window_sec: int = 60, overlap: float = 0.5):
        self.window_sec = window_sec
        self.overlap = overlap

    def extract(self, data: SensorData, **kwargs) -> FeatureSet:
        from src.data.wesad_loader import create_windowed_dataset
        from src.data.preprocessing import extract_ibi_from_ecg
        from src.features.hrv_features import extract_all_hrv_features
        from config import LABEL_MAP

        window_sec = kwargs.get("window_sec", self.window_sec)
        overlap = kwargs.get("overlap", self.overlap)

        if "ECG" not in data.signals:
            logger.warning("No ECG signal found in sensor data")
            return FeatureSet(dataframe=pd.DataFrame(), feature_names=self.feature_names)

        ecg = data.signals["ECG"]
        fs = data.sampling_rates.get("ECG", 700.0)
        labels = data.labels

        # Window the signal
        valid_labels = kwargs.get("valid_labels", [1, 2])
        windows = create_windowed_dataset(
            ecg, labels, fs=int(fs),
            window_sec=window_sec,
            overlap=overlap,
            valid_labels=valid_labels,
        )

        all_rows = []
        for i, w in enumerate(windows):
            ecg_window = w["ecg_window"]
            rr = extract_ibi_from_ecg(ecg_window, fs=fs)
            if len(rr) < 10:
                continue

            hrv = extract_all_hrv_features(rr)
            row = {
                "subject_id": data.subject_id,
                "window_idx": i,
                "label": w["label"],
                "label_name": LABEL_MAP.get(w["label"], "unknown"),
            }
            row.update(hrv)
            all_rows.append(row)

        df = pd.DataFrame(all_rows)

        # Drop rows with too many NaN features
        if len(df) > 0:
            feat_cols = [c for c in df.columns if c in self.feature_names]
            df = df.dropna(subset=feat_cols, thresh=max(1, int(len(feat_cols) * 0.7)))

        logger.info("Extracted %d HRV feature vectors from %s", len(df), data.subject_id)

        return FeatureSet(
            dataframe=df,
            feature_names=self.feature_names,
            metadata={"window_sec": window_sec, "overlap": overlap},
        )
