"""
Fatigue Feature Extractor Plugin
==================================
Estimates pilot fatigue from HRV trends and circadian-aware
heuristics.

Fatigue Indicators
------------------
- HF power decline over time (vagal withdrawal)
- Heart rate trend (gradual increase)
- HRV complexity reduction (SampEn decline)
- Time-on-task effect
- Circadian modulation (time-of-day)

This provides a third readiness channel alongside stress and
performance for more comprehensive pilot monitoring.
"""

from __future__ import annotations

import logging
from typing import Optional, List

import numpy as np
import pandas as pd

from src.core.base import BaseFeatureExtractor, FeatureSet, SensorData

logger = logging.getLogger("pilot_readiness.extractors.fatigue")


class FatigueExtractor(BaseFeatureExtractor):
    """
    Extract fatigue-related features from physiological signals.

    Features produced:
    - ``hr_trend``: slope of heart rate over the session
    - ``hf_power_decline``: rate of HF power decrease (vagal withdrawal)
    - ``hrv_complexity_trend``: SampEn trend (declining = fatigue)
    - ``sdnn_trend``: SDNN trend over windows
    - ``time_on_task_hours``: hours since session start
    - ``fatigue_composite``: weighted composite fatigue score ∈ [0, 1]
    """

    name = "fatigue"
    feature_names = [
        "hr_trend",
        "hf_power_decline",
        "hrv_complexity_trend",
        "sdnn_trend",
        "time_on_task_hours",
        "fatigue_composite",
    ]

    def __init__(self, session_start_hour: float = 8.0):
        """
        Parameters
        ----------
        session_start_hour : float
            Hour of day (24h) when the monitoring session started.
            Used for circadian modulation.
        """
        self.session_start_hour = session_start_hour

    def extract(self, data: SensorData, **kwargs) -> FeatureSet:
        """
        Extract fatigue features from pre-computed HRV feature windows.

        Expects ``kwargs["hrv_features_df"]`` — a DataFrame with columns
        like MeanHR, HF_power, SampEn, SDNN produced by the HRV extractor.
        If not provided, will try to extract from sensor data directly.
        """
        hrv_df = kwargs.get("hrv_features_df")

        if hrv_df is None:
            logger.info("No HRV DataFrame provided — extracting from sensor data")
            hrv_df = self._extract_hrv_from_sensor(data, **kwargs)

        if hrv_df is None or len(hrv_df) < 3:
            logger.warning("Insufficient data for fatigue estimation (need ≥3 windows)")
            return FeatureSet(
                dataframe=pd.DataFrame(),
                feature_names=self.feature_names,
            )

        return self._compute_fatigue_features(hrv_df)

    def _extract_hrv_from_sensor(self, data: SensorData, **kwargs) -> Optional[pd.DataFrame]:
        """Try to extract HRV features from raw sensor data."""
        try:
            from src.plugins.extractors.hrv_extractor import HRVExtractor
            extractor = HRVExtractor()
            result = extractor.extract(data, **kwargs)
            return result.dataframe
        except Exception as exc:
            logger.warning("HRV extraction failed: %s", exc)
            return None

    def _compute_fatigue_features(self, hrv_df: pd.DataFrame) -> FeatureSet:
        """Compute fatigue trends from an HRV feature DataFrame."""
        n_windows = len(hrv_df)
        time_axis = np.arange(n_windows, dtype=float)

        rows = []

        for idx in range(n_windows):
            # Use all windows up to current point for trend computation
            window_slice = hrv_df.iloc[:idx + 1]
            n = len(window_slice)

            row = {"window_idx": idx, "subject_id": hrv_df.iloc[idx].get("subject_id", "unknown")}

            # ── HR trend (slope of MeanHR over time) ──────────
            hr_trend = 0.0
            if "MeanHR" in window_slice.columns and n >= 2:
                hr = window_slice["MeanHR"].dropna().values
                if len(hr) >= 2:
                    t = np.arange(len(hr))
                    hr_trend = float(np.polyfit(t, hr, 1)[0])
            row["hr_trend"] = hr_trend

            # ── HF power decline (vagal withdrawal) ──────────
            hf_decline = 0.0
            if "HF_power" in window_slice.columns and n >= 2:
                hf = window_slice["HF_power"].dropna().values
                if len(hf) >= 2:
                    t = np.arange(len(hf))
                    hf_decline = float(-np.polyfit(t, hf, 1)[0])  # Positive = declining
            row["hf_power_decline"] = hf_decline

            # ── HRV complexity trend (SampEn) ─────────────────
            complexity_trend = 0.0
            if "SampEn" in window_slice.columns and n >= 2:
                se = window_slice["SampEn"].dropna().values
                if len(se) >= 2:
                    t = np.arange(len(se))
                    complexity_trend = float(-np.polyfit(t, se, 1)[0])  # Positive = declining
            row["hrv_complexity_trend"] = complexity_trend

            # ── SDNN trend ────────────────────────────────────
            sdnn_trend = 0.0
            if "SDNN" in window_slice.columns and n >= 2:
                sdnn = window_slice["SDNN"].dropna().values
                if len(sdnn) >= 2:
                    t = np.arange(len(sdnn))
                    sdnn_trend = float(-np.polyfit(t, sdnn, 1)[0])
            row["sdnn_trend"] = sdnn_trend

            # ── Time on task ──────────────────────────────────
            # Assume 1-minute windows
            time_on_task = (idx + 1) / 60.0  # hours
            row["time_on_task_hours"] = time_on_task

            # ── Composite fatigue score ───────────────────────
            # Normalise each component to [0, 1] and combine
            norm_hr = float(np.clip(hr_trend / 0.5, 0, 1))       # 0.5 bpm/window = max
            norm_hf = float(np.clip(hf_decline / 100.0, 0, 1))   # 100 units/window = max
            norm_cx = float(np.clip(complexity_trend / 0.01, 0, 1))
            norm_time = float(np.clip(time_on_task / 8.0, 0, 1)) # 8 hours = max

            # Circadian modulation: fatigue increases in afternoon
            hour = self.session_start_hour + time_on_task
            circadian = self._circadian_factor(hour)

            composite = (
                0.25 * norm_hr
                + 0.25 * norm_hf
                + 0.15 * norm_cx
                + 0.20 * norm_time
                + 0.15 * circadian
            )
            row["fatigue_composite"] = float(np.clip(composite, 0, 1))

            rows.append(row)

        df = pd.DataFrame(rows)

        # Copy labels if present
        if "label" in hrv_df.columns:
            df["label"] = hrv_df["label"].values[:len(df)]
        if "label_name" in hrv_df.columns:
            df["label_name"] = hrv_df["label_name"].values[:len(df)]

        return FeatureSet(
            dataframe=df,
            feature_names=self.feature_names,
            metadata={"session_start_hour": self.session_start_hour},
        )

    @staticmethod
    def _circadian_factor(hour: float) -> float:
        """
        Estimate circadian fatigue based on time of day.

        Returns ∈ [0, 1]: higher = more fatigue expected.
        Peak fatigue: 14:00-16:00 (post-lunch dip) and 02:00-04:00.
        """
        hour = hour % 24
        # Two-harmonic model of circadian alertness
        # Inverted: high value = MORE fatigue
        alertness = (
            0.5
            + 0.3 * np.cos(2 * np.pi * (hour - 10) / 24)    # main cycle
            + 0.2 * np.cos(2 * np.pi * (hour - 14) / 12)     # post-lunch dip
        )
        fatigue = 1.0 - float(np.clip(alertness, 0, 1))
        return fatigue
