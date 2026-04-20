"""
Weighted Linear Fusion Plugin
==============================
Wraps the existing weighted linear risk fusion as a
``BaseFusionEngine``.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from src.core.base import BaseFusionEngine

logger = logging.getLogger("pilot_readiness.fusion.weighted")


class WeightedLinearFusion(BaseFusionEngine):
    """
    Fuse multi-modal scores with a weighted linear combination:

        R = Σ wᵢ × sᵢ / Σ wᵢ

    Supports dynamic signal-quality adjustment: if a ``signal_quality``
    key is present in kwargs, the physiological channel weight is
    scaled down proportionally.
    """

    name = "weighted_linear"

    def __init__(self, default_weights: Optional[Dict[str, float]] = None):
        self.default_weights = default_weights or {"stress": 0.6, "performance": 0.4}

    def fuse(
        self,
        scores: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> float:
        w = dict(weights or self.default_weights)
        signal_quality = kwargs.get("signal_quality")

        # Dynamic weight adjustment based on signal quality
        if signal_quality is not None:
            sq = float(np.clip(signal_quality, 0.0, 1.0))
            if "stress" in w:
                original_stress_w = w["stress"]
                w["stress"] *= sq
                # Redistribute lost weight
                if "performance" in w:
                    w["performance"] += original_stress_w * (1 - sq) * 0.5

        # Normalize weights
        w_total = sum(w.values())
        if w_total <= 0:
            return 0.5

        risk = sum(
            (w.get(ch, 0.0) / w_total) * float(np.clip(s, 0, 1))
            for ch, s in scores.items()
            if ch in w
        )

        return float(np.clip(risk, 0.0, 1.0))

    def fuse_batch(
        self,
        scores: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Vectorized batch fusion for efficiency."""
        w = dict(weights or self.default_weights)
        signal_quality_arr = kwargs.get("signal_quality")

        # Get all channels
        channels = [ch for ch in scores if ch in w]
        if not channels:
            n = len(next(iter(scores.values())))
            return np.full(n, 0.5)

        n = len(scores[channels[0]])

        if signal_quality_arr is not None:
            # Per-sample weighting
            sq = np.clip(signal_quality_arr, 0.0, 1.0)
            adjusted_w = {}
            for ch in channels:
                if ch == "stress":
                    adjusted_w[ch] = w[ch] * sq
                elif ch == "performance":
                    adjusted_w[ch] = np.full(n, w[ch]) + w.get("stress", 0) * (1 - sq) * 0.5
                else:
                    adjusted_w[ch] = np.full(n, w[ch])

            w_total = sum(adjusted_w[ch] for ch in channels)
            w_total = np.where(w_total > 0, w_total, 1.0)

            risk = sum(
                (adjusted_w[ch] / w_total) * np.clip(scores[ch], 0, 1)
                for ch in channels
            )
        else:
            w_total = sum(w[ch] for ch in channels)
            if w_total <= 0:
                return np.full(n, 0.5)

            risk = sum(
                (w[ch] / w_total) * np.clip(scores[ch], 0, 1)
                for ch in channels
            )

        return np.clip(risk, 0.0, 1.0)
