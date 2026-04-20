"""
Bayesian Fusion Engine
=======================
Non-linear fusion strategy that combines model outputs using
Bayesian inference with uncertainty estimates.

Instead of a fixed weighted average, this engine:
  - Treats each channel's score as a likelihood
  - Maintains a prior belief about pilot readiness
  - Incorporates model confidence / uncertainty
  - Produces a posterior risk score with a credible interval

This yields more principled fusion when:
  - Channels have different noise levels
  - Model confidence varies across samples
  - Temporal context matters (prior from recent history)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, Optional

import numpy as np

from src.core.base import BaseFusionEngine

logger = logging.getLogger("pilot_readiness.fusion.bayesian")


class BayesianFusion(BaseFusionEngine):
    """
    Bayesian multi-modal fusion engine.

    Models the prior probability of risk using a Beta distribution,
    updates it with each channel's evidence (likelihood), and outputs
    the posterior mean as the fused risk score.

    Parameters
    ----------
    prior_alpha : float
        Beta prior α (higher = bias toward higher risk).
    prior_beta : float
        Beta prior β (higher = bias toward lower risk).
    temporal_decay : float
        How much prior weight decays per time step (0-1).
        0 = no memory (pure likelihood), 1 = max memory.
    min_confidence : float
        Minimum confidence to accept a channel's contribution.
    """

    name = "bayesian"

    def __init__(
        self,
        prior_alpha: float = 2.0,
        prior_beta: float = 5.0,
        temporal_decay: float = 0.8,
        min_confidence: float = 0.1,
        history_size: int = 20,
    ):
        self.prior_alpha_init = prior_alpha
        self.prior_beta_init = prior_beta
        self.temporal_decay = temporal_decay
        self.min_confidence = min_confidence

        # Running prior (updated with each fusion call)
        self._alpha = prior_alpha
        self._beta = prior_beta

        # History for temporal context
        self._history: deque = deque(maxlen=history_size)

    def fuse(
        self,
        scores: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> float:
        """
        Bayesian fusion of multi-modal scores.

        Parameters
        ----------
        scores : dict
            Channel name → score ∈ [0, 1].
        weights : dict, optional
            Channel weights (used as confidence multipliers).
        **kwargs
            ``confidences`` : dict — per-channel confidence ∈ [0, 1].
            ``signal_quality`` : float — ECG quality metric.

        Returns
        -------
        float
            Posterior risk score ∈ [0, 1].
        """
        w = weights or {}
        confidences = kwargs.get("confidences", {})
        signal_quality = kwargs.get("signal_quality")

        # Start from the current prior
        alpha = self._alpha
        beta = self._beta

        for channel, score in scores.items():
            score = float(np.clip(score, 0.001, 0.999))

            # Get channel confidence
            conf = confidences.get(channel, w.get(channel, 1.0))
            if signal_quality is not None and channel == "stress":
                conf *= float(np.clip(signal_quality, 0.1, 1.0))

            if conf < self.min_confidence:
                continue

            # Beta-Bernoulli update:
            # Treating score as a "soft observation" weighted by confidence
            # If score > 0.5 → more evidence for high risk (increase α)
            # If score < 0.5 → more evidence for low risk (increase β)
            pseudo_count = conf * 2.0  # weight of this observation
            alpha += pseudo_count * score
            beta += pseudo_count * (1.0 - score)

        # Posterior mean of Beta(α, β)
        posterior_mean = alpha / (alpha + beta)

        # Update the running prior with temporal decay
        # Decayed prior moves toward the initial prior
        self._alpha = (
            self.temporal_decay * alpha
            + (1 - self.temporal_decay) * self.prior_alpha_init
        )
        self._beta = (
            self.temporal_decay * beta
            + (1 - self.temporal_decay) * self.prior_beta_init
        )

        # Track history
        self._history.append(float(posterior_mean))

        return float(np.clip(posterior_mean, 0.0, 1.0))

    def fuse_batch(
        self,
        scores: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Batch fusion (processes sequentially to maintain temporal context)."""
        n = len(next(iter(scores.values())))
        out = np.empty(n)

        for i in range(n):
            row = {k: float(v[i]) for k, v in scores.items()}
            out[i] = self.fuse(row, weights, **kwargs)

        return out

    def get_credible_interval(self, credibility: float = 0.95) -> tuple:
        """
        Return the credible interval for the current risk estimate.

        Uses the current Beta posterior parameters.
        """
        from scipy.stats import beta as beta_dist
        low = beta_dist.ppf((1 - credibility) / 2, self._alpha, self._beta)
        high = beta_dist.ppf(1 - (1 - credibility) / 2, self._alpha, self._beta)
        return (float(low), float(high))

    def get_posterior_mean(self) -> float:
        """Current posterior mean risk."""
        return self._alpha / (self._alpha + self._beta)

    def get_posterior_variance(self) -> float:
        """Current posterior variance."""
        a, b = self._alpha, self._beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def reset(self):
        """Reset to initial prior."""
        self._alpha = self.prior_alpha_init
        self._beta = self.prior_beta_init
        self._history.clear()

    def get_trend(self) -> Optional[float]:
        """Trend in posterior mean over recent fusions."""
        if len(self._history) < 3:
            return None
        recent = np.array(list(self._history)[-5:])
        return float(np.polyfit(range(len(recent)), recent, 1)[0])
