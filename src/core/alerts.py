"""
Multi-Level Alert System
=========================
Provides temporal persistence, cooldown, escalation, and
configurable alert handlers for multi-level pilot readiness alerts.

Alert Levels
------------
  NOMINAL  → Everything is fine
  CAUTION  → Early warning, monitor closely
  WARNING  → Active concern, consider intervention
  CRITICAL → Immediate attention required
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.core.base import AlertLevel, BaseAlertHandler, PredictionResult

logger = logging.getLogger("pilot_readiness.alerts")


@dataclass
class AlertEvent:
    """Record of a single alert event."""
    timestamp: float
    level: AlertLevel
    risk_score: float
    decision: str
    suppressed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class LogAlertHandler(BaseAlertHandler):
    """Default handler — logs alerts to the standard logger."""

    name = "log"

    def handle(self, alert_level: AlertLevel, prediction: PredictionResult, **kwargs):
        if alert_level == AlertLevel.CRITICAL:
            logger.critical(
                "🚨 CRITICAL ALERT: risk=%.3f stress=%.3f",
                prediction.risk_score, prediction.stress_probability,
            )
        elif alert_level == AlertLevel.WARNING:
            logger.warning(
                "⚠️  WARNING: risk=%.3f stress=%.3f",
                prediction.risk_score, prediction.stress_probability,
            )
        elif alert_level == AlertLevel.CAUTION:
            logger.info(
                "🔶 CAUTION: risk=%.3f stress=%.3f",
                prediction.risk_score, prediction.stress_probability,
            )

    def should_fire(self, alert_level: AlertLevel) -> bool:
        return alert_level.value >= AlertLevel.CAUTION.value


class CallbackAlertHandler(BaseAlertHandler):
    """Alert handler that invokes a user-provided callback function."""

    name = "callback"

    def __init__(self, callback: Callable, min_level: AlertLevel = AlertLevel.CAUTION):
        self._callback = callback
        self._min_level = min_level

    def handle(self, alert_level: AlertLevel, prediction: PredictionResult, **kwargs):
        self._callback(alert_level, prediction, **kwargs)

    def should_fire(self, alert_level: AlertLevel) -> bool:
        return alert_level.value >= self._min_level.value


class WebhookAlertHandler(BaseAlertHandler):
    """Alert handler that sends alerts to a webhook URL."""

    name = "webhook"

    def __init__(self, url: str, min_level: AlertLevel = AlertLevel.WARNING):
        self._url = url
        self._min_level = min_level

    def handle(self, alert_level: AlertLevel, prediction: PredictionResult, **kwargs):
        import json
        import urllib.request

        payload = json.dumps({
            "alert_level": alert_level.name,
            "risk_score": prediction.risk_score,
            "stress_probability": prediction.stress_probability,
            "decision": prediction.decision,
            "timestamp": time.time(),
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                self._url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as exc:
            logger.error("Webhook alert failed: %s", exc)

    def should_fire(self, alert_level: AlertLevel) -> bool:
        return alert_level.value >= self._min_level.value


class AlertManager:
    """
    Manages alert lifecycle: escalation, persistence, and cooldown.

    Features
    --------
    - **Temporal persistence**: require ``escalation_window`` consecutive
      high-risk predictions before escalating the alert level.
    - **Cooldown**: suppress repeated alerts within ``cooldown_seconds``.
    - **Escalation**: automatically escalate if sustained high risk.
    - **De-escalation**: step down if risk drops below threshold for
      a sustained period.
    """

    def __init__(
        self,
        escalation_window: int = 3,
        cooldown_seconds: float = 30.0,
        handlers: Optional[List[BaseAlertHandler]] = None,
    ):
        self.escalation_window = escalation_window
        self.cooldown_seconds = cooldown_seconds
        self.handlers: List[BaseAlertHandler] = handlers or [LogAlertHandler()]

        # State
        self._current_level = AlertLevel.NOMINAL
        self._level_history: deque = deque(maxlen=max(escalation_window * 2, 10))
        self._last_alert_time: Dict[AlertLevel, float] = {}
        self._event_log: List[AlertEvent] = []

    @property
    def current_level(self) -> AlertLevel:
        return self._current_level

    @property
    def event_log(self) -> List[AlertEvent]:
        return list(self._event_log)

    def process(self, prediction: PredictionResult) -> AlertEvent:
        """
        Process a new prediction through the alert system.

        1. Determine the raw alert level from the prediction.
        2. Apply temporal persistence (escalation window).
        3. Check cooldown.
        4. Fire handlers if not suppressed.

        Parameters
        ----------
        prediction : PredictionResult

        Returns
        -------
        AlertEvent
        """
        now = time.time()

        # Determine raw level from the prediction
        raw_level = AlertLevel[prediction.alert_level]
        self._level_history.append(raw_level)

        # Apply temporal persistence
        effective_level = self._apply_persistence(raw_level)

        # Check cooldown
        suppressed = self._check_cooldown(effective_level, now)

        # Create event
        event = AlertEvent(
            timestamp=now,
            level=effective_level,
            risk_score=prediction.risk_score,
            decision=prediction.decision,
            suppressed=suppressed,
        )

        # Fire handlers if not suppressed and level changed or is high
        if not suppressed and effective_level.value >= AlertLevel.CAUTION.value:
            for handler in self.handlers:
                if handler.should_fire(effective_level):
                    try:
                        handler.handle(effective_level, prediction)
                    except Exception as exc:
                        logger.error("Alert handler %s failed: %s", handler.name, exc)

            self._last_alert_time[effective_level] = now

        # Update current level
        self._current_level = effective_level

        # Log event
        self._event_log.append(event)
        if len(self._event_log) > 1000:
            self._event_log = self._event_log[-500:]

        return event

    def _apply_persistence(self, raw_level: AlertLevel) -> AlertLevel:
        """
        Apply temporal persistence — require N consecutive alerts before escalating.
        """
        if len(self._level_history) < self.escalation_window:
            # Not enough history — use current level or NOMINAL
            return min(raw_level, self._current_level) if self._current_level != AlertLevel.NOMINAL else raw_level

        # Check last N levels
        recent = list(self._level_history)[-self.escalation_window:]
        min_recent = min(lev.value for lev in recent)

        # Escalation: all recent levels are at or above the raw level
        if all(lev.value >= raw_level.value for lev in recent):
            return raw_level

        # De-escalation: if recent levels are lower, step down
        if min_recent < self._current_level.value:
            # Step down by one level if sustained
            if all(lev.value <= self._current_level.value - 1 for lev in recent):
                new_level = AlertLevel(max(0, self._current_level.value - 1))
                return new_level

        return self._current_level

    def _check_cooldown(self, level: AlertLevel, now: float) -> bool:
        """Return True if the alert should be suppressed due to cooldown."""
        if level not in self._last_alert_time:
            return False
        elapsed = now - self._last_alert_time[level]
        return elapsed < self.cooldown_seconds

    def reset(self):
        """Reset all alert state."""
        self._current_level = AlertLevel.NOMINAL
        self._level_history.clear()
        self._last_alert_time.clear()
        self._event_log.clear()

    def add_handler(self, handler: BaseAlertHandler):
        """Add an alert handler."""
        self.handlers.append(handler)

    def get_statistics(self) -> Dict[str, Any]:
        """Return alert statistics from the event log."""
        if not self._event_log:
            return {"total_events": 0}

        levels = [e.level for e in self._event_log]
        return {
            "total_events": len(self._event_log),
            "nominal": sum(1 for l in levels if l == AlertLevel.NOMINAL),
            "caution": sum(1 for l in levels if l == AlertLevel.CAUTION),
            "warning": sum(1 for l in levels if l == AlertLevel.WARNING),
            "critical": sum(1 for l in levels if l == AlertLevel.CRITICAL),
            "suppressed": sum(1 for e in self._event_log if e.suppressed),
            "current_level": self._current_level.name,
        }
