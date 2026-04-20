"""
Pilot Readiness AI Framework — Core Module
============================================
Contains the abstract base classes, plugin registry, orchestrator,
and all core subsystems (alerts, calibration, live pipeline, signal quality).
"""

from src.core.base import (
    BaseSensorAdapter,
    BaseFeatureExtractor,
    BaseModel,
    BaseFusionEngine,
    BaseThresholdStrategy,
    BaseAlertHandler,
)
from src.core.registry import ComponentRegistry
from src.core.framework import PilotReadinessFramework
from src.core.config_schema import FrameworkConfig

__all__ = [
    "BaseSensorAdapter",
    "BaseFeatureExtractor",
    "BaseModel",
    "BaseFusionEngine",
    "BaseThresholdStrategy",
    "BaseAlertHandler",
    "ComponentRegistry",
    "PilotReadinessFramework",
    "FrameworkConfig",
]
