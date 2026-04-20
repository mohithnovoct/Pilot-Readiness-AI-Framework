"""
Pilot Readiness AI Framework
==============================
A Lightweight AI Framework for Pilot Readiness Monitoring
Using Stress-Correlated Performance Indicators.

Quick Start
-----------
>>> from src import PilotReadinessFramework
>>> fw = PilotReadinessFramework()
>>> fw.configure({"dataset": "wesad"})
>>> fw.fit(features_df=my_df)
>>> results = fw.predict(X=my_features)
"""

from src.core.framework import PilotReadinessFramework
from src.core.config_schema import FrameworkConfig
from src.core.base import (
    BaseSensorAdapter,
    BaseFeatureExtractor,
    BaseModel,
    BaseFusionEngine,
    BaseThresholdStrategy,
    BaseAlertHandler,
    AlertLevel,
    SensorData,
    FeatureSet,
    PredictionResult,
)

__version__ = "1.0.0"

__all__ = [
    "PilotReadinessFramework",
    "FrameworkConfig",
    "BaseSensorAdapter",
    "BaseFeatureExtractor",
    "BaseModel",
    "BaseFusionEngine",
    "BaseThresholdStrategy",
    "BaseAlertHandler",
    "AlertLevel",
    "SensorData",
    "FeatureSet",
    "PredictionResult",
]
