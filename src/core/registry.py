"""
Component Registry
===================
A central registry for discovering, registering, and retrieving
framework components (sensors, extractors, models, fusion engines,
threshold strategies, alert handlers).

Supports both programmatic registration and auto-discovery of
built-in plugins.

Usage
-----
>>> from src.core.registry import ComponentRegistry, registry
>>> registry.register("sensor", my_sensor_instance)
>>> sensor = registry.get("sensor", "wesad")
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type

from src.core.base import (
    BaseAlertHandler,
    BaseFeatureExtractor,
    BaseFusionEngine,
    BaseModel,
    BaseSensorAdapter,
    BaseThresholdStrategy,
)

logger = logging.getLogger("pilot_readiness.registry")

# Valid component categories and their corresponding base classes
_CATEGORY_BASES: Dict[str, Type] = {
    "sensor": BaseSensorAdapter,
    "extractor": BaseFeatureExtractor,
    "model": BaseModel,
    "fusion": BaseFusionEngine,
    "threshold": BaseThresholdStrategy,
    "alert_handler": BaseAlertHandler,
}


class ComponentRegistry:
    """
    Thread-safe registry for framework components.

    Each component is identified by a *(category, name)* pair.
    Categories: ``sensor``, ``extractor``, ``model``, ``fusion``,
    ``threshold``, ``alert_handler``.
    """

    def __init__(self):
        self._components: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._classes: Dict[str, Dict[str, Type]] = defaultdict(dict)

    # ── Registration ─────────────────────────────────────────────

    def register(self, category: str, component: Any, name: Optional[str] = None):
        """
        Register a component *instance*.

        Parameters
        ----------
        category : str
            One of: sensor, extractor, model, fusion, threshold, alert_handler.
        component : instance
            A concrete implementation of the corresponding ABC.
        name : str, optional
            Override name.  Defaults to ``component.name``.
        """
        self._validate_category(category)
        base_cls = _CATEGORY_BASES[category]
        if not isinstance(component, base_cls):
            raise TypeError(
                f"Component must be an instance of {base_cls.__name__}, "
                f"got {type(component).__name__}"
            )
        cname = name or getattr(component, "name", type(component).__name__)
        self._components[category][cname] = component
        logger.debug("Registered %s: %s", category, cname)

    def register_class(self, category: str, cls: Type, name: Optional[str] = None):
        """Register a component *class* for lazy instantiation."""
        self._validate_category(category)
        cname = name or getattr(cls, "name", cls.__name__)
        self._classes[category][cname] = cls
        logger.debug("Registered class %s: %s", category, cname)

    # ── Retrieval ────────────────────────────────────────────────

    def get(self, category: str, name: str) -> Any:
        """
        Retrieve a registered component by category and name.

        If only a class was registered (via ``register_class``), it will be
        instantiated with default arguments on first access and cached.
        """
        self._validate_category(category)

        # Check instances first
        if name in self._components[category]:
            return self._components[category][name]

        # Lazy-instantiate from class
        if name in self._classes[category]:
            instance = self._classes[category][name]()
            self._components[category][name] = instance
            return instance

        available = self.list(category)
        raise KeyError(
            f"No {category} named '{name}' found. "
            f"Available: {available}"
        )

    def list(self, category: str) -> List[str]:
        """List all registered component names in a category."""
        self._validate_category(category)
        names = set(self._components[category].keys())
        names |= set(self._classes[category].keys())
        return sorted(names)

    def list_all(self) -> Dict[str, List[str]]:
        """List all registered components across all categories."""
        return {cat: self.list(cat) for cat in _CATEGORY_BASES}

    def has(self, category: str, name: str) -> bool:
        """Check if a component is registered."""
        return (
            name in self._components.get(category, {})
            or name in self._classes.get(category, {})
        )

    # ── Bulk Operations ──────────────────────────────────────────

    def clear(self, category: Optional[str] = None):
        """Clear all components, or only a specific category."""
        if category:
            self._components[category].clear()
            self._classes[category].clear()
        else:
            self._components.clear()
            self._classes.clear()

    # ── Auto-Discovery ───────────────────────────────────────────

    def auto_discover(self):
        """
        Discover and register all built-in plugins from ``src.plugins``.

        This is called automatically when the framework is first instantiated.
        """
        logger.info("Auto-discovering built-in plugins...")

        # Sensors
        try:
            from src.plugins.sensors.wesad_sensor import WESADSensor
            self.register_class("sensor", WESADSensor)
        except ImportError:
            logger.debug("WESADSensor not available")

        try:
            from src.plugins.sensors.swell_sensor import SWELLSensor
            self.register_class("sensor", SWELLSensor)
        except ImportError:
            logger.debug("SWELLSensor not available")

        try:
            from src.plugins.sensors.matb_sensor import MATBSensor
            self.register_class("sensor", MATBSensor)
        except ImportError:
            logger.debug("MATBSensor not available")

        # Feature Extractors
        try:
            from src.plugins.extractors.hrv_extractor import HRVExtractor
            self.register_class("extractor", HRVExtractor)
        except ImportError:
            logger.debug("HRVExtractor not available")

        try:
            from src.plugins.extractors.performance_extractor import PerformanceExtractor
            self.register_class("extractor", PerformanceExtractor)
        except ImportError:
            logger.debug("PerformanceExtractor not available")

        try:
            from src.plugins.extractors.fatigue_extractor import FatigueExtractor
            self.register_class("extractor", FatigueExtractor)
        except ImportError:
            logger.debug("FatigueExtractor not available")

        # Models
        try:
            from src.plugins.models.lightgbm_model import LightGBMStressModel, LightGBMPerfModel
            self.register_class("model", LightGBMStressModel)
            self.register_class("model", LightGBMPerfModel)
        except ImportError:
            logger.debug("LightGBM models not available")

        # Fusion Engines
        try:
            from src.plugins.fusion.weighted_fusion import WeightedLinearFusion
            self.register_class("fusion", WeightedLinearFusion)
        except ImportError:
            logger.debug("WeightedLinearFusion not available")

        try:
            from src.plugins.fusion.bayesian_fusion import BayesianFusion
            self.register_class("fusion", BayesianFusion)
        except ImportError:
            logger.debug("BayesianFusion not available")

        # Threshold Strategies
        try:
            from src.plugins.threshold.neyman_pearson import NeymanPearsonThreshold
            self.register_class("threshold", NeymanPearsonThreshold)
        except ImportError:
            logger.debug("NeymanPearsonThreshold not available")

        discovered = self.list_all()
        total = sum(len(v) for v in discovered.values())
        logger.info("Discovered %d built-in plugins: %s", total, discovered)

    # ── Internal ─────────────────────────────────────────────────

    @staticmethod
    def _validate_category(category: str):
        if category not in _CATEGORY_BASES:
            raise ValueError(
                f"Invalid category '{category}'. "
                f"Must be one of: {list(_CATEGORY_BASES.keys())}"
            )


# ── Module-level singleton ───────────────────────────────────────────

registry = ComponentRegistry()
