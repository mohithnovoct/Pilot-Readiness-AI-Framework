"""
Framework Configuration Schema
================================
YAML / dict-based configuration with validation and sensible defaults.

A ``FrameworkConfig`` can be loaded from:
  - A YAML file:   ``FrameworkConfig.from_yaml("config.yaml")``
  - A Python dict:  ``FrameworkConfig.from_dict({...})``
  - Defaults:       ``FrameworkConfig()``
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SensorConfig:
    name: str = "wesad"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractorConfig:
    name: str = "hrv"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str = "lightgbm_stress"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionConfig:
    name: str = "weighted_linear"
    weights: Dict[str, float] = field(default_factory=lambda: {
        "stress": 0.6,
        "performance": 0.4,
    })
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThresholdConfig:
    name: str = "neyman_pearson"
    alpha: float = 0.05
    scenario: str = "operational"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertConfig:
    levels: List[str] = field(default_factory=lambda: [
        "NOMINAL", "CAUTION", "WARNING", "CRITICAL",
    ])
    escalation_window: int = 3          # consecutive alerts before escalation
    cooldown_seconds: float = 30.0      # seconds before re-alerting
    caution_threshold: float = 0.40
    warning_threshold: float = 0.60
    critical_threshold: float = 0.80
    handlers: List[str] = field(default_factory=lambda: ["log"])


@dataclass
class LivePipelineConfig:
    window_seconds: int = 60
    overlap: float = 0.5
    buffer_size: int = 256
    quality_threshold: float = 0.5
    reconnect_attempts: int = 5
    reconnect_delay_seconds: float = 2.0


@dataclass
class CalibrationConfig:
    duration_seconds: int = 300         # 5-minute baseline
    min_windows: int = 5
    profile_dir: str = "output/profiles"


@dataclass
class EdgeConfig:
    export_c: bool = True
    export_python: bool = True
    export_onnx: bool = False
    output_dir: str = "output/edge"


@dataclass
class PipelineConfig:
    window_sec: int = 60
    overlap: float = 0.5
    per_subject_norm: bool = False
    valid_labels: List[int] = field(default_factory=lambda: [1, 2])


@dataclass
class FrameworkConfig:
    """Top-level framework configuration."""

    # Data
    dataset: str = "wesad"
    data_dir: str = "Data"
    output_dir: str = "output"

    # Pipeline stages
    sensors: List[SensorConfig] = field(default_factory=lambda: [SensorConfig()])
    extractors: List[ExtractorConfig] = field(default_factory=lambda: [
        ExtractorConfig(name="hrv"),
        ExtractorConfig(name="performance"),
    ])
    models: List[ModelConfig] = field(default_factory=lambda: [
        ModelConfig(name="lightgbm_stress"),
        ModelConfig(name="lightgbm_perf"),
    ])
    fusion: FusionConfig = field(default_factory=FusionConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)

    # Sub-systems
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    live: LivePipelineConfig = field(default_factory=LivePipelineConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)

    # Misc
    log_level: str = "INFO"
    random_seed: int = 42

    # ── Factory Methods ──────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> "FrameworkConfig":
        """Load configuration from a YAML file."""
        import yaml
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FrameworkConfig":
        """Create configuration from a plain dictionary."""
        cfg = cls()

        # Top-level scalars
        for key in ("dataset", "data_dir", "output_dir", "log_level", "random_seed"):
            if key in d:
                setattr(cfg, key, d[key])

        # Sensors
        if "sensors" in d:
            cfg.sensors = [
                SensorConfig(**s) if isinstance(s, dict) else SensorConfig(name=s)
                for s in d["sensors"]
            ]

        # Extractors
        if "extractors" in d:
            cfg.extractors = [
                ExtractorConfig(**e) if isinstance(e, dict) else ExtractorConfig(name=e)
                for e in d["extractors"]
            ]

        # Models
        if "models" in d:
            cfg.models = [
                ModelConfig(**m) if isinstance(m, dict) else ModelConfig(name=m)
                for m in d["models"]
            ]

        # Fusion
        if "fusion" in d:
            fd = d["fusion"]
            cfg.fusion = FusionConfig(**fd) if isinstance(fd, dict) else FusionConfig(name=fd)

        # Threshold
        if "threshold" in d:
            td = d["threshold"]
            cfg.threshold = ThresholdConfig(**td) if isinstance(td, dict) else ThresholdConfig(name=td)

        # Alerts
        if "alerts" in d and isinstance(d["alerts"], dict):
            cfg.alerts = AlertConfig(**d["alerts"])

        # Pipeline
        if "pipeline" in d and isinstance(d["pipeline"], dict):
            cfg.pipeline = PipelineConfig(**d["pipeline"])

        # Live
        if "live" in d and isinstance(d["live"], dict):
            cfg.live = LivePipelineConfig(**d["live"])

        # Calibration
        if "calibration" in d and isinstance(d["calibration"], dict):
            cfg.calibration = CalibrationConfig(**d["calibration"])

        # Edge
        if "edge" in d and isinstance(d["edge"], dict):
            cfg.edge = EdgeConfig(**d["edge"])

        return cfg

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a plain dictionary."""
        import dataclasses
        return dataclasses.asdict(self)

    def save_yaml(self, path: str):
        """Save configuration to a YAML file."""
        import yaml
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
