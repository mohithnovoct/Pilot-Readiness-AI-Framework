# API Reference

Complete reference for the Pilot Readiness AI Framework public API.

---

## `PilotReadinessFramework`

**Module:** `src.core.framework`

The main orchestrator class.

### Constructor

```python
PilotReadinessFramework(config=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `str \| dict \| FrameworkConfig \| None` | Configuration source (YAML path, dict, or object) |

### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `configure(config)` | `str \| dict \| FrameworkConfig` | `self` | Load and apply configuration |
| `fit(X, y, features_df, feature_cols)` | arrays or DataFrame | `dict` | Train all registered models |
| `predict(X, features_df, feature_cols)` | arrays or DataFrame | `List[PredictionResult]` | Batch inference |
| `predict_single(x)` | `np.ndarray` (1-D) | `PredictionResult` | Single-sample inference |
| `calibrate(pilot_id, baseline_features)` | `str`, `np.ndarray` | `dict` | Per-pilot calibration |
| `export(format, output_dir)` | `str`, `str` | `dict` | Edge model export |
| `register_sensor(sensor)` | `BaseSensorAdapter` | `self` | Register custom sensor |
| `register_extractor(extractor)` | `BaseFeatureExtractor` | `self` | Register custom extractor |
| `register_model(model)` | `BaseModel` | `self` | Register custom model |
| `register_fusion(fusion)` | `BaseFusionEngine` | `self` | Register custom fusion |
| `register_threshold(threshold)` | `BaseThresholdStrategy` | `self` | Register custom threshold |
| `register_alert_handler(handler)` | `BaseAlertHandler` | `self` | Register alert handler |
| `list_components()` | — | `dict` | List all registered components |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `config` | `FrameworkConfig` | Current configuration |
| `is_fitted` | `bool` | Whether models have been trained |
| `training_metrics` | `dict` | Metrics from last training run |

---

## Data Containers

### `SensorData`

```python
@dataclass
class SensorData:
    signals: Dict[str, np.ndarray]       # channel → 1-D array
    sampling_rates: Dict[str, float]     # channel → Hz
    labels: Optional[np.ndarray]         # per-sample labels
    metadata: Dict[str, Any]
    subject_id: str
```

### `FeatureSet`

```python
@dataclass
class FeatureSet:
    dataframe: pd.DataFrame              # rows = windows, cols = features
    feature_names: List[str]
    metadata: Dict[str, Any]
```

### `PredictionResult`

```python
@dataclass
class PredictionResult:
    risk_score: float                    # fused risk ∈ [0, 1]
    stress_probability: float
    performance_score: float
    alert_level: str                     # NOMINAL/CAUTION/WARNING/CRITICAL
    decision: str                        # READY/CAUTION/WARNING/CRITICAL
    confidence: float
    top_features: Dict[str, float]
    metadata: Dict[str, Any]
```

### `AlertLevel` (Enum)

```python
class AlertLevel(Enum):
    NOMINAL = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3
```

---

## Abstract Base Classes

### `BaseSensorAdapter`

| Method | Required | Description |
|--------|----------|-------------|
| `load(**kwargs)` | ✅ | Load batch data → `SensorData` |
| `stream(callback)` | ❌ | Stream real-time data |
| `validate(data)` | ❌ | Validate data quality |

### `BaseFeatureExtractor`

| Method | Required | Description |
|--------|----------|-------------|
| `extract(data, **kwargs)` | ✅ | Extract features → `FeatureSet` |
| `get_feature_names()` | ❌ | Return ordered feature names |

### `BaseModel`

| Method | Required | Description |
|--------|----------|-------------|
| `train(X, y, **kwargs)` | ✅ | Train → metrics dict |
| `predict(X)` | ✅ | Inference → `np.ndarray` |
| `save(path)` | ❌ | Persist model |
| `load(path)` | ❌ | Load model |
| `get_feature_importance()` | ❌ | Feature importances |

### `BaseFusionEngine`

| Method | Required | Description |
|--------|----------|-------------|
| `fuse(scores, weights)` | ✅ | Fuse scores → `float` |
| `fuse_batch(scores, weights)` | ❌ | Vectorised batch fusion |

### `BaseThresholdStrategy`

| Method | Required | Description |
|--------|----------|-------------|
| `compute_threshold(baseline, **kw)` | ✅ | Calibrate threshold |
| `decide(risk_score)` | ✅ | Decision → `(AlertLevel, str)` |

### `BaseAlertHandler`

| Method | Required | Description |
|--------|----------|-------------|
| `handle(level, prediction)` | ✅ | Execute alert action |
| `should_fire(level)` | ❌ | Filter by level |

---

## Built-in Plugins

### Sensors

| Name | Class | Module |
|------|-------|--------|
| `wesad` | `WESADSensor` | `src.plugins.sensors.wesad_sensor` |
| `swell` | `SWELLSensor` | `src.plugins.sensors.swell_sensor` |
| `matb` | `MATBSensor` | `src.plugins.sensors.matb_sensor` |

### Feature Extractors

| Name | Class | Module |
|------|-------|--------|
| `hrv` | `HRVExtractor` | `src.plugins.extractors.hrv_extractor` |
| `performance` | `PerformanceExtractor` | `src.plugins.extractors.performance_extractor` |
| `fatigue` | `FatigueExtractor` | `src.plugins.extractors.fatigue_extractor` |

### Models

| Name | Class | Module |
|------|-------|--------|
| `lightgbm_stress` | `LightGBMStressModel` | `src.plugins.models.lightgbm_model` |
| `lightgbm_perf` | `LightGBMPerfModel` | `src.plugins.models.lightgbm_model` |

### Fusion Engines

| Name | Class | Module |
|------|-------|--------|
| `weighted_linear` | `WeightedLinearFusion` | `src.plugins.fusion.weighted_fusion` |
| `bayesian` | `BayesianFusion` | `src.plugins.fusion.bayesian_fusion` |

### Threshold Strategies

| Name | Class | Module |
|------|-------|--------|
| `neyman_pearson` | `NeymanPearsonThreshold` | `src.plugins.threshold.neyman_pearson` |

### Alert Handlers

| Name | Class | Module |
|------|-------|--------|
| `log` | `LogAlertHandler` | `src.core.alerts` |
| `callback` | `CallbackAlertHandler` | `src.core.alerts` |
| `webhook` | `WebhookAlertHandler` | `src.core.alerts` |

---

## REST API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/status` | Framework status |
| `GET` | `/config` | Current configuration |
| `GET` | `/components` | Registered components |
| `POST` | `/predict` | Batch inference |
| `POST` | `/calibrate` | Per-pilot calibration |
| `GET` | `/profiles` | List pilot profiles |
| `GET` | `/profiles/{id}` | Get specific profile |

See `src/api/rest_api.py` for full request/response schemas.
