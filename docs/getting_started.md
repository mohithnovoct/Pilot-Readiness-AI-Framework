# Getting Started with Pilot Readiness AI Framework

This guide covers installation, basic usage, and running your first
pilot readiness analysis.

## Installation

### From Source (recommended for development)

```bash
git clone https://github.com/mohithbutta/Pilot-Readiness-AI-Framework.git
cd Pilot-Readiness-AI-Framework
pip install -e ".[all]"
```

### Minimal Install

```bash
pip install -e .
```

### With Optional Features

```bash
pip install -e ".[streaming]"     # Flask + SocketIO streaming demo
pip install -e ".[api]"           # FastAPI REST API
pip install -e ".[edge]"          # m2cgen + ONNX edge export
pip install -e ".[dev]"           # Testing & linting tools
pip install -e ".[all]"           # Everything
```

---

## Quick Start: Python API

### 1. Import and Create Framework

```python
from src import PilotReadinessFramework

fw = PilotReadinessFramework()
```

### 2. Configure (Optional)

```python
# From a YAML file
fw.configure("default_config.yaml")

# Or from a dictionary
fw.configure({
    "dataset": "wesad",
    "fusion": {
        "name": "weighted_linear",
        "weights": {"stress": 0.7, "performance": 0.3}
    },
    "threshold": {"alpha": 0.05}
})
```

### 3. Train Models

```python
import pandas as pd

# Load pre-extracted features
features_df = pd.read_csv("output/features/wesad_features.csv")

# Add binary stress label if not present
features_df["stress_label"] = (features_df["label"] == 2).astype(int)

# Train
metrics = fw.fit(features_df=features_df)
print(metrics)
```

### 4. Run Inference

```python
# Batch prediction
results = fw.predict(features_df=features_df)

for r in results[:5]:
    print(f"Risk: {r.risk_score:.3f} | {r.alert_level} | {r.decision}")
```

### 5. Calibrate for a Specific Pilot

```python
import numpy as np

# Use baseline (non-stress) features for calibration
baseline_mask = features_df["label"] == 1
baseline_X = features_df.loc[baseline_mask, feature_cols].values

profile = fw.calibrate(pilot_id="CPT_SMITH", baseline_features=baseline_X)
print(f"Personalised threshold: {profile['threshold']:.3f}")
```

### 6. Export for Edge Deployment

```python
exported = fw.export(format="all")
print(exported)
# {'lightgbm_stress_c': 'output/edge/lightgbm_stress.c', ...}
```

---

## Quick Start: CLI

The existing CLI still works:

```bash
# Full pipeline
python main.py

# Quick test
python main.py --quick-test

# SWELL dataset
python main.py --dataset swell

# Streaming demo
python streaming_demo.py
```

---

## Quick Start: REST API

```bash
# Start API server
uvicorn src.api.rest_api:app --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/health

# List components
curl http://localhost:8000/components

# Predict (after training)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[0.5, 0.3, 0.7, ...]]}'
```

---

## Configuration Reference

See `default_config.yaml` for all available settings. Key sections:

| Section | Description |
|---------|-------------|
| `sensors` | Which data sources to use (wesad, swell, matb) |
| `extractors` | Feature extraction methods (hrv, performance, fatigue) |
| `models` | ML models (lightgbm_stress, lightgbm_perf) |
| `fusion` | Score combination strategy (weighted_linear, bayesian) |
| `threshold` | Decision strategy (neyman_pearson) |
| `alerts` | Multi-level alert configuration |
| `calibration` | Per-pilot baseline settings |
| `edge` | Export formats and paths |
