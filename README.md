# 🛫 Pilot Readiness Monitoring Framework

**A Lightweight AI Framework for Pilot Readiness Monitoring Using Stress-Correlated Performance Indicators**

> Team 104 — Minor Project (2025-26), Dept. of CSE, School of Engineering

[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](#-docker)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](#-quick-start)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This framework provides a non-clinical, edge-deployable AI system for real-time pilot readiness assessment. It uses:

- **Physiological signals** (ECG/HRV from WESAD dataset) to detect stress states
- **Behavioral metrics** (reaction time, tracking error from NASA MATB-II) to detect performance degradation
- **Multi-modal fusion** with a weighted risk score and Neyman-Pearson tunable thresholds

The system outputs a **Continuous Readiness Risk Score** ∈ [0, 1] that indicates the pilot's operational fitness without making clinical claims.

---

## Architecture

```
┌──────────────┐    ┌──────────────┐
│  ECG / HRV   │    │  MATB-II     │
│  (WESAD)     │    │  Performance │
└──────┬───────┘    └──────┬───────┘
       │                    │
  ┌────▼────┐         ┌────▼────┐
  │ HRV     │         │ Perf.   │
  │ Features│         │ Features│
  │ (15)    │         │ (13)    │
  └────┬────┘         └────┬────┘
       │                    │
  ┌────▼────┐         ┌────▼────┐
  │LightGBM │         │LightGBM │
  │Classifier│        │Regressor│
  │ P_stress │        │ P_perf  │
  └────┬────┘         └────┬────┘
       │                    │
       └──────┬─────────────┘
         ┌────▼────────────┐
         │  Risk Fusion    │
         │ R = w₁·P_s +   │
         │     w₂·P_p     │
         └────┬────────────┘
              │
      ┌───────▼────────┐
      │ Neyman-Pearson │
      │ Threshold (γ)  │
      │ P(alert|ready) │
      │    ≤ α         │
      └───────┬────────┘
              │
        ┌─────▼─────┐
        │  READY /   │
        │  ALERT     │
        └────────────┘
```

---

## Project Structure

```
Minor Project/
├── Data/
│   ├── WESAD/              # 15 subjects (S2-S17)
│   └── 0_SWELL/            # SWELL Knowledge Work Dataset
├── src/
│   ├── data/
│   │   ├── wesad_loader.py     # WESAD .pkl loader & windowing
│   │   ├── swell_loader.py     # SWELL dataset loader
│   │   ├── matb_parser.py      # MATB-II log file parser
│   │   ├── matb_simulator.py   # Synthetic performance simulator
│   │   └── preprocessing.py    # ECG filtering, R-peak detection
│   ├── features/
│   │   ├── hrv_features.py     # HRV: SDNN, RMSSD, LF/HF, SampEn
│   │   ├── performance_features.py  # CVRT, Lag-1, Inceptor Entropy
│   │   └── feature_pipeline.py # Orchestrator
│   ├── models/
│   │   ├── stress_classifier.py  # LightGBM stress (LOSO CV + Bootstrap CI)
│   │   └── performance_model.py  # LightGBM performance regressor
│   ├── risk/
│   │   ├── fusion.py           # Weighted multi-modal fusion
│   │   └── threshold.py        # Neyman-Pearson tunable thresholds
│   ├── edge/
│   │   └── export_model.py     # C/Python export via m2cgen
│   ├── experiments/
│   │   ├── model_comparison.py # LightGBM vs RF/SVM/XGBoost/LR
│   │   └── cross_dataset.py    # Train WESAD ↔ Test SWELL
│   └── visualization/
│       ├── plots.py            # Static matplotlib/seaborn plots
│       └── dashboard.py        # Interactive Plotly HTML dashboard
├── templates/
│   └── streaming.html          # Real-time streaming demo UI
├── output/
│   ├── features/      # Extracted feature CSVs
│   ├── models/        # Trained model pickles
│   ├── plots/         # Static plot PNGs
│   ├── edge/          # Exported C/Python models
│   ├── experiments/   # Comparison & ablation results
│   └── dashboard.html # Interactive dashboard
├── tests/             # Unit & integration tests
├── config.py          # Central configuration
├── main.py            # End-to-end pipeline
├── streaming_demo.py  # Real-time Flask + WebSocket demo
├── Dockerfile         # Docker container
├── docker-compose.yml # Multi-service orchestration
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Quick Test (Single Subject)
```bash
python main.py --quick-test
```

### 3. Run Full Pipeline (All 15 Subjects)
```bash
python main.py
```

### 4. Run with SWELL Dataset
```bash
python main.py --dataset swell
```

### 5. View Dashboard
Open `output/dashboard.html` in your browser.

### 6. Launch Streaming Demo
```bash
python streaming_demo.py
# Open http://localhost:5000 in your browser
```

---

## 🐳 Docker

### Pull & Run (DockerHub)
```bash
# Pull the image
docker pull pilot-readiness:latest

# Run streaming demo
docker run --rm -p 5000:5000 pilot-readiness

# Open http://localhost:5000 in your browser
```

### Build Locally
```bash
# Build the image
docker build -t pilot-readiness .

# Run streaming demo
docker run --rm -p 5000:5000 pilot-readiness

# Run full pipeline (with data volume)
docker run --rm -v $(pwd)/Data:/app/Data -v $(pwd)/output:/app/output \
    pilot-readiness python main.py --dataset swell --skip-extraction

# Run tests
docker run --rm pilot-readiness python -m pytest tests/ -v
```

### Docker Compose
```bash
# Start streaming demo
docker compose up demo

# Run pipeline (requires Data/ directory)
docker compose --profile pipeline up pipeline

# Run tests
docker compose --profile test run tests

# Run experiments
docker compose --profile experiments run experiments
```

---

## ⚙️ CLI Options

| Argument | Default | Description |
|---------|---------|-------------|
| `--quick-test` | off | Run with S2 only for fast validation |
| `--dataset` | `wesad` | Dataset pipeline: `wesad` or `swell` |
| `--skip-extraction` | off | Use cached feature CSVs if available |
| `--skip-training` | off | Skip model training (use cached models) |
| `--per-subject-norm` | off | Per-pilot Z-score normalization for personalized baselines |
| `--window-sec` | 60 | ECG window duration (seconds) |
| `--overlap` | 0.5 | Window overlap fraction (0-1) |
| `--n-sessions` | 30 | Simulated MATB-II sessions per workload |
| `--log-level` | INFO | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## 📊 Features Extracted

### HRV (Physiological) — 15 features
| Feature | Domain | Description |
|---------|--------|-------------|
| MeanNN, MedianNN | Time | Central tendency of NN intervals |
| SDNN | Time | Overall HRV — global cardiac health index |
| RMSSD | Time | Vagal tone — acute stress indicator |
| pNN50 | Time | Successive interval variability (%) |
| MeanHR, SDHR | Time | Heart rate statistics |
| LF_power | Frequency | 0.04–0.15 Hz band power |
| HF_power | Frequency | 0.15–0.4 Hz band power (vagal) |
| LF_HF_ratio | Frequency | Sympathovagal balance |
| VLF_power, Total_power | Frequency | Very low freq & total power |
| LF_norm, HF_norm | Frequency | Normalized band powers |
| SampEn | Non-linear | Regularity/complexity of HR signal |

### Performance (Behavioral) — 13 features
| Feature | Source | Description |
|---------|--------|-------------|
| MeanRT, MedianRT, StdRT | SYSMON | Reaction time statistics |
| CVRT | SYSMON | Coefficient of variation (fatigue marker) |
| Lag1_Autocorr | SYSMON | Sequential dependency ("clumping") |
| RT_Skewness, RT_Kurtosis | SYSMON | Distribution shape |
| MeanRMSD, StdRMSD, MaxRMSD | TRACK | Tracking error statistics |
| Inceptor_Entropy | TRACK | Control input randomness |
| MeanCommRT, CommAccuracy, TimeoutRate | COMM | Communication metrics |

---

## 🔬 Methodology

1. **Data**: WESAD (15 subjects × baseline/stress conditions) + SWELL Knowledge Worker Dataset + synthesized MATB-II data
2. **Preprocessing**: Butterworth bandpass (0.5–40 Hz), R-peak detection, RR artifact removal
3. **Windowing**: 60-second epochs, 50% overlap, ≥80% label purity
4. **Models**: LightGBM with LOSO cross-validation + Grid Search + Bootstrap 95% CIs
5. **Fusion**: Weighted linear combination with dynamic signal-quality weighting
6. **Thresholds**: Neyman-Pearson based — configurable false-alarm rate (α)
7. **Edge**: Model exported to C code via m2cgen (<256KB RAM, <10ms latency)
8. **Validation**: Cross-dataset evaluation (WESAD ↔ SWELL) + multi-model comparison

---

## 🧪 Experiments

### Model Comparison
Compare LightGBM against baseline classifiers using LOSO CV:
```bash
python -m src.experiments.model_comparison --dataset wesad
```
Models tested: LightGBM, Random Forest, SVM (RBF), XGBoost, Logistic Regression

### Feature Ablation
Evaluate impact of different feature subsets:
- All features (15) vs Time-domain only (7) vs Frequency-domain only (7)
- Top-5 SHAP features vs full set

### Cross-Dataset Validation
Test generalizability by training on one dataset, testing on another:
```bash
python -m src.experiments.cross_dataset
```

---

## 📈 Results

### WESAD Stress Classification (LOSO CV)
| Metric | Score |
|--------|-------|
| Accuracy | See `output/experiments/model_comparison.csv` |
| F1-Score | Generated after running pipeline |
| ROC-AUC | With 95% bootstrap confidence intervals |

### Edge Deployment
| Metric | Value |
|--------|-------|
| Model RAM | < 3 KB |
| Inference Latency | < 1 ms |
| Edge Target | ✓ Real-time capable |

*Run `python main.py` to generate full results.*

---

## 👥 Team 104

| Name | USN |
|------|-----|
| Chinmay M R | ENG23CS0047 |
| M S N S Aditya | ENG23CS0098 |
| Mohith Butta | ENG23CS0115 |
| Mourya Vardhan B K | ENG23CS0119 |

**Guide:** Prof. Dharmendra D P

---

## 📚 Key References

- Schmidt et al. (2018) — WESAD dataset
- Koldijk et al. (2014) — SWELL Knowledge Work dataset
- NASA MATB-II (TM-2011-217164)
- Task Force ESA/NASPE (1996) — HRV standards
- LightGBM (Ke et al., 2017)
- Neyman-Pearson (optimal detection theory)
