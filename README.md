# 🛫 Pilot Readiness Monitoring Framework

**A Lightweight AI Framework for Pilot Readiness Monitoring Using Stress-Correlated Performance Indicators**

> Team 104 — Minor Project (2025-26), Dept. of CSE, School of Engineering

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
│   └── NASA/               # MATB-II 2.0 software & sample data
├── src/
│   ├── data/
│   │   ├── wesad_loader.py     # WESAD .pkl loader & windowing
│   │   ├── matb_parser.py      # MATB-II log file parser
│   │   ├── matb_simulator.py   # Synthetic performance simulator
│   │   └── preprocessing.py    # ECG filtering, R-peak detection
│   ├── features/
│   │   ├── hrv_features.py     # HRV: SDNN, RMSSD, LF/HF, SampEn
│   │   ├── performance_features.py  # CVRT, Lag-1, Inceptor Entropy
│   │   └── feature_pipeline.py # Orchestrator
│   ├── models/
│   │   ├── stress_classifier.py  # LightGBM stress (LOSO CV)
│   │   └── performance_model.py  # LightGBM performance regressor
│   ├── risk/
│   │   ├── fusion.py           # Weighted multi-modal fusion
│   │   └── threshold.py        # Neyman-Pearson tunable thresholds
│   ├── edge/
│   │   └── export_model.py     # C/Python export via m2cgen
│   └── visualization/
│       ├── plots.py            # Static matplotlib/seaborn plots
│       └── dashboard.py        # Interactive Plotly HTML dashboard
├── output/
│   ├── features/      # Extracted feature CSVs
│   ├── models/        # Trained model pickles
│   ├── plots/         # Static plot PNGs
│   ├── edge/          # Exported C/Python models
│   └── dashboard.html # Interactive dashboard
├── tests/             # Unit tests
├── config.py          # Central configuration
├── main.py            # End-to-end pipeline
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

### 4. View Dashboard
Open `output/dashboard.html` in your browser.

---

## ⚙️ CLI Options

| Argument | Default | Description |
|---------|---------|-------------|
| `--quick-test` | off | Run with S2 only for fast validation |
| `--skip-extraction` | off | Use cached feature CSVs if available |
| `--window-sec` | 60 | ECG window duration (seconds) |
| `--overlap` | 0.5 | Window overlap fraction (0-1) |
| `--n-sessions` | 30 | Simulated MATB-II sessions per workload |

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

1. **Data**: WESAD (15 subjects × baseline/stress conditions) + synthesized MATB-II data
2. **Preprocessing**: Butterworth bandpass (0.5–40 Hz), R-peak detection, RR artifact removal
3. **Windowing**: 60-second epochs, 50% overlap, ≥80% label purity
4. **Models**: LightGBM with LOSO cross-validation + Grid Search
5. **Fusion**: Weighted linear combination with dynamic signal-quality weighting
6. **Thresholds**: Neyman-Pearson based — configurable false-alarm rate (α)
7. **Edge**: Model exported to C code via m2cgen (<256KB RAM, <10ms latency)

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
- NASA MATB-II (TM-2011-217164)
- Task Force ESA/NASPE (1996) — HRV standards
- LightGBM (Ke et al., 2017)
- Neyman-Pearson (optimal detection theory)
