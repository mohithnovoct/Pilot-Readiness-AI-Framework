"""
Project Configuration
=====================
Central configuration for all paths, hyperparameters, and constants.
Edit values here instead of modifying individual source modules.
"""

import os

# ---- Project Paths ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
WESAD_DIR = os.path.join(DATA_DIR, "WESAD")
NASA_DIR = os.path.join(DATA_DIR, "NASA")
MATB_DATA_DIR = os.path.join(NASA_DIR, "MATB-II_2.0", "MATB-II 2.0", "Data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# ---- WESAD ----
SUBJECT_IDS = [f"S{i}" for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]]

LABEL_MAP = {
    0: "not_defined",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
}

# ---- Signal Processing ----
ECG_SAMPLING_RATE = 700       # Hz (RespiBAN chest device)
BANDPASS_LOW = 0.5            # Hz
BANDPASS_HIGH = 40.0          # Hz
BANDPASS_ORDER = 4

# ---- Windowing ----
DEFAULT_WINDOW_SEC = 60       # seconds
DEFAULT_OVERLAP = 0.5         # fraction
LABEL_PURITY_THRESHOLD = 0.80 # minimum fraction for majority-vote label
VALID_LABELS = [1, 2]         # baseline, stress

# ---- HRV Features ----
HRV_FEATURE_COLS = [
    "MeanNN", "MedianNN", "SDNN", "RMSSD", "pNN50",
    "MeanHR", "SDHR",
    "VLF_power", "LF_power", "HF_power", "Total_power",
    "LF_HF_ratio", "LF_norm", "HF_norm",
    "SampEn",
]

# ---- Performance Features ----
PERF_FEATURE_COLS = [
    "mean_rt", "std_rt", "cv_rt", "median_rt",
    "lag1_autocorr", "rt_skewness", "rt_kurtosis",
    "mean_track_rmsd", "std_track_rmsd", "inceptor_entropy",
    "mean_comm_rt", "comm_accuracy", "n_timeouts",
]

# ---- Model Hyperparameters ----
LIGHTGBM_STRESS_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "max_depth": -1,
    "min_child_samples": 10,
    "class_weight": "balanced",
    "verbose": -1,
    "random_state": 42,
}

LIGHTGBM_PERF_PARAMS = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "verbose": -1,
    "random_state": 42,
}

STRESS_GRID_SEARCH = {
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 200, 300],
    "max_depth": [-1, 5, 10],
}

PERF_GRID_SEARCH = {
    "num_leaves": [15, 31],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 200],
    "max_depth": [-1, 5],
}

# ---- Risk Fusion ----
FUSION_W_PHYS = 0.6           # weight for physiological channel
FUSION_W_PERF = 0.4           # weight for performance channel

# ---- Threshold Scenarios ----
SCENARIO_PRESETS = {
    "training":    {"alpha": 0.10, "description": "High sensitivity — training missions"},
    "operational": {"alpha": 0.05, "description": "Balanced — standard operations"},
    "critical":    {"alpha": 0.01, "description": "Low false-alarm — critical missions"},
}

# ---- Simulation ----
DEFAULT_N_SESSIONS = 30       # per workload level
SIMULATION_SEED = 42
