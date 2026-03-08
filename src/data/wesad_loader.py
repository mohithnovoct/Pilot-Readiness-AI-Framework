"""
WESAD Dataset Loader
====================
Loads the WESAD (Wearable Stress and Affect Detection) dataset.
Each subject folder contains:
  - S{id}.pkl   : Pickled dict with chest (RespiBAN) and wrist (E4) sensor data + labels
  - S{id}_E4_Data/ : Raw E4 CSV files (IBI, BVP, EDA, HR, ACC, TEMP)
  - S{id}_quest.csv : Self-report questionnaires (PANAS, STAI, DIM, SSSQ)
  - S{id}_readme.txt : Demographics and pre-requisites

Labels in the .pkl:
  0 = not defined / transient
  1 = baseline
  2 = stress (TSST)
  3 = amusement
  4 = meditation
  5-7 = other conditions (not in all subjects)
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# ----- Constants -----
WESAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Data", "WESAD"
)

SUBJECT_IDS = [f"S{i}" for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]]

LABEL_MAP = {
    0: "not_defined",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
}

# Sampling rates for RespiBAN chest device (Hz)
CHEST_SAMPLING_RATES = {
    "ECG": 700,
    "EDA": 700,
    "EMG": 700,
    "Temp": 700,
    "Resp": 700,
    "ACC": 700,     # 3-axis
}

# Sampling rates for Empatica E4 wrist device (Hz)
WRIST_SAMPLING_RATES = {
    "BVP": 64,
    "EDA": 4,
    "TEMP": 4,
    "ACC": 32,      # 3-axis
}


def load_subject_pkl(subject_id: str, data_dir: Optional[str] = None) -> dict:
    """
    Load the raw .pkl file for a single subject.

    Parameters
    ----------
    subject_id : str
        Subject identifier, e.g. 'S2'.
    data_dir : str, optional
        Path to WESAD root directory. Defaults to project Data/WESAD.

    Returns
    -------
    dict
        Raw pickled data with keys: 'subject', 'signal' (chest/wrist), 'label'.
    """
    base = data_dir or WESAD_DIR
    pkl_path = os.path.join(base, subject_id, f"{subject_id}.pkl")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    return data


def extract_chest_signals(raw_data: dict) -> Dict[str, np.ndarray]:
    """
    Extract chest (RespiBAN) signals from loaded pickle data.

    Returns
    -------
    dict
        Keys: 'ECG', 'EDA', 'EMG', 'Temp', 'Resp', 'ACC' (each as ndarray).
    """
    signals = {}
    chest = raw_data["signal"]["chest"]
    for key in ["ECG", "EDA", "EMG", "Temp", "Resp", "ACC"]:
        arr = chest[key]
        if arr.ndim > 1 and arr.shape[1] == 1:
            arr = arr.flatten()
        signals[key] = arr
    return signals


def extract_wrist_signals(raw_data: dict) -> Dict[str, np.ndarray]:
    """
    Extract wrist (Empatica E4) signals from loaded pickle data.

    Returns
    -------
    dict
        Keys: 'BVP', 'EDA', 'TEMP', 'ACC' (each as ndarray).
    """
    signals = {}
    wrist = raw_data["signal"]["wrist"]
    for key in ["BVP", "EDA", "TEMP", "ACC"]:
        arr = wrist[key]
        if arr.ndim > 1 and arr.shape[1] == 1:
            arr = arr.flatten()
        signals[key] = arr
    return signals


def extract_labels(raw_data: dict) -> np.ndarray:
    """
    Extract label array from loaded pickle data.
    Labels are sampled at 700 Hz (chest rate), aligned with chest signals.
    """
    labels = raw_data["label"]
    if labels.ndim > 1:
        labels = labels.flatten()
    return labels


def load_e4_ibi(subject_id: str, data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load Inter-Beat Interval (IBI) data from the E4 CSV file.

    Parameters
    ----------
    subject_id : str
        Subject identifier, e.g. 'S2'.
    data_dir : str, optional
        Path to WESAD root directory.

    Returns
    -------
    pd.DataFrame
        Columns: ['time_offset', 'ibi_seconds']
        time_offset is relative to session start (in seconds).
    """
    base = data_dir or WESAD_DIR
    ibi_path = os.path.join(base, subject_id, f"{subject_id}_E4_Data", "IBI.csv")

    if not os.path.exists(ibi_path):
        raise FileNotFoundError(f"IBI file not found: {ibi_path}")

    # First line has the Unix start timestamp
    with open(ibi_path, "r") as f:
        header = f.readline().strip()
    start_ts = float(header.split(",")[0])

    # Remaining lines: time_offset, IBI_duration
    df = pd.read_csv(ibi_path, skiprows=1, header=None, names=["time_offset", "ibi_seconds"])
    df["unix_time"] = start_ts + df["time_offset"]

    return df


def create_windowed_dataset(
    ecg_signal: np.ndarray,
    labels: np.ndarray,
    fs: int = 700,
    window_sec: int = 60,
    overlap: float = 0.5,
    valid_labels: Optional[List[int]] = None,
) -> List[Dict]:
    """
    Segment signals into sliding windows and assign majority-vote labels.

    Parameters
    ----------
    ecg_signal : np.ndarray
        1-D ECG signal from the chest device.
    labels : np.ndarray
        Label array aligned with the ECG signal (same length).
    fs : int
        Sampling frequency in Hz.
    window_sec : int
        Window duration in seconds.
    overlap : float
        Fractional overlap between consecutive windows (0-1).
    valid_labels : list of int, optional
        Only keep windows where majority label is in this list.
        Defaults to [1, 2] (baseline and stress).

    Returns
    -------
    list of dict
        Each dict has: 'ecg_window', 'label', 'start_idx', 'end_idx'.
    """
    if valid_labels is None:
        valid_labels = [1, 2]

    window_size = int(fs * window_sec)
    step_size = int(window_size * (1 - overlap))
    n_samples = min(len(ecg_signal), len(labels))

    windows = []
    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        window_labels = labels[start:end]

        # Majority-vote label
        unique, counts = np.unique(window_labels, return_counts=True)
        majority_label = unique[np.argmax(counts)]

        if majority_label not in valid_labels:
            continue

        # Require ≥ 80 % purity
        purity = counts[np.argmax(counts)] / len(window_labels)
        if purity < 0.80:
            continue

        windows.append({
            "ecg_window": ecg_signal[start:end],
            "label": int(majority_label),
            "start_idx": start,
            "end_idx": end,
        })

    return windows


def load_subject_windowed(
    subject_id: str,
    data_dir: Optional[str] = None,
    window_sec: int = 60,
    overlap: float = 0.5,
) -> Tuple[List[Dict], dict]:
    """
    Convenience function: load a subject's data and return windowed ECG segments.

    Returns
    -------
    windows : list of dict
        Windowed ECG data with labels.
    metadata : dict
        Subject info including demographics.
    """
    raw = load_subject_pkl(subject_id, data_dir)
    chest = extract_chest_signals(raw)
    labels = extract_labels(raw)

    ecg = chest["ECG"]
    windows = create_windowed_dataset(
        ecg, labels, fs=700, window_sec=window_sec, overlap=overlap
    )

    # Parse readme for demographics
    metadata = {"subject_id": subject_id}
    base = data_dir or WESAD_DIR
    readme = os.path.join(base, subject_id, f"{subject_id}_readme.txt")
    if os.path.exists(readme):
        with open(readme, "r") as f:
            for line in f:
                if ":" in line and "###" not in line:
                    key, val = line.split(":", 1)
                    metadata[key.strip().lower().replace(" ", "_")] = val.strip()

    return windows, metadata


def load_all_subjects(
    data_dir: Optional[str] = None,
    window_sec: int = 60,
    overlap: float = 0.5,
    subject_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load and window ECG data for all (or selected) subjects.
    Returns a summary DataFrame (one row per window).

    Returns
    -------
    pd.DataFrame
        Columns: ['subject_id', 'window_idx', 'label', 'label_name',
                   'n_samples', 'start_idx', 'end_idx']
        The raw ECG windows are stored separately to avoid bloating the DataFrame.
    """
    ids = subject_ids or SUBJECT_IDS
    rows = []
    all_windows = {}

    for sid in ids:
        try:
            windows, meta = load_subject_windowed(sid, data_dir, window_sec, overlap)
            all_windows[sid] = windows
            for i, w in enumerate(windows):
                rows.append({
                    "subject_id": sid,
                    "window_idx": i,
                    "label": w["label"],
                    "label_name": LABEL_MAP.get(w["label"], "unknown"),
                    "n_samples": len(w["ecg_window"]),
                    "start_idx": w["start_idx"],
                    "end_idx": w["end_idx"],
                })
            print(f"  [OK] {sid}: {len(windows)} windows "
                  f"(baseline={sum(1 for w in windows if w['label']==1)}, "
                  f"stress={sum(1 for w in windows if w['label']==2)})")
        except Exception as e:
            print(f"  [SKIP] {sid}: {e}")

    df = pd.DataFrame(rows)
    return df, all_windows
