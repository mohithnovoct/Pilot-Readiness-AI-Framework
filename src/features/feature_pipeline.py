"""
Feature Pipeline
================
Orchestrates feature extraction across all subjects/windows,
combining HRV and performance features into unified feature matrices.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
import warnings

from src.data.wesad_loader import (
    load_subject_pkl, extract_chest_signals, extract_labels,
    create_windowed_dataset, SUBJECT_IDS, WESAD_DIR, LABEL_MAP
)
from src.data.preprocessing import (
    bandpass_filter, extract_ibi_from_ecg
)
from src.features.hrv_features import extract_all_hrv_features
from src.data.matb_simulator import generate_full_simulation
from src.features.performance_features import extract_all_performance_features


def extract_wesad_features(
    subject_ids: Optional[List[str]] = None,
    data_dir: Optional[str] = None,
    window_sec: int = 60,
    overlap: float = 0.5,
    valid_labels: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Extract HRV features from all WESAD subjects.

    Pipeline per subject:
      1. Load .pkl → extract chest ECG + labels
      2. Window the signal (60s, 50% overlap)
      3. For each window: filter ECG → detect R-peaks → compute RR intervals → HRV features
      4. Combine into a DataFrame

    Parameters
    ----------
    subject_ids : list of str, optional
        Which subjects to process. Defaults to all 15.
    data_dir : str, optional
        WESAD data directory.
    window_sec : int
        Window duration in seconds.
    overlap : float
        Overlap fraction.
    valid_labels : list of int, optional
        Which labels to keep. Default: [1, 2] (baseline, stress).

    Returns
    -------
    pd.DataFrame
        Feature matrix with one row per window.
        Columns: subject_id, window_idx, label, label_name, + all HRV features.
    """
    ids = subject_ids or SUBJECT_IDS
    if valid_labels is None:
        valid_labels = [1, 2]

    all_rows = []

    for sid in ids:
        try:
            print(f"\nProcessing {sid}...")

            # Load raw data
            raw = load_subject_pkl(sid, data_dir)
            chest = extract_chest_signals(raw)
            labels = extract_labels(raw)
            ecg = chest["ECG"]

            # Window the signal
            windows = create_windowed_dataset(
                ecg, labels, fs=700, window_sec=window_sec,
                overlap=overlap, valid_labels=valid_labels
            )

            n_baseline = sum(1 for w in windows if w["label"] == 1)
            n_stress = sum(1 for w in windows if w["label"] == 2)
            print(f"  {sid}: {len(windows)} windows "
                  f"(baseline={n_baseline}, stress={n_stress})")

            # Extract features per window
            for i, w in enumerate(windows):
                ecg_window = w["ecg_window"]

                # ECG → RR intervals
                rr = extract_ibi_from_ecg(ecg_window, fs=700.0)

                if len(rr) < 10:
                    continue

                # HRV features
                hrv = extract_all_hrv_features(rr)

                row = {
                    "subject_id": sid,
                    "window_idx": i,
                    "label": w["label"],
                    "label_name": LABEL_MAP.get(w["label"], "unknown"),
                }
                row.update(hrv)
                all_rows.append(row)

            print(f"  {sid}: {sum(1 for r in all_rows if r['subject_id'] == sid)} "
                  f"feature vectors extracted")

        except Exception as e:
            print(f"  [ERROR] {sid}: {e}")
            continue

    df = pd.DataFrame(all_rows)

    # Drop rows with too many NaN features
    feature_cols = [c for c in df.columns if c not in
                    ["subject_id", "window_idx", "label", "label_name"]]
    df = df.dropna(subset=feature_cols, thresh=int(len(feature_cols) * 0.7))

    print(f"\n=== Total: {len(df)} feature vectors from {df['subject_id'].nunique()} subjects ===")
    if len(df) > 0:
        print(f"  Baseline: {(df['label'] == 1).sum()}")
        print(f"  Stress:   {(df['label'] == 2).sum()}")

    return df


def extract_performance_features(
    n_sessions_per_level: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate simulated MATB-II performance data and return feature DataFrame.

    Returns
    -------
    pd.DataFrame
        Performance feature matrix with workload labels.
    """
    print("\nGenerating simulated MATB-II performance data...")
    df = generate_full_simulation(
        n_sessions_per_level=n_sessions_per_level,
        seed=seed,
    )
    return df


def build_combined_feature_matrix(
    wesad_df: pd.DataFrame,
    performance_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the combined feature matrices for model training.

    Since WESAD and MATB-II data come from different experiments, they are
    trained as separate models and fused at the risk-scoring stage.

    Returns
    -------
    physio_features : pd.DataFrame
        Physiological (HRV) feature matrix with binary stress labels.
    perf_features : pd.DataFrame
        Performance feature matrix with workload labels.
    """
    # Physiological features: binary classification (baseline=0, stress=1)
    physio = wesad_df.copy()
    physio["stress_label"] = (physio["label"] == 2).astype(int)

    # Performance features: already has workload_level column
    perf = performance_df.copy()

    return physio, perf


def save_features(
    df: pd.DataFrame,
    filename: str,
    output_dir: Optional[str] = None,
) -> str:
    """Save feature DataFrame to CSV."""
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "output"
        )
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    print(f"Saved features to {path}")
    return path
