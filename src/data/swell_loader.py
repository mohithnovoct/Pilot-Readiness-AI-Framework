"""
SWELL Dataset Loader
====================
Loads the SWELL Knowledge Work Dataset features.
It extracts physiological features (HR, RMSSD, SCL) and
behavioral features (keystrokes, mouse activity) under different stress conditions.

Conditions:
  'R', 'N' (Relax, Neutral) -> Baseline (0)
  'T', 'I' (Time Pressure, Interruption) -> Stress (1)
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import warnings

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import SWELL_DIR, SWELL_PHYSIO_COLS, SWELL_BEH_COLS

# Map SWELL explicit conditions to binary labels
CONDITION_MAP = {
    'R': 0, # Relax
    'N': 0, # Neutral
    'T': 1, # Time pressure
    'I': 1  # Interruption
}


def load_swell_physio(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load SWELL physiological features (HR, RMSSD, SCL).
    
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with 'subject_id', 'label', 'Condition', and HRV features.
    """
    base = data_dir or SWELL_DIR
    physio_file = os.path.join(
        base, "3 - Feature dataset", "per sensor",
        "D - Physiology features (HR_HRV_SCL - final).csv"
    )
    
    if not os.path.exists(physio_file):
        raise FileNotFoundError(f"SWELL physio file not found: {physio_file}")
    
    df = pd.read_csv(physio_file)
    
    # Clean up and normalize column names for pipeline compatibility
    df = df.rename(columns={"PP": "subject_id"})
    
    # Map conditions to binary stress label (1=Stress, 0=Baseline)
    df["label"] = df["Condition"].map(CONDITION_MAP)
    
    # Convert features to numeric, coerce errors to NaN
    for c in SWELL_PHYSIO_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Drop rows with unidentified conditions or missing features
    df = df.dropna(subset=["label", "subject_id"] + SWELL_PHYSIO_COLS)
    df["label"] = df["label"].astype(int)
    
    # For compatibility downstream, map 'label' to 'stress_label'
    df["stress_label"] = df["label"]
    
    print(f"[SWELL] Loaded Physiology Data: {len(df)} samples")
    print(f"        Class dist: Baseline={sum(df['label']==0)}, Stress={sum(df['label']==1)}")
    
    return df


def load_swell_behavioral(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load SWELL computer interaction (behavioral) features.
    
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with keystrokes and mouse dynamics per minute.
    """
    base = data_dir or SWELL_DIR
    csv_pattern = os.path.join(
        base, "3 - Feature dataset", "per sensor",
        "A - Computer interaction features*csv"
    )
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        raise FileNotFoundError(f"No SWELL behavioral CSVs found: {csv_pattern}")
        
    dfs = []
    for f in csv_files:
        _df = pd.read_csv(f)
        dfs.append(_df)
        
    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={"PP": "subject_id"})
    
    # Map conditions to binary label (here acting as workload_level for performance model)
    df["workload_level"] = df["Condition"].map(CONDITION_MAP)
    
    df = df.dropna(subset=["workload_level", "subject_id"])
    df["workload_level"] = df["workload_level"].astype(int)
    
    # Ensure behavioral feature columns exist
    available_cols = [c for c in SWELL_BEH_COLS if c in df.columns]
    
    # Convert features to numeric, coerce errors to NaN
    for c in available_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    df = df.dropna(subset=available_cols)
    
    print(f"[SWELL] Loaded Behavioral Data: {len(df)} samples, {len(available_cols)} features")
    return df
