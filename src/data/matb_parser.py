"""
NASA MATB-II Data Parser
========================
Parses the sample data files from the Multi-Attribute Task Battery II.

File types handled:
  - SYSM_*.txt : System Monitoring (reaction times to lights/scales)
  - TRCK_*.csv : Tracking task (RMSD of joystick control)
  - COMM_*.txt : Communications (response times to radio freq changes)
  - RMAN_*.csv : Resource Management (fuel tank levels)
"""

import os
import re
import sys
import pandas as pd
import numpy as np
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import MATB_DATA_DIR


def _parse_timestamp(ts_str: str) -> float:
    """Convert HH:MM:SS.S string to total seconds."""
    ts_str = ts_str.strip()
    parts = ts_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(ts_str)


def parse_sysmon(filepath: str) -> pd.DataFrame:
    """
    Parse a SYSMON (System Monitoring) data file.

    Returns
    -------
    pd.DataFrame
        Columns: ['elapsed_time_sec', 'response_time_sec', 'system', 'item', 'correct']
    """
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                elapsed = _parse_timestamp(parts[0])
                rt = float(parts[1])
                system = parts[2]       # 'Light' or 'Scale'
                item = parts[3]         # Color / Number
                correct = parts[4].upper() == "TRUE"

                rows.append({
                    "elapsed_time_sec": elapsed,
                    "response_time_sec": rt,
                    "system": system,
                    "item": item,
                    "correct": correct,
                })
            except (ValueError, IndexError):
                continue

    return pd.DataFrame(rows)


def parse_tracking(filepath: str) -> pd.DataFrame:
    """
    Parse a TRACKING task CSV file.

    Returns
    -------
    pd.DataFrame
        Columns from CSV including RMSD_C values for interval, session, and run.
    """
    df = pd.read_csv(filepath, skipinitialspace=True)
    # Clean column names
    df.columns = [c.strip().rstrip(",") for c in df.columns]

    # Parse elapsed time to seconds
    if "ELAPSED_TIME" in df.columns:
        df["elapsed_time_sec"] = df["ELAPSED_TIME"].apply(_parse_timestamp)

    return df


def parse_comm(filepath: str) -> pd.DataFrame:
    """
    Parse a COMMUNICATIONS data file.

    Returns
    -------
    pd.DataFrame
        Columns: ['elapsed_time_sec', 'response_time_sec', 'ship', 'radio_target',
                   'freq_target', 'radio_selected', 'freq_selected', 'radio_ok',
                   'freq_ok', 'timed_out', 'remarks']
    """
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                elapsed = _parse_timestamp(parts[0])

                # Check if this is a response event or just a timer-reset line
                if "Event Timer Reset" in line:
                    continue

                rt_str = parts[1]
                timed_out = rt_str.startswith("+")
                rt = float(rt_str.replace("+", ""))

                # Parse remaining fields
                ship = parts[2] if len(parts) > 2 else "UNKNOWN"
                radio_t = parts[3] if len(parts) > 3 else ""
                freq_t = parts[4] if len(parts) > 4 else ""
                radio_s = parts[5] if len(parts) > 5 else ""
                freq_s = parts[6] if len(parts) > 6 else ""
                r_ok = parts[7].upper() == "TRUE" if len(parts) > 7 else False
                f_ok = parts[8].upper() == "TRUE" if len(parts) > 8 else False

                remarks = ""
                if "-" in line:
                    remark_idx = line.rfind("- ")
                    if remark_idx > 0:
                        remarks = line[remark_idx + 2:].strip()

                rows.append({
                    "elapsed_time_sec": elapsed,
                    "response_time_sec": rt,
                    "ship": ship,
                    "radio_target": radio_t,
                    "freq_target": freq_t,
                    "radio_selected": radio_s,
                    "freq_selected": freq_s,
                    "radio_ok": r_ok,
                    "freq_ok": f_ok,
                    "timed_out": timed_out,
                    "remarks": remarks,
                })
            except (ValueError, IndexError):
                continue

    return pd.DataFrame(rows)


def parse_resman(filepath: str) -> pd.DataFrame:
    """
    Parse a Resource Management CSV file.

    Returns
    -------
    pd.DataFrame
        Fuel tank levels and pump states over time.
    """
    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = [c.strip().rstrip(",") for c in df.columns]
    return df


def load_all_matb_data(data_dir: Optional[str] = None) -> dict:
    """
    Load all MATB-II sample data files.

    Returns
    -------
    dict
        Keys: 'sysmon', 'tracking', 'comm', 'resman'
        Each value is a list of (filename, DataFrame) tuples.
    """
    base = data_dir or MATB_DATA_DIR
    result = {"sysmon": [], "tracking": [], "comm": [], "resman": []}

    # SYSMON
    sysmon_dir = os.path.join(base, "SYSMON")
    if os.path.isdir(sysmon_dir):
        for f in sorted(os.listdir(sysmon_dir)):
            if f.endswith(".txt"):
                df = parse_sysmon(os.path.join(sysmon_dir, f))
                result["sysmon"].append((f, df))
                print(f"  [SYSMON] {f}: {len(df)} events")

    # TRACKING
    track_dir = os.path.join(base, "TRACK")
    if os.path.isdir(track_dir):
        for f in sorted(os.listdir(track_dir)):
            if f.endswith(".csv"):
                df = parse_tracking(os.path.join(track_dir, f))
                result["tracking"].append((f, df))
                print(f"  [TRACK]  {f}: {len(df)} intervals")

    # COMM
    comm_dir = os.path.join(base, "COMM")
    if os.path.isdir(comm_dir):
        for f in sorted(os.listdir(comm_dir)):
            if f.endswith(".txt"):
                df = parse_comm(os.path.join(comm_dir, f))
                result["comm"].append((f, df))
                print(f"  [COMM]   {f}: {len(df)} events")

    # RESMAN
    resman_dir = os.path.join(base, "RESMAN")
    if os.path.isdir(resman_dir):
        for f in sorted(os.listdir(resman_dir)):
            if f.endswith(".csv"):
                df = parse_resman(os.path.join(resman_dir, f))
                result["resman"].append((f, df))
                print(f"  [RESMAN] {f}: {len(df)} records")

    return result
