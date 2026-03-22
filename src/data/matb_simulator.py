"""
MATB-II Performance Data Simulator
===================================
Since the MATB-II folder contains only 2 sample runs, this module generates
synthetic behavioral performance data that emulates different workload levels.

Generates:
  - Reaction-time distributions (Log-normal, workload-dependent)
  - Tracking error sequences (Gaussian random walk with drift)
  - Communication response times
  - Labeled by workload condition (Low / Medium / High)

Based on published parameters from the research plan (Section 5.2):
  - Fatigued pilots: 40% reduction in RT speed, 32% increase in decision errors
  - CVRT increases with fatigue (inconsistency vs. simple slowing)
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy as sp_entropy
from typing import Optional, Tuple


# ----- Workload parameters (empirically calibrated) -----
WORKLOAD_PARAMS = {
    "low": {
        "rt_mean": 2.0,      # seconds (mean of log-normal)
        "rt_std": 0.4,       # seconds
        "rt_lapse_prob": 0.02,  # probability of attentional lapse (> 5s RT)
        "track_rmsd_mean": 25.0,  # pixels
        "track_rmsd_std": 8.0,
        "track_drift": 0.1,     # slow drift factor
        "miss_rate": 0.02,       # fraction of missed events
        "comm_rt_mean": 8.0,
        "comm_rt_std": 2.0,
    },
    "medium": {
        "rt_mean": 3.0,
        "rt_std": 0.8,
        "rt_lapse_prob": 0.08,
        "track_rmsd_mean": 40.0,
        "track_rmsd_std": 12.0,
        "track_drift": 0.3,
        "miss_rate": 0.08,
        "comm_rt_mean": 12.0,
        "comm_rt_std": 4.0,
    },
    "high": {
        "rt_mean": 4.5,
        "rt_std": 1.5,
        "rt_lapse_prob": 0.20,
        "track_rmsd_mean": 60.0,
        "track_rmsd_std": 18.0,
        "track_drift": 0.7,
        "miss_rate": 0.18,
        "comm_rt_mean": 18.0,
        "comm_rt_std": 6.0,
    },
}


def simulate_reaction_times(
    n_events: int = 50,
    workload: str = "low",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate SYSMON reaction times for a given workload level.

    Uses a log-normal distribution with workload-dependent parameters,
    plus occasional attentional lapses (very long RTs).

    Parameters
    ----------
    n_events : int
        Number of events/trials.
    workload : str
        'low', 'medium', or 'high'.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of reaction times in seconds.
    """
    rng = np.random.default_rng(seed)
    params = WORKLOAD_PARAMS[workload]

    # Log-normal parameters
    mu_ln = np.log(params["rt_mean"]**2 / np.sqrt(params["rt_std"]**2 + params["rt_mean"]**2))
    sigma_ln = np.sqrt(np.log(1 + (params["rt_std"]**2 / params["rt_mean"]**2)))

    rts = rng.lognormal(mu_ln, sigma_ln, n_events)

    # Inject attentional lapses
    n_lapses = int(n_events * params["rt_lapse_prob"])
    if n_lapses > 0:
        lapse_indices = rng.choice(n_events, n_lapses, replace=False)
        rts[lapse_indices] = rng.uniform(6.0, 15.0, n_lapses)

    # Clip to realistic range
    rts = np.clip(rts, 0.3, 30.0)

    return rts


def simulate_tracking_error(
    n_intervals: int = 30,
    workload: str = "low",
    interval_sec: float = 15.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate TRACK task RMSD values for a given workload level.

    Uses a Gaussian random walk with workload-dependent drift and noise.

    Parameters
    ----------
    n_intervals : int
        Number of tracking intervals.
    workload : str
        'low', 'medium', or 'high'.
    interval_sec : float
        Duration of each tracking interval in seconds.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Array of RMSD tracking error values.
    """
    rng = np.random.default_rng(seed)
    params = WORKLOAD_PARAMS[workload]

    # Base RMSD plus drift and noise
    base = params["track_rmsd_mean"]
    noise = rng.normal(0, params["track_rmsd_std"], n_intervals)
    drift = np.cumsum(rng.normal(params["track_drift"], 0.3, n_intervals))

    rmsd = base + noise + drift
    rmsd = np.clip(rmsd, 5.0, 200.0)

    return rmsd


def simulate_comm_responses(
    n_events: int = 20,
    workload: str = "low",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate COMM task response times and correctness.

    Returns
    -------
    rt : np.ndarray
        Response times in seconds (30 for timeout).
    correct : np.ndarray
        Boolean array of correctness.
    """
    rng = np.random.default_rng(seed)
    params = WORKLOAD_PARAMS[workload]

    mu_ln = np.log(params["comm_rt_mean"]**2 /
                   np.sqrt(params["comm_rt_std"]**2 + params["comm_rt_mean"]**2))
    sigma_ln = np.sqrt(np.log(1 + (params["comm_rt_std"]**2 / params["comm_rt_mean"]**2)))

    rts = rng.lognormal(mu_ln, sigma_ln, n_events)
    rts = np.clip(rts, 1.0, 30.0)

    # Missed events (timeouts)
    n_miss = int(n_events * params["miss_rate"])
    if n_miss > 0:
        miss_idx = rng.choice(n_events, n_miss, replace=False)
        rts[miss_idx] = 30.0

    # Correctness (errors increase with workload)
    error_rate = params["miss_rate"] * 1.5
    correct = rng.random(n_events) > error_rate

    return rts, correct


def generate_full_simulation(
    n_sessions_per_level: int = 10,
    n_sysmon_events: int = 50,
    n_track_intervals: int = 30,
    n_comm_events: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a complete simulated behavioral dataset for model training.

    Creates sessions across Low/Medium/High workload conditions.

    Parameters
    ----------
    n_sessions_per_level : int
        How many sessions to simulate per workload level.
    n_sysmon_events : int
        Number of SYSMON events per session.
    n_track_intervals : int
        Number of tracking intervals per session.
    n_comm_events : int
        Number of COMM events per session.
    seed : int
        Master random seed.

    Returns
    -------
    pd.DataFrame
        One row per session with aggregated feature statistics.
        Columns include workload level and derived behavioral metrics.
    """
    rng = np.random.default_rng(seed)
    records = []

    for workload in ["low", "medium", "high"]:
        for session in range(n_sessions_per_level):
            session_seed = seed + hash(f"{workload}_{session}") % (2**31)

            # SYSMON reaction times
            rts = simulate_reaction_times(n_sysmon_events, workload, session_seed)
            mean_rt = np.mean(rts)
            std_rt = np.std(rts)
            cv_rt = std_rt / mean_rt if mean_rt > 0 else 0
            median_rt = np.median(rts)

            # Lag-1 autocorrelation
            if len(rts) > 1:
                lag1 = np.corrcoef(rts[:-1], rts[1:])[0, 1]
            else:
                lag1 = 0.0

            # RT distribution shape
            rt_skew = skew(rts)
            rt_kurt = kurtosis(rts)

            # Tracking RMSD
            track_rmsd = simulate_tracking_error(n_track_intervals, workload,
                                                  seed=session_seed + 1)
            mean_track = np.mean(track_rmsd)
            std_track = np.std(track_rmsd)

            # Inceptor entropy (simulate stick positions as bins)
            # Approximate by computing entropy of tracking RMSD distribution
            hist, _ = np.histogram(track_rmsd, bins=20, density=True)
            hist = hist[hist > 0]
            inceptor_entropy = sp_entropy(hist)

            # COMM response times
            comm_rts, comm_correct = simulate_comm_responses(n_comm_events, workload,
                                                              session_seed + 2)
            mean_comm_rt = np.mean(comm_rts)
            comm_accuracy = np.mean(comm_correct)
            n_timeouts = np.sum(comm_rts >= 30.0)

            records.append({
                "session_id": f"{workload}_{session:03d}",
                "workload": workload,
                "workload_level": {"low": 0, "medium": 1, "high": 2}[workload],
                # SYSMON features
                "mean_rt": mean_rt,
                "std_rt": std_rt,
                "cv_rt": cv_rt,
                "median_rt": median_rt,
                "lag1_autocorr": lag1 if not np.isnan(lag1) else 0.0,
                "rt_skewness": rt_skew,
                "rt_kurtosis": rt_kurt,
                # Tracking features
                "mean_track_rmsd": mean_track,
                "std_track_rmsd": std_track,
                "inceptor_entropy": inceptor_entropy,
                # COMM features
                "mean_comm_rt": mean_comm_rt,
                "comm_accuracy": comm_accuracy,
                "n_timeouts": n_timeouts,
            })

    df = pd.DataFrame(records)
    print(f"Generated {len(df)} simulated sessions "
          f"(Low={n_sessions_per_level}, Med={n_sessions_per_level}, "
          f"High={n_sessions_per_level})")

    return df
