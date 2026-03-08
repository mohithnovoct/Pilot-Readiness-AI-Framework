"""
Edge Model Export
=================
Exports trained LightGBM models for edge deployment.

Methods:
  1. m2cgen: Convert model to pure C/Python/Java code
  2. Model size profiling and latency benchmarking
  3. Parameter counting and RAM estimation

Based on research plan Section 6.1.1:
  - Decision tree ensembles → C++ if/else structures
  - Target: KB-level RAM, negligible latency
"""

import os
import time
import numpy as np
import pickle
from typing import Optional, Dict
import warnings


def export_to_c_code(
    model,
    output_path: str,
    indent: int = 4,
) -> str:
    """
    Export a trained LightGBM model to C code using m2cgen.

    Parameters
    ----------
    model : LGBMClassifier or LGBMRegressor
        Trained model.
    output_path : str
        Path to save the generated C code.
    indent : int
        Indentation width for generated code.

    Returns
    -------
    str
        Path to the generated C file.
    """
    try:
        import m2cgen as m2c
    except ImportError:
        raise ImportError("m2cgen is required for C code export. "
                         "Install with: pip install m2cgen")

    c_code = m2c.export_to_c(model, indent=indent)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(c_code)

    # Compute code size
    code_size = len(c_code.encode("utf-8"))
    print(f"Exported C code to {output_path}")
    print(f"  Code size: {code_size:,} bytes ({code_size / 1024:.1f} KB)")

    return output_path


def export_to_python_code(
    model,
    output_path: str,
    indent: int = 4,
) -> str:
    """
    Export model to pure Python code (no dependencies required).

    Useful for lightweight deployment without LightGBM runtime.
    """
    try:
        import m2cgen as m2c
    except ImportError:
        raise ImportError("m2cgen is required. Install with: pip install m2cgen")

    py_code = m2c.export_to_python(model, indent=indent)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(py_code)

    code_size = len(py_code.encode("utf-8"))
    print(f"Exported Python code to {output_path}")
    print(f"  Code size: {code_size:,} bytes ({code_size / 1024:.1f} KB)")

    return output_path


def profile_model(model) -> Dict:
    """
    Profile model size and complexity.

    Returns
    -------
    dict
        Model statistics including parameter count, tree count,
        estimated RAM, and model file size.
    """
    profile = {}

    if hasattr(model, "n_estimators_"):
        profile["n_estimators"] = model.n_estimators_
    elif hasattr(model, "n_estimators"):
        profile["n_estimators"] = model.n_estimators

    if hasattr(model, "booster_"):
        booster = model.booster_
        try:
            profile["n_trees"] = booster.num_trees()
        except Exception:
            pass
        try:
            # num_leaves may not be available in all LightGBM versions
            model_str = booster.model_to_string()
            n_leaves = model_str.count("leaf_value")
            profile["n_leaves_total"] = n_leaves
        except Exception:
            pass

    # Estimate model size via pickle serialization
    import io
    buf = io.BytesIO()
    pickle.dump(model, buf)
    model_bytes = buf.tell()
    profile["model_size_bytes"] = model_bytes
    profile["model_size_kb"] = model_bytes / 1024

    # RAM estimation for edge (rough: 4 bytes per parameter)
    if "n_leaves_total" in profile:
        # Each leaf stores a value (4 bytes) + split conditions
        est_ram = profile["n_leaves_total"] * 12  # ~12 bytes per leaf node
        profile["estimated_ram_bytes"] = est_ram
        profile["estimated_ram_kb"] = est_ram / 1024
    else:
        profile["estimated_ram_kb"] = model_bytes / 1024

    return profile


def benchmark_inference(
    model,
    X_sample: np.ndarray,
    n_iterations: int = 1000,
) -> Dict:
    """
    Benchmark inference latency.

    Parameters
    ----------
    model : trained model
        The model to benchmark.
    X_sample : np.ndarray
        Sample input(s) for inference.
    n_iterations : int
        Number of iterations to average over.

    Returns
    -------
    dict
        Latency statistics in milliseconds.
    """
    # Warm up
    for _ in range(10):
        if hasattr(model, "predict_proba"):
            model.predict_proba(X_sample[:1])
        else:
            model.predict(X_sample[:1])

    # Measure single-sample latency
    latencies = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        if hasattr(model, "predict_proba"):
            model.predict_proba(X_sample[:1])
        else:
            model.predict(X_sample[:1])
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)

    result = {
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "throughput_samples_per_sec": float(1000.0 / np.mean(latencies)),
    }

    print(f"\nInference Latency Benchmark ({n_iterations} iterations):")
    print(f"  Mean:   {result['mean_latency_ms']:.3f} ms")
    print(f"  Median: {result['median_latency_ms']:.3f} ms")
    print(f"  P95:    {result['p95_latency_ms']:.3f} ms")
    print(f"  P99:    {result['p99_latency_ms']:.3f} ms")
    print(f"  Throughput: {result['throughput_samples_per_sec']:.0f} samples/sec")

    return result


def generate_edge_report(
    model,
    X_sample: np.ndarray,
    output_dir: str,
    model_name: str = "stress_classifier",
) -> Dict:
    """
    Generate a comprehensive edge deployment report.

    Exports model, profiles it, and benchmarks inference.

    Returns
    -------
    dict
        Combined report with all metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    report = {"model_name": model_name}

    # Profile
    print(f"\n{'='*50}")
    print(f"Edge Deployment Report: {model_name}")
    print(f"{'='*50}")

    profile = profile_model(model)
    report["profile"] = profile
    print(f"\nModel Profile:")
    for k, v in profile.items():
        print(f"  {k}: {v}")

    # Export to C
    try:
        c_path = os.path.join(output_dir, f"{model_name}.c")
        export_to_c_code(model, c_path)
        report["c_code_path"] = c_path
        report["c_code_size_kb"] = os.path.getsize(c_path) / 1024
    except Exception as e:
        print(f"  C export failed: {e}")
        report["c_code_path"] = None

    # Export to Python
    try:
        py_path = os.path.join(output_dir, f"{model_name}_pure.py")
        export_to_python_code(model, py_path)
        report["python_code_path"] = py_path
    except Exception as e:
        print(f"  Python export failed: {e}")

    # Benchmark
    benchmark = benchmark_inference(model, X_sample)
    report["benchmark"] = benchmark

    # Edge feasibility summary
    print(f"\n{'='*50}")
    print(f"Edge Feasibility Summary:")
    ram_kb = profile.get("estimated_ram_kb", profile.get("model_size_kb", 0))
    latency_ms = benchmark["mean_latency_ms"]
    print(f"  Estimated RAM:     {ram_kb:.1f} KB", end="")
    print(f" {'✓' if ram_kb < 256 else '✗ (>256KB)'}")
    print(f"  Mean Latency:      {latency_ms:.3f} ms", end="")
    print(f" {'✓' if latency_ms < 10 else '✗ (>10ms)'}")
    print(f"  Real-time capable: {'Yes' if latency_ms < 100 else 'No'}")
    print(f"{'='*50}")

    return report
