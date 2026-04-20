"""
ONNX Model Export
==================
Export trained LightGBM models to ONNX format for cross-platform
edge deployment using ONNX Runtime.

Advantages over m2cgen C export:
  - Standardised format supported by many runtimes
  - Hardware-accelerated inference (TensorRT, DirectML, etc.)
  - Broader language support (C++, C#, Java, JavaScript, …)
  - Model optimisation passes (quantisation, graph fusion)

Usage
-----
    from src.edge.onnx_export import export_to_onnx, benchmark_onnx
    export_to_onnx(model, "output/edge/model.onnx", n_features=15)
    benchmark_onnx("output/edge/model.onnx", X_sample)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("pilot_readiness.edge.onnx")


def export_to_onnx(
    model,
    output_path: str,
    n_features: Optional[int] = None,
    model_name: str = "pilot_readiness_model",
    opset_version: int = 12,
) -> str:
    """
    Export a scikit-learn–compatible model (LightGBM, XGBoost, RF, etc.)
    to ONNX format.

    Parameters
    ----------
    model : trained model
        Must have a ``predict`` or ``predict_proba`` method.
    output_path : str
        Path to save the .onnx file.
    n_features : int, optional
        Number of input features.  Auto-detected if possible.
    model_name : str
        Name embedded in the ONNX model metadata.
    opset_version : int
        ONNX opset version.

    Returns
    -------
    str
        Path to the saved ONNX file.
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        raise ImportError(
            "skl2onnx is required for ONNX export. "
            "Install with: pip install pilot-readiness[edge]"
        )

    # Auto-detect n_features
    if n_features is None:
        if hasattr(model, "n_features_"):
            n_features = model.n_features_
        elif hasattr(model, "n_features_in_"):
            n_features = model.n_features_in_
        elif hasattr(model, "feature_importances_"):
            n_features = len(model.feature_importances_)
        else:
            raise ValueError(
                "Cannot auto-detect n_features. "
                "Pass n_features explicitly."
            )

    # Define input type
    initial_type = [("input", FloatTensorType([None, n_features]))]

    # Convert
    logger.info("Converting model to ONNX (n_features=%d)...", n_features)
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=opset_version,
        options={id(model): {"zipmap": False}},  # return raw probabilities
    )

    # Set model metadata
    onnx_model.doc_string = f"Pilot Readiness: {model_name}"

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    file_size = os.path.getsize(output_path)
    logger.info(
        "Saved ONNX model to %s (%.1f KB)",
        output_path, file_size / 1024,
    )

    return output_path


def validate_onnx(
    onnx_path: str,
    model,
    X_sample: np.ndarray,
    atol: float = 1e-4,
) -> bool:
    """
    Validate that the ONNX model produces the same output as the
    original model.

    Returns True if outputs match within tolerance.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not available — skipping validation")
        return True

    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name

    # ONNX inference
    X_float = X_sample.astype(np.float32)
    onnx_out = sess.run(None, {input_name: X_float})

    # Original model inference
    if hasattr(model, "predict_proba"):
        original_out = model.predict_proba(X_sample)
    else:
        original_out = model.predict(X_sample)

    # Compare
    if isinstance(onnx_out, list) and len(onnx_out) > 1:
        # Classification: onnx_out[1] is probabilities
        onnx_probs = np.array(onnx_out[1])
        if onnx_probs.ndim == 1:
            onnx_probs = onnx_probs.reshape(-1, 1)
        matches = np.allclose(onnx_probs, original_out, atol=atol)
    else:
        onnx_preds = np.array(onnx_out[0]).flatten()
        original_preds = np.array(original_out).flatten()
        matches = np.allclose(onnx_preds, original_preds, atol=atol)

    if matches:
        logger.info("ONNX validation passed (atol=%g)", atol)
    else:
        logger.warning("ONNX validation FAILED — outputs differ beyond atol=%g", atol)

    return matches


def benchmark_onnx(
    onnx_path: str,
    X_sample: np.ndarray,
    n_iterations: int = 1000,
) -> Dict[str, Any]:
    """
    Benchmark ONNX Runtime inference latency.

    Returns
    -------
    dict
        Latency statistics in milliseconds.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime required for benchmarking")

    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    X_float = X_sample[:1].astype(np.float32)

    # Warm up
    for _ in range(10):
        sess.run(None, {input_name: X_float})

    # Benchmark
    latencies = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        sess.run(None, {input_name: X_float})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    latencies = np.array(latencies)

    result = {
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "min_latency_ms": float(np.min(latencies)),
        "throughput_samples_per_sec": float(1000.0 / np.mean(latencies)),
        "model_size_kb": os.path.getsize(onnx_path) / 1024,
    }

    logger.info("ONNX Benchmark (%d iters):", n_iterations)
    logger.info("  Mean:       %.3f ms", result["mean_latency_ms"])
    logger.info("  P95:        %.3f ms", result["p95_latency_ms"])
    logger.info("  Throughput: %.0f samples/s", result["throughput_samples_per_sec"])

    return result
