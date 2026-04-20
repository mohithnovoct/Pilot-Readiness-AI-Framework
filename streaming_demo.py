"""
Real-Time Streaming Inference Demo
====================================
A Flask + SocketIO application that simulates real-time pilot monitoring
by replaying pre-computed features through the edge model.

Displays:
  - Animated risk score gauge (0-1 with color zones)
  - Scrolling risk timeline chart
  - Multi-level alert status (NOMINAL / CAUTION / WARNING / CRITICAL)
  - Top feature contributions driving current score

Part of the Pilot Readiness AI Framework.

Usage:
    python streaming_demo.py                        # Default: WESAD dataset, 1x speed
    python streaming_demo.py --speed 2              # 2x replay speed
    python streaming_demo.py --dataset swell        # Use SWELL data
    python streaming_demo.py --port 8080            # Custom port
"""

import argparse
import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
import pickle
import threading

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

from src.risk.fusion import compute_risk_score, get_risk_category
from config import HRV_FEATURE_COLS, SWELL_PHYSIO_COLS, SWELL_BEH_COLS

logger = logging.getLogger("streaming_demo")

app = Flask(__name__)
app.config["SECRET_KEY"] = "pilot-readiness-demo"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


class StreamingEngine:
    """Manages the real-time data replay and inference."""

    def __init__(self, dataset="wesad", speed=1.0):
        self.dataset = dataset
        self.speed = speed
        self.running = False
        self.current_index = 0
        self.risk_history = []
        self.feature_names = []

        # Load pre-trained model
        self.model = None
        self.features_df = None
        self.threshold = 0.5
        self.alpha = 0.05  # Default operational scenario
        self._load_resources()

    def _load_resources(self):
        """Load model and features."""
        output_dir = os.path.join(PROJECT_ROOT, "output")
        model_dir = os.path.join(output_dir, "models")
        features_dir = os.path.join(output_dir, "features")

        # Load model
        model_path = os.path.join(model_dir, f"{self.dataset}_stress_classifier.pkl")
        if not os.path.exists(model_path):
            # Fallback to default
            model_path = os.path.join(model_dir, "stress_classifier.pkl")

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(f"Model not found: {model_path}. Using synthetic data.")

        # Load features
        if self.dataset == "wesad":
            feat_path = os.path.join(features_dir, "wesad_features.csv")
            self.feature_names = [c for c in HRV_FEATURE_COLS]
        else:
            feat_path = os.path.join(features_dir, "swell_physio.csv")
            self.feature_names = [c for c in SWELL_PHYSIO_COLS + SWELL_BEH_COLS]

        if os.path.exists(feat_path):
            self.features_df = pd.read_csv(feat_path)
            logger.info(f"Loaded {len(self.features_df)} feature rows from {feat_path}")
        else:
            logger.warning(f"Features not found: {feat_path}. Generating synthetic.")
            self._generate_synthetic()

        # Compute threshold from baseline data
        if self.features_df is not None and self.model is not None:
            self._compute_threshold()

    def _generate_synthetic(self):
        """Generate synthetic features for demo when data isn't available."""
        np.random.seed(42)
        n = 200
        # Create plausible HRV-like features
        data = {
            "subject_id": ["DEMO"] * n,
            "label": ([1] * (n // 2)) + ([2] * (n // 2)),
        }
        for feat in self.feature_names:
            data[feat] = np.random.normal(0.5, 0.2, n)

        self.features_df = pd.DataFrame(data)

    def _compute_threshold(self):
        """Compute operational threshold from feature data."""
        available_cols = [c for c in self.feature_names if c in self.features_df.columns]
        if not available_cols or self.model is None:
            return

        X = self.features_df[available_cols].values
        # Handle NaN
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = col_medians[j] if not np.isnan(col_medians[j]) else 0

        try:
            probs = self.model.predict_proba(X)[:, 1]
            if "label" in self.features_df.columns:
                baseline_mask = self.features_df["label"].isin([0, 1])
                baseline_probs = probs[baseline_mask]
                if len(baseline_probs) > 0:
                    self.threshold = float(np.quantile(baseline_probs, 1.0 - self.alpha))
        except Exception as e:
            logger.warning(f"Threshold computation failed: {e}")

    def get_current_prediction(self):
        """Get the prediction for the current time step."""
        if self.features_df is None:
            return self._synthetic_prediction()

        idx = self.current_index % len(self.features_df)
        row = self.features_df.iloc[idx]

        available_cols = [c for c in self.feature_names if c in self.features_df.columns]

        if self.model is not None and available_cols:
            features = row[available_cols].values.astype(float).reshape(1, -1)
            # Handle NaN
            features = np.nan_to_num(features, nan=0.0)

            try:
                stress_prob = float(self.model.predict_proba(features)[0, 1])
            except Exception:
                stress_prob = float(np.random.uniform(0.2, 0.8))

            # Simulated performance score (correlated with stress)
            perf_score = np.clip(stress_prob * 0.6 + np.random.normal(0, 0.1), 0, 1)
            risk = compute_risk_score(stress_prob, perf_score)

            # Top feature contributions
            top_features = {}
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
                top_idx = np.argsort(importances)[::-1][:5]
                for i in top_idx:
                    if i < len(available_cols):
                        top_features[available_cols[i]] = float(features[0, i])
        else:
            risk = self._synthetic_prediction()["risk_score"]
            stress_prob = risk * 0.7
            perf_score = risk * 0.3
            top_features = {}

        category, color = get_risk_category(risk)
        is_alert = risk > self.threshold
        true_label = int(row.get("label", 0)) if hasattr(row, "get") else 0

        # Multi-level alert classification
        if risk > 0.80:
            alert_level = "CRITICAL"
        elif risk > 0.60:
            alert_level = "WARNING"
        elif risk > 0.40:
            alert_level = "CAUTION"
        else:
            alert_level = "NOMINAL"

        self.risk_history.append(risk)
        self.current_index += 1

        subject_id = str(row.get("subject_id", "Unknown")) if hasattr(row, "get") else "Unknown"

        return {
            "timestamp": self.current_index,
            "risk_score": round(risk, 4),
            "stress_prob": round(stress_prob, 4),
            "perf_score": round(perf_score, 4),
            "category": category,
            "color": color,
            "is_alert": is_alert,
            "alert_level": alert_level,
            "decision": alert_level if is_alert else "READY",
            "threshold": round(self.threshold, 4),
            "subject_id": subject_id,
            "true_label": "Stress" if true_label == 2 else "Baseline" if true_label == 1 else "Unknown",
            "top_features": top_features,
            "total_processed": self.current_index,
            "total_alerts": sum(1 for r in self.risk_history if r > self.threshold),
        }

    def _synthetic_prediction(self):
        """Generate synthetic prediction for demo."""
        t = self.current_index * 0.1
        risk = 0.3 + 0.3 * np.sin(t) + np.random.normal(0, 0.05)
        risk = float(np.clip(risk, 0, 1))
        return {"risk_score": risk}

    def reset(self):
        """Reset the replay."""
        self.current_index = 0
        self.risk_history = []


# Global engine instance
engine = None


def streaming_thread():
    """Background thread that pushes data at specified speed."""
    global engine
    interval = 1.0 / engine.speed  # seconds between updates

    while engine.running:
        prediction = engine.get_current_prediction()
        socketio.emit("prediction", prediction)
        time.sleep(interval)

        # Loop back if we reach the end
        if engine.current_index >= len(engine.features_df):
            engine.current_index = 0
            socketio.emit("info", {"message": "Replay restarting from beginning..."})


@app.route("/")
def index():
    return render_template("streaming.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "dataset": engine.dataset if engine else "none"})


@app.route("/api/status")
def api_status():
    if engine is None:
        return jsonify({"error": "Engine not initialized"})
    return jsonify({
        "dataset": engine.dataset,
        "speed": engine.speed,
        "running": engine.running,
        "current_index": engine.current_index,
        "total_samples": len(engine.features_df) if engine.features_df is not None else 0,
        "threshold": engine.threshold,
    })


@socketio.on("connect")
def handle_connect():
    logger.info("Client connected")
    socketio.emit("info", {
        "message": f"Connected. Dataset: {engine.dataset}, Speed: {engine.speed}x",
        "threshold": engine.threshold,
    })


@socketio.on("start")
def handle_start():
    global engine
    if not engine.running:
        engine.running = True
        thread = threading.Thread(target=streaming_thread, daemon=True)
        thread.start()
        socketio.emit("info", {"message": "Streaming started"})


@socketio.on("stop")
def handle_stop():
    global engine
    engine.running = False
    socketio.emit("info", {"message": "Streaming paused"})


@socketio.on("reset")
def handle_reset():
    global engine
    engine.running = False
    engine.reset()
    socketio.emit("info", {"message": "Reset to beginning"})


@socketio.on("set_speed")
def handle_speed(data):
    global engine
    engine.speed = float(data.get("speed", 1.0))
    socketio.emit("info", {"message": f"Speed set to {engine.speed}x"})


@socketio.on("set_threshold")
def handle_threshold(data):
    global engine
    engine.threshold = float(data.get("threshold", engine.threshold))
    socketio.emit("info", {
        "message": f"Threshold manually adjusted to {engine.threshold:.3f}",
        "threshold": engine.threshold
    })


def main():
    global engine

    parser = argparse.ArgumentParser(
        description="Pilot Readiness — Real-Time Streaming Demo"
    )
    parser.add_argument("--dataset", default="wesad", choices=["wesad", "swell"],
                       help="Dataset to replay")
    parser.add_argument("--speed", type=float, default=1.0,
                       help="Replay speed multiplier")
    parser.add_argument("--port", type=int, default=5000,
                       help="Server port")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Server host")
    parser.add_argument("--dry-run", action="store_true",
                       help="Test initialization only (don't start server)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                       format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    engine = StreamingEngine(dataset=args.dataset, speed=args.speed)

    if args.dry_run:
        print("Dry run: Engine initialized successfully.")
        pred = engine.get_current_prediction()
        print(f"Sample prediction: {json.dumps(pred, indent=2)}")
        return

    # Log framework info
    try:
        from src.core.framework import PilotReadinessFramework
        fw = PilotReadinessFramework()
        components = fw.list_components()
        fw_status = f"Framework: {sum(len(v) for v in components.values())} plugins"
    except Exception:
        fw_status = "Framework: standalone mode"

    print(f"\n{'='*50}")
    print(f"  🛫 Pilot Readiness Streaming Demo")
    print(f"{'='*50}")
    print(f"  Dataset:   {args.dataset.upper()}")
    print(f"  Speed:     {args.speed}x")
    print(f"  Threshold: {engine.threshold:.3f}")
    print(f"  Samples:   {len(engine.features_df) if engine.features_df is not None else 0}")
    print(f"  Alerts:    Multi-level (NOMINAL → CAUTION → WARNING → CRITICAL)")
    print(f"  {fw_status}")
    print(f"  URL:       http://localhost:{args.port}")
    print(f"{'='*50}\n")

    socketio.run(app, host=args.host, port=args.port,
                debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
