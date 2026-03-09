"""
Pilot Readiness Monitoring — Main Pipeline
===========================================
End-to-end orchestrator that runs:
  1. Data loading (WESAD + MATB-II)
  2. Preprocessing & windowing
  3. Feature extraction (HRV + Performance)
  4. Model training (LightGBM stress classifier + performance regressor)
  5. Risk score fusion (Neyman-Pearson thresholds)
  6. Edge model export
  7. Visualization & dashboard generation

Usage:
    python main.py                    # Full pipeline
    python main.py --quick-test       # Single subject quick test
    python main.py --skip-training    # Use cached models
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data.wesad_loader import SUBJECT_IDS
from src.features.feature_pipeline import (
    extract_wesad_features, extract_performance_features,
    build_combined_feature_matrix, save_features
)
from src.models.stress_classifier import (
    prepare_training_data, train_loso_cv, train_final_model,
    compute_shap_importance, save_model, HRV_FEATURE_COLS
)
from src.models.performance_model import (
    prepare_performance_data, train_performance_model,
    save_model as save_perf_model, PERF_FEATURE_COLS
)
from src.risk.fusion import compute_risk_batch, get_risk_category
from src.risk.threshold import ThresholdManager, evaluate_threshold_performance
from src.edge.export_model import generate_edge_report
from src.visualization.plots import (
    plot_confusion_matrix, plot_roc_curve, plot_feature_importance,
    plot_risk_timeline, plot_hrv_comparison, plot_loso_results,
    save_all_plots
)
from src.visualization.dashboard import generate_dashboard


def setup_output_dirs():
    """Create output directories."""
    dirs = [
        os.path.join(PROJECT_ROOT, "output"),
        os.path.join(PROJECT_ROOT, "output", "models"),
        os.path.join(PROJECT_ROOT, "output", "plots"),
        os.path.join(PROJECT_ROOT, "output", "features"),
        os.path.join(PROJECT_ROOT, "output", "edge"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs


def run_pipeline(args):
    """Execute the full pipeline."""
    start_time = time.time()

    print("=" * 70)
    print("  PILOT READINESS MONITORING FRAMEWORK")
    print("  A Lightweight AI Framework for Stress-Correlated")
    print("  Performance Indicators")
    print("=" * 70)

    setup_output_dirs()
    output_dir = os.path.join(PROJECT_ROOT, "output")

    # ============================================================
    # STEP 1: Feature Extraction
    # ============================================================
    print("\n" + "=" * 70)
    print("  STEP 1: Feature Extraction")
    print("=" * 70)

    # --- WESAD features ---
    subject_ids = ["S2"] if args.quick_test else None
    print("\n[1a] Extracting HRV features from WESAD data...")

    features_path = os.path.join(output_dir, "features", "wesad_features.csv")
    if os.path.exists(features_path) and args.skip_extraction:
        print(f"  Loading cached features from {features_path}")
        wesad_df = pd.read_csv(features_path)
    else:
        wesad_df = extract_wesad_features(
            subject_ids=subject_ids,
            window_sec=args.window_sec,
            overlap=args.overlap,
        )
        save_features(wesad_df, "wesad_features.csv",
                      os.path.join(output_dir, "features"))

    # --- Performance features ---
    print("\n[1b] Generating simulated performance data...")

    perf_path = os.path.join(output_dir, "features", "performance_features.csv")
    if os.path.exists(perf_path) and args.skip_extraction:
        print(f"  Loading cached features from {perf_path}")
        perf_df = pd.read_csv(perf_path)
    else:
        perf_df = extract_performance_features(
            n_sessions_per_level=args.n_sessions,
            seed=42,
        )
        save_features(perf_df, "performance_features.csv",
                      os.path.join(output_dir, "features"))

    # Combine
    physio_df, perf_train_df = build_combined_feature_matrix(wesad_df, perf_df)

    # ============================================================
    # STEP 2: Model Training
    # ============================================================
    print("\n" + "=" * 70)
    print("  STEP 2: Model Training")
    print("=" * 70)

    stress_model_path = os.path.join(output_dir, "models", "stress_classifier.pkl")
    perf_model_path = os.path.join(output_dir, "models", "performance_model.pkl")

    # --- Stress classifier ---
    print("\n[2a] Training LightGBM Stress Classifier...")

    feature_cols = [c for c in HRV_FEATURE_COLS if c in physio_df.columns]
    X_stress, y_stress, groups = prepare_training_data(physio_df, feature_cols)

    if len(X_stress) < 10:
        print("  WARNING: Too few samples for robust training.")
        print("  Consider running without --quick-test for full dataset.")

    # LOSO cross-validation
    if len(np.unique(groups)) > 1:
        cv_results = train_loso_cv(X_stress, y_stress, groups)
    else:
        print("  Skipping LOSO CV (only 1 subject in quick test)")
        cv_results = None

    # Train final model
    stress_model = train_final_model(
        X_stress, y_stress,
        do_grid_search=not args.quick_test,
    )
    save_model(stress_model, stress_model_path)

    # SHAP importance
    importance_df = compute_shap_importance(stress_model, X_stress, feature_cols)

    # --- Performance model ---
    print("\n[2b] Training Performance Regression Model...")

    perf_feature_cols = [c for c in PERF_FEATURE_COLS if c in perf_train_df.columns]
    X_perf, y_perf, scaler = prepare_performance_data(perf_train_df, perf_feature_cols)
    perf_model = train_performance_model(X_perf, y_perf, do_grid_search=not args.quick_test)
    save_perf_model(perf_model, perf_model_path)

    # ============================================================
    # STEP 3: Risk Scoring & Threshold Tuning
    # ============================================================
    print("\n" + "=" * 70)
    print("  STEP 3: Risk Score Fusion & Threshold Optimization")
    print("=" * 70)

    # Generate stress probabilities for all windows
    stress_probs = stress_model.predict_proba(X_stress)[:, 1]

    # Simulate performance scores (map from the same windows for demo)
    # In production, these would come from real MATB-II data
    np.random.seed(42)
    perf_scores = np.clip(
        stress_probs * 0.7 + np.random.normal(0, 0.15, len(stress_probs)),
        0, 1
    )

    # Fused risk scores
    risk_scores = compute_risk_batch(stress_probs, perf_scores, w_phys=0.6, w_perf=0.4)

    # Threshold tuning
    baseline_mask = y_stress == 0
    stress_mask = y_stress == 1

    baseline_risks = risk_scores[baseline_mask]
    stress_risks = risk_scores[stress_mask]

    # Initialize ThresholdManager with baseline distribution
    tm = ThresholdManager(baseline_scores=baseline_risks, alpha=0.05)

    print(f"\n  Threshold Analysis:")
    for scenario in ["training", "operational", "critical"]:
        gamma = tm.set_scenario(scenario)
        metrics = evaluate_threshold_performance(baseline_risks, stress_risks, gamma)
        print(f"  {scenario.upper():12s}  γ={gamma:.3f}  "
              f"FAR={metrics['false_alarm_rate']:.3f}  "
              f"DR={metrics['detection_rate']:.3f}  "
              f"F1={metrics['f1']:.3f}")

    # Use operational threshold for dashboard
    threshold = tm.set_scenario("operational")

    # ============================================================
    # STEP 4: Edge Optimization
    # ============================================================
    print("\n" + "=" * 70)
    print("  STEP 4: Edge Model Export & Benchmarking")
    print("=" * 70)

    edge_dir = os.path.join(output_dir, "edge")
    edge_report = generate_edge_report(
        stress_model, X_stress,
        output_dir=edge_dir,
        model_name="stress_classifier",
    )

    # ============================================================
    # STEP 5: Visualization
    # ============================================================
    print("\n" + "=" * 70)
    print("  STEP 5: Generating Visualizations")
    print("=" * 70)

    plots_dir = os.path.join(output_dir, "plots")

    plots = {}

    # Confusion matrix
    if cv_results is not None and "confusion_matrix" in cv_results:
        plots["confusion_matrix"] = plot_confusion_matrix(
            cv_results["confusion_matrix"],
            save_path=os.path.join(plots_dir, "confusion_matrix.png"),
        )

    # Feature importance
    if importance_df is not None:
        plots["feature_importance"] = plot_feature_importance(
            importance_df,
            save_path=os.path.join(plots_dir, "feature_importance.png"),
        )

    # HRV comparison
    plots["hrv_comparison"] = plot_hrv_comparison(
        wesad_df,
        save_path=os.path.join(plots_dir, "hrv_comparison.png"),
    )

    # Risk timeline
    timestamps = np.arange(len(risk_scores))
    plots["risk_timeline"] = plot_risk_timeline(
        timestamps, risk_scores, threshold, labels=y_stress,
        save_path=os.path.join(plots_dir, "risk_timeline.png"),
    )

    # LOSO results
    if cv_results is not None:
        plots["loso_results"] = plot_loso_results(
            cv_results["fold_results"],
            save_path=os.path.join(plots_dir, "loso_results.png"),
        )

    # ROC curve
    if cv_results is not None:
        from src.risk.threshold import compute_roc_curve
        fars, drs, _ = compute_roc_curve(baseline_risks, stress_risks)
        plots["roc_curve"] = plot_roc_curve(
            fars, drs, auc_score=cv_results.get("overall_auc"),
            save_path=os.path.join(plots_dir, "roc_curve.png"),
        )

    save_all_plots(plots_dir, plots)

    # Interactive Dashboard
    print("\n[5b] Generating interactive dashboard...")
    dashboard_path = generate_dashboard(
        risk_scores=risk_scores,
        timestamps=timestamps,
        threshold=threshold,
        features_df=wesad_df,
        feature_importance=importance_df,
        cv_results=cv_results,
        output_path=os.path.join(output_dir, "dashboard.html"),
    )

    # ============================================================
    # SUMMARY
    # ============================================================
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Total time: {elapsed:.1f} seconds")

    if cv_results:
        print(f"\n  Model Performance (LOSO CV):")
        print(f"    Accuracy: {cv_results['overall_accuracy']:.4f}")
        print(f"    F1-Score: {cv_results['overall_f1']:.4f}")
        print(f"    ROC-AUC:  {cv_results['overall_auc']:.4f}")

    print(f"\n  Edge Deployment:")
    if edge_report.get("profile"):
        ram = edge_report["profile"].get("estimated_ram_kb",
              edge_report["profile"].get("model_size_kb", "N/A"))
        print(f"    Model RAM:  {ram:.1f} KB" if isinstance(ram, float) else f"    Model RAM:  {ram}")
    if edge_report.get("benchmark"):
        print(f"    Latency:    {edge_report['benchmark']['mean_latency_ms']:.3f} ms")

    print(f"\n  Open dashboard: file://{os.path.abspath(dashboard_path)}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Pilot Readiness Monitoring Framework — End-to-End Pipeline"
    )
    parser.add_argument(
        "--quick-test", action="store_true",
        help="Run quick test with single subject (S2 only)"
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip feature extraction if cached CSVs exist"
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip model training (use cached models)"
    )
    parser.add_argument(
        "--window-sec", type=int, default=60,
        help="ECG window duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--overlap", type=float, default=0.5,
        help="Window overlap fraction (default: 0.5)"
    )
    parser.add_argument(
        "--n-sessions", type=int, default=30,
        help="Number of simulated MATB-II sessions per workload level (default: 30)"
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
