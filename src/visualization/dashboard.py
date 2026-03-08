"""
Interactive HTML Dashboard
===========================
Generates an interactive HTML dashboard using Plotly for:
  - Real-time risk score timeline (replay validation)
  - Feature contribution waterfall
  - Sensitivity slider for threshold adjustment
  - Per-subject physiological state
"""

import numpy as np
import pandas as pd
import os
from typing import Optional, Dict, List

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def create_risk_timeline_plot(
    timestamps: np.ndarray,
    risk_scores: np.ndarray,
    threshold: float = 0.5,
    labels: Optional[np.ndarray] = None,
) -> "go.Figure":
    """Create interactive risk score timeline with Plotly."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for the dashboard.")

    fig = go.Figure()

    # Risk score line
    fig.add_trace(go.Scatter(
        x=timestamps, y=risk_scores,
        mode="lines+markers",
        name="Risk Score",
        line=dict(color="#3b82f6", width=2),
        marker=dict(
            size=5,
            color=risk_scores,
            colorscale=[[0, "#22c55e"], [0.25, "#22c55e"],
                       [0.25, "#eab308"], [0.5, "#eab308"],
                       [0.5, "#f97316"], [0.75, "#f97316"],
                       [0.75, "#ef4444"], [1.0, "#ef4444"]],
            cmin=0, cmax=1,
        ),
        hovertemplate="Time: %{x}<br>Risk: %{y:.3f}<extra></extra>",
    ))

    # Threshold line
    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Threshold (γ = {threshold:.2f})")

    # Risk zone shading
    fig.add_hrect(y0=0, y1=0.25, fillcolor="#22c55e", opacity=0.1,
                  annotation_text="LOW", annotation_position="top left")
    fig.add_hrect(y0=0.25, y1=0.5, fillcolor="#eab308", opacity=0.1,
                  annotation_text="MODERATE", annotation_position="top left")
    fig.add_hrect(y0=0.5, y1=0.75, fillcolor="#f97316", opacity=0.1,
                  annotation_text="ELEVATED", annotation_position="top left")
    fig.add_hrect(y0=0.75, y1=1.0, fillcolor="#ef4444", opacity=0.1,
                  annotation_text="HIGH", annotation_position="top left")

    fig.update_layout(
        title="Pilot Readiness Risk Score – Real-Time Timeline",
        xaxis_title="Time (window index)",
        yaxis_title="Readiness Risk Score",
        yaxis_range=[-0.05, 1.05],
        template="plotly_white",
        height=500,
    )

    return fig


def create_feature_waterfall(
    feature_names: List[str],
    feature_values: np.ndarray,
    base_value: float = 0.5,
) -> "go.Figure":
    """Create a SHAP-style waterfall chart for feature contributions."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required.")

    # Sort by absolute value
    sort_idx = np.argsort(np.abs(feature_values))[::-1]
    names = [feature_names[i] for i in sort_idx[:10]]
    values = [feature_values[i] for i in sort_idx[:10]]

    colors = ["#ef4444" if v > 0 else "#22c55e" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title="Feature Contribution to Risk Score",
        xaxis_title="SHAP Value (impact on risk)",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        height=400,
    )

    return fig


def create_threshold_analysis_plot(
    alphas: np.ndarray,
    thresholds: np.ndarray,
    detection_rates: np.ndarray,
    false_alarm_rates: np.ndarray,
) -> "go.Figure":
    """Create interactive threshold sensitivity analysis plot."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required.")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Threshold vs. False Alarm Rate",
                       "Detection Rate vs. False Alarm Rate (ROC)"),
    )

    # Threshold vs alpha
    fig.add_trace(go.Scatter(
        x=alphas * 100, y=thresholds,
        mode="lines+markers",
        name="Threshold (γ)",
        line=dict(color="#3b82f6", width=2),
    ), row=1, col=1)

    # ROC curve
    fig.add_trace(go.Scatter(
        x=false_alarm_rates * 100, y=detection_rates * 100,
        mode="lines+markers",
        name="ROC",
        line=dict(color="#8b5cf6", width=2),
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100],
        mode="lines",
        name="Random",
        line=dict(color="gray", dash="dash"),
    ), row=1, col=2)

    fig.update_xaxes(title_text="False Alarm Rate α (%)", row=1, col=1)
    fig.update_yaxes(title_text="Threshold γ", row=1, col=1)
    fig.update_xaxes(title_text="False Alarm Rate (%)", row=1, col=2)
    fig.update_yaxes(title_text="Detection Rate (%)", row=1, col=2)

    fig.update_layout(
        title="Neyman-Pearson Threshold Sensitivity Analysis",
        template="plotly_white",
        height=400,
        showlegend=True,
    )

    return fig


def create_subject_overview(features_df: pd.DataFrame) -> "go.Figure":
    """Create per-subject feature overview with parallel coordinates."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required.")

    feature_cols = ["SDNN", "RMSSD", "pNN50", "LF_HF_ratio", "MeanHR"]
    available = [c for c in feature_cols if c in features_df.columns]

    if not available or "label" not in features_df.columns:
        return go.Figure()

    fig = px.parallel_coordinates(
        features_df,
        dimensions=available,
        color="label",
        color_continuous_scale=[[0, "#22c55e"], [1, "#ef4444"]],
        title="Physiological Feature Profiles: Baseline vs. Stress",
    )

    fig.update_layout(template="plotly_white", height=500)
    return fig


def generate_dashboard(
    risk_scores: np.ndarray,
    timestamps: np.ndarray,
    threshold: float,
    features_df: Optional[pd.DataFrame] = None,
    feature_importance: Optional[pd.DataFrame] = None,
    cv_results: Optional[Dict] = None,
    output_path: str = "output/dashboard.html",
) -> str:
    """
    Generate a comprehensive interactive HTML dashboard.

    Parameters
    ----------
    risk_scores : np.ndarray
        Continuous risk scores for timeline.
    timestamps : np.ndarray
        Time indices for risk scores.
    threshold : float
        Current alert threshold.
    features_df : pd.DataFrame, optional
        Feature data for overview plots.
    feature_importance : pd.DataFrame, optional
        SHAP feature importance.
    cv_results : dict, optional
        Cross-validation results.
    output_path : str
        Path to save the HTML dashboard.

    Returns
    -------
    str
        Path to the saved dashboard.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for the dashboard. "
                         "Install with: pip install plotly")

    # Build individual plots
    timeline_fig = create_risk_timeline_plot(timestamps, risk_scores, threshold)

    # Combine into single HTML
    html_parts = ["""
<!DOCTYPE html>
<html>
<head>
    <title>Pilot Readiness Monitoring Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            margin: 0;
            padding: 20px;
        }
        .header {
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, #1e293b, #334155);
            border-radius: 16px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }
        .header h1 {
            font-size: 2em;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        .header p {
            color: #94a3b8;
            font-size: 1.1em;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: linear-gradient(135deg, #1e293b, #334155);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }
        .metric-card .value {
            font-size: 2em;
            font-weight: bold;
            margin: 8px 0;
        }
        .metric-card .label {
            color: #94a3b8;
            font-size: 0.9em;
        }
        .plot-container {
            background: #1e293b;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }
        .green { color: #22c55e; }
        .yellow { color: #eab308; }
        .orange { color: #f97316; }
        .red { color: #ef4444; }
        .blue { color: #3b82f6; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛫 Pilot Readiness Monitoring Dashboard</h1>
        <p>A Lightweight AI Framework for Stress-Correlated Performance Indicators</p>
    </div>
"""]

    # Metrics cards
    mean_risk = np.mean(risk_scores)
    max_risk = np.max(risk_scores)
    alert_pct = np.mean(risk_scores > threshold) * 100
    n_alerts = np.sum(risk_scores > threshold)

    risk_color = "green" if mean_risk < 0.25 else "yellow" if mean_risk < 0.5 else "orange" if mean_risk < 0.75 else "red"

    html_parts.append(f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="label">Mean Risk Score</div>
            <div class="value {risk_color}">{mean_risk:.3f}</div>
        </div>
        <div class="metric-card">
            <div class="label">Peak Risk Score</div>
            <div class="value {'red' if max_risk > 0.7 else 'orange'}">{max_risk:.3f}</div>
        </div>
        <div class="metric-card">
            <div class="label">Alert Rate</div>
            <div class="value {'red' if alert_pct > 20 else 'yellow'}">{alert_pct:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="label">Total Alerts</div>
            <div class="value blue">{n_alerts}</div>
        </div>
        <div class="metric-card">
            <div class="label">Threshold (γ)</div>
            <div class="value blue">{threshold:.3f}</div>
        </div>
        <div class="metric-card">
            <div class="label">N Observations</div>
            <div class="value blue">{len(risk_scores)}</div>
        </div>
    </div>
""")

    # Risk timeline
    html_parts.append('<div class="plot-container">')
    html_parts.append(timeline_fig.to_html(full_html=False, include_plotlyjs="cdn"))
    html_parts.append("</div>")

    # Feature importance if available
    if feature_importance is not None and len(feature_importance) > 0:
        imp_col = feature_importance.columns[1]
        top_features = feature_importance.head(10)
        fig_imp = go.Figure(go.Bar(
            x=top_features[imp_col].values[::-1],
            y=top_features.iloc[:, 0].values[::-1],
            orientation="h",
            marker_color="#8b5cf6",
        ))
        fig_imp.update_layout(
            title="Top 10 Feature Importance (SHAP)",
            xaxis_title="Mean |SHAP Value|",
            template="plotly_white",
            height=400,
        )
        html_parts.append('<div class="plot-container">')
        html_parts.append(fig_imp.to_html(full_html=False, include_plotlyjs=False))
        html_parts.append("</div>")

    # Subject overview if available
    if features_df is not None and len(features_df) > 0:
        try:
            overview_fig = create_subject_overview(features_df)
            html_parts.append('<div class="plot-container">')
            html_parts.append(overview_fig.to_html(full_html=False, include_plotlyjs=False))
            html_parts.append("</div>")
        except Exception:
            pass

    # CV results if available
    if cv_results is not None:
        html_parts.append(f"""
    <div class="plot-container">
        <h2 style="color: #e2e8f0;">Model Performance (LOSO Cross-Validation)</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="label">Accuracy</div>
                <div class="value green">{cv_results.get('overall_accuracy', 0):.3f}</div>
            </div>
            <div class="metric-card">
                <div class="label">F1-Score</div>
                <div class="value green">{cv_results.get('overall_f1', 0):.3f}</div>
            </div>
            <div class="metric-card">
                <div class="label">ROC-AUC</div>
                <div class="value green">{cv_results.get('overall_auc', 0):.3f}</div>
            </div>
        </div>
    </div>
""")

    html_parts.append("""
    <div style="text-align: center; padding: 20px; color: #64748b;">
        <p>Team 104 — A Lightweight AI Framework for Pilot Readiness Monitoring</p>
        <p>Using Stress-Correlated Performance Indicators</p>
    </div>
</body>
</html>
""")

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(html_parts))

    print(f"Dashboard saved to {output_path}")
    return output_path
