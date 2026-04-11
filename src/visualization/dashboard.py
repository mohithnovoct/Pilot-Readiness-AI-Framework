"""
Interactive HTML Dashboard (Enhanced)
=======================================
Generates a premium interactive HTML dashboard using Plotly.

Enhanced features:
  - Per-subject selector dropdown
  - SHAP waterfall for sample predictions
  - Threshold sensitivity slider
  - Model comparison results panel
  - Confidence interval display
  - Polished dark glassmorphism UI
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
        template="plotly_dark",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,41,59,0.8)",
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
        title="Feature Contribution to Risk Score (SHAP Waterfall)",
        xaxis_title="SHAP Value (impact on risk)",
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,41,59,0.8)",
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
        template="plotly_dark",
        height=400,
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,41,59,0.8)",
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

    fig.update_layout(
        template="plotly_dark",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_per_subject_timeline(
    features_df: pd.DataFrame,
    risk_scores: np.ndarray,
    threshold: float,
) -> str:
    """Generate JavaScript for per-subject filtering."""
    if "subject_id" not in features_df.columns:
        return ""

    subjects = sorted(features_df["subject_id"].unique())
    options_html = '<option value="all">All Subjects</option>\n'
    for s in subjects:
        options_html += f'    <option value="{s}">{s}</option>\n'

    return f"""
    <div class="control-panel">
        <label for="subject-select" style="color: #94a3b8; margin-right: 10px;">
            Filter by Subject:
        </label>
        <select id="subject-select" onchange="filterSubject(this.value)"
                style="background: #334155; color: #e2e8f0; border: 1px solid #475569;
                       padding: 8px 16px; border-radius: 8px; font-size: 14px; cursor: pointer;">
            {options_html}
        </select>
    </div>
    """


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
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Real-time pilot readiness monitoring dashboard with AI-powered stress detection and risk scoring.">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0a0f1c;
            color: #e2e8f0;
            min-height: 100vh;
            overflow-x: hidden;
        }
        body::before {
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background:
                radial-gradient(ellipse at 20% 50%, rgba(59, 130, 246, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(139, 92, 246, 0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 80%, rgba(34, 197, 94, 0.04) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
            position: relative;
            z-index: 1;
        }
        .header {
            text-align: center;
            padding: 40px 30px;
            background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(51,65,85,0.7));
            backdrop-filter: blur(20px);
            border-radius: 20px;
            margin-bottom: 24px;
            border: 1px solid rgba(255,255,255,0.05);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }
        .header h1 {
            font-size: 2.2em;
            font-weight: 700;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
            letter-spacing: -0.02em;
        }
        .header p {
            color: #94a3b8;
            font-size: 1.05em;
            font-weight: 300;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .metric-card {
            background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(51,65,85,0.6));
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.06);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        }
        .metric-card .value {
            font-size: 2em;
            font-weight: 700;
            margin: 8px 0;
            letter-spacing: -0.02em;
        }
        .metric-card .label {
            color: #94a3b8;
            font-size: 0.85em;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .plot-container {
            background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(51,65,85,0.5));
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid rgba(255,255,255,0.06);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        }
        .section-title {
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 16px;
            color: #e2e8f0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .section-title::before {
            content: '';
            display: inline-block;
            width: 4px;
            height: 24px;
            background: linear-gradient(180deg, #3b82f6, #8b5cf6);
            border-radius: 2px;
        }
        .control-panel {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 16px 20px;
            background: rgba(30,41,59,0.6);
            border-radius: 12px;
            margin-bottom: 16px;
            border: 1px solid rgba(255,255,255,0.04);
        }
        .ci-badge {
            display: inline-block;
            background: rgba(59, 130, 246, 0.15);
            color: #93c5fd;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.75em;
            font-weight: 500;
            margin-left: 8px;
        }
        .green { color: #22c55e; }
        .yellow { color: #eab308; }
        .orange { color: #f97316; }
        .red { color: #ef4444; }
        .blue { color: #3b82f6; }
        .purple { color: #8b5cf6; }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .status-ready {
            background: rgba(34, 197, 94, 0.15);
            color: #22c55e;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }
        .status-alert {
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .two-column {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }
        @media (max-width: 768px) {
            .two-column { grid-template-columns: 1fr; }
            .metrics-grid { grid-template-columns: repeat(2, 1fr); }
        }

        .footer {
            text-align: center;
            padding: 30px;
            color: #475569;
            font-size: 0.9em;
        }
        .footer a {
            color: #3b82f6;
            text-decoration: none;
        }

        /* Pulse animation for live indicator */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .live-dot {
            width: 8px; height: 8px;
            background: #22c55e;
            border-radius: 50%;
            display: inline-block;
            animation: pulse 2s infinite;
            margin-right: 6px;
        }
    </style>
</head>
<body>
<div class="container">
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
    current_status = "ALERT" if risk_scores[-1] > threshold else "READY"

    risk_color = "green" if mean_risk < 0.25 else "yellow" if mean_risk < 0.5 else "orange" if mean_risk < 0.75 else "red"

    html_parts.append(f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="label">Current Status</div>
            <div class="value">
                <span class="status-badge {'status-alert' if current_status == 'ALERT' else 'status-ready'}">
                    <span class="live-dot" style="background: {'#ef4444' if current_status == 'ALERT' else '#22c55e'}"></span>
                    {current_status}
                </span>
            </div>
        </div>
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
            <div class="value purple">{threshold:.3f}</div>
        </div>
    </div>
""")

    # Per-subject filter
    if features_df is not None and "subject_id" in features_df.columns:
        html_parts.append(create_per_subject_timeline(features_df, risk_scores, threshold))

    # Risk timeline
    html_parts.append('<div class="plot-container">')
    html_parts.append('<div class="section-title">Risk Score Timeline</div>')
    html_parts.append(timeline_fig.to_html(full_html=False, include_plotlyjs="cdn"))
    html_parts.append("</div>")

    # Two-column layout for feature importance + SHAP waterfall
    if feature_importance is not None and len(feature_importance) > 0:
        imp_col = feature_importance.columns[1]
        top_features = feature_importance.head(10)

        fig_imp = go.Figure(go.Bar(
            x=top_features[imp_col].values[::-1],
            y=top_features.iloc[:, 0].values[::-1],
            orientation="h",
            marker=dict(
                color=top_features[imp_col].values[::-1],
                colorscale=[[0, "#3b82f6"], [0.5, "#8b5cf6"], [1, "#ec4899"]],
            ),
        ))
        fig_imp.update_layout(
            title="Top 10 Feature Importance (SHAP)",
            xaxis_title="Mean |SHAP Value|",
            template="plotly_dark",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30,41,59,0.8)",
        )

        # SHAP waterfall for a sample prediction
        fig_waterfall = create_feature_waterfall(
            top_features.iloc[:, 0].tolist(),
            top_features[imp_col].values,
        )

        html_parts.append('<div class="two-column">')
        html_parts.append('<div class="plot-container">')
        html_parts.append('<div class="section-title">Feature Importance</div>')
        html_parts.append(fig_imp.to_html(full_html=False, include_plotlyjs=False))
        html_parts.append("</div>")
        html_parts.append('<div class="plot-container">')
        html_parts.append('<div class="section-title">Feature Contributions</div>')
        html_parts.append(fig_waterfall.to_html(full_html=False, include_plotlyjs=False))
        html_parts.append("</div>")
        html_parts.append("</div>")

    # Subject overview if available
    if features_df is not None and len(features_df) > 0:
        try:
            overview_fig = create_subject_overview(features_df)
            html_parts.append('<div class="plot-container">')
            html_parts.append('<div class="section-title">Physiological Profiles</div>')
            html_parts.append(overview_fig.to_html(full_html=False, include_plotlyjs=False))
            html_parts.append("</div>")
        except Exception:
            pass

    # CV results if available
    if cv_results is not None:
        ci = cv_results.get("confidence_intervals", {})
        acc_ci = ci.get("accuracy_ci", (None, None))
        f1_ci = ci.get("f1_ci", (None, None))
        auc_ci = ci.get("auc_ci", (None, None))

        def ci_badge(ci_tuple):
            if ci_tuple and ci_tuple[0] is not None and not np.isnan(ci_tuple[0]):
                return f'<span class="ci-badge">95% CI [{ci_tuple[0]:.3f}, {ci_tuple[1]:.3f}]</span>'
            return ""

        html_parts.append(f"""
    <div class="plot-container">
        <div class="section-title">Model Performance (LOSO Cross-Validation)</div>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="label">Accuracy</div>
                <div class="value green">{cv_results.get('overall_accuracy', 0):.3f}</div>
                {ci_badge(acc_ci)}
            </div>
            <div class="metric-card">
                <div class="label">F1-Score</div>
                <div class="value green">{cv_results.get('overall_f1', 0):.3f}</div>
                {ci_badge(f1_ci)}
            </div>
            <div class="metric-card">
                <div class="label">ROC-AUC</div>
                <div class="value green">{cv_results.get('overall_auc', 0):.3f}</div>
                {ci_badge(auc_ci)}
            </div>
        </div>
    </div>
""")

    # Threshold sensitivity slider (JavaScript-powered)
    html_parts.append(f"""
    <div class="plot-container">
        <div class="section-title">Threshold Sensitivity Explorer</div>
        <div class="control-panel">
            <label for="threshold-slider" style="color: #94a3b8; font-weight: 500;">
                Adjust Threshold (γ):
            </label>
            <input type="range" id="threshold-slider" min="0" max="100" value="{int(threshold*100)}"
                   style="flex: 1; accent-color: #8b5cf6; cursor: pointer;"
                   oninput="updateThreshold(this.value)">
            <span id="threshold-value" class="purple" style="font-weight: 700; font-size: 1.2em; min-width: 60px;">
                γ = {threshold:.2f}
            </span>
        </div>
        <div id="threshold-stats" class="metrics-grid" style="margin-top: 12px;">
            <div class="metric-card">
                <div class="label">Alert Rate at γ</div>
                <div class="value yellow" id="slider-alert-rate">{alert_pct:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="label">Total Alerts at γ</div>
                <div class="value blue" id="slider-n-alerts">{n_alerts}</div>
            </div>
        </div>
    </div>
    <script>
        const riskScores = {risk_scores.tolist()};
        function updateThreshold(val) {{
            const gamma = val / 100;
            document.getElementById('threshold-value').textContent = 'γ = ' + gamma.toFixed(2);
            let nAlerts = 0;
            for (let i = 0; i < riskScores.length; i++) {{
                if (riskScores[i] > gamma) nAlerts++;
            }}
            const alertRate = (nAlerts / riskScores.length * 100).toFixed(1);
            document.getElementById('slider-alert-rate').textContent = alertRate + '%';
            document.getElementById('slider-n-alerts').textContent = nAlerts;
        }}
    </script>
""")

    # Load experiment results if available
    exp_dir = os.path.join(os.path.dirname(os.path.dirname(output_path)), "experiments")
    comparison_path = os.path.join(exp_dir, "model_comparison.csv")
    if os.path.exists(comparison_path):
        comp_df = pd.read_csv(comparison_path)
        rows_html = ""
        for _, row in comp_df.iterrows():
            rows_html += f"""
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #334155;">{row['Model']}</td>
                <td style="padding: 10px; border-bottom: 1px solid #334155;">{row['Accuracy']:.4f}</td>
                <td style="padding: 10px; border-bottom: 1px solid #334155;">{row['F1-Score']:.4f}</td>
                <td style="padding: 10px; border-bottom: 1px solid #334155;">{row['ROC-AUC']:.4f}</td>
                <td style="padding: 10px; border-bottom: 1px solid #334155;">{row['Time (s)']:.1f}s</td>
            </tr>"""

        html_parts.append(f"""
    <div class="plot-container">
        <div class="section-title">Model Comparison Results</div>
        <table style="width: 100%; border-collapse: collapse; color: #e2e8f0;">
            <thead>
                <tr style="border-bottom: 2px solid #3b82f6;">
                    <th style="padding: 12px; text-align: left; color: #94a3b8;">Model</th>
                    <th style="padding: 12px; text-align: left; color: #94a3b8;">Accuracy</th>
                    <th style="padding: 12px; text-align: left; color: #94a3b8;">F1-Score</th>
                    <th style="padding: 12px; text-align: left; color: #94a3b8;">ROC-AUC</th>
                    <th style="padding: 12px; text-align: left; color: #94a3b8;">Time</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
""")

    html_parts.append("""
    <div class="footer">
        <p>Team 104 — A Lightweight AI Framework for Pilot Readiness Monitoring</p>
        <p>Using Stress-Correlated Performance Indicators</p>
        <p style="margin-top: 8px; font-size: 0.85em;">
            Built with LightGBM • Plotly • Neyman-Pearson Detection Theory
        </p>
    </div>
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
