"""
Visualization utilities for IoT-IDS dashboard.

This module provides reusable plotting functions for displaying
detection results, metrics, and analysis.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# =============================================================================
# CONFIGURATION
# =============================================================================

# Color schemes
THREAT_COLORS = {
    "Benign": "#2ecc71",      # Green
    "DDoS": "#e74c3c",        # Red
    "DoS": "#e67e22",         # Orange
    "Brute_Force": "#c0392b", # Dark red
    "Spoofing": "#f39c12",    # Yellow-orange
    "MITM": "#d35400",        # Dark orange
    "Scan": "#3498db",        # Blue
    "Recon": "#9b59b6",       # Purple
}

SEVERITY_COLORS = {
    "normal": "#2ecc71",
    "low": "#3498db",
    "medium": "#f39c12",
    "high": "#e67e22",
    "critical": "#e74c3c",
}

# =============================================================================
# CONFUSION MATRIX
# =============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    normalize: bool = False,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Create confusion matrix heatmap using seaborn.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize values
        title: Plot title

    Returns:
        Matplotlib figure

    Example:
        >>> fig = plot_confusion_matrix(y_true, y_pred, class_names)
        >>> st.pyplot(fig)
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    return fig

# =============================================================================
# TEMPORAL CHARTS
# =============================================================================

def plot_temporal_detections(
    timestamps: List[float],
    predictions: List[str],
    confidences: List[float],
    window_size: int = 60
) -> go.Figure:
    """
    Create temporal line chart of detections.

    Args:
        timestamps: List of timestamps (seconds)
        predictions: List of prediction labels
        confidences: List of confidence values (0-100)
        window_size: Number of recent points to show

    Returns:
        Plotly figure

    Example:
        >>> fig = plot_temporal_detections(times, preds, confs)
        >>> st.plotly_chart(fig)
    """
    # Limit to window
    if len(timestamps) > window_size:
        timestamps = timestamps[-window_size:]
        predictions = predictions[-window_size:]
        confidences = confidences[-window_size:]

    # Create dataframe
    df = pd.DataFrame({
        'Time': timestamps,
        'Prediction': predictions,
        'Confidence': confidences
    })

    # Create figure
    fig = go.Figure()

    # Add trace for each threat type
    for threat_type in df['Prediction'].unique():
        mask = df['Prediction'] == threat_type
        subset = df[mask]

        fig.add_trace(go.Scatter(
            x=subset['Time'],
            y=subset['Confidence'],
            mode='markers+lines',
            name=threat_type,
            marker=dict(
                size=8,
                color=THREAT_COLORS.get(threat_type, '#95a5a6')
            ),
            line=dict(width=2),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'Time: %{x:.1f}s<br>' +
                          'Confidence: %{y:.1f}%<br>' +
                          '<extra></extra>'
        ))

    fig.update_layout(
        title='Real-Time Threat Detection',
        xaxis_title='Time (seconds)',
        yaxis_title='Confidence (%)',
        hovermode='closest',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

# =============================================================================
# DISTRIBUTION CHARTS
# =============================================================================

def plot_class_distribution(
    predictions: List[str],
    title: str = "Threat Distribution"
) -> go.Figure:
    """
    Create pie chart of threat type distribution.

    Args:
        predictions: List of prediction labels
        title: Chart title

    Returns:
        Plotly figure
    """
    # Count occurrences
    counts = pd.Series(predictions).value_counts()

    # Create colors list
    colors = [THREAT_COLORS.get(name, '#95a5a6') for name in counts.index]

    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        marker=dict(colors=colors),
        hole=0.3,
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>' +
                      'Count: %{value}<br>' +
                      'Percentage: %{percent}<br>' +
                      '<extra></extra>'
    )])

    fig.update_layout(
        title=title,
        height=400,
        showlegend=True
    )

    return fig

def plot_top_threats(
    predictions: List[str],
    top_n: int = 5,
    title: str = "Top Detected Threats"
) -> go.Figure:
    """
    Create bar chart of top N threats.

    Args:
        predictions: List of prediction labels
        top_n: Number of top threats to show
        title: Chart title

    Returns:
        Plotly figure
    """
    # Count and get top N
    counts = pd.Series(predictions).value_counts().head(top_n)

    # Create colors
    colors = [THREAT_COLORS.get(name, '#95a5a6') for name in counts.index]

    fig = go.Figure(data=[go.Bar(
        x=counts.index,
        y=counts.values,
        marker=dict(color=colors),
        text=counts.values,
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>' +
                      'Detections: %{y}<br>' +
                      '<extra></extra>'
    )])

    fig.update_layout(
        title=title,
        xaxis_title='Threat Type',
        yaxis_title='Count',
        height=400,
        showlegend=False
    )

    return fig

# =============================================================================
# COMPARISON CHARTS
# =============================================================================

def plot_confidence_comparison(
    predictions_a: List[str],
    confidences_a: List[float],
    predictions_b: List[str],
    confidences_b: List[float],
    labels: tuple = ("Model A", "Model B")
) -> go.Figure:
    """
    Compare confidence levels between two models.

    Args:
        predictions_a: Predictions from model A
        confidences_a: Confidences from model A
        predictions_b: Predictions from model B
        confidences_b: Confidences from model B
        labels: Tuple of model names

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Model A
    fig.add_trace(go.Bar(
        name=labels[0],
        x=['Sample ' + str(i) for i in range(len(predictions_a))],
        y=confidences_a,
        text=[f"{p}<br>{c:.1f}%" for p, c in zip(predictions_a, confidences_a)],
        textposition='outside',
        marker_color='#3498db'
    ))

    # Model B
    fig.add_trace(go.Bar(
        name=labels[1],
        x=['Sample ' + str(i) for i in range(len(predictions_b))],
        y=confidences_b,
        text=[f"{p}<br>{c:.1f}%" for p, c in zip(predictions_b, confidences_b)],
        textposition='outside',
        marker_color='#e74c3c'
    ))

    fig.update_layout(
        title='Model Confidence Comparison',
        xaxis_title='Sample',
        yaxis_title='Confidence (%)',
        barmode='group',
        height=500,
        showlegend=True
    )

    return fig

# =============================================================================
# METRICS VISUALIZATION
# =============================================================================

def plot_metrics_radar(
    metrics: Dict[str, float],
    title: str = "Model Performance Metrics"
) -> go.Figure:
    """
    Create radar chart for multiple metrics.

    Args:
        metrics: Dictionary of metric_name: value (0-1 or 0-100)
        title: Chart title

    Returns:
        Plotly figure

    Example:
        >>> metrics = {'Accuracy': 97, 'Precision': 96, 'Recall': 95}
        >>> fig = plot_metrics_radar(metrics)
    """
    categories = list(metrics.keys())
    values = list(metrics.values())

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Performance',
        marker=dict(color='#3498db'),
        line=dict(color='#2980b9', width=2)
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title=title,
        height=450,
        showlegend=False
    )

    return fig

def create_risk_gauge(
    risk_score: float,
    title: str = "Current Risk Level"
) -> go.Figure:
    """
    Create gauge chart for risk level.

    Args:
        risk_score: Risk score (0-100)
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        title={'text': title},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "#2ecc71"},
                {'range': [25, 50], 'color': "#f39c12"},
                {'range': [50, 75], 'color': "#e67e22"},
                {'range': [75, 100], 'color': "#e74c3c"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))

    fig.update_layout(height=300)

    return fig

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_metrics_table(metrics: Dict[str, Any]) -> go.Figure:
    """
    Create formatted table for displaying metrics.

    Args:
        metrics: Dictionary of metric names and values

    Returns:
        Plotly table figure
    """
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Metric</b>', '<b>Value</b>'],
            fill_color='#3498db',
            font=dict(color='white', size=12),
            align='left'
        ),
        cells=dict(
            values=[
                list(metrics.keys()),
                [f"{v:.2f}" if isinstance(v, float) else str(v) for v in metrics.values()]
            ],
            fill_color='#ecf0f1',
            align='left',
            height=30
        )
    )])

    fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))

    return fig

def format_metric_card(
    value: float,
    label: str,
    format_str: str = "{:.2f}%"
) -> str:
    """
    Format metric for display in Streamlit metric card.

    Args:
        value: Metric value
        label: Metric label
        format_str: Python format string

    Returns:
        Formatted string
    """
    return format_str.format(value)
