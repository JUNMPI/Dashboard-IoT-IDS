"""
Metrics Page - Display model performance metrics and visualizations.
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
from pathlib import Path

from utils.visualizations import (
    plot_metrics_radar,
    create_metrics_table
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Métricas del Modelo",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Métricas y Rendimiento del Modelo")
st.markdown("Visualiza el desempeño y características del modelo entrenado")

# =============================================================================
# CHECK MODEL
# =============================================================================

if not st.session_state.get('model_loaded', False):
    st.error("⚠️ No hay modelo cargado. Por favor selecciona un modelo desde la página principal.")
    st.stop()

model = st.session_state.model
metadata = st.session_state.metadata
class_names = st.session_state.class_names
model_type = st.session_state.selected_model

st.success(f"✅ Modelo activo: **{model_type.upper()}**")

# =============================================================================
# MODEL METADATA
# =============================================================================

st.markdown("---")
st.subheader("ℹ️ Información General del Modelo")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.markdown("**Arquitectura**")
    st.write(f"Tipo: {metadata.get('model_type', 'N/A')}")
    st.write(f"Input Dim: {metadata.get('input_dim', 16)}")
    st.write(f"Latent Dim: {metadata.get('latent_dim', 6)}")
    st.write(f"Num. Clases: {metadata.get('num_classes', 6)}")

with info_col2:
    st.markdown("**Entrenamiento**")
    st.write(f"Dataset: {metadata.get('dataset_type', 'N/A')}")
    st.write(f"Épocas: {metadata.get('epochs_trained', 'N/A')}")
    st.write(f"Batch Size: {metadata.get('batch_size', 64)}")
    st.write(f"Optimizador: {metadata.get('optimizer', 'adam')}")

with info_col3:
    st.markdown("**Pesos de Pérdida**")
    lambda_recon = metadata.get('lambda_peso_reconstruction', 0.3)
    lambda_class = metadata.get('lambda_peso_classification', 0.7)
    st.write(f"Reconstrucción: {lambda_recon}")
    st.write(f"Clasificación: {lambda_class}")
    st.write(f"**Total Loss:** {lambda_recon}×MSE + {lambda_class}×CE")

# =============================================================================
# DATASET STATISTICS
# =============================================================================

st.markdown("---")
st.subheader("📊 Estadísticas del Dataset")

data_col1, data_col2, data_col3 = st.columns(3)

with data_col1:
    st.metric(
        "Muestras de Entrenamiento",
        f"{metadata.get('training_samples', 0):,}"
    )

with data_col2:
    st.metric(
        "Muestras de Validación",
        f"{metadata.get('validation_samples', 0):,}"
    )

with data_col3:
    st.metric(
        "Muestras de Prueba",
        f"{metadata.get('test_samples', 0):,}"
    )

# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

st.markdown("---")
st.subheader("🎯 Métricas de Rendimiento")

# Test set metrics
perf_col1, perf_col2 = st.columns(2)

with perf_col1:
    st.markdown("**Métricas en Conjunto de Prueba**")

    accuracy = metadata.get('test_accuracy', 0) * 100
    loss_total = metadata.get('test_loss_total', 0)
    loss_recon = metadata.get('test_loss_reconstruction', 0)
    loss_class = metadata.get('test_loss_classification', 0)

    # Display metrics as cards
    metric_grid1, metric_grid2 = st.columns(2)

    with metric_grid1:
        st.metric("Accuracy", f"{accuracy:.2f}%")
        st.metric("Loss Total", f"{loss_total:.4f}")

    with metric_grid2:
        st.metric("Loss Reconstrucción", f"{loss_recon:.4f}")
        st.metric("Loss Clasificación", f"{loss_class:.4f}")

with perf_col2:
    st.markdown("**Radar de Métricas**")

    # Create radar chart
    radar_metrics = {
        'Accuracy': accuracy,
        'Reconstruction': (1 - loss_recon) * 100,  # Invert loss to score
        'Classification': (1 - loss_class) * 100,
        'Overall': (1 - loss_total) * 100
    }

    fig_radar = plot_metrics_radar(radar_metrics, title="Rendimiento del Modelo")
    st.plotly_chart(fig_radar, use_container_width=True)

# =============================================================================
# CLASS INFORMATION
# =============================================================================

st.markdown("---")
st.subheader("🏷️ Clases Detectadas")

st.info(f"**Total de clases:** {len(class_names)}")

# Display classes in a nice grid
class_grid_cols = st.columns(3)

for idx, class_name in enumerate(class_names):
    col_idx = idx % 3
    with class_grid_cols[col_idx]:
        # Color coding for threat severity
        if class_name == 'normal':
            emoji = "🟢"
        elif class_name in ['scan']:
            emoji = "🔵"
        elif class_name in ['spoofing']:
            emoji = "🟡"
        elif class_name in ['brute_force', 'mitm']:
            emoji = "🟠"
        else:  # ddos
            emoji = "🔴"

        st.markdown(f"{emoji} **{class_name}**")

# =============================================================================
# TRAINING HISTORY (if available)
# =============================================================================

st.markdown("---")
st.subheader("📉 Historial de Entrenamiento")

# Try to load training history
history_file = None
if model_type == 'synthetic':
    history_path = Path("models/synthetic/training_history.json")
elif model_type == 'real':
    history_path = Path("models/real/training_history_REAL.json")

if history_path.exists():
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)

        # Convert to DataFrame for plotting
        history_df = pd.DataFrame(history)

        # Plot training curves
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Accuracy', 'Total Loss',
                'Reconstruction Loss', 'Classification Loss'
            )
        )

        # Accuracy
        if 'accuracy' in history_df.columns:
            fig.add_trace(
                go.Scatter(x=history_df.index, y=history_df['accuracy'],
                          name='Train Accuracy', line=dict(color='blue')),
                row=1, col=1
            )
        if 'val_accuracy' in history_df.columns:
            fig.add_trace(
                go.Scatter(x=history_df.index, y=history_df['val_accuracy'],
                          name='Val Accuracy', line=dict(color='orange')),
                row=1, col=1
            )

        # Total Loss
        if 'loss' in history_df.columns:
            fig.add_trace(
                go.Scatter(x=history_df.index, y=history_df['loss'],
                          name='Train Loss', line=dict(color='blue')),
                row=1, col=2
            )
        if 'val_loss' in history_df.columns:
            fig.add_trace(
                go.Scatter(x=history_df.index, y=history_df['val_loss'],
                          name='Val Loss', line=dict(color='orange')),
                row=1, col=2
            )

        # Reconstruction Loss
        recon_cols = [col for col in history_df.columns if 'autoencoder' in col.lower() and 'loss' in col.lower()]
        if recon_cols:
            train_recon = recon_cols[0]
            fig.add_trace(
                go.Scatter(x=history_df.index, y=history_df[train_recon],
                          name='Train Recon', line=dict(color='blue')),
                row=2, col=1
            )
            if f'val_{train_recon}' in history_df.columns:
                fig.add_trace(
                    go.Scatter(x=history_df.index, y=history_df[f'val_{train_recon}'],
                              name='Val Recon', line=dict(color='orange')),
                    row=2, col=1
                )

        # Classification Loss
        class_cols = [col for col in history_df.columns if 'fnn' in col.lower() and 'loss' in col.lower()]
        if class_cols:
            train_class = class_cols[0]
            fig.add_trace(
                go.Scatter(x=history_df.index, y=history_df[train_class],
                          name='Train Class', line=dict(color='blue')),
                row=2, col=2
            )
            if f'val_{train_class}' in history_df.columns:
                fig.add_trace(
                    go.Scatter(x=history_df.index, y=history_df[f'val_{train_class}'],
                              name='Val Class', line=dict(color='orange')),
                    row=2, col=2
                )

        fig.update_layout(height=600, showlegend=True)
        fig.update_xaxes(title_text="Época")

        st.plotly_chart(fig, use_container_width=True)

        # Show training summary
        with st.expander("📊 Ver Datos de Entrenamiento"):
            st.dataframe(history_df, use_container_width=True)

    except Exception as e:
        st.warning(f"No se pudo cargar el historial de entrenamiento: {str(e)}")
else:
    st.info("Historial de entrenamiento no disponible para este modelo")

# =============================================================================
# CONFUSION MATRIX (if available)
# =============================================================================

st.markdown("---")
st.subheader("🎯 Matriz de Confusión")

# Try to load confusion matrix image
cm_file = None
if model_type == 'synthetic':
    cm_path = Path("models/synthetic/confusion_matrix_normalized_synthetic.png")
elif model_type == 'real':
    cm_path = Path("models/real/confusion_matrix_normalized_REAL.png")

if cm_path.exists():
    st.image(str(cm_path), caption="Matriz de Confusión Normalizada (Conjunto de Prueba)",
            use_container_width=True)
else:
    st.info("Imagen de matriz de confusión no disponible")

# =============================================================================
# PER-CLASS METRICS (if available)
# =============================================================================

st.markdown("---")
st.subheader("📊 Métricas por Clase")

# Try to load per-class metrics chart
metrics_file = None
if model_type == 'synthetic':
    metrics_path = Path("models/synthetic/metrics_per_class_barchart_synthetic.png")
elif model_type == 'real':
    metrics_path = Path("models/real/metrics_per_class_barchart_REAL.png")

if metrics_path.exists():
    st.image(str(metrics_path), caption="Precision, Recall y F1-Score por Clase",
            use_container_width=True)
else:
    st.info("Gráfico de métricas por clase no disponible")

# =============================================================================
# MODEL COMPARISON
# =============================================================================

st.markdown("---")
st.subheader("⚖️ Comparación entre Modelos")

# Load both metadata
synthetic_meta_path = Path("models/synthetic/model_metadata_synthetic.json")
real_meta_path = Path("models/real/model_metadata_REAL.json")

if synthetic_meta_path.exists() and real_meta_path.exists():
    with open(synthetic_meta_path, 'r') as f:
        synth_meta = json.load(f)
    with open(real_meta_path, 'r') as f:
        real_meta = json.load(f)

    # Create comparison table
    comparison_data = {
        'Métrica': [
            'Accuracy (%)',
            'Loss Total',
            'Loss Reconstrucción',
            'Loss Clasificación',
            'Muestras Entrenamiento',
            'Épocas',
            'Dataset'
        ],
        'Modelo Sintético': [
            f"{synth_meta.get('test_accuracy', 0) * 100:.2f}",
            f"{synth_meta.get('test_loss_total', 0):.4f}",
            f"{synth_meta.get('test_loss_reconstruction', 0):.4f}",
            f"{synth_meta.get('test_loss_classification', 0):.4f}",
            f"{synth_meta.get('training_samples', 0):,}",
            synth_meta.get('epochs_trained', 'N/A'),
            synth_meta.get('dataset_type', 'N/A')
        ],
        'Modelo Real': [
            f"{real_meta.get('test_accuracy', 0) * 100:.2f}",
            f"{real_meta.get('test_loss_total', 0):.4f}",
            f"{real_meta.get('test_loss_reconstruction', 0):.4f}",
            f"{real_meta.get('test_loss_classification', 0):.4f}",
            f"{real_meta.get('training_samples', 0):,}",
            real_meta.get('epochs_trained', 'N/A'),
            real_meta.get('dataset_type', 'N/A')
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.info("""
    **Observaciones:**
    - El modelo sintético tiene mayor accuracy debido a datos más limpios y balanceados
    - El modelo real es más robusto ante variaciones del mundo real
    - El gap de ~13% es esperado en datasets reales vs sintéticos
    """)
else:
    st.warning("No se pudieron cargar los metadatos de ambos modelos para comparación")

# =============================================================================
# MODEL ARCHITECTURE SUMMARY
# =============================================================================

st.markdown("---")
st.subheader("🏗️ Resumen de Arquitectura")

with st.expander("Ver detalles de la arquitectura"):
    arch_col1, arch_col2 = st.columns(2)

    with arch_col1:
        st.markdown("""
        **Encoder (Autoencoder):**
        ```
        Input(16)
          → Dense(8, relu)
          → Dense(6, relu) [LATENT]
        ```

        **Decoder (Autoencoder):**
        ```
        Latent(6)
          → Dense(8, relu)
          → Dense(16, linear) [RECONSTRUCTION]
        ```
        """)

    with arch_col2:
        st.markdown("""
        **Classifier (FNN):**
        ```
        Latent(6)
          → Dense(16, relu)
          → Dropout(0.3)
          → Dense(6, softmax) [CLASSIFICATION]
        ```

        **Loss Function:**
        ```
        L = 0.3 × MSE(recon) + 0.7 × CE(class)
        ```
        """)

# Footer
st.markdown("---")
st.caption("""
**Métricas del Modelo** - Visualización detallada del rendimiento, arquitectura
y características del modelo Autoencoder-FNN multi-tarea entrenado.
""")
