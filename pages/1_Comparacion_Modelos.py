"""
Model Comparison Page - Compare predictions from both models side-by-side.
"""

import streamlit as st
import numpy as np
import pandas as pd

from utils.model_loader import load_model, predict_sample
from utils.data_simulator import generate_traffic_sample, get_all_threat_types
from utils.visualizations import plot_confidence_comparison
from utils.report_generator import generate_comparison_report, save_report_to_file
from utils.sidebar_component import render_sidebar

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Comparación de Modelos",
    layout="wide"
)

# Render shared sidebar for consistency
render_sidebar()

st.title("Comparación de Modelos")
st.markdown("Analiza y compara predicciones de ambos modelos en las mismas muestras")

# =============================================================================
# LOAD BOTH MODELS
# =============================================================================

@st.cache_resource
def load_both_models():
    """Load both synthetic and real models."""
    try:
        synthetic = load_model('synthetic')
        real = load_model('real')
        return synthetic, real, True
    except Exception as e:
        st.error(f"Error cargando modelos: {str(e)}")
        return None, None, False

synthetic_artifacts, real_artifacts, models_loaded = load_both_models()

if not models_loaded:
    st.error("No se pudieron cargar los modelos. Verifica que todos los archivos estén presentes.")
    st.stop()

synthetic_model, synthetic_scaler, synthetic_encoder, synthetic_classes, synthetic_meta = synthetic_artifacts
real_model, real_scaler, real_encoder, real_classes, real_meta = real_artifacts

st.success("Ambos modelos cargados correctamente")

# =============================================================================
# COMPARISON INTERFACE
# =============================================================================

st.markdown("---")

# Comparison settings
col1, col2, col3 = st.columns(3)

with col1:
    num_samples = st.slider(
        "Número de muestras a comparar:",
        min_value=1,
        max_value=50,
        value=10,
        step=1
    )

with col2:
    sample_source = st.selectbox(
        "Fuente de muestras:",
        options=['random', 'specific_threat', 'upload'],
        format_func=lambda x: {
            'random': 'Aleatorias',
            'specific_threat': 'Tipo específico',
            'upload': 'Cargar desde archivo'
        }[x]
    )

with col3:
    if sample_source == 'specific_threat':
        threat_filter = st.selectbox(
            "Tipo de amenaza:",
            options=get_all_threat_types()
        )
    else:
        threat_filter = None

# Generate comparison button
if st.button("Comparar Modelos", type="primary"):
    with st.spinner("Generando muestras y comparando predicciones..."):

        # Generate samples
        samples = []
        true_labels = []

        for _ in range(num_samples):
            if sample_source == 'specific_threat':
                sample, label = generate_traffic_sample(threat_filter)
            else:
                sample, label = generate_traffic_sample()
            samples.append(sample)
            true_labels.append(label)

        # Make predictions with both models
        synthetic_predictions = []
        synthetic_confidences = []
        real_predictions = []
        real_confidences = []

        for sample in samples:
            # Synthetic model
            pred_s, _, conf_s = predict_sample(
                synthetic_model, synthetic_scaler, synthetic_encoder,
                synthetic_classes, sample
            )
            synthetic_predictions.append(pred_s)
            synthetic_confidences.append(conf_s)

            # Real model
            pred_r, _, conf_r = predict_sample(
                real_model, real_scaler, real_encoder,
                real_classes, sample
            )
            real_predictions.append(pred_r)
            real_confidences.append(conf_r)

        # Store results in session state
        st.session_state.comparison_results = {
            'samples': samples,
            'true_labels': true_labels,
            'synthetic_preds': synthetic_predictions,
            'synthetic_confs': synthetic_confidences,
            'real_preds': real_predictions,
            'real_confs': real_confidences
        }

        st.success(f"Comparación completada para {num_samples} muestras")

# =============================================================================
# DISPLAY RESULTS
# =============================================================================

if 'comparison_results' in st.session_state:
    results = st.session_state.comparison_results

    st.markdown("---")
    st.subheader("Resultados de la Comparación")

    # Calculate concordance rate
    matches = sum(
        1 for p_s, p_r in zip(results['synthetic_preds'], results['real_preds'])
        if p_s == p_r
    )
    concordance_rate = (matches / len(results['synthetic_preds'])) * 100

    # Metrics row
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("Tasa de Concordancia", f"{concordance_rate:.1f}%")

    with metric_col2:
        st.metric("Muestras Comparadas", len(results['samples']))

    with metric_col3:
        avg_conf_synthetic = np.mean(results['synthetic_confs'])
        st.metric("Confianza Promedio (Sintético)", f"{avg_conf_synthetic:.1f}%")

    with metric_col4:
        avg_conf_real = np.mean(results['real_confs'])
        st.metric("Confianza Promedio (Real)", f"{avg_conf_real:.1f}%")

    # Comparison table
    st.markdown("---")
    st.subheader("Tabla Comparativa")

    comparison_df = pd.DataFrame({
        'Muestra': range(1, len(results['samples']) + 1),
        'Etiqueta Real': results['true_labels'],
        'Pred. Sintético': results['synthetic_preds'],
        'Conf. Sintético': [f"{c:.1f}%" for c in results['synthetic_confs']],
        'Pred. Real': results['real_preds'],
        'Conf. Real': [f"{c:.1f}%" for c in results['real_confs']],
        'Coinciden': ['SI' if p_s == p_r else 'NO'
                     for p_s, p_r in zip(results['synthetic_preds'], results['real_preds'])]
    })

    # Color rows based on match
    def highlight_match(row):
        if row['Coinciden'] == 'SI':
            return ['background-color: #d4edda'] * len(row)
        else:
            return ['background-color: #f8d7da'] * len(row)

    styled_df = comparison_df.style.apply(highlight_match, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=400)

    # Visualizations
    st.markdown("---")
    st.subheader("Visualizaciones")

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown("**Comparación de Confianzas**")
        if len(results['samples']) <= 20:  # Only for reasonable number of samples
            fig = plot_confidence_comparison(
                results['synthetic_preds'],
                results['synthetic_confs'],
                results['real_preds'],
                results['real_confs'],
                labels=("Sintético", "Real")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Demasiadas muestras para visualizar. Reduce el número a 20 o menos.")

    with viz_col2:
        st.markdown("**Análisis de Discrepancias**")

        # Find disagreements
        disagreements = []
        for i, (p_s, p_r, label) in enumerate(zip(
            results['synthetic_preds'],
            results['real_preds'],
            results['true_labels']
        )):
            if p_s != p_r:
                disagreements.append({
                    'Muestra': i + 1,
                    'Etiqueta': label,
                    'Sintético': p_s,
                    'Real': p_r
                })

        if disagreements:
            st.warning(f"Se encontraron {len(disagreements)} discrepancias")
            disagreements_df = pd.DataFrame(disagreements)
            st.dataframe(disagreements_df, use_container_width=True)
        else:
            st.success("Todos los modelos coinciden en sus predicciones")

    # Export options
    st.markdown("---")
    st.subheader("Exportar Resultados")

    export_col1, export_col2 = st.columns(2)

    with export_col1:
        # CSV export
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            label="Descargar CSV",
            data=csv,
            file_name="comparacion_modelos.csv",
            mime="text/csv"
        )

    with export_col2:
        # PDF export
        if st.button("Generar Reporte PDF"):
            with st.spinner("Generando reporte PDF..."):
                try:
                    # Create results DataFrames for PDF
                    results_synthetic_df = pd.DataFrame({
                        'prediction': results['synthetic_preds'],
                        'confidence': results['synthetic_confs'],
                        'true_label': results['true_labels']
                    })
                    results_real_df = pd.DataFrame({
                        'prediction': results['real_preds'],
                        'confidence': results['real_confs'],
                        'true_label': results['true_labels']
                    })

                    pdf_bytes = generate_comparison_report(
                        results_synthetic_df,
                        results_real_df,
                        concordance_rate
                    )

                    st.download_button(
                        label="Descargar PDF",
                        data=pdf_bytes,
                        file_name="comparacion_modelos.pdf",
                        mime="application/pdf"
                    )
                    st.success("Reporte PDF generado")
                except Exception as e:
                    st.error(f"Error generando PDF: {str(e)}")

else:
    st.info("Configura los parámetros arriba y presiona 'Comparar Modelos' para comenzar")

# Footer
st.markdown("---")
st.caption("""
**Análisis Comparativo** - Evalúa el desempeño de ambos modelos en las mismas muestras
para identificar fortalezas y debilidades de cada enfoque (sintético vs. real).
""")
