"""
File Analysis Page - Batch analysis of CSV files with network traffic data.
"""

import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO

from utils.model_loader import predict_batch
from utils.visualizations import (
    plot_class_distribution,
    plot_top_threats,
    create_risk_gauge
)
from utils.report_generator import generate_analysis_report

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Análisis de Archivo",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Análisis de Archivo CSV")
st.markdown("Carga un archivo CSV con datos de tráfico de red y obtén análisis detallado")

# =============================================================================
# CHECK MODEL
# =============================================================================

if not st.session_state.get('model_loaded', False):
    st.error("⚠️ No hay modelo cargado. Por favor selecciona un modelo desde la página principal.")
    st.stop()

model = st.session_state.model
scaler = st.session_state.scaler
label_encoder = st.session_state.label_encoder
class_names = st.session_state.class_names
metadata = st.session_state.metadata

st.success(f"✅ Modelo activo: **{st.session_state.selected_model.upper()}**")

# =============================================================================
# FILE UPLOAD
# =============================================================================

st.markdown("---")
st.subheader("📁 Cargar Archivo")

st.info("""
**Formato requerido del archivo CSV:**
- El archivo debe contener exactamente 16 columnas con los componentes PCA (PC1-PC16)
- Los nombres de columnas pueden ser: PC1, PC2, ..., PC16 o feature_1, feature_2, etc.
- Opcionalmente puede incluir una columna 'label' con la etiqueta verdadera
- Sin valores faltantes o no numéricos en las columnas de features
""")

uploaded_file = st.file_uploader(
    "Selecciona un archivo CSV:",
    type=['csv'],
    help="Archivo CSV con 16 componentes PCA"
)

# =============================================================================
# FILE PROCESSING
# =============================================================================

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        st.success(f"✅ Archivo cargado: {uploaded_file.name}")
        st.write(f"**Dimensiones:** {df.shape[0]} filas x {df.shape[1]} columnas")

        # Display preview
        with st.expander("👁️ Vista previa de los datos"):
            st.dataframe(df.head(20), use_container_width=True)

        st.markdown("---")

        # Identify feature columns
        # Try common column naming patterns
        feature_cols = []

        # Pattern 1: PC1, PC2, ..., PC16
        if all(f'PC{i}' in df.columns for i in range(1, 17)):
            feature_cols = [f'PC{i}' for i in range(1, 17)]

        # Pattern 2: feature_1, feature_2, ..., feature_16
        elif all(f'feature_{i}' in df.columns for i in range(1, 17)):
            feature_cols = [f'feature_{i}' for i in range(1, 17)]

        # Pattern 3: First 16 numeric columns
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 16:
                feature_cols = numeric_cols[:16]
            else:
                st.error(f"❌ No se encontraron suficientes columnas numéricas. "
                        f"Se necesitan 16, se encontraron {len(numeric_cols)}")
                st.stop()

        st.info(f"**Columnas de features detectadas:** {', '.join(feature_cols[:5])}... (16 total)")

        # Check for label column
        has_labels = 'label' in df.columns or 'Label' in df.columns
        if has_labels:
            label_col = 'label' if 'label' in df.columns else 'Label'
            st.info(f"✅ Se detectó columna de etiquetas: '{label_col}'")
        else:
            st.warning("⚠️ No se detectó columna de etiquetas. Solo se mostrarán predicciones.")
            label_col = None

        # Analysis settings
        st.subheader("⚙️ Configuración de Análisis")

        analysis_col1, analysis_col2 = st.columns(2)

        with analysis_col1:
            max_samples = st.number_input(
                "Máximo de muestras a procesar:",
                min_value=1,
                max_value=len(df),
                value=min(1000, len(df)),
                step=100,
                help="Limita el número de muestras para acelerar el procesamiento"
            )

        with analysis_col2:
            batch_size = st.selectbox(
                "Tamaño de lote (batch):",
                options=[16, 32, 64, 128, 256],
                index=1,
                help="Mayor tamaño = procesamiento más rápido pero más memoria"
            )

        # Analyze button
        if st.button("🚀 Analizar Archivo", type="primary"):
            with st.spinner(f"Analizando {max_samples} muestras..."):

                # Extract features
                X = df[feature_cols].head(max_samples).values

                # Check for NaN values
                if np.isnan(X).any():
                    st.error("❌ El archivo contiene valores faltantes (NaN). "
                            "Por favor limpia los datos antes de procesarlos.")
                    st.stop()

                # Get true labels if available
                if has_labels:
                    y_true = df[label_col].head(max_samples).values
                else:
                    y_true = None

                # Make predictions
                predictions, confidences = predict_batch(
                    model, scaler, label_encoder, class_names,
                    X, batch_size=batch_size
                )

                # Store results
                results_df = pd.DataFrame({
                    'sample_id': range(1, len(predictions) + 1),
                    'prediction': predictions,
                    'confidence': confidences
                })

                if y_true is not None:
                    results_df['true_label'] = y_true
                    results_df['correct'] = results_df['prediction'] == results_df['true_label']

                st.session_state.analysis_results = results_df
                st.session_state.analysis_metadata = {
                    'filename': uploaded_file.name,
                    'total_samples': len(df),
                    'analyzed_samples': max_samples,
                    'batch_size': batch_size,
                    'model_type': st.session_state.selected_model
                }

                st.success(f"✅ Análisis completado para {len(predictions)} muestras")
                st.rerun()

    except Exception as e:
        st.error(f"❌ Error procesando archivo: {str(e)}")
        st.exception(e)

# =============================================================================
# DISPLAY RESULTS
# =============================================================================

if 'analysis_results' in st.session_state:
    results_df = st.session_state.analysis_results
    analysis_meta = st.session_state.analysis_metadata

    st.markdown("---")
    st.subheader("📈 Resultados del Análisis")

    # Summary metrics
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    total_analyzed = len(results_df)
    threat_count = sum(results_df['prediction'] != 'normal')
    avg_confidence = results_df['confidence'].mean()

    with metric_col1:
        st.metric("Muestras Analizadas", total_analyzed)

    with metric_col2:
        st.metric("Amenazas Detectadas", threat_count)

    with metric_col3:
        st.metric("Confianza Promedio", f"{avg_confidence:.2f}%")

    with metric_col4:
        if 'correct' in results_df.columns:
            accuracy = (results_df['correct'].sum() / len(results_df)) * 100
            st.metric("Precisión", f"{accuracy:.2f}%")
        else:
            st.metric("Etiquetas", "No disponibles")

    # Threat distribution
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Distribución de Amenazas**")
        fig_dist = plot_class_distribution(results_df['prediction'].tolist())
        st.plotly_chart(fig_dist, use_container_width=True)

    with chart_col2:
        st.markdown("**Top Amenazas Detectadas**")
        fig_top = plot_top_threats(results_df['prediction'].tolist(), top_n=5)
        st.plotly_chart(fig_top, use_container_width=True)

    # Detailed results table
    st.markdown("---")
    st.subheader("📋 Detalle de Predicciones")

    # Format results for display
    display_df = results_df.copy()
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2f}%")

    if 'correct' in display_df.columns:
        display_df['correct'] = display_df['correct'].apply(lambda x: '✓' if x else '✗')

    # Add color coding
    st.dataframe(display_df, use_container_width=True, height=400)

    # Confusion analysis (if labels available)
    if 'true_label' in results_df.columns:
        st.markdown("---")
        st.subheader("🎯 Análisis de Confusión")

        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Classification report
        report = classification_report(
            results_df['true_label'],
            results_df['prediction'],
            output_dict=True,
            zero_division=0
        )

        # Display per-class metrics
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df[report_df.index != 'accuracy']  # Remove accuracy row
        report_df = report_df[['precision', 'recall', 'f1-score', 'support']]
        report_df = report_df.round(3)

        st.dataframe(report_df, use_container_width=True)

        # Confusion matrix heatmap
        st.markdown("**Matriz de Confusión**")

        from utils.visualizations import plot_confusion_matrix

        fig_cm = plot_confusion_matrix(
            results_df['true_label'].values,
            results_df['prediction'].values,
            class_names.tolist(),
            normalize=True
        )
        st.pyplot(fig_cm)

    # Export options
    st.markdown("---")
    st.subheader("📥 Exportar Resultados")

    export_col1, export_col2, export_col3 = st.columns(3)

    with export_col1:
        # CSV export
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📄 Descargar Resultados (CSV)",
            data=csv,
            file_name=f"analisis_{analysis_meta['filename'].replace('.csv', '')}_resultados.csv",
            mime="text/csv"
        )

    with export_col2:
        # Full data with predictions
        full_results = results_df.copy()
        full_csv = full_results.to_csv(index=False)
        st.download_button(
            label="📊 Descargar Datos Completos",
            data=full_csv,
            file_name=f"analisis_{analysis_meta['filename'].replace('.csv', '')}_completo.csv",
            mime="text/csv"
        )

    with export_col3:
        # PDF report
        if st.button("📑 Generar Reporte PDF"):
            with st.spinner("Generando reporte PDF..."):
                try:
                    pdf_bytes = generate_analysis_report(
                        results_df,
                        model_name=st.session_state.selected_model.upper(),
                        metadata=metadata,
                        include_plots=True
                    )

                    st.download_button(
                        label="⬇️ Descargar Reporte PDF",
                        data=pdf_bytes,
                        file_name=f"reporte_{analysis_meta['filename'].replace('.csv', '')}.pdf",
                        mime="application/pdf"
                    )
                    st.success("✅ Reporte PDF generado")
                except Exception as e:
                    st.error(f"Error generando PDF: {str(e)}")
                    st.exception(e)

else:
    st.info("👆 Carga un archivo CSV para comenzar el análisis")

# Footer
st.markdown("---")
st.caption("""
**Análisis por Lotes** - Procesa grandes volúmenes de datos de tráfico de red
almacenados en archivos CSV para obtener análisis detallado y métricas de rendimiento.
""")
