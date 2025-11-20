"""
Real-Time Simulation Page - Simulate IoT traffic and detect threats in real-time.
"""

import streamlit as st
import numpy as np
import pandas as pd
import time

from utils.model_loader import predict_sample
from utils.data_simulator import (
    generate_scenario_traffic,
    get_threat_severity,
    calculate_risk_score
)
from utils.visualizations import (
    plot_temporal_detections,
    plot_class_distribution,
    create_risk_gauge
)
from utils.sidebar_component import render_sidebar

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Simulación en Tiempo Real",
    layout="wide"
)

# Render shared sidebar for consistency
render_sidebar()

st.title("Detección de Amenazas en Tiempo Real")
st.markdown("Simula tráfico de red IoT y observa la detección de amenazas en vivo")

# =============================================================================
# CHECK MODEL
# =============================================================================

if not st.session_state.get('model_loaded', False):
    st.error("No hay modelo cargado. Por favor selecciona un modelo desde la página principal.")
    st.stop()

model = st.session_state.model
scaler = st.session_state.scaler
label_encoder = st.session_state.label_encoder
class_names = st.session_state.class_names
metadata = st.session_state.metadata

st.success(f"Modelo activo: **{st.session_state.selected_model.upper()}**")

# =============================================================================
# SIMULATION SETTINGS
# =============================================================================

st.markdown("---")
st.subheader("Configuración de Simulación")

col1, col2, col3 = st.columns(3)

with col1:
    scenario = st.selectbox(
        "Escenario de Tráfico:",
        options=['normal', 'under_attack', 'scanning', 'mixed'],
        format_func=lambda x: {
            'normal': 'Normal (5% amenazas)',
            'under_attack': 'Bajo Ataque (80% DDoS)',
            'scanning': 'Escaneo (60% scan)',
            'mixed': 'Mixto (30% amenazas)'
        }[x]
    )

with col2:
    duration = st.slider(
        "Duración (segundos):",
        min_value=10,
        max_value=120,
        value=30,
        step=10
    )

with col3:
    speed = st.slider(
        "Velocidad de Simulación:",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="1.0 = tiempo real, 2.0 = 2x más rápido, 0.5 = 2x más lento"
    )

# =============================================================================
# SIMULATION CONTROL
# =============================================================================

st.markdown("---")

# Initialize session state for simulation
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = {
        'timestamps': [],
        'predictions': [],
        'confidences': [],
        'true_labels': []
    }

# Control buttons
button_col1, button_col2, button_col3 = st.columns([1, 1, 3])

with button_col1:
    if st.button("Iniciar Simulación", type="primary", disabled=st.session_state.simulation_running):
        st.session_state.simulation_running = True
        # Clear previous data
        st.session_state.simulation_data = {
            'timestamps': [],
            'predictions': [],
            'confidences': [],
            'true_labels': []
        }
        st.rerun()

with button_col2:
    if st.button("Detener", disabled=not st.session_state.simulation_running):
        st.session_state.simulation_running = False
        st.rerun()

with button_col3:
    if st.button("Reiniciar Datos"):
        st.session_state.simulation_data = {
            'timestamps': [],
            'predictions': [],
            'confidences': [],
            'true_labels': []
        }
        st.session_state.simulation_running = False
        st.rerun()

# =============================================================================
# SIMULATION EXECUTION
# =============================================================================

if st.session_state.simulation_running:
    # Generate traffic scenario
    traffic_timeline = generate_scenario_traffic(scenario, duration)

    # Create placeholders for live updates
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    details_placeholder = st.empty()

    # Progress bar
    progress_bar = st.progress(0)

    # Run simulation
    for idx, (timestamp, sample, true_label) in enumerate(traffic_timeline):
        # Make prediction
        prediction, probs, confidence = predict_sample(
            model, scaler, label_encoder, class_names, sample
        )

        # Store data
        st.session_state.simulation_data['timestamps'].append(timestamp)
        st.session_state.simulation_data['predictions'].append(prediction)
        st.session_state.simulation_data['confidences'].append(confidence)
        st.session_state.simulation_data['true_labels'].append(true_label)

        # Update progress
        progress = (idx + 1) / len(traffic_timeline)
        progress_bar.progress(progress)

        # Get severity
        severity = get_threat_severity(prediction)

        # Status update
        severity_labels = {
            'normal': '[NORMAL]',
            'low': '[LOW]',
            'medium': '[MEDIUM]',
            'high': '[HIGH]',
            'critical': '[CRITICAL]'
        }
        status_label = severity_labels.get(severity, '[UNKNOWN]')

        status_placeholder.info(
            f"{status_label} **t={timestamp:.1f}s** | Detectado: **{prediction}** | "
            f"Confianza: {confidence:.1f}% | Real: {true_label}"
        )

        # Update metrics every 5 samples
        if len(st.session_state.simulation_data['predictions']) >= 5:
            data = st.session_state.simulation_data

            # Calculate metrics
            total_detections = len(data['predictions'])
            threat_detections = sum(1 for p in data['predictions'] if p != 'normal')
            avg_confidence = np.mean(data['confidences'])
            risk_score = calculate_risk_score(data['predictions'][-20:])  # Last 20 samples

            # Display metrics
            with metrics_placeholder.container():
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    st.metric("Total Muestras", total_detections)
                with m_col2:
                    st.metric("Amenazas Detectadas", threat_detections)
                with m_col3:
                    st.metric("Confianza Promedio", f"{avg_confidence:.1f}%")
                with m_col4:
                    st.metric("Nivel de Riesgo", f"{risk_score:.0f}/100")

            # Update chart
            with chart_placeholder.container():
                if len(data['timestamps']) > 2:
                    fig = plot_temporal_detections(
                        data['timestamps'],
                        data['predictions'],
                        data['confidences'],
                        window_size=min(60, len(data['timestamps']))
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Simulate time delay based on speed
        time.sleep(1.0 / speed)

    # Simulation complete
    st.session_state.simulation_running = False
    progress_bar.empty()
    status_placeholder.success(f"Simulación completada | {duration} segundos procesados")

# =============================================================================
# RESULTS DISPLAY (when not running)
# =============================================================================

if not st.session_state.simulation_running and len(st.session_state.simulation_data['predictions']) > 0:
    data = st.session_state.simulation_data

    st.markdown("---")
    st.subheader("Resultados de la Simulación")

    # Summary metrics
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    total_samples = len(data['predictions'])
    threat_count = sum(1 for p in data['predictions'] if p != 'normal')
    avg_confidence = np.mean(data['confidences'])
    risk_score = calculate_risk_score(data['predictions'])

    with metric_col1:
        st.metric("Muestras Analizadas", total_samples)
    with metric_col2:
        st.metric("Amenazas Detectadas", threat_count)
    with metric_col3:
        st.metric("Confianza Promedio", f"{avg_confidence:.1f}%")
    with metric_col4:
        st.metric("Riesgo Global", f"{risk_score:.0f}/100")

    # Visualizations
    st.markdown("---")
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown("**Timeline de Detecciones**")
        fig_timeline = plot_temporal_detections(
            data['timestamps'],
            data['predictions'],
            data['confidences']
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    with viz_col2:
        st.markdown("**Distribución de Amenazas**")
        fig_dist = plot_class_distribution(data['predictions'])
        st.plotly_chart(fig_dist, use_container_width=True)

    # Risk gauge
    st.markdown("---")
    gauge_col1, gauge_col2, gauge_col3 = st.columns([1, 2, 1])
    with gauge_col2:
        fig_gauge = create_risk_gauge(risk_score)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Detailed table
    st.markdown("---")
    st.subheader("Detalle de Detecciones")

    details_df = pd.DataFrame({
        'Tiempo (s)': data['timestamps'],
        'Predicción': data['predictions'],
        'Confianza (%)': [f"{c:.1f}" for c in data['confidences']],
        'Etiqueta Real': data['true_labels'],
        'Correcto': ['SI' if p == t else 'NO'
                    for p, t in zip(data['predictions'], data['true_labels'])]
    })

    st.dataframe(details_df, use_container_width=True, height=300)

    # Accuracy calculation
    correct = sum(1 for p, t in zip(data['predictions'], data['true_labels']) if p == t)
    accuracy = (correct / total_samples) * 100
    st.info(f"**Precisión en esta simulación:** {accuracy:.2f}% ({correct}/{total_samples} correctos)")

    # Export
    st.markdown("---")
    csv = details_df.to_csv(index=False)
    st.download_button(
        label="Descargar Resultados (CSV)",
        data=csv,
        file_name=f"simulacion_{scenario}_{duration}s.csv",
        mime="text/csv"
    )

else:
    if not st.session_state.simulation_running:
        st.info("Configura el escenario y presiona 'Iniciar Simulación' para comenzar")

# Footer
st.markdown("---")
st.caption("""
**Simulación en Tiempo Real** - Genera tráfico IoT sintético según escenarios predefinidos
y observa cómo el modelo detecta amenazas en tiempo real.
""")
