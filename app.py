"""
IoT Intrusion Detection System Dashboard - Main Application

This is the main entry point for the Streamlit application demonstrating
an Autoencoder-FNN multi-task model for IoT network threat detection.
"""

import streamlit as st
import numpy as np
from pathlib import Path

from utils.model_loader import load_model, get_model_info, check_model_files

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="IoT-IDS Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """
        **IoT Intrusion Detection System**

        Sistema de detección de intrusiones para IoT basado en
        Autoencoder-FNN multi-tarea con reducción dimensional PCA.

        Universidad Señor de Sipán
        """
    }
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'label_encoder' not in st.session_state:
        st.session_state.label_encoder = None
    if 'class_names' not in st.session_state:
        st.session_state.class_names = None
    if 'metadata' not in st.session_state:
        st.session_state.metadata = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'synthetic'

init_session_state()

# =============================================================================
# SIDEBAR - MODEL SELECTOR
# =============================================================================

with st.sidebar:
    st.title("IoT-IDS")
    st.markdown("---")

    # Model selection
    st.subheader("Configuración")

    model_type = st.radio(
        "Seleccionar Modelo:",
        options=['synthetic', 'real'],
        format_func=lambda x: {
            'synthetic': 'Modelo Sintético (97.24%)',
            'real': 'Modelo Real (84.48%)'
        }[x],
        key='model_selector'
    )

    # Load model if changed
    if model_type != st.session_state.selected_model or not st.session_state.model_loaded:
        st.session_state.selected_model = model_type

        with st.spinner(f'Cargando modelo {model_type}...'):
            try:
                # Check if files exist
                file_status = check_model_files(model_type)
                missing_files = [k for k, v in file_status.items() if not v]

                if missing_files:
                    st.error(f"Archivos faltantes: {', '.join(missing_files)}")
                    st.session_state.model_loaded = False
                else:
                    # Load model
                    model, scaler, encoder, classes, metadata = load_model(model_type)

                    # Store in session state
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.session_state.label_encoder = encoder
                    st.session_state.class_names = classes
                    st.session_state.metadata = metadata
                    st.session_state.model_loaded = True

                    st.success(f"Modelo {model_type} cargado exitosamente")
            except Exception as e:
                st.error(f"Error al cargar modelo: {str(e)}")
                st.session_state.model_loaded = False

    # Model info (if loaded)
    if st.session_state.model_loaded:
        st.markdown("---")
        st.subheader("Info del Modelo")

        metadata = st.session_state.metadata

        # Display key metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Accuracy",
                f"{metadata.get('test_accuracy', 0) * 100:.2f}%"
            )
        with col2:
            st.metric(
                "Clases",
                metadata.get('num_classes', 6)
            )

        # Additional info in expander
        with st.expander("Ver más detalles"):
            st.write(f"**Tipo:** {metadata.get('model_type', 'N/A')}")
            st.write(f"**Dataset:** {metadata.get('dataset_type', 'N/A')}")
            st.write(f"**Input Dim:** {metadata.get('input_dim', 16)}")
            st.write(f"**Latent Dim:** {metadata.get('latent_dim', 6)}")
            st.write(f"**Épocas:** {metadata.get('epochs_trained', 'N/A')}")
            st.write(f"**Batch Size:** {metadata.get('batch_size', 64)}")

            # Loss components
            st.markdown("**Pérdidas:**")
            st.write(f"- Total: {metadata.get('test_loss_total', 0):.4f}")
            st.write(f"- Reconstrucción: {metadata.get('test_loss_reconstruction', 0):.4f}")
            st.write(f"- Clasificación: {metadata.get('test_loss_classification', 0):.4f}")

    st.markdown("---")
    st.caption("Universidad Señor de Sipán")

# =============================================================================
# MAIN CONTENT - HOMEPAGE
# =============================================================================

st.title("Sistema de Detección de Intrusiones para IoT")
st.markdown("### Dashboard de Análisis y Detección de Amenazas")

# Welcome message
if st.session_state.model_loaded:
    st.success(f"Sistema listo | Modelo activo: **{st.session_state.selected_model.upper()}**")
else:
    st.warning("Selecciona un modelo en la barra lateral para comenzar")

st.markdown("---")

# Project overview
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### Objetivo

    Demostración de un sistema de detección de intrusiones para redes IoT
    utilizando aprendizaje profundo multi-tarea (Autoencoder-FNN).
    """)

with col2:
    st.markdown("""
    #### Tecnología

    - **Autoencoder:** Reducción dimensional y reconstrucción
    - **FNN:** Clasificación de amenazas
    - **PCA:** 16 componentes principales
    """)

with col3:
    st.markdown("""
    #### Amenazas Detectadas

    1. Normal (tráfico benigno)
    2. Brute Force
    3. DDoS
    4. MITM
    5. Scan
    6. Spoofing
    """)

st.markdown("---")

# Features overview
st.subheader("Funcionalidades del Dashboard")

feature_col1, feature_col2 = st.columns(2)

with feature_col1:
    st.markdown("""
    **Comparación de Modelos**
    - Análisis lado a lado de ambos modelos
    - Tasa de concordancia entre predicciones
    - Exportación de resultados comparativos

    **Simulación en Tiempo Real**
    - Generación de tráfico sintético
    - Detección instantánea de amenazas
    - Visualización temporal de alertas
    """)

with feature_col2:
    st.markdown("""
    **Análisis de Archivos**
    - Carga de datasets CSV
    - Procesamiento por lotes
    - Métricas detalladas y reportes PDF

    **Métricas y Rendimiento**
    - Matriz de confusión
    - Distribución de amenazas
    - Métricas de clasificación
    """)

st.markdown("---")

# Model architecture info (if model is loaded)
if st.session_state.model_loaded:
    st.subheader("Arquitectura del Modelo")

    arch_col1, arch_col2 = st.columns([2, 1])

    with arch_col1:
        st.markdown("""
        **Pipeline de Inferencia:**

        1. **Entrada:** 16 componentes PCA (PC1-PC16)
        2. **Normalización:** StandardScaler
        3. **Autoencoder:**
           - Encoder: 16 → 8 → 6 (latente)
           - Decoder: 6 → 8 → 16 (reconstrucción)
        4. **Clasificador FNN:**
           - Input: 6 features (latente)
           - Hidden: 16 neurons + Dropout (0.3)
           - Output: 6 clases (Softmax)
        5. **Salidas:**
           - Reconstrucción (16 valores)
           - Clasificación (6 probabilidades)
        """)

    with arch_col2:
        st.info("""
        **Función de Pérdida Multi-tarea:**

        ```
        Loss = 0.3 × MSE(recon)
             + 0.7 × CrossEntropy(class)
        ```

        **Optimizador:** Adam

        **Ventajas:**
        - Aprendizaje de representaciones
        - Detección de anomalías
        - Alta precisión
        """)

# Instructions
st.markdown("---")
st.subheader("Cómo Usar el Dashboard")

st.markdown("""
1. **Selecciona un modelo** en la barra lateral (Sintético o Real)
2. **Navega a las páginas** usando el menú lateral:
   - **Comparación de Modelos:** Compara ambos modelos en las mismas muestras
   - **Tiempo Real:** Simula tráfico de red en tiempo real
   - **Análisis de Archivo:** Carga y analiza datasets CSV
   - **Métricas:** Visualiza el rendimiento del modelo
3. **Explora las visualizaciones** interactivas
4. **Exporta resultados** en formato PDF o CSV

**Tip:** El modelo sintético tiene mayor precisión pero el modelo real es más robusto
ante variaciones de datos del mundo real.
""")

# Footer
st.markdown("---")
st.caption("""
**Proyecto de Tesis - Universidad Señor de Sipán**
Sistema de Detección de Intrusiones para IoT mediante Autoencoder-FNN Multi-tarea
Desarrollado para demostración académica
""")
