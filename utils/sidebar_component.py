"""
Shared sidebar component for all pages
Ensures consistent sidebar design across the multi-page Streamlit app
"""

import streamlit as st
from utils.model_loader import load_model, check_model_files


def render_sidebar():
    """
    Render the enhanced professional sidebar.
    Should be called at the top of every page to ensure consistency.

    Returns:
        str: The selected model type ('synthetic' or 'real')
    """
    with st.sidebar:
        # Header with logo style
        st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <h1 style='color: #1f77b4; font-size: 2.2rem; margin: 0;'>IoT-IDS</h1>
                <p style='color: #666; font-size: 0.9rem; margin: 0;'>Sistema de Detección de Intrusiones</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Model selection with improved UI
        st.markdown("### Configuración del Modelo")

        # Create visual cards for model selection
        st.markdown("""
            <style>
            .model-card {
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 0.5rem;
                border: 2px solid #e0e0e0;
                transition: all 0.3s;
            }
            .model-card:hover {
                border-color: #1f77b4;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .model-card.selected {
                border-color: #1f77b4;
                background-color: #e3f2fd;
            }
            .model-name {
                font-weight: bold;
                font-size: 1.1rem;
                color: #1f77b4;
            }
            .model-accuracy {
                font-size: 1.5rem;
                font-weight: bold;
                color: #2e7d32;
            }
            .model-type {
                color: #666;
                font-size: 0.85rem;
            }
            </style>
        """, unsafe_allow_html=True)

        model_type = st.radio(
            "Seleccionar Modelo:",
            options=['synthetic', 'real'],
            format_func=lambda x: {
                'synthetic': '🔬 Modelo Sintético',
                'real': '🌐 Modelo Real (CICIoT2023)'
            }[x],
            key='model_selector',
            label_visibility="collapsed"
        )

        # Display model cards
        if model_type == 'synthetic':
            st.markdown("""
                <div class='model-card selected'>
                    <div class='model-name'>🔬 Modelo Sintético</div>
                    <div class='model-accuracy'>97.24%</div>
                    <div class='model-type'>Alta precisión | Dataset balanceado</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='model-card selected'>
                    <div class='model-name'>🌐 Modelo Real</div>
                    <div class='model-accuracy'>84.48%</div>
                    <div class='model-type'>Dataset CICIoT2023 | Mayor robustez</div>
                </div>
            """, unsafe_allow_html=True)

        # Load model if changed
        if model_type != st.session_state.get('selected_model') or not st.session_state.get('model_loaded', False):
            st.session_state.selected_model = model_type

            with st.spinner('⏳ Cargando modelo...'):
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

                        st.success(f"Modelo cargado exitosamente")
                except Exception as e:
                    st.error(f"Error al cargar modelo: {str(e)}")
                    st.session_state.model_loaded = False

        # Model info (if loaded) - Enhanced design
        if st.session_state.get('model_loaded', False):
            st.markdown("---")
            st.markdown("### Información del Modelo")

            metadata = st.session_state.metadata

            # Display key metrics with enhanced style
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 1.5rem; border-radius: 0.5rem; color: white; margin-bottom: 1rem;'>
                    <div style='font-size: 0.9rem; opacity: 0.9;'>Accuracy del Modelo</div>
                    <div style='font-size: 2.5rem; font-weight: bold;'>
                        {metadata.get('test_accuracy', 0) * 100:.2f}%
                    </div>
                    <div style='font-size: 0.85rem; opacity: 0.8;'>
                        {metadata.get('num_classes', 6)} clases detectadas
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Performance indicators
            st.markdown("**📊 Rendimiento:**")
            col1, col2 = st.columns(2)
            with col1:
                loss_recon = metadata.get('test_loss_reconstruction', 0)
                st.metric("Loss Recon.", f"{loss_recon:.4f}",
                         delta=None, delta_color="off")
            with col2:
                loss_class = metadata.get('test_loss_classification', 0)
                st.metric("Loss Class.", f"{loss_class:.4f}",
                         delta=None, delta_color="off")

            # Additional info in expander with better formatting
            with st.expander("🔍 Ver especificaciones técnicas"):
                st.markdown(f"""
                **Configuración del Modelo:**
                - **Tipo:** `{metadata.get('model_type', 'N/A')}`
                - **Dataset:** `{metadata.get('dataset_type', 'N/A')}`
                - **Dimensión de entrada:** `{metadata.get('input_dim', 16)} features (PCA)`
                - **Espacio latente:** `{metadata.get('latent_dim', 6)} dimensiones`

                **Hiperparámetros:**
                - **Épocas:** `{metadata.get('epochs_trained', 'N/A')}`
                - **Batch Size:** `{metadata.get('batch_size', 64)}`

                **Función de Pérdida:**
                - Total: `{metadata.get('test_loss_total', 0):.4f}`
                - Reconstrucción: `{metadata.get('test_loss_reconstruction', 0):.4f}`
                - Clasificación: `{metadata.get('test_loss_classification', 0):.4f}`
                """)

        # Navigation guide
        st.markdown("---")
        st.markdown("### 📚 Navegación")
        st.markdown("""
        **Páginas disponibles:**
        - **Comparación**: Análisis comparativo
        - **Tiempo Real**: Simulación en vivo
        - **Análisis**: Procesamiento de archivos
        - **Métricas**: Dashboard completo
        """)

        # Footer with branding
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <p style='color: #666; font-size: 0.8rem; margin: 0;'>
                    <b>Universidad Señor de Sipán</b><br>
                    Sistema de Tesis - 2025
                </p>
            </div>
        """, unsafe_allow_html=True)

    return model_type
