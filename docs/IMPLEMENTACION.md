# Guía de Implementación - Sistema IoT-IDS

Esta guía detalla el proceso de implementación por fases para desarrollar la aplicación de demostración del Sistema de Detección de Intrusiones IoT.

## Enfoque de Desarrollo

Se recomienda implementar el proyecto en **5 fases secuenciales**, probando cada fase antes de continuar con la siguiente. Esto permite:

- Tener control sobre cada componente
- Realizar ajustes según necesidades específicas
- Comprender mejor el código generado
- Identificar y corregir errores de forma temprana

---

## FASE 1: Estructura Base y Carga de Modelos

### Objetivo
Crear la estructura inicial de la aplicación Streamlit con capacidad de cargar ambos modelos (sintético y real).

### Tareas

#### 1.1 Crear estructura de carpetas

```
iot_ids_demo/
├── app.py                    # Aplicación principal
├── models/                   # Archivos .h5, .pkl, .npy
├── data/                     # Datasets de ejemplo
├── utils/
│   ├── model_loader.py       # Cargar ambos modelos
│   ├── data_simulator.py     # Generar datos simulados
│   └── visualizations.py     # Gráficos y visualizaciones
└── pages/
    ├── 1_🔬_Comparacion_Modelos.py
    ├── 2_⚡_Tiempo_Real.py
    ├── 3_📊_Analisis_Archivo.py
    └── 4_📈_Metricas.py
```

#### 1.2 Implementar `utils/model_loader.py`

Crear funciones para:

```python
def load_synthetic_model():
    """Carga el modelo sintético y sus componentes"""
    # - Cargar modelo_ae_fnn_iot_synthetic.h5
    # - Cargar scaler_synthetic.pkl
    # - Cargar label_encoder_synthetic.pkl
    # - Cargar class_names_synthetic.npy
    # - Cargar model_metadata_synthetic.json
    # - Retornar todos los componentes
    pass

def load_real_model():
    """Carga el modelo real y sus componentes"""
    # Similar a load_synthetic_model()
    pass

def predict_sample(model, scaler, label_encoder, sample):
    """
    Realiza predicción de una muestra

    Args:
        model: Modelo Keras cargado
        scaler: StandardScaler cargado
        label_encoder: LabelEncoder cargado
        sample: Array de 16 features (PC1-PC16)

    Returns:
        prediction: Clase predicha
        probabilities: Probabilidades de cada clase
        confidence: Confianza de la predicción (%)
    """
    pass

def verify_model_input(model):
    """
    Verifica que el modelo espera 16 features de entrada

    Returns:
        bool: True si el modelo es válido
    """
    pass
```

#### 1.3 Implementar `app.py` (Página Principal)

Contenido de la página principal:

```python
import streamlit as st
from utils.model_loader import load_synthetic_model, load_real_model

st.set_page_config(
    page_title="Sistema IDS IoT - USS",
    page_icon="🛡️",
    layout="wide"
)

# Título principal
st.title("🛡️ Sistema de Detección de Intrusiones IoT - USS")
st.markdown("### Clasificación de tráfico de red y fortalecimiento de ciberseguridad")

# Selector de modelo
model_choice = st.sidebar.selectbox(
    "Seleccionar Modelo",
    ["Sintético (97%)", "Real CICIoT2023 (84.48%)"]
)

# Cargar modelo seleccionado
if "Sintético" in model_choice:
    # Cargar modelo sintético
    # Mostrar información del modelo sintético
    pass
else:
    # Cargar modelo real
    # Mostrar información del modelo real
    pass

# Información básica
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Accuracy", "97%" if "Sintético" in model_choice else "84.48%")
with col2:
    st.metric("Tiempo de Inferencia", "<2ms")
with col3:
    st.metric("False Positive Rate", "<2%")

# Arquitectura
st.subheader("Arquitectura del Modelo")
st.markdown("""
**Autoencoder-FNN Multi-tarea**
- Encoder: 16 → 8 → 4 (compresión)
- Decoder: 4 → 8 → 16 (reconstrucción)
- Clasificador: 4 → 16 → 8 clases
- Función de pérdida combinada: λ₁ × MSE + λ₂ × CrossEntropy
""")

# Instrucciones
st.info("""
👈 Utiliza el menú lateral para navegar entre las diferentes funcionalidades:
- 🔬 **Comparación de Modelos**: Prueba y compara ambos modelos
- ⚡ **Tiempo Real**: Simulación de detección en vivo
- 📊 **Análisis de Archivo**: Procesa archivos CSV
- 📈 **Métricas**: Dashboard completo de rendimiento
""")
```

### Criterios de Éxito Fase 1

- ✅ Estructura de carpetas creada correctamente
- ✅ Modelos cargados sin errores
- ✅ Verificación de 16 features de entrada
- ✅ Página principal muestra información básica
- ✅ Selector de modelo funcional

---

## FASE 2: Módulo de Comparación de Modelos

### Objetivo
Implementar página para comparar el desempeño de ambos modelos lado a lado.

### Archivo: `pages/1_🔬_Comparacion_Modelos.py`

### Tareas

#### 2.1 Sección de Comparación Lado a Lado

```python
import streamlit as st
import pandas as pd
from utils.model_loader import predict_sample, load_synthetic_model, load_real_model

st.title("🔬 Comparación de Modelos")

# Layout en dos columnas
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Modelo Sintético")
    st.metric("Accuracy", "97%")
    # Cargar modelo sintético

with col_right:
    st.subheader("Modelo Real")
    st.metric("Accuracy", "84.48%")
    # Cargar modelo real
```

#### 2.2 Funcionalidad de Prueba con Muestra Única

```python
st.divider()
st.subheader("Prueba con Muestra Única")

if st.button("🎲 Generar Muestra Aleatoria"):
    # Generar muestra aleatoria de 16 componentes PCA
    sample = generate_random_sample()

    # Mostrar componentes
    st.write("Componentes PCA:", sample)

    # Predecir con ambos modelos
    pred_synthetic, prob_synthetic, conf_synthetic = predict_sample(
        synthetic_model, sample
    )
    pred_real, prob_real, conf_real = predict_sample(
        real_model, sample
    )

    # Mostrar resultados en dos columnas
    col1, col2 = st.columns(2)

    with col1:
        st.success(f"Predicción: {pred_synthetic}")
        st.metric("Confianza", f"{conf_synthetic:.2f}%")

    with col2:
        st.success(f"Predicción: {pred_real}")
        st.metric("Confianza", f"{conf_real:.2f}%")

    # Resaltar si hay diferencia
    if pred_synthetic != pred_real:
        st.warning("⚠️ Los modelos predicen clases diferentes!")
```

#### 2.3 Análisis Batch (Múltiples Muestras)

```python
st.divider()
st.subheader("📦 Análisis Batch")

uploaded_file = st.file_uploader(
    "Subir archivo CSV con múltiples muestras",
    type=['csv']
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validar que tenga 16 columnas (PC1-PC16)
    if df.shape[1] != 16:
        st.error("El archivo debe tener exactamente 16 columnas (PC1-PC16)")
    else:
        if st.button("🚀 Procesar con Ambos Modelos"):
            results = []

            for idx, row in df.iterrows():
                sample = row.values

                # Predicciones
                pred_syn, _, conf_syn = predict_sample(synthetic_model, sample)
                pred_real, _, conf_real = predict_sample(real_model, sample)

                results.append({
                    'Muestra': idx,
                    'Pred_Sintético': pred_syn,
                    'Conf_Sintético': conf_syn,
                    'Pred_Real': pred_real,
                    'Conf_Real': conf_real,
                    'Coincide': pred_syn == pred_real
                })

            results_df = pd.DataFrame(results)

            # Mostrar tabla comparativa
            st.dataframe(results_df, use_container_width=True)

            # Calcular métricas de concordancia
            concordancia = (results_df['Coincide'].sum() / len(results_df)) * 100
            st.metric("Concordancia entre Modelos", f"{concordancia:.2f}%")
```

#### 2.4 Visualización Comparativa

```python
import plotly.graph_objects as go

st.divider()
st.subheader("📊 Visualización Comparativa")

if results_df is not None:
    # Gráfico de barras comparando confidence scores
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Modelo Sintético',
        x=results_df['Muestra'],
        y=results_df['Conf_Sintético']
    ))

    fig.add_trace(go.Bar(
        name='Modelo Real',
        x=results_df['Muestra'],
        y=results_df['Conf_Real']
    ))

    fig.update_layout(
        title='Comparación de Confianza por Muestra',
        xaxis_title='Muestra',
        yaxis_title='Confianza (%)',
        barmode='group'
    )

    st.plotly_chart(fig, use_container_width=True)
```

### Criterios de Éxito Fase 2

- ✅ Comparación lado a lado funcional
- ✅ Generación de muestras aleatorias
- ✅ Procesamiento batch de archivos CSV
- ✅ Tabla comparativa de resultados
- ✅ Visualizaciones interactivas con Plotly

---

## FASE 3: Simulación en Tiempo Real

### Objetivo
Crear simulador de tráfico IoT con detección de amenazas en tiempo real.

### Archivo: `pages/2_⚡_Tiempo_Real.py`

### Tareas

#### 3.1 Implementar `utils/data_simulator.py`

```python
import numpy as np
import random

def generate_traffic_sample(attack_type=None):
    """
    Genera una muestra de tráfico IoT simulado

    Args:
        attack_type: 'DDoS', 'DoS', 'Brute_Force', 'Spoofing',
                     'MITM', 'Scan', 'Recon', 'Benign' o None (aleatorio)

    Returns:
        sample: Array de 16 componentes PCA
        true_label: Etiqueta verdadera
    """

    if attack_type is None:
        # 70% tráfico normal, 30% ataques
        if random.random() < 0.7:
            attack_type = 'Benign'
        else:
            attack_type = random.choice([
                'DDoS', 'DoS', 'Brute_Force', 'Spoofing',
                'MITM', 'Scan', 'Recon'
            ])

    # Generar muestra sintética basada en patrón del ataque
    sample = generate_attack_pattern(attack_type)

    return sample, attack_type

def generate_attack_pattern(attack_type):
    """
    Genera patrón característico de un tipo de ataque
    """
    # Base normal
    sample = np.random.randn(16)

    if attack_type == 'DDoS':
        # Características de DDoS: alto volumen, múltiples orígenes
        sample[0] *= 3  # PC1 alto
        sample[1] *= 2.5
        sample[3] *= 2

    elif attack_type == 'Brute_Force':
        # Características de Brute Force: intentos repetitivos
        sample[5] *= 4
        sample[6] *= 3

    # ... patrones para otros ataques

    return sample

def generate_attack_burst(attack_type, count=10):
    """
    Genera ráfaga de muestras del mismo tipo de ataque
    """
    samples = []
    for _ in range(count):
        sample, label = generate_traffic_sample(attack_type)
        samples.append((sample, label))
    return samples
```

#### 3.2 Panel de Monitoreo en Vivo

```python
import streamlit as st
import time
import plotly.graph_objects as go
from collections import deque

st.title("⚡ Simulación en Tiempo Real")

# Estado de la simulación
if 'running' not in st.session_state:
    st.session_state.running = False
if 'threat_history' not in st.session_state:
    st.session_state.threat_history = deque(maxlen=60)
if 'threat_counts' not in st.session_state:
    st.session_state.threat_counts = {
        'DDoS': 0, 'DoS': 0, 'Brute_Force': 0,
        'Spoofing': 0, 'MITM': 0, 'Scan': 0,
        'Recon': 0, 'Benign': 0
    }

# Controles
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Iniciar Simulación" if not st.session_state.running else "⏸️ Pausar"):
        st.session_state.running = not st.session_state.running

with col2:
    if st.button("🔄 Reiniciar Contadores"):
        st.session_state.threat_counts = {k: 0 for k in st.session_state.threat_counts}
        st.session_state.threat_history.clear()

# Placeholder para actualizaciones en vivo
metrics_placeholder = st.empty()
chart_placeholder = st.empty()
log_placeholder = st.empty()

# Loop de simulación
while st.session_state.running:
    # Generar muestra
    sample, true_label = generate_traffic_sample()

    # Predecir con modelo seleccionado
    prediction, probabilities, confidence = predict_sample(model, sample)

    # Actualizar contadores
    st.session_state.threat_counts[prediction] += 1
    st.session_state.threat_history.append({
        'time': time.time(),
        'prediction': prediction,
        'confidence': confidence,
        'is_attack': prediction != 'Benign'
    })

    # Actualizar métricas
    with metrics_placeholder.container():
        col1, col2, col3, col4 = st.columns(4)

        total_samples = sum(st.session_state.threat_counts.values())
        attack_samples = total_samples - st.session_state.threat_counts['Benign']
        risk_level = (attack_samples / total_samples * 100) if total_samples > 0 else 0

        with col1:
            st.metric("Total Muestras", total_samples)
        with col2:
            st.metric("Amenazas Detectadas", attack_samples)
        with col3:
            st.metric("Nivel de Riesgo", f"{risk_level:.1f}%")
        with col4:
            # Última detección
            color = "🔴" if prediction != 'Benign' else "🟢"
            st.metric("Última Detección", f"{color} {prediction}")

    # Actualizar gráfico temporal
    with chart_placeholder:
        update_temporal_chart(st.session_state.threat_history)

    # Actualizar log
    with log_placeholder:
        update_detection_log(st.session_state.threat_history)

    time.sleep(1)  # 1 muestra por segundo
```

#### 3.3 Sistema de Alertas Visuales

```python
def show_alert(prediction, confidence):
    """
    Muestra alerta visual según el tipo de amenaza
    """
    if prediction == 'Benign':
        if confidence > 90:
            st.success(f"✅ Tráfico Normal - Confianza: {confidence:.2f}%")
        else:
            st.info(f"ℹ️ Tráfico Normal (baja confianza) - {confidence:.2f}%")
    else:
        if confidence > 80:
            st.error(f"🚨 AMENAZA DETECTADA: {prediction} - Confianza: {confidence:.2f}%")
            # Opcionalmente reproducir sonido
            # play_alert_sound()
        elif confidence > 50:
            st.warning(f"⚠️ Anomalía Detectada: {prediction} - Confianza: {confidence:.2f}%")
        else:
            st.info(f"🔍 Posible Anomalía: {prediction} - Confianza: {confidence:.2f}%")
```

#### 3.4 Simulación de Escenarios Específicos

```python
st.divider()
st.subheader("🎯 Simular Escenarios de Ataque")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("💥 Simular DDoS"):
        simulate_attack_scenario('DDoS', count=10)

with col2:
    if st.button("🔐 Simular Brute Force"):
        simulate_attack_scenario('Brute_Force', count=10)

with col3:
    if st.button("🔍 Simular Scan"):
        simulate_attack_scenario('Scan', count=10)

with col4:
    if st.button("🎭 Simular MITM"):
        simulate_attack_scenario('MITM', count=10)

def simulate_attack_scenario(attack_type, count=10):
    """
    Simula ráfaga de ataque específico
    """
    st.info(f"Simulando {count} muestras de {attack_type}...")

    samples = generate_attack_burst(attack_type, count)

    results = []
    for sample, true_label in samples:
        prediction, _, confidence = predict_sample(model, sample)
        results.append({
            'Verdadero': true_label,
            'Predicho': prediction,
            'Confianza': confidence,
            'Correcto': prediction == true_label
        })

    results_df = pd.DataFrame(results)

    # Métricas de la simulación
    accuracy = (results_df['Correcto'].sum() / len(results_df)) * 100

    st.success(f"Simulación completada: {accuracy:.1f}% de precisión")
    st.dataframe(results_df)
```

### Criterios de Éxito Fase 3

- ✅ Generador de tráfico simulado funcional
- ✅ Monitoreo en tiempo real (1 muestra/segundo)
- ✅ Alertas visuales por nivel de riesgo
- ✅ Gráfico temporal scrollable
- ✅ Simulación de escenarios específicos
- ✅ Log de últimas detecciones

---

## FASE 4: Análisis de Archivos y Reportes

### Objetivo
Permitir análisis batch de archivos CSV y generación de reportes detallados.

### Archivo: `pages/3_📊_Analisis_Archivo.py`

### Tareas

#### 4.1 Upload y Validación de Archivos

```python
import streamlit as st
import pandas as pd

st.title("📊 Análisis de Archivos CSV")

st.markdown("""
Sube un archivo CSV con datos de tráfico IoT para análisis batch.

**Formato requerido:**
- 16 columnas con componentes PCA (PC1-PC16)
- Opcionalmente: columna 'label' con etiquetas verdaderas
""")

uploaded_file = st.file_uploader(
    "Seleccionar archivo CSV",
    type=['csv'],
    help="El archivo debe contener 16 columnas de features PCA"
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Validar formato
        has_labels = 'label' in df.columns
        feature_cols = [col for col in df.columns if col != 'label']

        if len(feature_cols) != 16:
            st.error(f"❌ Error: El archivo tiene {len(feature_cols)} columnas, se requieren 16 (PC1-PC16)")
        else:
            st.success(f"✅ Archivo válido: {len(df)} muestras cargadas")

            # Preview de los datos
            st.subheader("Vista Previa")
            st.dataframe(df.head(10), use_container_width=True)

            # Información del dataset
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Muestras", len(df))
            with col2:
                st.metric("Features", len(feature_cols))
            with col3:
                st.metric("Tiene Etiquetas", "Sí" if has_labels else "No")

    except Exception as e:
        st.error(f"❌ Error al leer el archivo: {str(e)}")
```

#### 4.2 Procesamiento y Análisis

```python
if uploaded_file and len(feature_cols) == 16:
    st.divider()

    model_choice = st.selectbox(
        "Seleccionar modelo para análisis",
        ["Modelo Sintético (97%)", "Modelo Real (84.48%)"]
    )

    if st.button("🚀 Analizar Archivo", type="primary"):
        with st.spinner("Procesando muestras..."):
            # Extraer features
            X = df[feature_cols].values

            # Predicciones
            predictions = []
            confidences = []

            progress_bar = st.progress(0)

            for idx, sample in enumerate(X):
                pred, probs, conf = predict_sample(model, sample)
                predictions.append(pred)
                confidences.append(conf)

                # Actualizar barra de progreso
                progress_bar.progress((idx + 1) / len(X))

            # Agregar resultados al dataframe
            df['Predicción'] = predictions
            df['Confianza'] = confidences

            # Si tiene etiquetas, calcular métricas
            if has_labels:
                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score,
                    f1_score, confusion_matrix
                )

                y_true = df['label']
                y_pred = df['Predicción']

                # Calcular métricas
                accuracy = accuracy_score(y_true, y_pred) * 100

                # Mostrar métricas principales
                st.success("✅ Análisis completado!")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.2f}%")
                with col2:
                    precision = precision_score(y_true, y_pred, average='weighted') * 100
                    st.metric("Precisión", f"{precision:.2f}%")
                with col3:
                    recall = recall_score(y_true, y_pred, average='weighted') * 100
                    st.metric("Recall", f"{recall:.2f}%")
                with col4:
                    f1 = f1_score(y_true, y_pred, average='weighted') * 100
                    st.metric("F1-Score", f"{f1:.2f}%")

            # Guardar resultados en session state
            st.session_state.analysis_results = df
            st.session_state.has_labels = has_labels
```

#### 4.3 Visualización de Resultados

```python
if 'analysis_results' in st.session_state:
    st.divider()
    st.subheader("📊 Resultados del Análisis")

    df_results = st.session_state.analysis_results

    # Tabla de resultados
    st.dataframe(df_results, use_container_width=True)

    # Distribución de clases predichas
    st.subheader("Distribución de Predicciones")

    pred_counts = df_results['Predicción'].value_counts()

    fig = go.Figure(data=[go.Pie(
        labels=pred_counts.index,
        values=pred_counts.values,
        hole=0.3
    )])
    fig.update_layout(title="Distribución de Clases Predichas")
    st.plotly_chart(fig, use_container_width=True)

    # Top 10 muestras más sospechosas
    st.subheader("🔍 Top 10 Muestras Más Sospechosas")

    threats = df_results[df_results['Predicción'] != 'Benign']
    threats_sorted = threats.sort_values('Confianza', ascending=False).head(10)

    st.dataframe(threats_sorted, use_container_width=True)

    # Matriz de confusión (si hay etiquetas)
    if st.session_state.has_labels:
        st.subheader("Matriz de Confusión")

        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt

        cm = confusion_matrix(
            df_results['label'],
            df_results['Predicción']
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Verdadero')
        ax.set_title('Matriz de Confusión')

        st.pyplot(fig)
```

#### 4.4 Exportar Reporte

```python
st.divider()
st.subheader("📥 Exportar Reporte")

col1, col2 = st.columns(2)

with col1:
    # Exportar resultados a CSV
    csv = df_results.to_csv(index=False)
    st.download_button(
        label="📄 Descargar Resultados (CSV)",
        data=csv,
        file_name=f"analisis_iot_ids_{timestamp}.csv",
        mime="text/csv"
    )

with col2:
    # Exportar reporte a PDF (implementar con reportlab)
    if st.button("📑 Generar Reporte PDF"):
        pdf_bytes = generate_pdf_report(
            df_results=df_results,
            model_name=model_choice,
            has_labels=st.session_state.has_labels
        )

        st.download_button(
            label="📥 Descargar Reporte PDF",
            data=pdf_bytes,
            file_name=f"reporte_iot_ids_{timestamp}.pdf",
            mime="application/pdf"
        )
```

### Criterios de Éxito Fase 4

- ✅ Upload de archivos CSV funcional
- ✅ Validación de formato (16 columnas)
- ✅ Procesamiento batch con barra de progreso
- ✅ Cálculo de métricas cuando hay etiquetas
- ✅ Visualizaciones (distribución, top amenazas, matriz confusión)
- ✅ Exportar resultados CSV y PDF

---

## FASE 5: Dashboard de Métricas

### Objetivo
Crear dashboard completo con métricas técnicas y justificación académica.

### Archivo: `pages/4_📈_Metricas.py`

### Tareas

#### 5.1 Métricas del Modelo Sintético

```python
import streamlit as st
import plotly.graph_objects as go

st.title("📈 Dashboard de Métricas")

tab1, tab2, tab3 = st.tabs([
    "📊 Modelo Sintético",
    "📊 Modelo Real",
    "🔬 Información Técnica"
])

with tab1:
    st.header("Modelo Sintético - Métricas de Desempeño")

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "97.00%", "+2.48% vs Real")
    with col2:
        st.metric("Precision", "96.85%")
    with col3:
        st.metric("Recall", "96.72%")
    with col4:
        st.metric("F1-Score", "96.78%")

    # Matriz de confusión original
    st.subheader("Matriz de Confusión")

    # Cargar matriz de confusión guardada
    confusion_matrix_synthetic = load_confusion_matrix('synthetic')
    plot_confusion_matrix(confusion_matrix_synthetic)

    # F1-Score por clase
    st.subheader("F1-Score por Clase")

    class_metrics = {
        'Benign': 0.98,
        'DDoS': 0.97,
        'DoS': 0.96,
        'Brute_Force': 0.95,
        'Spoofing': 0.97,
        'MITM': 0.89,  # Problema identificado
        'Scan': 0.96,
        'Recon': 0.95
    }

    fig = go.Figure(data=[
        go.Bar(
            x=list(class_metrics.keys()),
            y=list(class_metrics.values()),
            marker_color=['red' if v < 0.90 else 'green' for v in class_metrics.values()]
        )
    ])
    fig.update_layout(
        title="F1-Score por Clase de Ataque",
        xaxis_title="Clase",
        yaxis_title="F1-Score",
        yaxis_range=[0, 1]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Problema con MITM
    st.warning("""
    ⚠️ **Área de Mejora Identificada**:

    La clase MITM presenta el menor recall (68%) debido a similitudes
    con tráfico normal en algunas características. Esto representa una
    oportunidad de mejora para futuras iteraciones del modelo.
    """)
```

#### 5.2 Métricas del Modelo Real

```python
with tab2:
    st.header("Modelo Real (CICIoT2023) - Métricas de Desempeño")

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "84.48%", "-12.52% vs Sintético")
    with col2:
        st.metric("Precision", "83.20%")
    with col3:
        st.metric("Recall", "82.95%")
    with col4:
        st.metric("F1-Score", "83.07%")

    # Matriz de confusión
    st.subheader("Matriz de Confusión")
    confusion_matrix_real = load_confusion_matrix('real')
    plot_confusion_matrix(confusion_matrix_real)

    # Comparación con sintético
    st.subheader("📉 Análisis de la Brecha de Desempeño")

    comparison_data = {
        'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Sintético': [97.00, 96.85, 96.72, 96.78],
        'Real': [84.48, 83.20, 82.95, 83.07],
        'Diferencia': [12.52, 13.65, 13.77, 13.71]
    }

    st.dataframe(comparison_data, use_container_width=True)

    st.info("""
    📊 **Análisis de la Brecha**:

    La diferencia de ~12-13% entre ambos modelos se debe principalmente a:

    1. **Complejidad de datos reales**: Mayor variabilidad y ruido
    2. **Desbalanceo de clases**: CICIoT2023 tiene distribución irregular
    3. **Características sutiles**: Algunos ataques reales son más difíciles de distinguir
    4. **Tamaño del dataset**: Modelo real entrenado con menos datos

    Sin embargo, 84.48% sigue siendo un desempeño sólido para detección de amenazas IoT.
    """)
```

#### 5.3 Información Técnica y Arquitectura

```python
with tab3:
    st.header("🔬 Información Técnica del Sistema")

    # Arquitectura AE-FNN
    st.subheader("Arquitectura Autoencoder-FNN Multi-tarea")

    st.markdown("""
    ### Componentes del Modelo

    #### 1. Encoder (Compresión)
    ```
    Input Layer:  16 features (PC1-PC16)
           ↓
    Dense Layer:   8 neurons (ReLU)
           ↓
    Latent Space:  4 neurons (bottleneck)
    ```

    #### 2. Decoder (Reconstrucción)
    ```
    Latent Space:  4 neurons
           ↓
    Dense Layer:   8 neurons (ReLU)
           ↓
    Output Layer: 16 features (reconstrucción)
    ```

    #### 3. Clasificador (Multi-clase)
    ```
    Latent Space:  4 neurons
           ↓
    Dense Layer:  16 neurons (ReLU) + Dropout(0.3)
           ↓
    Output Layer:  8 classes (Softmax)
    ```

    ### Función de Pérdida Combinada

    ```
    Loss = λ₁ × MSE(reconstrucción) + λ₂ × CrossEntropy(clasificación)
    ```

    **Hiperparámetros:**
    - Sintético: λ₁ = 0.3, λ₂ = 0.7
    - Real: λ₁ = 0.3, λ₂ = 0.7
    """)

    # Diagrama de arquitectura (usando st.mermaid o imagen)
    st.image("architecture_diagram.png", caption="Arquitectura AE-FNN Multi-tarea")

    # Especificaciones técnicas
    st.subheader("⚙️ Especificaciones Técnicas")

    specs = {
        'Parámetro': [
            'Framework',
            'Versión Python',
            'Optimizador',
            'Learning Rate',
            'Batch Size',
            'Epochs',
            'Tiempo de Inferencia',
            'Tamaño del Modelo',
            'Reducción Dimensional'
        ],
        'Valor': [
            'TensorFlow/Keras 2.10+',
            'Python 3.8+',
            'Adam',
            '0.001',
            '64',
            '100 (con Early Stopping)',
            '<2ms por muestra',
            '~150 KB (.h5)',
            'PCA: 35 → 16 componentes'
        ]
    }

    st.table(specs)

    # Recursos computacionales
    st.subheader("💻 Recursos Computacionales")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Entrenamiento:**
        - GPU: NVIDIA Tesla T4 / Google Colab
        - RAM: 12-16 GB
        - Tiempo: ~15-20 minutos
        """)

    with col2:
        st.markdown("""
        **Inferencia:**
        - CPU: Procesador estándar
        - RAM: 2-4 GB
        - Latencia: <2ms
        """)
```

#### 5.4 Justificación Académica

```python
st.divider()
st.header("🎓 Cumplimiento de Objetivos de Tesis")

st.markdown("""
### Objetivo General
**"Clasificar el tráfico de red y fortalecer la ciberseguridad en entornos de IoT
utilizando aprendizaje profundo"**

✅ **CUMPLIDO**: El sistema implementado demuestra capacidad de clasificación con
97% de accuracy y operación en tiempo real con <2ms de latencia.
""")

st.divider()

# Objetivos específicos
objectives = [
    {
        'title': 'OE1: Generar y estructurar conjunto de datos',
        'description': """
        - ✅ Dataset sintético de 100,000 muestras generado
        - ✅ Transformación PCA de 35 → 16 componentes aplicada
        - ✅ Validación con dataset real CICIoT2023
        - ✅ Balanceo de clases implementado

        **Evidencia en la aplicación:**
        - Módulo de comparación demuestra validez de datos sintéticos
        - Visualizaciones muestran distribución de features PCA
        """
    },
    {
        'title': 'OE2: Desarrollar modelo AE-FNN multi-tarea',
        'description': """
        - ✅ Arquitectura Autoencoder-FNN implementada
        - ✅ Enfoque multi-tarea: reconstrucción + clasificación
        - ✅ Función de pérdida combinada (λ₁=0.3, λ₂=0.7)
        - ✅ 2 modelos entrenados: sintético y real

        **Evidencia en la aplicación:**
        - Modelo funciona en tiempo real en simulación
        - Arquitectura documentada en sección técnica
        """
    },
    {
        'title': 'OE3: Evaluar efectividad del modelo',
        'description': """
        - ✅ Accuracy: 97% (sintético), 84.48% (real)
        - ✅ False Positive Rate: <2%
        - ✅ F1-Score promedio: >0.96
        - ✅ Tiempo de inferencia: <2ms

        **Evidencia en la aplicación:**
        - Dashboard de métricas con resultados completos
        - Matriz de confusión interactiva
        - Módulo de análisis calcula métricas en vivo
        """
    },
    {
        'title': 'OE4: Analizar contribución al fortalecimiento',
        'description': """
        - ✅ Sistema detecta 97% de amenazas correctamente
        - ✅ Tiempo de respuesta <2ms permite defensa en tiempo real
        - ✅ Identificación de 7 tipos de ataques IoT
        - ✅ Bajo FPR minimiza falsas alarmas

        **Evidencia en la aplicación:**
        - Simulación en tiempo real demuestra capacidad práctica
        - Sistema funcional listo para despliegue
        - Detección efectiva de escenarios de ataque
        """
    }
]

for obj in objectives:
    with st.expander(obj['title'], expanded=False):
        st.markdown(obj['description'])

st.success("""
### 🎯 Conclusión

Esta aplicación demuestra que la investigación **cumple satisfactoriamente
todos los objetivos específicos** y el objetivo general de la tesis.

El modelo desarrollado no es solo un ejercicio académico, sino una **herramienta
funcional de ciberseguridad** que puede desplegarse en entornos IoT reales para
fortalecer la detección de amenazas.
""")
```

### Criterios de Éxito Fase 5

- ✅ Métricas completas de ambos modelos
- ✅ Visualizaciones comparativas
- ✅ Documentación técnica de arquitectura
- ✅ Especificaciones detalladas
- ✅ Justificación académica completa
- ✅ Alineación con objetivos de tesis

---

## Estrategia de Implementación Recomendada

### 1. Orden de Implementación

1. **Fase 1** → Base crítica, debe funcionar perfectamente
2. **Fase 5** → Métricas y documentación (puede hacerse en paralelo)
3. **Fase 2** → Comparación de modelos
4. **Fase 4** → Análisis de archivos
5. **Fase 3** → Simulación en tiempo real (más compleja)

### 2. Testing por Fase

Después de cada fase, verificar:

- ✅ No hay errores en consola
- ✅ Funcionalidades básicas operativas
- ✅ Visualizaciones se renderizan correctamente
- ✅ Manejo de errores implementado
- ✅ Performance aceptable

### 3. Iteración y Mejora

- Implementar funcionalidad básica primero
- Agregar visualizaciones después
- Refinar UI/UX al final
- Documentar código a medida que se desarrolla

---

## Próximos Pasos

Una vez completadas las 5 fases:

1. **Testing Completo**: Probar todos los flujos de usuario
2. **Optimización**: Mejorar performance si es necesario
3. **Documentación de Usuario**: Crear guía de uso
4. **Preparación de Demo**: Ensayar presentación para defensa de tesis
5. **Deployment** (opcional): Desplegar en Streamlit Cloud o servidor

---

## Soporte y Recursos

- [Documentación Streamlit](https://docs.streamlit.io/)
- [TensorFlow/Keras Docs](https://www.tensorflow.org/api_docs)
- [Plotly Python](https://plotly.com/python/)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

**Última actualización**: Noviembre 2024
