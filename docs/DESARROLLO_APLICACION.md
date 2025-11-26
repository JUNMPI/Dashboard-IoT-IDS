# DOCUMENTACIÓN TÉCNICA: DESARROLLO DE APLICACIÓN IoT-IDS

## Índice
1. [Descripción General del Sistema](#1-descripción-general-del-sistema)
2. [Estructura del Proyecto](#2-estructura-del-proyecto)
3. [Arquitectura Lógica](#3-arquitectura-lógica)
4. [Arquitectura Física/Despliegue](#4-arquitectura-físicadespliegue)
5. [Stack Tecnológico Completo](#5-stack-tecnológico-completo)
6. [Endpoints/APIs](#6-endpointsapis)
7. [Flujo de Datos Detallado](#7-flujo-de-datos-detallado)
8. [Archivos de Configuración](#8-archivos-de-configuración)
9. [Patrones de Diseño](#9-patrones-de-diseño)
10. [Información Adicional Específica](#10-información-adicional-específica)

---

## 1. DESCRIPCIÓN GENERAL DEL SISTEMA

### 1.1 Propósito Principal

Sistema de **demostración web interactiva** para detección de intrusiones en redes IoT utilizando un modelo de Deep Learning multi-tarea (Autoencoder-FNN). Desarrollado como herramienta de visualización y análisis para tesis de pregrado en Universidad Señor de Sipán.

### 1.2 Tipo de Aplicación

**Aplicación Web Interactiva** construida con Streamlit (framework Python para dashboards de ML/Data Science).

### 1.3 Naturaleza del Sistema

**NO es un testbed IoT físico**. Es una aplicación de **inferencia y visualización** que:

- Trabaja con datos pre-procesados (16 componentes PCA)
- Usa modelos pre-entrenados (no entrena en tiempo real)
- Simula tráfico sintético para demostración
- Permite análisis por lotes de archivos CSV

### 1.4 Flujo Principal de Datos

```
┌─────────────────┐
│  ENTRADA DE     │
│  DATOS          │
└────────┬────────┘
         │
         ├─→ Simulación: Generador sintético (data_simulator.py)
         ├─→ Archivo CSV: Upload de usuario
         └─→ Manual: Componentes PCA individuales
         │
         ▼
┌─────────────────┐
│ PREPROCESAMIENTO│
│ (Normalización) │
│ StandardScaler  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MODELO AE-FNN  │
│ (TensorFlow)    │
│ - Encoder       │
│ - Decoder       │
│ - Classifier    │
└────────┬────────┘
         │
         ▼ (2 salidas)
┌─────────────────┐
│  POST-PROCESO   │
│ - Argmax        │
│ - LabelEncoder  │
│ - Confidence %  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  VISUALIZACIÓN  │
│  (Streamlit UI) │
│ - Gráficos      │
│ - Métricas      │
│ - Reportes PDF  │
└─────────────────┘
```

---

## 2. ESTRUCTURA DEL PROYECTO

### 2.1 Árbol de Directorios Completo

```
Dashboard IoT-IDS/
│
├── app.py                              # Punto de entrada principal (Home)
│
├── pages/                              # Módulos multi-página Streamlit
│   ├── 1_Comparacion_Modelos.py        # Comparación lado a lado
│   ├── 2_Tiempo_Real.py                # Simulación en tiempo real
│   ├── 3_Analisis_Archivo.py           # Análisis batch de CSV
│   ├── 4_Metricas.py                   # Dashboard de métricas
│   └── README.md
│
├── utils/                              # Módulos reutilizables
│   ├── __init__.py
│   ├── model_loader.py                 # Carga de modelos y predicción
│   ├── data_simulator.py               # Generación de tráfico sintético
│   ├── visualizations.py               # Gráficos Plotly/Matplotlib
│   ├── report_generator.py             # Generación de reportes PDF
│   ├── sidebar_component.py            # Componente compartido sidebar
│   └── README.md
│
├── models/                             # Artefactos de modelos entrenados
│   ├── synthetic/                      # Modelo sintético (97.24% acc)
│   │   ├── modelo_ae_fnn_iot_synthetic.h5
│   │   ├── scaler_synthetic.pkl
│   │   ├── label_encoder_synthetic.pkl
│   │   ├── class_names_synthetic.npy
│   │   ├── model_metadata_synthetic.json
│   │   ├── training_history.json
│   │   ├── confusion_matrix_normalized_synthetic.png
│   │   ├── metrics_per_class_barchart_synthetic.png
│   │   └── README.md
│   │
│   ├── real/                           # Modelo real CICIoT2023 (84.48% acc)
│   │   ├── modelo_ae_fnn_iot_REAL.h5
│   │   ├── scaler_REAL.pkl
│   │   ├── label_encoder_REAL.pkl
│   │   ├── class_names_REAL.npy
│   │   ├── model_metadata_REAL.json
│   │   ├── training_history_REAL.json
│   │   ├── confusion_matrix_normalized_REAL.png
│   │   ├── metrics_per_class_barchart_REAL.png
│   │   └── README.md
│   └── README.md
│
├── data/                               # Datasets de ejemplo
│   ├── dataset_pca_capa3_iot_ultra_fixed_100k_dataset.csv
│   ├── CICIoT2023_samples.csv
│   └── README.md
│
├── docs/                               # Documentación técnica
│   ├── ARQUITECTURA.md
│   ├── MODELOS.md
│   ├── IMPLEMENTACION.md
│   ├── OBJETIVOS_TESIS.md
│   ├── DESARROLLO_APLICACION.md        # Este documento
│   ├── screenshots/
│   └── gifs/
│
├── requirements.txt                    # Dependencias Python
├── Dockerfile                          # Contenedor Docker
├── .dockerignore
├── CLAUDE.md
├── README.md
├── INSTRUCCIONES.md
├── TESTING.md
└── .gitignore
```

### 2.2 Responsabilidades por Módulo

| Módulo | Responsabilidad | Componentes Clave |
|--------|----------------|-------------------|
| `app.py` | Página principal, inicialización session state, selector de modelos | `init_session_state()`, `render_sidebar()` |
| `pages/1_Comparacion_Modelos.py` | Carga ambos modelos, comparación lado a lado, reporte comparativo | `load_both_models()`, tasa concordancia |
| `pages/2_Tiempo_Real.py` | Simulación de tráfico en tiempo real, visualización temporal | Loop con `time.sleep()`, placeholders dinámicos |
| `pages/3_Analisis_Archivo.py` | Upload CSV, procesamiento batch, métricas, reportes PDF | `predict_batch()`, `generate_analysis_report()` |
| `pages/4_Metricas.py` | Dashboard de rendimiento, curvas entrenamiento, comparación modelos | Lectura JSON metadata, gráficos Plotly |
| `utils/model_loader.py` | Carga modelos TensorFlow, gestión compatibilidad versiones, inferencia | `load_model()`, `predict_sample()`, `predict_batch()` |
| `utils/data_simulator.py` | Generación tráfico sintético basado en patrones estadísticos PCA | `ATTACK_PATTERNS`, `generate_traffic_sample()` |
| `utils/visualizations.py` | Gráficos interactivos Plotly y estáticos Matplotlib | `plot_confusion_matrix()`, `plot_temporal_detections()` |
| `utils/report_generator.py` | PDFs con ReportLab (tablas, métricas, gráficos) | `generate_analysis_report()`, `generate_comparison_report()` |
| `utils/sidebar_component.py` | Sidebar consistente en todas las páginas, carga modelos | `render_sidebar()`, gestión session state |

---

## 3. ARQUITECTURA LÓGICA

### 3.1 Componentes Lógicos del Sistema

#### Capa de Presentación (Frontend)

| Componente | Responsabilidad | Comunicación | Datos Manejados |
|------------|----------------|--------------|-----------------|
| **Streamlit UI** | Renderizado interfaz web, widgets interactivos | Session State (st.session_state) | DataFrames pandas, diccionarios Python |
| **Sidebar Component** | Selector modelos, visualización metadata | Llamadas a `model_loader` | Metadata JSON, objetos modelo |
| **Pages** | Vistas multi-página (4 páginas) | Session State global | Resultados predicciones, configuración |

#### Capa de Lógica de Negocio

| Componente | Responsabilidad | Comunicación | Datos Manejados |
|------------|----------------|--------------|-----------------|
| **Model Loader** | Gestión ciclo vida modelos, caché Streamlit | TensorFlow, pickle, numpy | Modelos H5, scalers PKL, arrays numpy |
| **Data Simulator** | Generación tráfico sintético IoT | Devuelve tuplas (sample, label) | Arrays numpy (16,), strings |
| **Visualizations** | Transformación datos → gráficos | Plotly/Matplotlib APIs | DataFrames, listas, diccionarios |
| **Report Generator** | Exportación resultados a PDF | ReportLab, Matplotlib | PDF bytes, imágenes PNG |

#### Capa de Servicios ML

| Componente | Responsabilidad | Comunicación | Datos Manejados |
|------------|----------------|--------------|-----------------|
| **Modelo AE-FNN** | Inferencia (reconstrucción + clasificación) | TensorFlow Keras API | Tensores (batch, 16) → (batch, 16) + (batch, 6) |
| **StandardScaler** | Normalización features PCA | scikit-learn transform() | Arrays numpy (n, 16) |
| **LabelEncoder** | Codificación/decodificación etiquetas | inverse_transform() | Índices → strings clases |

#### Capa de Datos

| Componente | Responsabilidad | Comunicación | Datos Manejados |
|------------|----------------|--------------|-----------------|
| **Model Artifacts** | Almacenamiento modelos entrenados | Disco local (models/) | H5, PKL, NPY, JSON, PNG |
| **Session State** | Estado temporal sesión usuario | Memoria RAM Streamlit | Diccionarios Python, objetos complejos |
| **Datasets CSV** | Datos estáticos para análisis | Pandas read_csv | DataFrames con 16+ columnas |

---

## 4. ARQUITECTURA FÍSICA/DESPLIEGUE

### 4.1 Topología de Despliegue

```
┌─────────────────────────────────────────────────────────┐
│                    CLIENTE (Navegador)                   │
│  - Chrome/Firefox/Edge                                   │
│  - JavaScript habilitado                                 │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP (8501)
                        ▼
┌─────────────────────────────────────────────────────────┐
│           SERVIDOR STREAMLIT (Python)                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Streamlit Server (Tornado)                     │   │
│  │  - Puerto: 8501                                 │   │
│  │  - WebSocket para actualizaciones en vivo      │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  App.py + Pages (Lógica de negocio)            │   │
│  │  - Session State compartido                     │   │
│  │  - Multi-threading para páginas                 │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Utils (Módulos Python)                         │   │
│  │  - model_loader: Caché @st.cache_resource      │   │
│  │  - visualizations: Plotly/Matplotlib           │   │
│  │  - data_simulator: Numpy/Random                │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  TensorFlow/Keras (ML Runtime)                  │   │
│  │  - CPU inference (no GPU required)              │   │
│  │  - Modelos cargados en RAM                      │   │
│  └─────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────┘
                        │ File I/O
                        ▼
┌─────────────────────────────────────────────────────────┐
│              SISTEMA DE ARCHIVOS LOCAL                   │
│  - models/synthetic/ (H5, PKL, NPY, JSON, PNG)          │
│  - models/real/ (H5, PKL, NPY, JSON, PNG)               │
│  - data/ (CSV datasets)                                  │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Tecnologías por Capa

| Capa | Componente | Tecnología | Versión | Propósito |
|------|-----------|------------|---------|-----------|
| **Frontend** | UI Framework | Streamlit | ≥1.25.0 | Interfaz web interactiva |
| **Frontend** | Visualización | Plotly | ≥5.14.0 | Gráficos interactivos |
| **Frontend** | Visualización | Matplotlib | ≥3.6.0 | Gráficos estáticos |
| **Frontend** | Visualización | Seaborn | ≥0.12.0 | Matrices de confusión |
| **Backend** | Runtime | Python | 3.8+ | Lenguaje principal |
| **Backend** | Web Server | Tornado | (incluido Streamlit) | Servidor HTTP/WebSocket |
| **Backend** | ML Framework | TensorFlow | 2.10-2.15 | Inferencia modelos |
| **Backend** | ML Framework | Keras | 2.15.0 | API modelos AE-FNN |
| **Backend** | Preprocesamiento | scikit-learn | ≥1.2.0 | StandardScaler, LabelEncoder |
| **Backend** | Computación | NumPy | <2.0.0 | Arrays numéricos |
| **Backend** | Datos | Pandas | ≥1.5.0 | DataFrames, CSV |
| **Backend** | PDF | ReportLab | ≥3.6.0 | Generación reportes |
| **Backend** | Imágenes | Pillow | ≥9.5.0 | Procesamiento imágenes |
| **Deployment** | Contenedor | Docker | - | Despliegue containerizado |
| **Storage** | Modelos | Formato H5 | - | TensorFlow SavedModel |
| **Storage** | Objetos Python | Pickle | - | Serialización scalers/encoders |

### 4.3 Comunicación entre Componentes

| Origen | Destino | Protocolo/Mecanismo | Datos Transferidos |
|--------|---------|---------------------|-------------------|
| Navegador | Streamlit Server | HTTP/WebSocket (Puerto 8501) | Eventos UI, comandos |
| Streamlit | Pages | Session State (dict Python) | Estado global aplicación |
| Pages | Utils | Llamadas función Python | Parámetros, arrays numpy |
| Utils | TensorFlow | API Keras (predict) | Tensores numpy |
| Utils | Filesystem | File I/O (open, pickle.load) | Binarios PKL, H5 |
| Visualizations | Frontend | JSON (Plotly), Base64 (Matplotlib) | Gráficos serializados |

### 4.4 Base de Datos

**NO hay base de datos persistente**. Todo el estado se maneja:

- **En memoria**: `st.session_state` (temporal por sesión)
- **En disco**: Archivos estáticos (modelos, datasets)

---

## 5. STACK TECNOLÓGICO COMPLETO

### 5.1 Frontend (UI)

| Categoría | Tecnología | Versión | Uso en el Proyecto |
|-----------|-----------|---------|-------------------|
| Framework UI | **Streamlit** | ≥1.25.0 | Sistema multi-página, widgets, layout |
| Gráficos Interactivos | **Plotly** | ≥5.14.0 | Scatter plots, bar charts, pie charts, radar charts, gauges |
| Gráficos Estáticos | **Matplotlib** | ≥3.6.0 | Confusion matrices, exportación PDF |
| Gráficos Estadísticos | **Seaborn** | ≥0.12.0 | Heatmaps para matrices de confusión |

### 5.2 Backend (Lógica)

| Categoría | Tecnología | Versión | Uso en el Proyecto |
|-----------|-----------|---------|-------------------|
| Lenguaje | **Python** | 3.8+ | Todo el código backend |
| Web Server | **Tornado** | (incluido) | Servidor Streamlit HTTP/WebSocket |

### 5.3 Machine Learning

| Categoría | Tecnología | Versión | Uso en el Proyecto |
|-----------|-----------|---------|-------------------|
| Framework ML | **TensorFlow** | 2.10-2.15 | Inferencia modelos AE-FNN |
| API Modelos | **Keras** | 2.15.0 | Definición arquitectura, load_model |
| Preprocesamiento | **scikit-learn** | ≥1.2.0 | StandardScaler, LabelEncoder, métricas (confusion_matrix, classification_report) |

### 5.4 Procesamiento de Datos

| Categoría | Tecnología | Versión | Uso en el Proyecto |
|-----------|-----------|---------|-------------------|
| Arrays Numéricos | **NumPy** | <2.0.0 | Manipulación datos PCA, predicciones |
| DataFrames | **Pandas** | ≥1.5.0 | Lectura CSV, resultados tabulares |
| Fechas | **python-dateutil** | ≥2.8.0 | Timestamps en reportes |

### 5.5 Visualización

| Categoría | Tecnología | Versión | Uso en el Proyecto |
|-----------|-----------|---------|-------------------|
| Gráficos Web | **Plotly Express** | (incluido) | Gráficos rápidos distribuciones |
| Gráficos Web | **Plotly Graph Objects** | (incluido) | Gráficos personalizados temporales |
| Gráficos Científicos | **Matplotlib** | ≥3.6.0 | Confusion matrices, gráficos para PDF |
| Heatmaps | **Seaborn** | ≥0.12.0 | Matrices de confusión visuales |

### 5.6 Generación de Reportes

| Categoría | Tecnología | Versión | Uso en el Proyecto |
|-----------|-----------|---------|-------------------|
| PDF | **ReportLab** | ≥3.6.0 | Reportes análisis, tablas, gráficos |
| Imágenes | **Pillow (PIL)** | ≥9.5.0 | Procesamiento imágenes para PDF |

### 5.7 Deployment

| Categoría | Tecnología | Versión | Uso en el Proyecto |
|-----------|-----------|---------|-------------------|
| Contenedor | **Docker** | - | Dockerfile multi-stage con Python 3.11-slim |
| Orquestación | (Manual) | - | Ejecutar vía `streamlit run app.py` |

### 5.8 Formato de Archivos

| Tipo | Extensión | Librería | Propósito |
|------|-----------|----------|-----------|
| Modelos TensorFlow | `.h5` | tensorflow.keras | Pesos + arquitectura modelo |
| Objetos Python | `.pkl` | pickle | StandardScaler, LabelEncoder |
| Arrays NumPy | `.npy` | numpy | class_names |
| Metadata | `.json` | json | Métricas entrenamiento, configuración |
| Datos tabulares | `.csv` | pandas | Datasets PCA |
| Imágenes | `.png` | matplotlib/PIL | Confusion matrices, gráficos |
| Reportes | `.pdf` | reportlab | Reportes exportados |

---

## 6. ENDPOINTS/APIs

### 6.1 Páginas Accesibles (Rutas Streamlit)

**No hay endpoints REST tradicionales**. Streamlit maneja la comunicación vía WebSocket automáticamente.

| Ruta | Descripción | Funciones Clave |
|------|-------------|-----------------|
| `/` | Página principal (Home) | Selector modelos, arquitectura modelo, instrucciones |
| `/Comparacion_Modelos` | Comparación lado a lado | `load_both_models()`, comparación predicciones |
| `/Tiempo_Real` | Simulación tiempo real | Loop simulación, `generate_scenario_traffic()` |
| `/Analisis_Archivo` | Upload y análisis CSV | `predict_batch()`, reportes PDF |
| `/Metricas` | Dashboard métricas | Visualización metadata, curvas entrenamiento |

### 6.2 Funciones de Predicción Internas

#### `predict_sample(model, scaler, encoder, class_names, sample)`

**Ubicación**: `utils/model_loader.py:378`

- **Entrada**: Array numpy `(16,)` con componentes PCA raw
- **Salida**: Tupla `(prediction: str, probabilities: array(6,), confidence: float)`
- **Ejemplo**:

```python
sample = np.random.randn(16)  # Simulado
pred, probs, conf = predict_sample(model, scaler, encoder, names, sample)
# pred: "ddos"
# probs: [0.01, 0.89, 0.02, 0.03, 0.02, 0.03]
# conf: 89.3
```

#### `predict_batch(model, scaler, encoder, class_names, samples, batch_size=32)`

**Ubicación**: `utils/model_loader.py:429`

- **Entrada**: Array numpy `(n, 16)` con múltiples muestras
- **Salida**: Tupla `(predictions: array(n,), confidences: array(n,))`
- **Ejemplo**:

```python
samples = np.random.randn(100, 16)
preds, confs = predict_batch(model, scaler, encoder, names, samples)
# preds: array(['normal', 'ddos', ...], dtype='<U12')
# confs: array([95.2, 87.3, ...])
```

---

## 7. FLUJO DE DATOS DETALLADO

### 7.1 Entrada de Datos (3 vías)

#### Opción A: Simulación en Tiempo Real

**Archivo**: `pages/2_Tiempo_Real.py`

```python
# Paso 1.1: Usuario selecciona escenario
scenario = 'under_attack'  # 80% DDoS

# Paso 1.2: Generador crea timeline
traffic = generate_scenario_traffic(scenario, duration=30)
# Retorna: [(timestamp, sample_16d, true_label), ...]

# Paso 1.3: Extrae sample
timestamp, sample, true_label = traffic[0]
# sample: array([3.2, 2.5, ..., 0.1])  # 16 valores PCA
```

#### Opción B: Upload CSV

**Archivo**: `pages/3_Analisis_Archivo.py`

```python
# Paso 1.1: Usuario carga archivo
uploaded_file = st.file_uploader(...)

# Paso 1.2: Pandas lee CSV
df = pd.read_csv(uploaded_file)

# Paso 1.3: Extrae columnas PCA
feature_cols = ['PC1', 'PC2', ..., 'PC16']
X = df[feature_cols].values  # shape: (n, 16)
```

#### Opción C: Manual

```python
# Paso 1: Array numpy directo
sample = np.array([0.5, -1.2, 0.8, ..., -0.3])  # 16 valores
```

### 7.2 Preprocesamiento

**Archivo**: `utils/model_loader.py`

```python
# Paso 2.1: Validar dimensiones
assert sample.shape == (16,), "Debe tener 16 features PCA"

# Paso 2.2: Reshape para scaler
sample_reshaped = sample.reshape(1, -1)  # (16,) → (1, 16)

# Paso 2.3: Normalización con StandardScaler
sample_scaled = scaler.transform(sample_reshaped)
# Aplica: (x - mean) / std para cada feature
# sample_scaled: array([[ 0.27, -1.89, 0.33, ..., -0.33]])
```

### 7.3 Inferencia con Modelo

```python
# Paso 3.1: Predicción (silenciosa)
predictions = model.predict(sample_scaled, verbose=0)
# Modelo tiene 2 salidas (multi-task):
# predictions[0]: Reconstrucción (16 valores)
# predictions[1]: Clasificación (6 probabilidades)

# Paso 3.2: Extraer output clasificación
class_probs = predictions[1]
# class_probs: array([[0.01, 0.89, 0.02, 0.03, 0.02, 0.03]])
#                     [brute, ddos, mitm, normal, scan, spoof]
```

### 7.4 Post-Procesamiento

```python
# Paso 4.1: Argmax para clase predicha
predicted_idx = np.argmax(class_probs[0])
# predicted_idx: 1 (índice de ddos)

# Paso 4.2: Decodificar con LabelEncoder
predicted_class = label_encoder.inverse_transform([predicted_idx])[0]
# predicted_class: "ddos"

# Paso 4.3: Calcular confianza
confidence = float(np.max(class_probs[0]) * 100)
# confidence: 89.3

# Paso 4.4: Retornar resultado
return predicted_class, class_probs[0], confidence
# ("ddos", array([0.01, 0.89, ...]), 89.3)
```

### 7.5 Entrega de Resultados

#### Tiempo Real

```python
# Paso 5.1: Actualizar métricas en vivo
st.metric("Amenaza Detectada", prediction)
st.metric("Confianza", f"{confidence:.1f}%")

# Paso 5.2: Actualizar gráfico temporal
fig = plot_temporal_detections(times, preds, confs)
st.plotly_chart(fig)

# Paso 5.3: Alertas visuales
if prediction != 'normal':
    severity = get_threat_severity(prediction)
    st.error(f"[{severity.upper()}] Amenaza: {prediction}")
```

#### Análisis Batch

```python
# Paso 5.1: Tabla de resultados
results_df = pd.DataFrame({
    'prediction': predictions,
    'confidence': confidences,
    'true_label': y_true
})
st.dataframe(results_df)

# Paso 5.2: Métricas agregadas
accuracy = (results_df['prediction'] == results_df['true_label']).mean()
st.metric("Accuracy", f"{accuracy*100:.2f}%")

# Paso 5.3: Visualizaciones
fig_dist = plot_class_distribution(predictions)
st.plotly_chart(fig_dist)

# Paso 5.4: Exportar PDF
pdf_bytes = generate_analysis_report(results_df, model_name, metadata)
st.download_button("Descargar PDF", pdf_bytes, "reporte.pdf")
```

---

## 8. ARCHIVOS DE CONFIGURACIÓN

### 8.1 `requirements.txt`

**Ubicación**: `C:\Dashboard IoT-IDS\requirements.txt`

```txt
# Core Python (3.8+)

# Machine Learning - TensorFlow/Keras
tensorflow>=2.10.0,<2.16.0
keras==2.15.0

# Machine Learning - Scikit-learn
scikit-learn>=1.2.0

# Numerical Computing
numpy>=1.23.0,<2.0.0

# Data Processing
pandas>=1.5.0

# Visualization
plotly>=5.14.0
matplotlib>=3.6.0
seaborn>=0.12.0

# Web Framework
streamlit>=1.25.0

# Report Generation
reportlab>=3.6.0
Pillow>=9.5.0

# Utilities
python-dateutil>=2.8.0
```

### 8.2 Versiones Críticas

| Dependencia | Versión Requerida | Razón |
|-------------|-------------------|-------|
| TensorFlow | 2.10-2.15 | Compatibilidad formato H5 modelos guardados |
| NumPy | <2.0.0 | Incompatibilidad NumPy 2.0 con TensorFlow 2.x |
| Streamlit | ≥1.25.0 | Funciones multi-página, session state avanzado |
| Keras | 2.15.0 | Sincronización con TensorFlow 2.15 |

### 8.3 `Dockerfile`

**Ubicación**: `C:\Dashboard IoT-IDS\Dockerfile`

```dockerfile
FROM python:3.11-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Copiar y instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicación
COPY . .

# Crear directorios
RUN mkdir -p data models/synthetic models/real

EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 8.4 `model_metadata_synthetic.json`

**Ubicación**: `models/synthetic/model_metadata_synthetic.json`

```json
{
    "model_type": "Autoencoder-FNN Multi-Task",
    "dataset_type": "Synthetic IoT Dataset",
    "num_classes": 6,
    "class_names": [
        "brute_force",
        "ddos",
        "mitm",
        "normal",
        "scan",
        "spoofing"
    ],
    "input_dim": 16,
    "latent_dim": 6,
    "lambda_peso_reconstruction": 0.3,
    "lambda_peso_classification": 0.7,
    "training_samples": 70000,
    "validation_samples": 15000,
    "test_samples": 15000,
    "epochs_trained": 53,
    "batch_size": 64,
    "optimizer": "adam",
    "test_accuracy": 0.9724,
    "test_loss_total": 0.1361,
    "test_loss_reconstruction": 0.3009,
    "test_loss_classification": 0.0655
}
```

---

## 9. PATRONES DE DISEÑO

### 9.1 Patrones Arquitectónicos

#### 1. Model-View-Controller (MVC) Modificado

- **Model**: `utils/model_loader.py`, artefactos en `models/`
- **View**: Páginas Streamlit (`pages/`)
- **Controller**: Funciones en `utils/`, Session State

#### 2. Repository Pattern

- `model_loader.py`: Abstrae acceso a modelos H5, PKL
- Funciones `load_model()`, `check_model_files()`

#### 3. Singleton (vía Caché Streamlit)

- `@st.cache_resource` en `load_synthetic_model()`, `load_real_model()`
- Asegura una sola instancia de modelo en memoria por tipo

#### 4. Strategy Pattern

- `generate_scenario_traffic(scenario)` con estrategias: 'normal', 'under_attack', 'scanning', 'mixed'
- Diferentes generadores de tráfico según escenario

#### 5. Factory Pattern

- `load_model(model_type)` retorna diferentes artifacts según 'synthetic' o 'real'

### 9.2 Patrones de Codificación

#### 6. Facade Pattern

- `predict_sample()`: Simplifica flujo completo (scale → predict → decode)
- `generate_analysis_report()`: Oculta complejidad ReportLab

#### 7. Template Method

- `render_sidebar()`: Componente reutilizable en todas las páginas
- Todas las páginas siguen plantilla: config → check model → procesar → visualizar

#### 8. Observer (implícito)

- Streamlit re-renderiza UI automáticamente cuando cambia `session_state`
- Reactivo: cambio modelo → re-carga → actualiza UI

### 9.3 Prácticas de Código Limpio

#### 9. Separation of Concerns

- Módulos independientes: visualización, simulación, inferencia
- Cada archivo `utils/` tiene una responsabilidad única

#### 10. DRY (Don't Repeat Yourself)

- `sidebar_component.py`: Evita duplicar código sidebar en cada página
- Funciones reutilizables en `visualizations.py`

---

## 10. INFORMACIÓN ADICIONAL ESPECÍFICA

### 10.1 Clases del Modelo (6 categorías)

| Clase | Descripción | Severidad | Patrón PCA Distintivo |
|-------|-------------|-----------|----------------------|
| `normal` | Tráfico benigno | Normal | Media ≈0, varianza baja |
| `brute_force` | Ataques de fuerza bruta | Alta | Picos en PC5-PC7 |
| `ddos` | Ataques de denegación de servicio | Crítica | Valores altos PC1-PC4 |
| `mitm` | Man-in-the-Middle | Alta | Distribución dispersa |
| `scan` | Escaneo de puertos | Baja | Picos en PC8-PC9 |
| `spoofing` | Suplantación IP/MAC | Media | Variación en PC2-PC5 |

### 10.2 Arquitectura Modelo AE-FNN

```
INPUT (16 PCA components)
    ↓
┌────────────────────┐
│ ENCODER            │
│  Dense(12, LeakyReLU) │
│  Dense(8, LeakyReLU)  │
│  Dense(6, LeakyReLU)  │ ← LATENT SPACE
└────────────────────┘
    ↓               ↓
    │               └──────────────────┐
    │                                  │
┌────────────────────┐    ┌────────────────────┐
│ DECODER            │    │ CLASSIFIER (FNN)   │
│  Dense(8, LeakyReLU)  │    │  Dense(64, LeakyReLU) │
│  Dense(12, LeakyReLU) │    │  Dense(32, LeakyReLU) │
│  Dense(16, linear)    │    │  Dense(6, softmax)    │
└────────────────────┘    └────────────────────┘
    ↓                         ↓
OUTPUT 1:                OUTPUT 2:
Reconstrucción (16)      Clasificación (6)
```

**Función de Pérdida**:

```
Loss_total = 0.3 × MSE(reconstrucción) + 0.7 × CrossEntropy(clasificación)
```

### 10.3 Comparación Modelos

| Característica | Modelo Sintético | Modelo Real (CICIoT2023) |
|----------------|------------------|--------------------------|
| **Accuracy** | 97.24% | 84.48% |
| **Muestras entrenamiento** | 70,000 | Variable (real dataset) |
| **Época óptima** | 53 | Variable |
| **Dataset** | Sintético balanceado PCA | CICIoT2023 (world real) |
| **Fortaleza** | Alta precisión, demo limpia | Robusto ante variaciones reales |
| **Debilidad** | Overfitting a patrones sintéticos | Menor accuracy, más noise |
| **Uso recomendado** | Demostración académica | Validación en entornos reales |

---

## RESUMEN EJECUTIVO

### Naturaleza del Sistema

Este sistema **NO es un testbed IoT físico**. Es una **aplicación web de demostración/visualización** que implementa:

1. **Dashboard interactivo Streamlit** con 4 páginas funcionales
2. **Modelo de ML multi-tarea** (Autoencoder-FNN) pre-entrenado en TensorFlow
3. **3 modos de operación**: Simulación tiempo real, análisis batch CSV, comparación modelos
4. **Pipeline completo**: Generación datos → Preprocesamiento → Inferencia → Visualización → Exportación PDF
5. **Stack tecnológico moderno**: Python 3.8+, TensorFlow 2.15, Streamlit 1.25, Plotly, ReportLab
6. **Arquitectura MVC** con separación clara: Presentación (Streamlit) → Lógica (utils) → ML (TensorFlow) → Datos (archivos estáticos)
7. **Despliegue containerizado** con Docker para portabilidad

### Contribución Académica

Sistema completo end-to-end para demostrar viabilidad de detección de intrusiones IoT con Deep Learning multi-tarea, con interfaz visual accesible para usuarios no técnicos.

### Alcance

- **Entrada**: Datos pre-procesados (16 componentes PCA)
- **Procesamiento**: Inferencia con modelos pre-entrenados
- **Salida**: Predicciones, visualizaciones interactivas, reportes PDF
- **NO incluye**: Captura de tráfico real, entrenamiento en vivo, dispositivos IoT físicos

---

**Documento generado para**: Tesis de Grado - Universidad Señor de Sipán
**Fecha**: 2025
**Sistema**: Dashboard IoT-IDS v1.0
