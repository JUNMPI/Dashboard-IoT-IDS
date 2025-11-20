# Arquitectura del Sistema IoT-IDS

## Diagrama de Arquitectura General

```
┌──────────────────────────────────────────────────────────────────────┐
│                         DASHBOARD IoT-IDS                            │
│                    (Streamlit Web Application)                       │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
        v                                         v
┌───────────────────┐                  ┌──────────────────────┐
│   MODEL LOADER    │                  │   DATA SIMULATOR     │
│   (utils/)        │                  │   (utils/)           │
├───────────────────┤                  ├──────────────────────┤
│ - Load models     │                  │ - Generate samples   │
│ - Load scalers    │                  │ - Create scenarios   │
│ - Load encoders   │                  │ - Simulate traffic   │
└────────┬──────────┘                  └──────────┬───────────┘
         │                                        │
         │                                        │
         v                                        v
┌─────────────────────────────────────────────────────────────┐
│                    PREDICTION ENGINE                        │
│                                                             │
│  ┌──────────────┐          ┌──────────────────────────┐   │
│  │   Scaler     │  ──────> │  Autoencoder-FNN Model   │   │
│  │ (Normalize)  │          │                          │   │
│  └──────────────┘          │  [Encoder → Latent Space]│   │
│                            │         ↓        ↓       │   │
│                            │    [Decoder] [Classifier]│   │
│                            └───────────┬──────────────┘   │
│                                        │                   │
│                                        v                   │
│                            ┌──────────────────────┐       │
│                            │  Label Encoder       │       │
│                            │  (Decode classes)    │       │
│                            └──────────┬───────────┘       │
└───────────────────────────────────────┼───────────────────┘
                                        │
                                        v
                            ┌──────────────────────┐
                            │   PREDICTIONS        │
                            │   - Class labels     │
                            │   - Confidences      │
                            │   - Threat detection │
                            └──────────┬───────────┘
                                       │
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        │                                                              │
        v                              v                              v
┌───────────────┐           ┌──────────────────┐         ┌─────────────────┐
│ VISUALIZATIONS│           │ REPORT GENERATOR │         │   WEB PAGES     │
│ (Plotly/MPL)  │           │ (PDF Reports)    │         │   (Streamlit)   │
├───────────────┤           ├──────────────────┤         ├─────────────────┤
│ - Confusion   │           │ - Analysis       │         │ - Comparacion   │
│ - Metrics     │           │ - Statistics     │         │ - Tiempo Real   │
│ - Time series │           │ - Charts         │         │ - Analisis      │
│ - Heatmaps    │           │ - Tables         │         │ - Metricas      │
└───────────────┘           └──────────────────┘         └─────────────────┘
```

## Flujo de Datos Detallado

### 1. Entrada de Datos

```
┌──────────────────────┐
│   INPUT SOURCES      │
├──────────────────────┤
│ 1. Simulated data    │ ──┐
│ 2. CSV files         │   │
│ 3. Real-time stream  │   │
└──────────────────────┘   │
                           │
                           v
              ┌────────────────────────┐
              │   DATA PREPROCESSING   │
              ├────────────────────────┤
              │ 1. Shape validation    │
              │    (Must be 16 dims)   │
              │ 2. StandardScaler      │
              │    (Mean=0, Std=1)     │
              │ 3. Format for model    │
              └───────────┬────────────┘
                          │
                          v
                  [16 normalized values]
```

### 2. Arquitectura del Modelo Autoencoder-FNN

```
INPUT: X (16 components from PCA)
  │
  │  ┌─────────────────────────────────────────────────┐
  │  │             ENCODER BRANCH                      │
  │  │                                                  │
  └──┤  Dense(12, activation=None)                     │
     │  LeakyReLU(alpha=0.3)                           │
     │  Dense(8, activation=None)                      │
     │  LeakyReLU(alpha=0.3)                           │
     │  Dense(6, activation=None)    [LATENT SPACE]    │
     │  LeakyReLU(alpha=0.3)                           │
     └────────────────┬────────────────────────────────┘
                      │
                      │ (6-dimensional representation)
                      │
         ┌────────────┴────────────────┐
         │                             │
         v                             v
  ┌──────────────────┐       ┌─────────────────────┐
  │ DECODER BRANCH   │       │ CLASSIFIER BRANCH   │
  │                  │       │                     │
  │ Dense(8)         │       │ Dense(64)           │
  │ LeakyReLU(0.3)   │       │ LeakyReLU(0.3)      │
  │ Dense(12)        │       │ Dense(32)           │
  │ LeakyReLU(0.3)   │       │ LeakyReLU(0.3)      │
  │ Dense(16)        │       │ Dense(6, softmax)   │
  │ [linear]         │       │                     │
  └────────┬─────────┘       └──────────┬──────────┘
           │                            │
           v                            v
    RECONSTRUCTION              CLASSIFICATION
    X' (16 values)              Y' (6 probabilities)

    MSE Loss                    Categorical CE Loss
         │                              │
         └──────────┬───────────────────┘
                    v
            TOTAL LOSS = 0.3×MSE + 0.7×CE
```

### 3. Pipeline de Predicción

```
[RAW DATA]
    │
    v
[VALIDATE SHAPE]
    │ (must be N×16)
    v
[NORMALIZE]
    │ (StandardScaler)
    v
[MODEL INPUT]
    │
    ├──> [ENCODER] ──> [DECODER] ──> Reconstruction
    │                                      │
    └──> [ENCODER] ──> [CLASSIFIER] ──> Probabilities
                                           │
                                           v
                                    [ARGMAX + THRESHOLD]
                                           │
                                           v
                                    ┌──────────────┐
                                    │  PREDICTION  │
                                    ├──────────────┤
                                    │ - Class name │
                                    │ - Confidence │
                                    │ - Is threat? │
                                    └──────────────┘
```

## Estructura de Archivos del Sistema

```
Dashboard-IoT-IDS/
│
├── app.py                          [MAIN ENTRY]
│   └─> Carga sidebar
│   └─> Inicializa session_state
│   └─> Muestra arquitectura
│
├── pages/                          [MULTIPAGE APP]
│   ├── 1_Comparacion_Modelos.py
│   │   ├─> Carga AMBOS modelos (synthetic + real)
│   │   ├─> Genera muestras
│   │   ├─> Compara predicciones
│   │   └─> Calcula concordancia
│   │
│   ├── 2_Tiempo_Real.py
│   │   ├─> Genera tráfico simulado
│   │   ├─> Predicción en tiempo real
│   │   ├─> Gráficos temporales
│   │   └─> Métricas live
│   │
│   ├── 3_Analisis_Archivo.py
│   │   ├─> Upload CSV
│   │   ├─> Validación formato
│   │   ├─> Predicción batch
│   │   └─> Reporte PDF
│   │
│   └── 4_Metricas.py
│       ├─> Carga metadata
│       ├─> Muestra métricas de entrenamiento
│       ├─> Confusion matrix
│       └─> Comparación modelos
│
├── utils/                          [CORE MODULES]
│   ├── __init__.py
│   │
│   ├── model_loader.py             [MODEL MANAGEMENT]
│   │   ├─> ModelLoader class
│   │   ├─> _load_model_with_compatibility()
│   │   ├─> _load_pickle() con fallbacks
│   │   ├─> predict() method
│   │   └─> predict_proba() method
│   │
│   ├── data_simulator.py           [DATA GENERATION]
│   │   ├─> DataSimulator class
│   │   ├─> generate_sample()
│   │   ├─> generate_batch()
│   │   └─> generate_scenario()
│   │
│   ├── visualizations.py           [CHARTS]
│   │   ├─> create_confusion_matrix()
│   │   ├─> create_metrics_chart()
│   │   ├─> create_time_series()
│   │   └─> create_distribution()
│   │
│   └── report_generator.py         [PDF EXPORT]
│       ├─> ReportGenerator class
│       ├─> generate_report()
│       └─> _create_charts()
│
└── models/                         [TRAINED MODELS]
    ├── synthetic/
    │   ├── modelo_ae_fnn_iot_synthetic.h5
    │   ├── scaler_synthetic.pkl
    │   ├── label_encoder_synthetic.pkl
    │   ├── class_names_synthetic.npy
    │   └── model_metadata_synthetic.json
    │
    └── real/
        ├── modelo_ae_fnn_iot_REAL.h5
        ├── scaler_REAL.pkl
        ├── label_encoder_REAL.pkl
        ├── class_names_REAL.npy
        └── model_metadata_REAL.json
```

## Flujo de Sesión de Usuario

```
[USER OPENS BROWSER]
        │
        v
[http://localhost:8501]
        │
        v
┌───────────────────────┐
│    app.py LOADS       │
│                       │
│ 1. Import utils       │
│ 2. Set page config    │
│ 3. Initialize sidebar │
│ 4. Load model         │
│    (from sidebar)     │
└───────┬───────────────┘
        │
        v
┌───────────────────────────────────┐
│  SESSION STATE INITIALIZED        │
│                                   │
│  st.session_state = {             │
│    'model_type': 'synthetic',     │
│    'model_loader': ModelLoader,   │
│    'current_model': Model,        │
│    'data': None,                  │
│    'predictions': None            │
│  }                                │
└──────────┬────────────────────────┘
           │
           v
    ┌──────────────┐
    │ USER CHOICE  │
    └──┬───────────┘
       │
       ├──> [Comparación]  ──> pages/1_Comparacion_Modelos.py
       │                        │
       │                        ├─> Load synthetic model
       │                        ├─> Load real model
       │                        ├─> Generate samples
       │                        ├─> Predict with both
       │                        └─> Show comparison table
       │
       ├──> [Tiempo Real]  ──> pages/2_Tiempo_Real.py
       │                        │
       │                        ├─> Select scenario
       │                        ├─> Generate stream
       │                        ├─> Predict continuously
       │                        └─> Update charts live
       │
       ├──> [Análisis]  ──> pages/3_Analisis_Archivo.py
       │                     │
       │                     ├─> Upload CSV
       │                     ├─> Validate format
       │                     ├─> Process batch
       │                     ├─> Show results
       │                     └─> Generate PDF
       │
       └──> [Métricas]  ──> pages/4_Metricas.py
                             │
                             ├─> Load metadata
                             ├─> Show training metrics
                             ├─> Display confusion matrix
                             └─> Compare models
```

## Clases Detectadas y Características

### 6 Clases de Tráfico IoT

```
┌────────────────────────────────────────────────────────┐
│                   CLASE: NORMAL                        │
├────────────────────────────────────────────────────────┤
│ Descripción: Tráfico benigno de dispositivos IoT      │
│ Severidad:   BAJA                                      │
│ Indicador:   [NORMAL]                                  │
│ Acción:      Ninguna                                   │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                 CLASE: BRUTE FORCE                     │
├────────────────────────────────────────────────────────┤
│ Descripción: Intentos de acceso no autorizado         │
│ Severidad:   ALTA                                      │
│ Indicador:   [ALTA]                                    │
│ Acción:      Bloquear IP, alertar administrador       │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                   CLASE: DDoS                          │
├────────────────────────────────────────────────────────┤
│ Descripción: Ataque de denegación de servicio         │
│ Severidad:   CRITICA                                   │
│ Indicador:   [CRITICAL]                                │
│ Acción:      Activar mitigación, alertar SOC          │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                   CLASE: MITM                          │
├────────────────────────────────────────────────────────┤
│ Descripción: Man-in-the-Middle, interceptación        │
│ Severidad:   CRITICA                                   │
│ Indicador:   [CRITICAL]                                │
│ Acción:      Aislar dispositivo, investigar           │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                   CLASE: SCAN                          │
├────────────────────────────────────────────────────────┤
│ Descripción: Escaneo de puertos y reconocimiento      │
│ Severidad:   MEDIA                                     │
│ Indicador:   [MEDIA]                                   │
│ Acción:      Monitorear, registrar evento             │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                 CLASE: SPOOFING                        │
├────────────────────────────────────────────────────────┤
│ Descripción: Suplantación de identidad                │
│ Severidad:   ALTA                                      │
│ Indicador:   [ALTA]                                    │
│ Acción:      Validar origen, alertar                  │
└────────────────────────────────────────────────────────┘
```

## Decisiones de Diseño

### 1. Multi-task Learning
- **Razón**: Mejorar la representación aprendida mediante dos objetivos
- **Ventajas**:
  - Mejor generalización
  - Detección de anomalías (via reconstrucción)
  - Clasificación precisa (via clasificador)

### 2. PCA Preprocessing (35 → 16 componentes)
- **Razón**: Reducir dimensionalidad manteniendo varianza
- **Ventajas**:
  - Modelo más eficiente
  - Menor overfitting
  - Faster inference

### 3. LeakyReLU (α=0.3)
- **Razón**: Evitar "dying ReLU" problem
- **Ventajas**:
  - Gradientes no-cero para valores negativos
  - Mejor convergencia

### 4. Dual Model Strategy
- **Modelo Sintético**: Alta precisión (97.24%) para demostración
- **Modelo Real**: Robustez (84.48%) con datos CICIoT2023
- **Razón**: Mostrar trade-off precisión vs. robustez

## Deployment Options

```
┌────────────────────────────────────────────────────┐
│            DEPLOYMENT METHODS                      │
├────────────────────────────────────────────────────┤
│                                                    │
│  1. LOCAL DEVELOPMENT                              │
│     python -m venv venv                            │
│     pip install -r requirements.txt                │
│     streamlit run app.py                           │
│                                                    │
│  2. DOCKER STANDALONE                              │
│     docker build -t iot-ids .                      │
│     docker run -p 8501:8501 iot-ids                │
│                                                    │
│  3. DOCKER COMPOSE                                 │
│     docker-compose up -d                           │
│                                                    │
│  4. CLOUD (Future)                                 │
│     - AWS ECS/Fargate                              │
│     - Google Cloud Run                             │
│     - Azure Container Instances                    │
│                                                    │
└────────────────────────────────────────────────────┘
```

## Performance Metrics

```
┌──────────────────────────────────────────────────────┐
│          SYNTHETIC MODEL PERFORMANCE                 │
├──────────────────────────────────────────────────────┤
│  Accuracy:              97.24%                       │
│  Total Loss:            0.0453                       │
│  Reconstruction Loss:   0.0183                       │
│  Classification Loss:   0.0501                       │
│  F1-Score (avg):        > 0.95                       │
│  False Positive Rate:   < 2%                         │
│  Inference Time:        < 2ms/sample                 │
│  Throughput:            > 500 samples/sec            │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│            REAL MODEL PERFORMANCE                    │
├──────────────────────────────────────────────────────┤
│  Accuracy:              84.48%                       │
│  Total Loss:            0.2547                       │
│  Reconstruction Loss:   0.1842                       │
│  Classification Loss:   0.2832                       │
│  Robustness:            High (real data)             │
│  Inference Time:        < 2ms/sample                 │
│  Throughput:            > 500 samples/sec            │
└──────────────────────────────────────────────────────┘
```
