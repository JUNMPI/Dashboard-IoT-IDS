# Arquitectura del Proyecto - Sistema IoT-IDS

## Descripción General

Este documento describe la arquitectura completa de la aplicación de demostración del Sistema de Detección de Intrusiones IoT, incluyendo la estructura de archivos, componentes, flujos de datos y dependencias.

---

## Estructura de Directorios

```
Dashboard IoT-IDS/
│
├── app.py                              # Aplicación principal Streamlit
│
├── pages/                              # Páginas multi-página de Streamlit
│   ├── 1_🔬_Comparacion_Modelos.py    # Comparación lado a lado
│   ├── 2_⚡_Tiempo_Real.py            # Simulación en tiempo real
│   ├── 3_📊_Analisis_Archivo.py       # Análisis batch de CSV
│   └── 4_📈_Metricas.py               # Dashboard de métricas
│
├── utils/                              # Módulos de utilidades
│   ├── __init__.py                     # Inicialización del paquete
│   ├── model_loader.py                 # Carga y gestión de modelos
│   ├── data_simulator.py               # Generación de datos sintéticos
│   ├── visualizations.py               # Gráficos y visualizaciones
│   └── report_generator.py             # Generación de reportes PDF
│
├── models/                             # Modelos entrenados
│   ├── modelo_ae_fnn_iot_synthetic.h5
│   ├── modelo_ae_fnn_iot_real.h5
│   ├── scaler_synthetic.pkl
│   ├── scaler_real.pkl
│   ├── label_encoder_synthetic.pkl
│   ├── label_encoder_real.pkl
│   ├── class_names_synthetic.npy
│   ├── class_names_real.npy
│   ├── model_metadata_synthetic.json
│   └── model_metadata_real.json
│
├── data/                               # Datasets de ejemplo
│   ├── dataset_pca_capa3_iot_ultra_fixed_100k_dataset.csv
│   ├── CICIoT2023_samples.csv
│   └── ejemplos/                       # Archivos de ejemplo para pruebas
│       ├── sample_normal.csv
│       ├── sample_ddos.csv
│       └── sample_mixed.csv
│
├── docs/                               # Documentación
│   ├── IMPLEMENTACION.md               # Guía de implementación
│   ├── MODELOS.md                      # Documentación técnica de modelos
│   ├── ARQUITECTURA.md                 # Este archivo
│   ├── OBJETIVOS_TESIS.md              # Alineación con objetivos
│   └── assets/                         # Recursos para documentación
│       ├── diagrams/
│       └── screenshots/
│
├── tests/                              # Tests unitarios (opcional)
│   ├── test_model_loader.py
│   ├── test_data_simulator.py
│   └── test_predictions.py
│
├── .streamlit/                         # Configuración de Streamlit
│   └── config.toml                     # Tema y configuraciones
│
├── requirements.txt                    # Dependencias de Python
├── .gitignore                          # Archivos ignorados por git
├── README.md                           # Documentación principal
└── LICENSE                             # Licencia del proyecto
```

---

## Componentes Principales

### 1. Aplicación Principal (`app.py`)

**Responsabilidades:**
- Configuración global de Streamlit
- Página de inicio/home
- Selección de modelo (Sintético vs Real)
- Carga inicial de modelos
- Navegación entre páginas

**Funciones principales:**
```python
def main():
    """Función principal de la aplicación"""
    - Configurar página (título, icono, layout)
    - Renderizar sidebar con selector de modelo
    - Cargar modelo seleccionado en session_state
    - Mostrar métricas generales
    - Instrucciones de uso

def load_selected_model(model_choice):
    """Carga el modelo seleccionado por el usuario"""
    - Verificar si el modelo ya está en cache
    - Cargar componentes (modelo, scaler, encoder)
    - Guardar en st.session_state
    - Retornar éxito/error
```

**Estado de sesión gestionado:**
```python
st.session_state = {
    'current_model': 'synthetic' | 'real',
    'model': <Keras Model>,
    'scaler': <StandardScaler>,
    'label_encoder': <LabelEncoder>,
    'class_names': <np.array>,
    'metadata': <dict>
}
```

---

### 2. Páginas Streamlit

#### 2.1 Comparación de Modelos (`1_🔬_Comparacion_Modelos.py`)

**Funcionalidades:**
- Comparación lado a lado de predicciones
- Generación de muestras aleatorias
- Análisis batch de archivos CSV
- Visualización comparativa

**Componentes UI:**
- Columnas izquierda/derecha para cada modelo
- Botón "Generar Muestra Aleatoria"
- File uploader para CSV
- Gráficos de barras comparativos
- Tabla de resultados

**Estado local:**
```python
st.session_state = {
    'comparison_results': pd.DataFrame,
    'last_sample': np.array,
    'concordance_rate': float
}
```

#### 2.2 Tiempo Real (`2_⚡_Tiempo_Real.py`)

**Funcionalidades:**
- Simulación de tráfico IoT continuo
- Monitoreo en tiempo real
- Alertas visuales por nivel de riesgo
- Simulación de escenarios específicos

**Componentes UI:**
- Botones Start/Pause/Reset
- Métricas en vivo (total muestras, amenazas, riesgo)
- Gráfico temporal scrollable
- Log de detecciones
- Botones de simulación de ataques

**Estado local:**
```python
st.session_state = {
    'simulation_running': bool,
    'threat_history': deque(maxlen=60),
    'threat_counts': dict,
    'total_samples': int,
    'last_detection': dict
}
```

**Loop de simulación:**
```python
while st.session_state.simulation_running:
    - Generar muestra simulada
    - Predecir con modelo
    - Actualizar contadores
    - Actualizar visualizaciones
    - Mostrar alerta si es amenaza
    - sleep(1)  # 1 muestra/segundo
```

#### 2.3 Análisis de Archivo (`3_📊_Analisis_Archivo.py`)

**Funcionalidades:**
- Upload de archivos CSV
- Validación de formato
- Procesamiento batch
- Generación de reportes
- Exportación de resultados

**Componentes UI:**
- File uploader
- Preview de datos
- Barra de progreso
- Tabla de resultados
- Visualizaciones (distribución, matriz confusión)
- Botones de descarga (CSV, PDF)

**Flujo de procesamiento:**
```
1. Usuario sube archivo CSV
2. Validar formato (16 columnas + opcional label)
3. Mostrar preview
4. Usuario selecciona modelo y presiona "Analizar"
5. Iterar sobre muestras con barra de progreso
6. Generar predicciones
7. Calcular métricas (si hay labels)
8. Visualizar resultados
9. Permitir descarga de reporte
```

#### 2.4 Métricas (`4_📈_Metricas.py`)

**Funcionalidades:**
- Visualización de métricas de ambos modelos
- Información técnica de arquitectura
- Justificación académica
- Cumplimiento de objetivos de tesis

**Componentes UI:**
- Tabs (Sintético, Real, Técnico)
- Tarjetas de métricas
- Gráficos de rendimiento
- Tablas comparativas
- Sección de justificación académica

---

### 3. Módulos de Utilidades

#### 3.1 `utils/model_loader.py`

**Propósito:** Gestionar carga de modelos y predicciones

**Funciones principales:**

```python
@st.cache_resource
def load_synthetic_model():
    """
    Carga el modelo sintético y todos sus componentes

    Returns:
        model, scaler, label_encoder, class_names, metadata
    """

@st.cache_resource
def load_real_model():
    """
    Carga el modelo real y todos sus componentes

    Returns:
        model, scaler, label_encoder, class_names, metadata
    """

def predict_sample(model, scaler, label_encoder, class_names, sample):
    """
    Predice la clase de una muestra

    Args:
        model: Modelo Keras
        scaler: StandardScaler
        label_encoder: LabelEncoder
        class_names: Array de nombres de clases
        sample: Array de 16 features

    Returns:
        prediction: Clase predicha (str)
        probabilities: Array de probabilidades
        confidence: Confianza en % (float)
    """

def predict_batch(model, scaler, label_encoder, class_names, X_batch):
    """
    Predice múltiples muestras en batch

    Args:
        X_batch: Array (n_samples, 16)

    Returns:
        predictions: Lista de predicciones
        confidences: Lista de confianzas
    """

def verify_model_compatibility(model):
    """
    Verifica que el modelo tenga la estructura esperada

    Returns:
        bool: True si es compatible
        str: Mensaje de error si no es compatible
    """
```

**Manejo de cache:**
```python
# Usa @st.cache_resource para cargar modelos una sola vez
# Persiste en memoria durante toda la sesión
# Comparte entre usuarios (multitenancy)
```

#### 3.2 `utils/data_simulator.py`

**Propósito:** Generar datos sintéticos para simulación

**Funciones principales:**

```python
def generate_traffic_sample(attack_type=None):
    """
    Genera una muestra de tráfico IoT

    Args:
        attack_type: 'DDoS', 'DoS', etc. o None (aleatorio)

    Returns:
        sample: Array de 16 componentes PCA
        label: Etiqueta verdadera
    """

def generate_attack_pattern(attack_type):
    """
    Genera patrón característico de un ataque

    Args:
        attack_type: Tipo de ataque

    Returns:
        sample: Array de 16 componentes con patrón del ataque
    """

def generate_attack_burst(attack_type, count=10):
    """
    Genera ráfaga de muestras del mismo ataque

    Args:
        attack_type: Tipo de ataque
        count: Número de muestras

    Returns:
        samples: Lista de (sample, label)
    """

def generate_mixed_traffic(duration_seconds=60):
    """
    Genera tráfico mixto para simulación temporal

    Args:
        duration_seconds: Duración de la simulación

    Returns:
        timeline: Lista de (timestamp, sample, label)
    """
```

**Patrones de ataque implementados:**
```python
ATTACK_PATTERNS = {
    'DDoS': {
        'pc1_multiplier': 3.0,
        'pc2_multiplier': 2.5,
        'pc3_multiplier': 2.0,
        'noise_level': 0.2
    },
    'Brute_Force': {
        'pc5_multiplier': 4.0,
        'pc6_multiplier': 3.0,
        'repetitive': True
    },
    # ... otros patrones
}
```

#### 3.3 `utils/visualizations.py`

**Propósito:** Funciones de visualización reutilizables

**Funciones principales:**

```python
def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Genera matriz de confusión con seaborn

    Returns:
        fig: Figura de matplotlib
    """

def plot_temporal_chart(threat_history):
    """
    Gráfico temporal de detecciones

    Args:
        threat_history: deque con histórico

    Returns:
        fig: Figura de Plotly
    """

def plot_class_distribution(predictions):
    """
    Gráfico de pastel con distribución de clases

    Returns:
        fig: Figura de Plotly
    """

def plot_confidence_comparison(results_df):
    """
    Gráfico de barras comparando confianzas

    Returns:
        fig: Figura de Plotly
    """

def plot_metrics_radar(metrics_dict):
    """
    Gráfico radar con múltiples métricas

    Returns:
        fig: Figura de Plotly
    """

def create_risk_gauge(risk_level):
    """
    Velocímetro de nivel de riesgo

    Args:
        risk_level: 0-100

    Returns:
        fig: Figura de Plotly
    """
```

#### 3.4 `utils/report_generator.py`

**Propósito:** Generar reportes PDF

**Funciones principales:**

```python
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table

def generate_pdf_report(results_df, model_name, metadata):
    """
    Genera reporte PDF completo

    Args:
        results_df: DataFrame con resultados
        model_name: Nombre del modelo usado
        metadata: Metadatos adicionales

    Returns:
        bytes: Contenido del PDF
    """

def add_header(pdf, title):
    """Agrega encabezado al PDF"""

def add_metrics_section(pdf, metrics):
    """Agrega sección de métricas"""

def add_visualizations(pdf, figures):
    """Agrega gráficos al PDF"""

def add_footer(pdf, timestamp):
    """Agrega pie de página con timestamp"""
```

---

## Flujos de Datos

### Flujo 1: Carga de Modelo

```
Usuario → Selecciona modelo en sidebar
    ↓
app.py → load_selected_model(choice)
    ↓
utils/model_loader.py → load_synthetic_model() o load_real_model()
    ↓
    ├─→ Cargar modelo.h5 (TensorFlow)
    ├─→ Cargar scaler.pkl (pickle)
    ├─→ Cargar label_encoder.pkl (pickle)
    ├─→ Cargar class_names.npy (numpy)
    └─→ Cargar metadata.json (json)
    ↓
Almacenar en st.session_state
    ↓
Modelo disponible para todas las páginas
```

### Flujo 2: Predicción de Muestra Única

```
Usuario → Genera muestra aleatoria o sube datos
    ↓
Página Streamlit → Obtener muestra (16 features)
    ↓
utils/model_loader.py → predict_sample(model, scaler, encoder, sample)
    ↓
    1. Normalizar muestra con scaler.transform()
    2. Predecir con model.predict()
    3. Extraer salida de clasificación (output[1])
    4. Obtener clase con argmax
    5. Decodificar con label_encoder.inverse_transform()
    6. Calcular confianza (max probability)
    ↓
Retornar → (prediction, probabilities, confidence)
    ↓
Página Streamlit → Mostrar resultados
```

### Flujo 3: Simulación en Tiempo Real

```
Usuario → Presiona "Iniciar Simulación"
    ↓
st.session_state.simulation_running = True
    ↓
Loop continuo (cada 1 segundo):
    ├─→ utils/data_simulator.py → generate_traffic_sample()
    │       └─→ Retorna (sample, true_label)
    ├─→ utils/model_loader.py → predict_sample(sample)
    │       └─→ Retorna (prediction, probs, confidence)
    ├─→ Actualizar threat_history (agregar nueva detección)
    ├─→ Actualizar threat_counts (incrementar contador)
    ├─→ Actualizar visualizaciones (gráfico temporal)
    ├─→ Mostrar alerta si es amenaza
    └─→ sleep(1)
    ↓
Usuario → Presiona "Pausar"
    ↓
st.session_state.simulation_running = False
```

### Flujo 4: Análisis de Archivo

```
Usuario → Sube archivo CSV
    ↓
Streamlit → file_uploader() retorna UploadedFile
    ↓
pd.read_csv() → DataFrame
    ↓
Validar formato:
    ├─→ Verificar 16 columnas de features
    ├─→ Detectar columna 'label' opcional
    └─→ Verificar tipos de datos
    ↓
Usuario → Presiona "Analizar"
    ↓
For each row in DataFrame:
    ├─→ Extraer sample (16 features)
    ├─→ predict_sample(sample)
    ├─→ Almacenar resultado
    └─→ Actualizar progress bar
    ↓
Si hay labels:
    ├─→ Calcular accuracy, precision, recall, f1
    └─→ Generar matriz de confusión
    ↓
Visualizar resultados:
    ├─→ Tabla de predicciones
    ├─→ Distribución de clases
    ├─→ Top amenazas
    └─→ Matriz de confusión
    ↓
Usuario → Descarga reporte (CSV o PDF)
    ↓
utils/report_generator.py → generate_pdf_report()
    ↓
Retornar bytes del PDF
```

---

## Gestión de Estado

### Session State en Streamlit

Streamlit es stateless por defecto. Usamos `st.session_state` para persistencia.

**Variables globales (app.py):**
```python
st.session_state = {
    # Modelo seleccionado
    'current_model': 'synthetic' | 'real',

    # Componentes del modelo cargado
    'model': <Keras Model>,
    'scaler': <StandardScaler>,
    'label_encoder': <LabelEncoder>,
    'class_names': np.array,
    'metadata': dict,

    # Flags de estado
    'model_loaded': bool,
    'first_run': bool
}
```

**Variables específicas de página (2_⚡_Tiempo_Real.py):**
```python
st.session_state = {
    'simulation_running': False,
    'threat_history': deque(maxlen=60),
    'threat_counts': {
        'Benign': 0, 'DDoS': 0, ...
    },
    'total_samples': 0,
    'start_time': timestamp
}
```

**Variables específicas de página (3_📊_Analisis_Archivo.py):**
```python
st.session_state = {
    'analysis_results': pd.DataFrame,
    'uploaded_file_hash': str,
    'has_labels': bool,
    'metrics': dict
}
```

### Cache de Streamlit

**@st.cache_resource:**
- Cachea modelos ML (persistencia en memoria)
- Comparte entre sesiones de usuarios
- No se serializa, almacena el objeto directamente

```python
@st.cache_resource
def load_synthetic_model():
    # Se ejecuta solo una vez
    # Resultado se comparte entre todos los usuarios
    pass
```

**@st.cache_data:**
- Cachea DataFrames y datos computados
- Serializa y deserializa automáticamente
- Ideal para procesamiento de datos

```python
@st.cache_data
def load_dataset(file_path):
    # Se ejecuta solo si file_path cambia
    return pd.read_csv(file_path)
```

---

## Dependencias del Proyecto

### requirements.txt

```txt
# Core
python>=3.8

# Machine Learning
tensorflow>=2.10.0
keras>=2.10.0
scikit-learn>=1.2.0
numpy>=1.23.0

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

### Diagrama de Dependencias

```
app.py
    ├── streamlit
    ├── utils.model_loader
    │       ├── tensorflow
    │       ├── scikit-learn
    │       └── numpy
    └── pages/
            ├── 1_Comparacion_Modelos.py
            │       ├── utils.model_loader
            │       ├── utils.visualizations
            │       └── plotly
            ├── 2_Tiempo_Real.py
            │       ├── utils.model_loader
            │       ├── utils.data_simulator
            │       ├── utils.visualizations
            │       └── plotly
            ├── 3_Analisis_Archivo.py
            │       ├── utils.model_loader
            │       ├── utils.visualizations
            │       ├── utils.report_generator
            │       ├── pandas
            │       └── plotly
            └── 4_Metricas.py
                    ├── utils.visualizations
                    └── plotly
```

---

## Configuración de Streamlit

### `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"

[server]
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[runner]
magicEnabled = true
fastReruns = true
```

---

## Consideraciones de Seguridad

### 1. Validación de Entrada

- **Archivos CSV**: Validar formato, tamaño máximo, tipos de datos
- **Muestras simuladas**: Limitar rangos de valores
- **Paths**: Evitar path traversal attacks

### 2. Manejo de Modelos

- **Verificación**: Checksum de archivos .h5 antes de cargar
- **Sandboxing**: Ejecutar predicciones en modo restringido
- **Timeout**: Límite de tiempo para inferencia

### 3. Gestión de Sesiones

- **Límite de datos**: Limpiar session_state periódicamente
- **Timeout de sesión**: Invalidar sesiones inactivas
- **Aislamiento**: Cada usuario tiene session_state separado

---

## Escalabilidad

### Limitaciones Actuales

- **Concurrencia**: Streamlit single-threaded por sesión
- **Memoria**: Modelos cargados en RAM (~150KB c/u)
- **Simulación**: Loop síncrono, bloquea UI

### Mejoras Futuras

1. **Async Processing**: Usar asyncio para simulación
2. **Background Workers**: Celery para procesamiento batch
3. **Database**: Persistir resultados en DB (PostgreSQL, MongoDB)
4. **Queue System**: RabbitMQ para manejar múltiples análisis
5. **Containerización**: Docker para despliegue consistente

---

## Deployment

### Opciones de Despliegue

#### 1. Streamlit Cloud (Recomendado para demo)
```bash
# Push a GitHub
git push origin main

# Conectar en streamlit.io
# Auto-deploy desde repositorio
```

#### 2. Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

#### 3. Cloud Platforms
- **Google Cloud Run**: Serverless, auto-scaling
- **AWS EC2**: VM tradicional
- **Heroku**: PaaS simplificado

---

## Monitoreo y Logging

### Logs de Aplicación

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Uso
logger.info("Modelo cargado exitosamente")
logger.warning("Confianza baja en predicción")
logger.error("Error al cargar archivo CSV")
```

### Métricas de Uso

```python
# Trackear en session_state
st.session_state.metrics = {
    'total_predictions': 0,
    'total_files_analyzed': 0,
    'simulation_time': 0,
    'avg_inference_time': 0.0
}
```

---

## Testing

### Estructura de Tests

```python
# tests/test_model_loader.py
import pytest
from utils.model_loader import load_synthetic_model, predict_sample

def test_load_synthetic_model():
    model, scaler, encoder, names, meta = load_synthetic_model()
    assert model is not None
    assert scaler is not None

def test_predict_sample():
    sample = np.random.randn(16)
    pred, probs, conf = predict_sample(model, scaler, encoder, names, sample)
    assert isinstance(pred, str)
    assert 0 <= conf <= 100
```

### Comandos de Testing

```bash
# Ejecutar todos los tests
pytest tests/

# Con cobertura
pytest --cov=utils tests/

# Test específico
pytest tests/test_model_loader.py::test_predict_sample
```

---

**Última actualización**: Noviembre 2024
