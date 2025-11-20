# Sistema de Detección de Intrusiones para IoT

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![License](https://img.shields.io/badge/license-Academic-green.svg)

Dashboard interactivo para demostración de sistema de detección de intrusiones en redes IoT utilizando modelos Autoencoder-FNN multi-tarea con aprendizaje profundo.

**Proyecto de Tesis** - Universidad Señor de Sipán (USS)

---

## Características Principales

- **Doble Modelo**: Comparación entre modelo sintético (97.24% accuracy) y modelo real (84.48% accuracy)
- **Detección en Tiempo Real**: Simulación de tráfico IoT con detección instantánea de amenazas
- **Análisis por Lotes**: Procesamiento de archivos CSV con reportes detallados
- **Métricas Completas**: Dashboard con visualizaciones de rendimiento y arquitectura
- **6 Tipos de Amenazas**: Normal, Brute Force, DDoS, MITM, Scan, Spoofing

---

## Arquitectura del Modelo

### Autoencoder-FNN Multi-tarea

```
Input (16 PCA components)
    ↓
┌─────────────────────────────────────────────┐
│ ENCODER                                     │
│   16 → Dense(12) → LeakyReLU(0.3)          │
│   12 → Dense(8)  → LeakyReLU(0.3)          │
│    8 → Dense(6)  → LeakyReLU(0.3) [LATENT] │
└─────────────────────────────────────────────┘
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
┌───────────────────┐   ┌───────────────────┐
│ DECODER           │   │ CLASSIFIER        │
│  6 → Dense(8)     │   │  6 → Dense(64)    │
│  8 → Dense(12)    │   │ 64 → Dense(32)    │
│ 12 → Dense(16)    │   │ 32 → Dense(6)     │
│     [RECON]       │   │     [CLASS]       │
└───────────────────┘   └───────────────────┘

Loss = 0.3 × MSE(reconstruction) + 0.7 × CrossEntropy(classification)
```

**Características Técnicas:**
- **Input**: 16 componentes PCA (reducción dimensional)
- **Latent Space**: 6 dimensiones
- **Activación**: LeakyReLU (α=0.3)
- **Optimizador**: Adam
- **Outputs**: Reconstrucción (16 valores) + Clasificación (6 clases)

---

## Modelos Disponibles

| Modelo | Accuracy | Dataset | Muestras | Épocas |
|--------|----------|---------|----------|--------|
| **Sintético** | 97.24% | PCA 16 componentes | 100,000 | 100 |
| **Real** | 84.48% | CICIoT2023 | Variable | 100 |

### Archivos por Modelo

**Modelo Sintético** (`models/synthetic/`):
- `modelo_ae_fnn_iot_synthetic.h5` - Pesos del modelo
- `scaler_synthetic.pkl` - StandardScaler
- `label_encoder_synthetic.pkl` - LabelEncoder
- `class_names_synthetic.npy` - Nombres de clases
- `model_metadata_synthetic.json` - Metadatos y métricas

**Modelo Real** (`models/real/`):
- `modelo_ae_fnn_iot_REAL.h5` - Pesos del modelo
- `scaler_REAL.pkl` - StandardScaler
- `label_encoder_REAL.pkl` - LabelEncoder
- `class_names_REAL.npy` - Nombres de clases
- `model_metadata_REAL.json` - Metadatos y métricas

---

## Tipos de Amenazas Detectadas

El sistema clasifica tráfico de red IoT en 6 categorías:

1. **Normal** - Tráfico benigno
2. **Brute Force** - Ataques de fuerza bruta
3. **DDoS** - Denegación de servicio distribuida
4. **MITM** - Man-in-the-Middle
5. **Scan** - Escaneo de puertos/reconocimiento
6. **Spoofing** - Suplantación de identidad

---

## Requisitos del Sistema

### Dependencias Principales

```
Python >= 3.8
TensorFlow >= 2.10.0, < 2.16.0
Keras == 2.15.0
Streamlit >= 1.25.0
scikit-learn >= 1.2.0
numpy >= 1.23.0, < 2.0.0
pandas >= 1.5.0
plotly >= 5.14.0
matplotlib >= 3.6.0
seaborn >= 0.12.0
```

Ver [requirements.txt](requirements.txt) para lista completa.

---

## Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/JUNMPI/Dashboard-IoT-IDS.git
cd Dashboard-IoT-IDS
```

### 2. Crear Entorno Virtual (Recomendado)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar Archivos de Modelos

Asegúrate de que la estructura de carpetas `models/` contenga todos los archivos necesarios:

```
models/
├── synthetic/
│   ├── modelo_ae_fnn_iot_synthetic.h5
│   ├── scaler_synthetic.pkl
│   ├── label_encoder_synthetic.pkl
│   ├── class_names_synthetic.npy
│   └── model_metadata_synthetic.json
└── real/
    ├── modelo_ae_fnn_iot_REAL.h5
    ├── scaler_REAL.pkl
    ├── label_encoder_REAL.pkl
    ├── class_names_REAL.npy
    └── model_metadata_REAL.json
```

### 5. Ejecutar la Aplicación

```bash
streamlit run app.py
```

El dashboard se abrirá automáticamente en `http://localhost:8501`

---

## Deployment con Docker

El proyecto incluye soporte completo para containerización con Docker, facilitando el deployment en cualquier entorno.

### Opción 1: Docker Compose (Recomendado)

**Requisitos:**
- Docker Desktop (Windows/Mac) o Docker Engine (Linux)
- Docker Compose v2.0+

**Pasos:**

```bash
# 1. Clonar el repositorio
git clone https://github.com/JUNMPI/Dashboard-IoT-IDS.git
cd Dashboard-IoT-IDS

# 2. Construir y ejecutar con un solo comando
docker-compose up -d

# 3. Verificar que el contenedor está corriendo
docker-compose ps

# 4. Ver logs en tiempo real
docker-compose logs -f

# 5. Acceder al dashboard
# Abrir navegador en http://localhost:8501
```

**Comandos útiles:**

```bash
# Detener el servicio
docker-compose down

# Reconstruir después de cambios en código
docker-compose up -d --build

# Reiniciar el servicio
docker-compose restart

# Ver logs de las últimas 100 líneas
docker-compose logs --tail=100
```

### Opción 2: Docker Manual

```bash
# 1. Construir la imagen
docker build -t iot-ids-dashboard .

# 2. Ejecutar el contenedor
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data:/app/data \
  --name iot-ids \
  iot-ids-dashboard

# 3. Ver logs
docker logs -f iot-ids

# 4. Detener y eliminar el contenedor
docker stop iot-ids
docker rm iot-ids
```

### Características del Contenedor

- **Base Image**: Python 3.11-slim
- **Puerto Expuesto**: 8501
- **Health Check**: Endpoint `/_stcore/health` cada 30s
- **Volúmenes**:
  - `./models:/app/models` - Modelos entrenados (solo lectura)
  - `./data:/app/data` - Datos de entrada/salida
- **Variables de Entorno**:
  - `STREAMLIT_SERVER_PORT=8501`
  - `STREAMLIT_SERVER_ADDRESS=0.0.0.0`
  - `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`

### Verificación del Deployment

```bash
# Verificar que el contenedor está saludable
docker ps
# Debe mostrar "healthy" en STATUS

# Probar el endpoint de salud
curl http://localhost:8501/_stcore/health

# Respuesta esperada: {"status": "ok"}
```

### Solución de Problemas con Docker

**1. Puerto 8501 ya está en uso:**
```bash
# Usar un puerto diferente
docker-compose down
# Editar docker-compose.yml: cambiar "8501:8501" a "8502:8501"
docker-compose up -d
```

**2. Error al montar volumen de modelos:**
```bash
# Verificar que la carpeta models/ existe y tiene los archivos
ls -la models/synthetic/
ls -la models/real/
```

**3. Contenedor se reinicia constantemente:**
```bash
# Ver logs para identificar el error
docker-compose logs --tail=50

# Verificar que requirements.txt está completo
docker-compose exec iot-ids-dashboard pip list
```

---

## Uso del Dashboard

### Página Principal

- Selecciona el modelo (Sintético o Real) en el sidebar
- Visualiza métricas y arquitectura del modelo activo
- Navega entre las 4 páginas disponibles

### 1. Comparación de Modelos

Compara las predicciones de ambos modelos en las mismas muestras.

**Funcionalidades:**
- Generación de muestras sintéticas aleatorias
- Filtrado por tipo de amenaza específica
- Cálculo de tasa de concordancia
- Análisis de discrepancias
- Exportación CSV/PDF

### 2. Tiempo Real

Simula tráfico IoT en tiempo real con detección de amenazas.

**Escenarios Disponibles:**
- **Normal**: 5% amenazas
- **Bajo Ataque**: 80% DDoS
- **Escaneo**: 60% scans
- **Mixto**: 30% amenazas variadas

**Características:**
- Visualización temporal de detecciones
- Métricas en vivo (total muestras, amenazas, confianza)
- Nivel de riesgo global
- Exportación de resultados

### 3. Análisis de Archivo

Procesa archivos CSV con tráfico de red en modo batch.

**Formato de CSV:**
- 16 columnas con componentes PCA (PC1-PC16)
- Opcional: columna 'label' con etiqueta verdadera
- Sin valores faltantes

**Salidas:**
- Predicciones con confianzas
- Distribución de amenazas
- Matriz de confusión (si hay labels)
- Métricas por clase
- Reportes PDF/CSV

### 4. Métricas

Visualiza el rendimiento completo de los modelos.

**Información Mostrada:**
- Estadísticas del dataset
- Métricas de rendimiento (Accuracy, Loss)
- Historial de entrenamiento
- Matriz de confusión
- Métricas por clase
- Comparación entre modelos

---

## Estructura del Proyecto

```
Dashboard-IoT-IDS/
├── app.py                          # Aplicación principal
├── requirements.txt                # Dependencias
├── README.md                       # Este archivo
├── LICENSE                         # Licencia MIT
│
├── Dockerfile                      # Container definition
├── docker-compose.yml              # Docker Compose config
├── .dockerignore                   # Docker exclusions
│
├── models/                         # Modelos entrenados
│   ├── synthetic/                  # Modelo sintético
│   └── real/                       # Modelo real (CICIoT2023)
│
├── pages/                          # Páginas del dashboard
│   ├── 1_Comparacion_Modelos.py   # Comparación lado a lado
│   ├── 2_Tiempo_Real.py           # Simulación en tiempo real
│   ├── 3_Analisis_Archivo.py      # Análisis de CSV
│   └── 4_Metricas.py              # Dashboard de métricas
│
└── utils/                          # Módulos de utilidades
    ├── __init__.py
    ├── model_loader.py            # Carga de modelos (compatibilidad)
    ├── data_simulator.py          # Generador de tráfico sintético
    ├── visualizations.py          # Gráficos y visualizaciones
    └── report_generator.py        # Generación de reportes PDF
```

---

## Rendimiento del Sistema

### Métricas Modelo Sintético

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 97.24% |
| **Loss Total** | 0.0453 |
| **Loss Reconstrucción** | 0.0183 |
| **Loss Clasificación** | 0.0501 |
| **F1-Score Promedio** | > 0.95 |
| **False Positive Rate** | < 2% |

### Métricas Modelo Real

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 84.48% |
| **Loss Total** | 0.2547 |
| **Loss Reconstrucción** | 0.1842 |
| **Loss Clasificación** | 0.2832 |
| **Robustez** | Alta (datos reales) |

### Rendimiento de Inferencia

- **Tiempo por muestra**: < 2ms
- **Throughput**: > 500 muestras/segundo
- **Carga de modelo**: ~5 segundos
- **Memoria**: ~500 MB por modelo

---

## Compatibilidad y Soporte

### Versiones Probadas

- **Python**: 3.8, 3.9, 3.10, 3.11
- **TensorFlow**: 2.15.1
- **Keras**: 2.15.0
- **Sistema Operativo**: Windows 10/11, Linux, macOS

### Problemas Conocidos y Soluciones

**1. Error de Keras/TensorFlow:**
```
Error when deserializing class 'InputLayer'
```
✅ **Solución**: El código incluye carga compatible automática.

**2. Pickle Compatibility:**
```
STACK_GLOBAL requires str
```
✅ **Solución**: Fallback automático a joblib.

**3. Warnings de sklearn:**
```
InconsistentVersionWarning
```
✅ **Normal**: Compatible entre versiones 1.6.1 y 1.7.2.

---

## Contribución a la Tesis

Este proyecto demuestra el cumplimiento de los objetivos específicos:

- **OE1**: Validación de generación y estructuración del dataset (PCA 35→16)
- **OE2**: Implementación funcional del modelo AE-FNN multi-tarea
- **OE3**: Evaluación de efectividad con 97.24% accuracy y FPR<2%
- **OE4**: Demostración de fortalecimiento de ciberseguridad IoT

### Metodología

1. **Preprocesamiento**: PCA para reducción dimensional
2. **Entrenamiento**: Modelos Autoencoder-FNN multi-tarea
3. **Evaluación**: Métricas en datasets sintéticos y reales
4. **Implementación**: Dashboard interactivo para demostración

---

## Solución de Problemas

### El modelo no carga correctamente

1. Verificar versión de Keras: `pip show keras` → debe ser 2.15.0
2. Reinstalar dependencias: `pip install -r requirements.txt --force-reinstall`
3. Verificar archivos en `models/synthetic/` y `models/real/`

### Errores de importación

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### La aplicación no inicia

1. Verificar puerto 8501 disponible
2. Ejecutar: `streamlit run app.py --server.port 8502`

---

## Licencia

Este proyecto es parte de una tesis de pregrado de la Universidad Señor de Sipán (USS) y está disponible solo para fines académicos y de demostración.

---

## Autor

**Junior Alvines**
Tesis de Pregrado - Universidad Señor de Sipán
GitHub: [@JUNMPI](https://github.com/JUNMPI)

---

## Agradecimientos

- Universidad Señor de Sipán (USS)
- Asesores y revisores de tesis
- Comunidad de TensorFlow/Keras
- Streamlit community

---

**Nota Importante**: Este es un sistema de demostración académica. Para uso en entornos de producción, se requieren ajustes adicionales de seguridad, escalabilidad y validación exhaustiva.
