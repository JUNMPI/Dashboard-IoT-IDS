# Resumen del Proyecto - Dashboard IoT-IDS

**Sistema de Detección de Intrusiones para Redes IoT**
**Proyecto de Tesis - Universidad Señor de Sipán (USS)**
**Autor**: Junior Alvines
**Año**: 2025

---

## Descripción General

Dashboard interactivo desarrollado con Streamlit para demostración de un sistema de detección de intrusiones en redes IoT utilizando modelos de aprendizaje profundo basados en arquitectura Autoencoder-FNN multi-tarea.

### Características Principales

- **Doble Modelo**: Comparación entre modelo sintético (97.24% accuracy) y real (84.48% accuracy)
- **Detección en Tiempo Real**: Simulación de tráfico IoT con detección instantánea
- **Análisis por Lotes**: Procesamiento de archivos CSV con reportes detallados
- **6 Tipos de Amenazas**: Normal, Brute Force, DDoS, MITM, Scan, Spoofing
- **Visualizaciones Interactivas**: Métricas, matrices de confusión, gráficos temporales
- **Exportación de Reportes**: CSV y PDF con análisis completo

---

## Tecnologías Utilizadas

### Backend y ML
- **Python 3.11** - Lenguaje de programación principal
- **TensorFlow 2.15** - Framework de deep learning
- **Keras 2.15** - API de alto nivel para redes neuronales
- **Scikit-learn 1.7** - Preprocesamiento y métricas
- **NumPy 1.26** - Computación numérica
- **Pandas 2.2** - Manipulación de datos

### Frontend y Visualización
- **Streamlit 1.41** - Framework web interactivo
- **Plotly 5.24** - Gráficos interactivos
- **Matplotlib 3.9** - Visualizaciones estáticas
- **Seaborn 0.13** - Visualizaciones estadísticas

### Reportes y Deployment
- **ReportLab 4.2** - Generación de PDFs
- **Docker** - Containerización
- **Docker Compose** - Orquestación de contenedores

---

## Arquitectura del Modelo

### Autoencoder-FNN Multi-tarea

```
Input (16 componentes PCA)
         ↓
    ┌─────────────┐
    │   ENCODER   │
    │  16→12→8→6  │
    └──────┬──────┘
           │ (Latent Space: 6 dims)
           │
    ┌──────┴────────┐
    ↓               ↓
┌─────────┐   ┌──────────────┐
│ DECODER │   │  CLASSIFIER  │
│  6→8→12 │   │    6→64→32   │
│   →16   │   │      →6      │
└─────────┘   └──────────────┘
    ↓               ↓
RECONSTRUCCIÓN  CLASIFICACIÓN
```

**Loss Function**: `L_total = 0.3 × MSE_recon + 0.7 × CE_class`

**Características Técnicas**:
- Activación: LeakyReLU (α=0.3)
- Optimizador: Adam
- Input: 16 componentes PCA
- Output: Reconstrucción (16 valores) + Clasificación (6 clases)

---

## Estructura del Proyecto

```
Dashboard-IoT-IDS/
├── app.py                           # Aplicación principal
├── requirements.txt                 # Dependencias Python
├── Dockerfile                       # Container definition
├── docker-compose.yml               # Docker orchestration
├── .dockerignore                    # Docker exclusions
│
├── README.md                        # Documentación principal
├── LICENSE                          # Licencia MIT
├── ARCHITECTURE.md                  # Diagramas de arquitectura
├── VISUAL_GUIDE.md                  # Guía de capturas visuales
├── TESTING.md                       # Guía de testing
├── PROJECT_SUMMARY.md               # Este archivo
│
├── models/                          # Modelos entrenados
│   ├── synthetic/                   # Modelo sintético (97.24%)
│   │   ├── modelo_ae_fnn_iot_synthetic.h5
│   │   ├── scaler_synthetic.pkl
│   │   ├── label_encoder_synthetic.pkl
│   │   ├── class_names_synthetic.npy
│   │   └── model_metadata_synthetic.json
│   │
│   └── real/                        # Modelo real (84.48%)
│       ├── modelo_ae_fnn_iot_REAL.h5
│       ├── scaler_REAL.pkl
│       ├── label_encoder_REAL.pkl
│       ├── class_names_REAL.npy
│       └── model_metadata_REAL.json
│
├── pages/                           # Páginas del dashboard
│   ├── 1_Comparacion_Modelos.py    # Comparación lado a lado
│   ├── 2_Tiempo_Real.py            # Simulación en tiempo real
│   ├── 3_Analisis_Archivo.py       # Análisis de CSV
│   └── 4_Metricas.py               # Dashboard de métricas
│
├── utils/                           # Módulos de utilidades
│   ├── __init__.py
│   ├── model_loader.py             # Carga de modelos con compatibilidad
│   ├── data_simulator.py           # Generador de tráfico sintético
│   ├── visualizations.py           # Gráficos y visualizaciones
│   └── report_generator.py         # Generación de reportes PDF
│
├── docs/                            # Documentación visual
│   ├── screenshots/                 # Capturas de pantalla
│   └── gifs/                        # GIFs de demostración
│
├── run_tests.py                     # Script de testing automatizado
└── test_system.py                   # Tests adicionales
```

---

## Funcionalidades del Dashboard

### 1. Página Principal
- Selector de modelo (Sintético / Real)
- Visualización de arquitectura del modelo
- Métricas de rendimiento
- Información del dataset

### 2. Comparación de Modelos
- Generación de muestras sintéticas
- Predicción con ambos modelos simultáneamente
- Cálculo de tasa de concordancia
- Tabla comparativa con confianzas
- Análisis de discrepancias
- Exportación de resultados

### 3. Simulación en Tiempo Real
**Escenarios disponibles**:
- Normal (5% amenazas)
- Bajo Ataque - DDoS (80% amenazas)
- Escaneo (60% scans)
- Mixto (30% amenazas variadas)

**Características**:
- Gráficos temporales actualizables
- Métricas en vivo
- Nivel de riesgo global
- Tabla de detecciones recientes
- Control de intervalo y cantidad

### 4. Análisis de Archivo
- Carga de archivos CSV (drag & drop)
- Validación de formato (16 columnas PCA)
- Procesamiento batch
- Distribución de amenazas
- Matriz de confusión (si hay labels)
- Métricas por clase
- Exportación CSV y PDF

### 5. Métricas del Sistema
- Estadísticas del dataset
- Métricas de entrenamiento
- Historial de accuracy y loss
- Matriz de confusión 6×6
- Métricas por clase (Precision, Recall, F1)
- Comparación entre modelos

---

## Amenazas Detectadas

| Clase | Descripción | Severidad |
|-------|-------------|-----------|
| **Normal** | Tráfico benigno de dispositivos IoT | BAJA |
| **Brute Force** | Intentos de acceso no autorizado | ALTA |
| **DDoS** | Ataque de denegación de servicio | CRÍTICA |
| **MITM** | Man-in-the-Middle, interceptación | CRÍTICA |
| **Scan** | Escaneo de puertos y reconocimiento | MEDIA |
| **Spoofing** | Suplantación de identidad | ALTA |

---

## Rendimiento del Sistema

### Modelo Sintético

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 97.24% |
| **Loss Total** | 0.0453 |
| **Loss Reconstrucción** | 0.0183 |
| **Loss Clasificación** | 0.0501 |
| **F1-Score Promedio** | > 0.95 |
| **False Positive Rate** | < 2% |
| **Tiempo Inferencia** | < 2ms/muestra |
| **Throughput** | > 500 muestras/segundo |

### Modelo Real

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 84.48% |
| **Loss Total** | 0.2547 |
| **Loss Reconstrucción** | 0.1842 |
| **Loss Clasificación** | 0.2832 |
| **Dataset** | CICIoT2023 |
| **Robustez** | Alta (datos reales) |
| **Tiempo Inferencia** | < 2ms/muestra |
| **Throughput** | > 500 muestras/segundo |

---

## Instalación y Deployment

### Método 1: Local (Desarrollo)

```bash
# 1. Clonar repositorio
git clone https://github.com/JUNMPI/Dashboard-IoT-IDS.git
cd Dashboard-IoT-IDS

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar aplicación
streamlit run app.py
```

### Método 2: Docker Compose (Recomendado)

```bash
# 1. Clonar repositorio
git clone https://github.com/JUNMPI/Dashboard-IoT-IDS.git
cd Dashboard-IoT-IDS

# 2. Ejecutar con Docker Compose
docker-compose up -d

# 3. Acceder al dashboard
# http://localhost:8501
```

### Método 3: Docker Manual

```bash
# Construir imagen
docker build -t iot-ids-dashboard .

# Ejecutar contenedor
docker run -d -p 8501:8501 \
  -v $(pwd)/models:/app/models:ro \
  --name iot-ids \
  iot-ids-dashboard
```

---

## Testing y Validación

### Tests Automatizados

```bash
# Ejecutar suite de tests
python run_tests.py
```

**Tests incluidos** (6 tests):
1. ✅ Importación de dependencias (11 módulos)
2. ✅ Estructura del proyecto (16 elementos)
3. ✅ Archivos de modelos (10 archivos)
4. ✅ Carga de modelos (ambos modelos)
5. ✅ Predicciones del modelo (10 muestras)
6. ✅ Páginas del dashboard (4 páginas)

**Resultado**: 6/6 tests PASARON EXITOSAMENTE

### Tests Manuales

Ver `TESTING.md` para 10 casos de prueba detallados (CP-01 a CP-10) que cubren:
- Carga inicial del sistema
- Predicciones con ambos modelos
- Simulación en tiempo real (escenarios normal y ataque)
- Análisis de archivos CSV
- Visualización de métricas
- Exportación de reportes

---

## Contribución a la Tesis

### Objetivos Específicos Cumplidos

| OE | Descripción | Cumplimiento |
|----|-------------|--------------|
| **OE1** | Validación de generación y estructuración del dataset | ✅ PCA 35→16 componentes |
| **OE2** | Implementación funcional del modelo AE-FNN | ✅ Arquitectura multi-tarea |
| **OE3** | Evaluación de efectividad | ✅ 97.24% accuracy, FPR<2% |
| **OE4** | Demostración de fortalecimiento | ✅ Dashboard interactivo |

### Metodología Aplicada

1. **Preprocesamiento**: Reducción dimensional con PCA
2. **Entrenamiento**: Modelos Autoencoder-FNN multi-tarea
3. **Evaluación**: Métricas en datasets sintéticos y reales
4. **Implementación**: Dashboard interactivo para demostración

---

## Documentación

### Archivos de Documentación

| Archivo | Descripción | Líneas |
|---------|-------------|--------|
| **README.md** | Documentación principal, instalación, uso | ~600 |
| **ARCHITECTURE.md** | Diagramas de arquitectura, flujos, decisiones | ~400 |
| **VISUAL_GUIDE.md** | Guía para screenshots y GIFs | ~700 |
| **TESTING.md** | Guía de testing, casos de prueba | ~500 |
| **PROJECT_SUMMARY.md** | Este archivo - resumen ejecutivo | ~400 |
| **LICENSE** | Licencia MIT con nota académica | ~40 |

### Guías Disponibles

- ✅ Instalación y configuración
- ✅ Uso de cada página del dashboard
- ✅ Deployment con Docker
- ✅ Testing automatizado y manual
- ✅ Creación de screenshots y GIFs
- ✅ Arquitectura del sistema
- ✅ Solución de problemas

---

## Compatibilidad

### Plataformas Soportadas

- ✅ **Windows** 10/11
- ✅ **Linux** (Ubuntu 20.04+, Debian 11+)
- ✅ **macOS** (10.15+)

### Versiones de Python

- ✅ Python 3.8
- ✅ Python 3.9
- ✅ Python 3.10
- ✅ Python 3.11

### Navegadores

- ✅ Google Chrome 90+
- ✅ Mozilla Firefox 88+
- ✅ Microsoft Edge 90+
- ✅ Safari 14+

---

## Próximos Pasos (Opcional)

### Mejoras Futuras Sugeridas

1. **Mejoras de Modelo**:
   - Entrenar con más datos del CICIoT2023
   - Implementar ensemble de modelos
   - Agregar detección de anomalías no supervisada

2. **Funcionalidades**:
   - Integración con fuentes de tráfico real
   - API REST para integración externa
   - Alertas en tiempo real (email, webhook)
   - Dashboard de administración

3. **Deployment**:
   - Desplegar en cloud (AWS, GCP, Azure)
   - Implementar CI/CD
   - Agregar autenticación de usuarios
   - Escalabilidad horizontal

4. **Visuales**:
   - Capturas de pantalla del sistema
   - GIFs de demostración
   - Video tutorial completo
   - Presentación interactiva

---

## Licencia

Este proyecto está licenciado bajo la **Licencia MIT** con nota académica.

Copyright (c) 2025 Edgard Junior Alvines Santa Cruz - Sheral Altamirano Vega - Universidad Señor de Sipán

El software es provisto "tal cual", sin garantía de ningún tipo. Ver archivo `LICENSE` para detalles completos.

---

## Contacto y Referencias

**Autor**: Junior Alvines
**Institución**: Universidad Señor de Sipán (USS)
**GitHub**: [@JUNMPI](https://github.com/JUNMPI)
**Repositorio**: [Dashboard-IoT-IDS](https://github.com/JUNMPI/Dashboard-IoT-IDS)

### Citación Académica

```
Alvines, J. (2025). "Clasificación de tráfico de red y fortalecimiento de la
ciberseguridad en entornos de IoT utilizando aprendizaje profundo".
Tesis de Pregrado, Universidad Señor de Sipán (USS), Perú.
```

---

## Agradecimientos

- Universidad Señor de Sipán (USS)
- Asesores y revisores de tesis
- Comunidad de TensorFlow/Keras
- Comunidad de Streamlit
- Dataset CICIoT2023

---

**Nota Final**: Este es un sistema de demostración académica desarrollado como parte de una tesis de pregrado. Para uso en entornos de producción, se requieren ajustes adicionales de seguridad, escalabilidad y validación exhaustiva.

---

**Última actualización**: 2025-01-20
**Versión del sistema**: 1.0
**Estado**: Completo y funcional
