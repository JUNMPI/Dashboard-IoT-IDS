# Sistema de Detección de Intrusiones IoT - USS

## Descripción del Proyecto

Aplicación de demostración para tesis de pregrado sobre **"Clasificación de tráfico de red y fortalecimiento de la ciberseguridad en entornos de IoT utilizando aprendizaje profundo"**.

Este proyecto implementa una aplicación interactiva que demuestra el funcionamiento de modelos Autoencoder-FNN (AE-FNN) para la detección de amenazas en redes IoT, comparando el desempeño entre modelos entrenados con datos sintéticos vs. datos reales.

## Características Principales

- **Comparación de Modelos**: Análisis lado a lado entre modelo sintético (97% accuracy) y modelo real (84.48% accuracy)
- **Simulación en Tiempo Real**: Detección de amenazas IoT en tiempo real con alertas visuales
- **Análisis de Archivos**: Procesamiento batch de archivos CSV con generación de reportes
- **Dashboard de Métricas**: Visualización completa de rendimiento y métricas técnicas

## Modelos Disponibles

### 1. Modelo con Datos Sintéticos
- **Accuracy**: 97%
- **Dataset**: PCA con 16 componentes (100k muestras)
- **Archivos**:
  - `modelo_ae_fnn_iot_synthetic.h5`
  - `scaler_synthetic.pkl`
  - `label_encoder_synthetic.pkl`
  - `class_names_synthetic.npy`
  - `model_metadata_synthetic.json`

### 2. Modelo con Datos Reales (CICIoT2023)
- **Accuracy**: 84.48%
- **Dataset**: CICIoT2023 preprocesado
- **Archivos**:
  - `modelo_ae_fnn_iot_real.h5`
  - `scaler_real.pkl`
  - `label_encoder_real.pkl`
  - `class_names_real.npy`
  - `model_metadata_real.json`

## Tipos de Ataques Detectados

El sistema es capaz de detectar los siguientes tipos de amenazas IoT:

- **DDoS** (Distributed Denial of Service)
- **DoS** (Denial of Service)
- **Brute Force** (Ataques de fuerza bruta)
- **Spoofing** (Suplantación)
- **MITM** (Man-in-the-Middle)
- **Scan** (Escaneo de puertos)
- **Recon** (Reconocimiento)
- **Tráfico Normal** (Benign)

## Estructura del Proyecto

```
iot_ids_demo/
├── app.py                              # Aplicación principal Streamlit
├── models/                             # Modelos entrenados y archivos relacionados
│   ├── modelo_ae_fnn_iot_synthetic.h5
│   ├── modelo_ae_fnn_iot_real.h5
│   ├── scaler_synthetic.pkl
│   ├── scaler_real.pkl
│   └── ...
├── data/                               # Datasets de ejemplo
│   ├── dataset_pca_capa3_iot_ultra_fixed_100k_dataset.csv
│   └── CICIoT2023_samples.csv
├── utils/                              # Utilidades y funciones auxiliares
│   ├── model_loader.py                 # Carga de modelos
│   ├── data_simulator.py               # Generador de datos simulados
│   └── visualizations.py               # Funciones de visualización
├── pages/                              # Páginas de la aplicación Streamlit
│   ├── 1_🔬_Comparacion_Modelos.py
│   ├── 2_⚡_Tiempo_Real.py
│   ├── 3_📊_Analisis_Archivo.py
│   └── 4_📈_Metricas.py
└── docs/                               # Documentación del proyecto
    ├── IMPLEMENTACION.md               # Guía de implementación por fases
    ├── MODELOS.md                      # Información técnica de modelos
    ├── ARQUITECTURA.md                 # Arquitectura detallada
    └── OBJETIVOS_TESIS.md              # Alineación con objetivos de tesis
```

## Requisitos del Sistema

### Dependencias Principales
```
python>=3.8
tensorflow>=2.10.0
streamlit>=1.25.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
plotly>=5.14.0
seaborn>=0.12.0
matplotlib>=3.6.0
```

### Instalación

1. Clonar el repositorio:
```bash
git clone <repository-url>
cd Dashboard\ IoT-IDS
```

2. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Colocar los archivos de modelos en la carpeta `models/`

5. Ejecutar la aplicación:
```bash
streamlit run app.py
```

## Uso de la Aplicación

### Página Principal
Selecciona el modelo a utilizar (Sintético o Real) y visualiza información básica sobre su arquitectura y rendimiento.

### 1. Comparación de Modelos 🔬
- Procesa la misma muestra con ambos modelos simultáneamente
- Visualiza diferencias en predicciones
- Analiza archivos CSV en modo batch
- Genera matrices de confusión comparativas

### 2. Simulación en Tiempo Real ⚡
- Monitorea tráfico IoT simulado en tiempo real
- Visualiza detecciones con alertas de seguridad
- Simula escenarios de ataque específicos
- Panel de métricas en vivo

### 3. Análisis de Archivos 📊
- Carga archivos CSV con datos de red IoT
- Procesa y clasifica múltiples muestras
- Genera reportes detallados con métricas
- Exporta resultados en formato PDF

### 4. Dashboard de Métricas 📈
- Visualiza métricas completas de ambos modelos
- Compara rendimiento entre modelos
- Información técnica de arquitectura AE-FNN
- Análisis de contribución a ciberseguridad

## Contribución a la Tesis

Este proyecto demuestra de manera práctica el cumplimiento de los objetivos específicos de la tesis:

- **OE1**: Validación de la generación y estructuración del conjunto de datos (PCA 35→16)
- **OE2**: Implementación funcional del modelo AE-FNN multi-tarea
- **OE3**: Evaluación en vivo de efectividad con 97% accuracy y FPR<2%
- **OE4**: Demostración práctica del fortalecimiento de ciberseguridad IoT

## Rendimiento del Sistema

- **Tiempo de inferencia**: <2ms por muestra
- **Accuracy (Sintético)**: 97%
- **Accuracy (Real)**: 84.48%
- **False Positive Rate**: <2%
- **F1-Score promedio**: >0.95 (modelo sintético)

## Documentación Adicional

Para más detalles técnicos, consulta:

- [Guía de Implementación](docs/IMPLEMENTACION.md) - Desarrollo paso a paso por fases
- [Documentación de Modelos](docs/MODELOS.md) - Arquitectura y especificaciones técnicas
- [Arquitectura del Proyecto](docs/ARQUITECTURA.md) - Estructura detallada de componentes
- [Objetivos de Tesis](docs/OBJETIVOS_TESIS.md) - Alineación con objetivos académicos

## Licencia

Este proyecto es parte de una tesis de pregrado de la Universidad Señor de Sipán (USS).

## Autor

**Junior** - Tesis de Pregrado
Universidad Señor de Sipán - USS

---

**Nota**: Este es un sistema de demostración académica. Para uso en entornos de producción, se recomienda realizar pruebas adicionales y ajustes de seguridad.
