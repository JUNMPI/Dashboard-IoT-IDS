# Instrucciones de Instalación y Uso

## Instalación Rápida

### 1. Clonar o descargar el repositorio

```bash
git clone <url-del-repositorio>
cd Dashboard\ IoT-IDS
```

### 2. Crear entorno virtual

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**En Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Colocar archivos de modelos

Coloca los siguientes archivos en la carpeta `models/`:

**Modelo Sintético:**
- `modelo_ae_fnn_iot_synthetic.h5`
- `scaler_synthetic.pkl`
- `label_encoder_synthetic.pkl`
- `class_names_synthetic.npy`
- `model_metadata_synthetic.json`

**Modelo Real:**
- `modelo_ae_fnn_iot_real.h5`
- `scaler_real.pkl`
- `label_encoder_real.pkl`
- `class_names_real.npy`
- `model_metadata_real.json`

### 5. Colocar datasets (opcional)

Coloca los datasets en la carpeta `data/`:
- `dataset_pca_capa3_iot_ultra_fixed_100k_dataset.csv`
- `CICIoT2023_samples.csv`

### 6. Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

---

## Verificación de Instalación

### Verificar versiones de Python

```bash
python --version  # Debe ser >= 3.8
```

### Verificar instalación de TensorFlow

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Verificar instalación de Streamlit

```bash
streamlit --version
```

---

## Solución de Problemas Comunes

### Error: "No module named 'tensorflow'"

```bash
pip install tensorflow
```

### Error: "No se encuentra el archivo modelo_ae_fnn_iot_synthetic.h5"

Asegúrate de que los archivos de modelos estén en la carpeta `models/`

### Error de memoria al cargar modelos

Los modelos son pequeños (~150KB), pero si tienes problemas:
- Cierra otras aplicaciones
- Reinicia Python
- Verifica que tienes al menos 2GB de RAM disponible

### La aplicación no se abre en el navegador

1. Verifica que el puerto 8501 no esté ocupado
2. Abre manualmente: `http://localhost:8501`
3. Intenta con otro puerto: `streamlit run app.py --server.port 8502`

---

## Uso de la Aplicación

### Página Principal
1. Selecciona el modelo en el sidebar (Sintético o Real)
2. Observa las métricas generales
3. Lee las instrucciones

### Comparación de Modelos
1. Ve a la página "Comparación de Modelos"
2. Haz clic en "Generar Muestra Aleatoria"
3. Observa las predicciones de ambos modelos
4. Opcionalmente, sube un archivo CSV para análisis batch

### Simulación en Tiempo Real
1. Ve a la página "Tiempo Real"
2. Haz clic en "Iniciar Simulación"
3. Observa las detecciones en vivo
4. Prueba los botones de simulación de ataques específicos
5. Haz clic en "Pausar" para detener

### Análisis de Archivo
1. Ve a la página "Análisis de Archivo"
2. Sube un archivo CSV con 16 columnas (PC1-PC16)
3. Haz clic en "Analizar Archivo"
4. Revisa los resultados y visualizaciones
5. Descarga el reporte si lo deseas

### Dashboard de Métricas
1. Ve a la página "Métricas"
2. Explora los tabs (Sintético, Real, Técnico)
3. Revisa las métricas detalladas
4. Lee la justificación académica

---

## Desarrollo

### Estructura del Proyecto

Consulta `docs/ARQUITECTURA.md` para detalles completos.

### Implementar nuevas funcionalidades

1. Lee `docs/IMPLEMENTACION.md` para la guía por fases
2. Modifica o crea archivos según necesites
3. Reinicia Streamlit para ver cambios: Ctrl+C y ejecuta `streamlit run app.py` nuevamente

### Testing

```bash
# Instalar pytest
pip install pytest pytest-cov

# Ejecutar tests
pytest tests/

# Con reporte de cobertura
pytest --cov=utils tests/
```

---

## Despliegue

### Streamlit Cloud (Recomendado)

1. Push a GitHub
2. Ve a [streamlit.io/cloud](https://streamlit.io/cloud)
3. Conecta tu repositorio
4. Selecciona el archivo `app.py`
5. Deploy

### Docker

```bash
# Construir imagen
docker build -t iot-ids-demo .

# Ejecutar contenedor
docker run -p 8501:8501 iot-ids-demo
```

---

## Recursos Adicionales

### Documentación

- [README.md](README.md) - Descripción general
- [docs/IMPLEMENTACION.md](docs/IMPLEMENTACION.md) - Guía de implementación
- [docs/MODELOS.md](docs/MODELOS.md) - Documentación técnica de modelos
- [docs/ARQUITECTURA.md](docs/ARQUITECTURA.md) - Arquitectura del sistema
- [docs/OBJETIVOS_TESIS.md](docs/OBJETIVOS_TESIS.md) - Alineación con tesis

### Links Útiles

- [Documentación Streamlit](https://docs.streamlit.io/)
- [TensorFlow/Keras](https://www.tensorflow.org/)
- [Plotly Python](https://plotly.com/python/)
- [Scikit-learn](https://scikit-learn.org/)

---

## Soporte

Para problemas o preguntas:
1. Revisa la documentación en `docs/`
2. Verifica la sección "Solución de Problemas" arriba
3. Consulta los README en cada carpeta del proyecto

---

**Última actualización**: Noviembre 2024
