# Páginas de la Aplicación

Esta carpeta contiene las páginas multi-página de Streamlit.

## Estructura de Páginas

Streamlit automáticamente detecta archivos en esta carpeta y los muestra en el menú lateral.
El orden de aparición se determina por el prefijo numérico en el nombre del archivo.

### Páginas a Implementar

#### `1_🔬_Comparacion_Modelos.py`
**Funcionalidad**: Comparar modelos sintético vs real
- Comparación lado a lado de predicciones
- Generación de muestras aleatorias
- Análisis batch de archivos CSV
- Visualizaciones comparativas
- Tabla de resultados con concordancia

**Componentes principales**:
- Layout en dos columnas
- Botón "Generar Muestra Aleatoria"
- File uploader
- Gráficos de barras de confianza
- Métricas de concordancia

---

#### `2_⚡_Tiempo_Real.py`
**Funcionalidad**: Simulación de detección en tiempo real
- Generación continua de tráfico simulado
- Monitoreo en vivo (1 muestra/segundo)
- Alertas visuales por nivel de riesgo
- Simulación de escenarios específicos
- Log de últimas detecciones

**Componentes principales**:
- Botones Start/Pause/Reset
- Métricas en vivo (total, amenazas, riesgo)
- Gráfico temporal scrollable (últimos 60s)
- Contadores por tipo de ataque
- Botones de simulación (DDoS, Brute Force, etc.)

---

#### `3_📊_Analisis_Archivo.py`
**Funcionalidad**: Análisis batch de archivos CSV
- Upload de archivos CSV
- Validación de formato (16 columnas)
- Procesamiento batch con progress bar
- Cálculo de métricas (si hay labels)
- Exportación de resultados (CSV, PDF)

**Componentes principales**:
- File uploader con validación
- Preview de datos
- Barra de progreso durante análisis
- Tabla de resultados
- Visualizaciones (distribución, top amenazas)
- Matriz de confusión (si hay labels)
- Botones de descarga

---

#### `4_📈_Metricas.py`
**Funcionalidad**: Dashboard de métricas y documentación técnica
- Métricas completas de ambos modelos
- Comparación sintético vs real
- Información técnica de arquitectura
- Justificación académica
- Cumplimiento de objetivos de tesis

**Componentes principales**:
- Tabs (Sintético, Real, Técnico)
- Tarjetas de métricas (accuracy, precision, etc.)
- Matriz de confusión
- F1-Score por clase
- Gráficos comparativos
- Especificaciones técnicas
- Sección de objetivos de tesis

---

## Convención de Nombres

Streamlit usa el siguiente formato para archivos de página:
```
[número]_[emoji]_[Nombre_Con_Guiones_Bajos].py
```

Ejemplos:
- ✅ `1_🔬_Comparacion_Modelos.py`
- ✅ `2_⚡_Tiempo_Real.py`
- ❌ `comparacion modelos.py` (sin número, sin emoji)
- ❌ `1-comparacion.py` (guión en lugar de guión bajo)

## Acceso a Session State

Todas las páginas tienen acceso a `st.session_state`, que se comparte entre páginas:

```python
# Acceder a modelo cargado en app.py
model = st.session_state.get('model')
scaler = st.session_state.get('scaler')

# Guardar datos para otras páginas
st.session_state['analysis_results'] = df
```

## Estructura Básica de una Página

```python
import streamlit as st
from utils.model_loader import predict_sample

st.set_page_config(page_title="Nombre Página", page_icon="🔬")

st.title("Título de la Página")

# Verificar que hay modelo cargado
if 'model' not in st.session_state:
    st.error("⚠️ Por favor selecciona un modelo en la página principal")
    st.stop()

# Tu código aquí
# ...
```

## Notas

- Cada página debe ser autocontenida
- Usa `st.session_state` para compartir datos entre páginas
- Implementa manejo de errores apropiado
- Agrega tooltips y ayuda contextual
- Usa progress bars para operaciones largas

Para más detalles de implementación, consulta `docs/IMPLEMENTACION.md`.
