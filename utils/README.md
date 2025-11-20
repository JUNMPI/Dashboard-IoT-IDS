# Utilidades

Esta carpeta contiene módulos de utilidades reutilizables para la aplicación.

## Módulos a Implementar

### `__init__.py`
Archivo de inicialización del paquete Python (puede estar vacío).

### `model_loader.py`
Funciones para cargar y usar los modelos:
- `load_synthetic_model()` - Carga modelo sintético
- `load_real_model()` - Carga modelo real
- `predict_sample()` - Predicción de muestra única
- `predict_batch()` - Predicción batch
- `verify_model_compatibility()` - Verifica estructura del modelo

### `data_simulator.py`
Generación de datos sintéticos para simulación:
- `generate_traffic_sample()` - Genera muestra aleatoria
- `generate_attack_pattern()` - Genera patrón de ataque específico
- `generate_attack_burst()` - Genera ráfaga de ataque
- `generate_mixed_traffic()` - Genera tráfico mixto temporal

### `visualizations.py`
Funciones de visualización con Plotly/Matplotlib:
- `plot_confusion_matrix()` - Matriz de confusión
- `plot_temporal_chart()` - Gráfico temporal
- `plot_class_distribution()` - Distribución de clases
- `plot_confidence_comparison()` - Comparación de confianzas
- `plot_metrics_radar()` - Gráfico radar de métricas
- `create_risk_gauge()` - Velocímetro de riesgo

### `report_generator.py`
Generación de reportes PDF:
- `generate_pdf_report()` - Genera reporte completo
- `add_header()` - Agrega encabezado
- `add_metrics_section()` - Agrega sección de métricas
- `add_visualizations()` - Agrega gráficos
- `add_footer()` - Agrega pie de página

## Uso

```python
# Importar módulos
from utils.model_loader import load_synthetic_model, predict_sample
from utils.data_simulator import generate_traffic_sample
from utils.visualizations import plot_confusion_matrix

# Usar funciones
model, scaler, encoder, names, meta = load_synthetic_model()
sample, label = generate_traffic_sample('DDoS')
pred, probs, conf = predict_sample(model, scaler, encoder, names, sample)
```

## Notas

- Todos los módulos deben usar docstrings claros
- Incluir type hints cuando sea posible
- Manejar errores apropiadamente
- Usar @st.cache_resource/@st.cache_data cuando corresponda

Para más detalles de implementación, consulta `docs/IMPLEMENTACION.md`.
