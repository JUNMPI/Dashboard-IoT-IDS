# Modelos Entrenados

Esta carpeta contiene los modelos entrenados y sus componentes de preprocesamiento.

## Archivos Requeridos

### Modelo Sintético
- `modelo_ae_fnn_iot_synthetic.h5` - Modelo Keras entrenado (97% accuracy)
- `scaler_synthetic.pkl` - StandardScaler para normalización
- `label_encoder_synthetic.pkl` - LabelEncoder para clases
- `class_names_synthetic.npy` - Array con nombres de las 8 clases
- `model_metadata_synthetic.json` - Metadatos (arquitectura, hiperparámetros)

### Modelo Real
- `modelo_ae_fnn_iot_real.h5` - Modelo Keras entrenado (84.48% accuracy)
- `scaler_real.pkl` - StandardScaler para normalización
- `label_encoder_real.pkl` - LabelEncoder para clases
- `class_names_real.npy` - Array con nombres de las 8 clases
- `model_metadata_real.json` - Metadatos (arquitectura, hiperparámetros)

## Instrucciones

1. Coloca todos los archivos listados arriba en esta carpeta
2. Asegúrate de que los archivos no estén corruptos
3. Verifica que los nombres de archivo coincidan exactamente

## Notas

- Los archivos .h5 contienen los pesos del modelo en formato Keras
- Los archivos .pkl son objetos Python serializados con pickle
- Los archivos .npy son arrays NumPy
- Los archivos .json contienen metadatos en formato JSON

Para más información sobre los modelos, consulta `docs/MODELOS.md`.
