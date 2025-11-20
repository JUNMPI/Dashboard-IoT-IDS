# Modelo Real (CICIoT2023) - Archivos Requeridos

Coloca aquí los archivos de tu **modelo entrenado con datos reales (84.48% accuracy)**:

## ✅ Archivos Necesarios

1. **`modelo_ae_fnn_iot_real.h5`**
   - Modelo Keras entrenado
   - Arquitectura AE-FNN multi-tarea
   - Input: 16 componentes PCA
   - Outputs: reconstrucción (16) + clasificación (8)

2. **`scaler_real.pkl`**
   - StandardScaler de scikit-learn
   - Usado para normalizar datos de entrada
   - Debe ser el mismo usado durante entrenamiento

3. **`label_encoder_real.pkl`**
   - LabelEncoder de scikit-learn
   - Mapea índices a nombres de clases
   - Debe corresponder al orden de entrenamiento

4. **`class_names_real.npy`**
   - Array NumPy con nombres de las 8 clases
   - Ejemplo: `['Benign', 'DDoS', 'DoS', 'Brute_Force', 'Spoofing', 'MITM', 'Scan', 'Recon']`
   - Orden debe coincidir con label_encoder

5. **`model_metadata_real.json`**
   - Metadatos del modelo en formato JSON
   - Incluye métricas de rendimiento
   - Ejemplo:
   ```json
   {
     "accuracy": 84.48,
     "precision": 83.20,
     "recall": 82.95,
     "f1_score": 83.07,
     "fpr": 3.5,
     "dataset": "CICIoT2023 real-world data",
     "n_features": 16,
     "n_classes": 8,
     "training_date": "2024-11-01",
     "epochs": 100,
     "batch_size": 64,
     "optimizer": "Adam",
     "learning_rate": 0.001
   }
   ```

## 📝 Notas

- Si tus archivos tienen nombres diferentes, actualiza las constantes en `utils/model_loader.py` (líneas 40-44)
- Todos los archivos deben ser generados del mismo entrenamiento
- El modelo debe aceptar input de forma (None, 16)
- El modelo debe retornar 2 outputs (reconstrucción + clasificación)
- Datos reales típicamente tienen menor accuracy por mayor complejidad
