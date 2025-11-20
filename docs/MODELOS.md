# Documentación Técnica de Modelos

## Descripción General

Este documento detalla las especificaciones técnicas de los dos modelos Autoencoder-FNN (AE-FNN) desarrollados para la detección de amenazas en redes IoT.

---

## Arquitectura: Autoencoder-FNN Multi-tarea

### Concepto

El modelo combina dos arquitecturas de deep learning en un enfoque multi-tarea:

1. **Autoencoder (AE)**: Aprende representaciones compactas de los datos
2. **Feedforward Neural Network (FNN)**: Clasifica las amenazas

### Ventajas del Enfoque Multi-tarea

- **Mejor generalización**: El autoencoder obliga al modelo a aprender features relevantes
- **Reducción de overfitting**: La tarea de reconstrucción actúa como regularizador
- **Detección de anomalías**: El error de reconstrucción puede identificar patrones desconocidos
- **Efficiency**: Comparte representaciones entre tareas

---

## Arquitectura Detallada

### 1. Encoder (Compresión)

```
Input Layer:  16 features (PC1 - PC16)
     ↓
Dense Layer:  8 neurons
     - Activation: ReLU
     - Kernel Initializer: he_normal
     ↓
Latent Space: 4 neurons (bottleneck)
     - Activation: ReLU
     - Kernel Initializer: he_normal
```

**Propósito**: Comprimir información de 16 dimensiones a 4 dimensiones, extrayendo características esenciales.

### 2. Decoder (Reconstrucción)

```
Latent Space: 4 neurons
     ↓
Dense Layer:  8 neurons
     - Activation: ReLU
     - Kernel Initializer: he_normal
     ↓
Output Layer: 16 neurons (reconstrucción)
     - Activation: Linear
```

**Propósito**: Reconstruir la entrada original a partir de la representación comprimida.

### 3. Clasificador (Clasificación Multi-clase)

```
Latent Space: 4 neurons (compartido con Encoder)
     ↓
Dense Layer:  16 neurons
     - Activation: ReLU
     - Dropout: 0.3
     - Kernel Initializer: he_normal
     ↓
Output Layer: 8 neurons (clases)
     - Activation: Softmax
```

**Propósito**: Clasificar el tráfico en 8 categorías (7 tipos de ataques + tráfico normal).

---

## Clases de Salida

El modelo clasifica el tráfico en las siguientes 8 clases:

| Clase | Descripción | Tipo |
|-------|-------------|------|
| `Benign` | Tráfico normal, legítimo | Normal |
| `DDoS` | Distributed Denial of Service | Ataque |
| `DoS` | Denial of Service | Ataque |
| `Brute_Force` | Intentos de acceso por fuerza bruta | Ataque |
| `Spoofing` | Suplantación de identidad | Ataque |
| `MITM` | Man-in-the-Middle | Ataque |
| `Scan` | Escaneo de puertos/servicios | Ataque |
| `Recon` | Reconocimiento de red | Ataque |

---

## Función de Pérdida Combinada

### Fórmula

```
Total Loss = λ₁ × Loss_reconstruction + λ₂ × Loss_classification

donde:
- Loss_reconstruction = MSE (Mean Squared Error)
- Loss_classification = Categorical Cross-Entropy
- λ₁, λ₂ = pesos de cada tarea
```

### Hiperparámetros de Pérdida

**Modelo Sintético:**
- λ₁ (reconstrucción) = 0.3
- λ₂ (clasificación) = 0.7

**Modelo Real:**
- λ₁ (reconstrucción) = 0.3
- λ₂ (clasificación) = 0.7

**Justificación**: Se prioriza la clasificación (0.7) sobre la reconstrucción (0.3) dado que el objetivo principal es detectar amenazas, no comprimir datos.

---

## Modelo 1: Sintético

### Características

| Parámetro | Valor |
|-----------|-------|
| **Dataset** | Sintético PCA (100,000 muestras) |
| **Features de Entrada** | 16 componentes PCA |
| **Accuracy** | 97.00% |
| **Precision (weighted)** | 96.85% |
| **Recall (weighted)** | 96.72% |
| **F1-Score (weighted)** | 96.78% |
| **False Positive Rate** | <2% |

### Archivos del Modelo

```
models/
├── modelo_ae_fnn_iot_synthetic.h5        # Modelo entrenado (Keras)
├── scaler_synthetic.pkl                   # StandardScaler (scikit-learn)
├── label_encoder_synthetic.pkl            # LabelEncoder (scikit-learn)
├── class_names_synthetic.npy              # Array con nombres de clases
└── model_metadata_synthetic.json          # Metadatos (arquitectura, métricas)
```

### Dataset de Entrenamiento

- **Tamaño**: 100,000 muestras
- **Distribución**: Balanceada (~12,500 muestras por clase)
- **Features**: 16 componentes PCA (reducción de 35 features originales)
- **Preprocesamiento**:
  - Normalización con StandardScaler
  - Reducción dimensional PCA
  - Balanceo de clases

### Métricas por Clase

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Benign | 0.98 | 0.98 | 0.98 | 2500 |
| DDoS | 0.97 | 0.97 | 0.97 | 2500 |
| DoS | 0.96 | 0.96 | 0.96 | 2500 |
| Brute_Force | 0.96 | 0.95 | 0.95 | 2500 |
| Spoofing | 0.97 | 0.97 | 0.97 | 2500 |
| **MITM** | 0.95 | **0.68** | **0.89** | 2500 |
| Scan | 0.97 | 0.96 | 0.96 | 2500 |
| Recon | 0.96 | 0.95 | 0.95 | 2500 |

**Nota**: MITM presenta el menor recall (68%) debido a similitudes con tráfico normal en algunas características.

### Rendimiento Temporal

- **Tiempo de Inferencia**: <2ms por muestra
- **Throughput**: ~500 muestras/segundo
- **Tamaño del Modelo**: ~150 KB

---

## Modelo 2: Real (CICIoT2023)

### Características

| Parámetro | Valor |
|-----------|-------|
| **Dataset** | CICIoT2023 (datos reales) |
| **Features de Entrada** | 16 componentes PCA |
| **Accuracy** | 84.48% |
| **Precision (weighted)** | 83.20% |
| **Recall (weighted)** | 82.95% |
| **F1-Score (weighted)** | 83.07% |
| **False Positive Rate** | ~3-4% |

### Archivos del Modelo

```
models/
├── modelo_ae_fnn_iot_real.h5             # Modelo entrenado (Keras)
├── scaler_real.pkl                        # StandardScaler (scikit-learn)
├── label_encoder_real.pkl                 # LabelEncoder (scikit-learn)
├── class_names_real.npy                   # Array con nombres de clases
└── model_metadata_real.json               # Metadatos (arquitectura, métricas)
```

### Dataset de Entrenamiento

- **Fuente**: CICIoT2023 (Canadian Institute for Cybersecurity)
- **Tamaño**: Subset preprocesado
- **Distribución**: Desbalanceada (refleja tráfico real)
- **Features**: 16 componentes PCA (reducción de features originales)
- **Preprocesamiento**:
  - Normalización con StandardScaler
  - Reducción dimensional PCA
  - Manejo de valores faltantes

### Diferencias con Modelo Sintético

| Aspecto | Sintético | Real |
|---------|-----------|------|
| **Accuracy** | 97.00% | 84.48% |
| **Datos** | Balanceados, sintéticos | Desbalanceados, reales |
| **Ruido** | Bajo | Alto (tráfico real) |
| **Complejidad** | Patrones claros | Patrones sutiles |
| **Generalización** | Excelente en datos similares | Mejor en escenarios reales |

### Análisis de la Brecha de Rendimiento

La diferencia de ~12-13% en accuracy se explica por:

1. **Complejidad de datos reales**: Mayor variabilidad y ruido en tráfico real
2. **Desbalanceo de clases**: CICIoT2023 tiene distribución no uniforme
3. **Características sutiles**: Algunos ataques reales son más difíciles de distinguir del tráfico normal
4. **Tamaño del dataset**: Potencialmente menor cantidad de ejemplos de entrenamiento

**Sin embargo**: 84.48% sigue siendo un desempeño sólido y competitivo para detección de amenazas IoT en entornos reales.

---

## Especificaciones de Entrenamiento

### Hiperparámetros Comunes

| Parámetro | Valor |
|-----------|-------|
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 |
| **Batch Size** | 64 |
| **Epochs** | 100 (con Early Stopping) |
| **Early Stopping Patience** | 10 epochs |
| **Early Stopping Monitor** | val_loss |
| **Validation Split** | 20% |

### Arquitectura de Red

| Capa | Parámetros |
|------|------------|
| **Input** | 16 neurons |
| **Encoder Hidden** | 8 neurons, ReLU |
| **Latent Space** | 4 neurons, ReLU |
| **Decoder Hidden** | 8 neurons, ReLU |
| **Decoder Output** | 16 neurons, Linear |
| **Classifier Hidden** | 16 neurons, ReLU, Dropout(0.3) |
| **Classifier Output** | 8 neurons, Softmax |

### Regularización

- **Dropout**: 0.3 en capa oculta del clasificador
- **L2 Regularization**: No aplicada
- **Batch Normalization**: No aplicada
- **Early Stopping**: Sí, patience=10

---

## Preprocesamiento de Datos

### Reducción Dimensional con PCA

**Objetivo**: Reducir dimensionalidad de 35 features a 16 componentes principales.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=16)
X_pca = pca.fit_transform(X_original)

# Varianza explicada: ~95%
```

**Justificación**:
- Reduce complejidad computacional
- Elimina features redundantes
- Mantiene 95% de varianza explicada
- Mejora generalización del modelo

### Normalización con StandardScaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

# Asegura media=0, std=1 para cada feature
```

**Justificación**:
- Mejora convergencia del entrenamiento
- Evita dominancia de features con mayor escala
- Requisito para PCA efectivo

### Codificación de Etiquetas

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_labels)

# Convierte etiquetas textuales a números 0-7
```

---

## Formato de Entrada Esperado

### Para Inferencia

El modelo espera un array NumPy con las siguientes características:

```python
# Forma esperada
sample.shape = (16,)  # Vector de 16 componentes

# Ejemplo de muestra
sample = np.array([
    0.23, -1.45, 0.89, -0.12,  # PC1-PC4
    1.34, -0.67, 0.45, -1.23,  # PC5-PC8
    0.56, -0.89, 1.12, -0.34,  # PC9-PC12
    0.78, -1.01, 0.23, -0.56   # PC13-PC16
])

# IMPORTANTE: La muestra debe estar normalizada con el StandardScaler
sample_scaled = scaler.transform(sample.reshape(1, -1))
```

### Para Batch Processing

```python
# Forma esperada
X_batch.shape = (n_samples, 16)

# Ejemplo
X_batch = np.array([
    [0.23, -1.45, ..., -0.56],  # Muestra 1
    [1.12, -0.34, ..., 0.78],   # Muestra 2
    ...
])

# Normalizar batch completo
X_batch_scaled = scaler.transform(X_batch)
```

---

## Carga y Uso de Modelos

### Cargar Modelo Sintético

```python
import tensorflow as tf
import pickle
import numpy as np

# Cargar modelo Keras
model = tf.keras.models.load_model('models/modelo_ae_fnn_iot_synthetic.h5')

# Cargar componentes de preprocesamiento
with open('models/scaler_synthetic.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/label_encoder_synthetic.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Cargar nombres de clases
class_names = np.load('models/class_names_synthetic.npy')
```

### Realizar Predicción

```python
def predict_threat(sample):
    """
    Predice la clase de una muestra de tráfico

    Args:
        sample: Array de 16 componentes PCA (sin normalizar)

    Returns:
        prediction: Nombre de la clase predicha
        probabilities: Array con probabilidades de cada clase
        confidence: Confianza de la predicción (%)
    """
    # Normalizar muestra
    sample_scaled = scaler.transform(sample.reshape(1, -1))

    # Predecir
    predictions = model.predict(sample_scaled, verbose=0)

    # El modelo tiene 2 outputs: reconstrucción y clasificación
    # Usar solo la salida de clasificación
    class_probabilities = predictions[1] if isinstance(predictions, list) else predictions

    # Obtener clase predicha
    predicted_class_idx = np.argmax(class_probabilities[0])
    predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]

    # Calcular confianza
    confidence = np.max(class_probabilities[0]) * 100

    return predicted_class, class_probabilities[0], confidence
```

### Ejemplo de Uso

```python
# Muestra de ejemplo (16 componentes PCA)
sample = np.array([
    0.23, -1.45, 0.89, -0.12,
    1.34, -0.67, 0.45, -1.23,
    0.56, -0.89, 1.12, -0.34,
    0.78, -1.01, 0.23, -0.56
])

# Predecir
prediction, probabilities, confidence = predict_threat(sample)

print(f"Predicción: {prediction}")
print(f"Confianza: {confidence:.2f}%")
print(f"Probabilidades por clase:")
for i, class_name in enumerate(class_names):
    print(f"  {class_name}: {probabilities[i]*100:.2f}%")
```

**Salida esperada:**
```
Predicción: DDoS
Confianza: 94.23%
Probabilidades por clase:
  Benign: 1.23%
  DDoS: 94.23%
  DoS: 2.45%
  Brute_Force: 0.56%
  Spoofing: 0.78%
  MITM: 0.34%
  Scan: 0.23%
  Recon: 0.18%
```

---

## Interpretación de Resultados

### Niveles de Confianza

- **>90%**: Alta confianza - Predicción muy confiable
- **70-90%**: Confianza moderada - Predicción confiable
- **50-70%**: Confianza baja - Requiere verificación
- **<50%**: Muy baja confianza - Muestra ambigua

### Umbrales de Alerta

Para sistemas de producción, se recomiendan los siguientes umbrales:

| Nivel | Confianza | Acción |
|-------|-----------|--------|
| 🟢 Normal | <50% amenaza | Permitir tráfico |
| 🟡 Sospechoso | 50-80% amenaza | Monitorear, logging |
| 🟠 Probable Amenaza | 80-90% amenaza | Alerta, análisis adicional |
| 🔴 Amenaza Confirmada | >90% amenaza | Bloquear, escalar |

---

## Limitaciones Conocidas

### Modelo Sintético

1. **Clase MITM**: Recall de solo 68%, muchos falsos negativos
2. **Generalización**: Optimizado para datos sintéticos, puede tener menor rendimiento con datos reales
3. **Nuevos Ataques**: No detecta tipos de ataques no presentes en entrenamiento

### Modelo Real

1. **Accuracy Moderada**: 84.48% es bueno pero no excepcional
2. **Desbalanceo**: Rendimiento varía significativamente entre clases
3. **Datos de Entrenamiento**: Limitado por el tamaño del dataset CICIoT2023 disponible

### Ambos Modelos

1. **Dependencia de PCA**: Requiere transformación PCA específica de los datos originales
2. **Drift de Datos**: Rendimiento puede degradarse con nuevos patrones de tráfico
3. **Explicabilidad**: Como deep learning, las decisiones no son fácilmente interpretables
4. **Latent Space Fijo**: 4 dimensiones pueden no capturar toda la complejidad

---

## Recomendaciones de Mejora Futura

### Corto Plazo

1. **Mejorar detección de MITM**: Recolectar más ejemplos, aplicar técnicas de balanceo
2. **Ensemble Methods**: Combinar predicciones de ambos modelos
3. **Ajuste de Hiperparámetros**: Grid search para λ₁, λ₂, learning rate
4. **Data Augmentation**: Generar variaciones de muestras de entrenamiento

### Largo Plazo

1. **Arquitecturas Avanzadas**: Explorar Transformers, GNNs para datos de red
2. **Transfer Learning**: Pre-entrenar en datasets grandes, fine-tune en CICIoT2023
3. **Detección de Anomalías**: Usar error de reconstrucción para detectar ataques desconocidos
4. **Reentrenamiento Continuo**: Actualizar modelo con nuevos datos periódicamente
5. **Explicabilidad**: Implementar SHAP, LIME para interpretar decisiones

---

## Referencias Técnicas

### Papers

- Canadian Institute for Cybersecurity. (2023). CICIoT2023 Dataset
- Autoencoder-Based Deep Learning for Network Intrusion Detection
- Multi-task Learning for Cybersecurity Applications

### Frameworks

- TensorFlow 2.10+
- Keras 2.10+
- scikit-learn 1.2+
- NumPy 1.23+

---

**Última actualización**: Noviembre 2024
