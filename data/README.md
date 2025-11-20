# Datasets

Esta carpeta contiene los datasets utilizados para entrenamiento y pruebas.

## Archivos Principales

### Dataset Sintético
- `dataset_pca_capa3_iot_ultra_fixed_100k_dataset.csv`
  - 100,000 muestras sintéticas
  - 16 componentes PCA (PC1-PC16)
  - Columna 'label' con clases
  - Balanceado (~12,500 muestras por clase)

### Dataset Real
- `CICIoT2023_samples.csv`
  - Muestras del dataset CICIoT2023
  - Preprocesado y transformado con PCA
  - 16 componentes PCA
  - Refleja distribución real de tráfico IoT

## Estructura de los CSV

```csv
PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,PC11,PC12,PC13,PC14,PC15,PC16,label
0.23,-1.45,0.89,-0.12,1.34,-0.67,0.45,-1.23,0.56,-0.89,1.12,-0.34,0.78,-1.01,0.23,-0.56,DDoS
...
```

## Carpeta de Ejemplos

Opcionalmente, crea una subcarpeta `ejemplos/` con archivos CSV más pequeños para pruebas:

- `sample_normal.csv` - Solo tráfico normal (100 muestras)
- `sample_ddos.csv` - Solo ataques DDoS (100 muestras)
- `sample_mixed.csv` - Mezcla de diferentes tipos (200 muestras)

## Clases Disponibles

1. Benign - Tráfico normal
2. DDoS - Distributed Denial of Service
3. DoS - Denial of Service
4. Brute_Force - Ataques de fuerza bruta
5. Spoofing - Suplantación
6. MITM - Man-in-the-Middle
7. Scan - Escaneo de puertos
8. Recon - Reconocimiento

## Notas

- Los archivos CSV deben tener encoding UTF-8
- Los valores son números flotantes (resultado de PCA)
- La columna 'label' es opcional (si no existe, solo se hacen predicciones)

Para más información sobre los datos, consulta `docs/IMPLEMENTACION.md`.
