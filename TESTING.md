# Guía de Testing - Dashboard IoT-IDS

Esta guía describe cómo realizar pruebas funcionales del sistema de detección de intrusiones IoT.

## Índice

1. [Tests Automatizados](#tests-automatizados)
2. [Tests Manuales](#tests-manuales)
3. [Casos de Prueba](#casos-de-prueba)
4. [Criterios de Aceptación](#criterios-de-aceptación)

---

## Tests Automatizados

### Script de Testing

El archivo `test_system.py` contiene tests automatizados para verificar:
- Importación de dependencias
- Estructura del proyecto
- Archivos de modelos
- Carga de modelos
- Simulación de datos
- Predicciones
- Visualizaciones
- Generación de reportes

### Ejecutar Tests

```bash
# Activar entorno virtual
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Ejecutar tests
python test_system.py
```

### Salida Esperada

```
=================================================================
      DASHBOARD IoT-IDS - SISTEMA DE TESTS
      Sistema de Deteccion de Intrusiones para IoT
=================================================================

============================================================
TEST 1: Importación de Dependencias
============================================================

[OK] Importar TensorFlow
[OK] Importar Keras
[OK] Importar Scikit-learn
...

============================================================
RESUMEN DE RESULTADOS
============================================================

[OK] Importación de Dependencias
[OK] Estructura del Proyecto
[OK] Archivos de Modelos
...

Tests Pasados: 9/9

TODOS LOS TESTS PASARON EXITOSAMENTE
```

---

## Tests Manuales

### 1. Test de Página Principal

**Objetivo**: Verificar que la aplicación inicia correctamente

**Pasos**:
1. Ejecutar `streamlit run app.py`
2. Abrir http://localhost:8501
3. Verificar que aparece:
   - Título: "Sistema de Detección de Intrusiones para IoT"
   - Sidebar con selector de modelo
   - Diagrama de arquitectura
   - Métricas del modelo

**Criterio de éxito**: Página carga sin errores en < 10 segundos

---

### 2. Test de Comparación de Modelos

**Objetivo**: Verificar funcionalidad de comparación entre modelos

**Pasos**:
1. Navegar a "Comparación de Modelos"
2. Configurar:
   - Número de muestras: 20
   - Filtro: "Todas las amenazas"
3. Click en "Generar Predicciones"
4. Verificar que aparece:
   - Tabla con predicciones
   - Columnas: Muestra, Sintético, Confianza, Real, Confianza, Coinciden
   - Métricas de concordancia

**Criterio de éxito**:
- Predicciones se generan en < 5 segundos
- Tabla muestra 20 filas
- Tasa de concordancia se calcula correctamente

**Casos de borde**:
- 1 muestra (mínimo)
- 100 muestras (máximo)
- Filtro específico (ej. solo "DDoS")

---

### 3. Test de Tiempo Real

**Objetivo**: Verificar simulación en tiempo real

**Pasos**:
1. Navegar a "Tiempo Real"
2. Seleccionar escenario: "Bajo Ataque (DDoS)"
3. Configurar:
   - Muestras por lote: 10
   - Intervalo: 2 segundos
4. Click en "Iniciar Simulación"
5. Observar durante 20 segundos
6. Click en "Detener Simulación"

**Verificar**:
- Gráfico se actualiza cada 2 segundos
- Métricas en vivo se actualizan
- Nivel de riesgo cambia según amenazas
- Tabla de detecciones recientes se llena

**Criterio de éxito**:
- Simulación corre sin interrupciones
- Gráficos se actualizan correctamente
- Stop detiene la simulación inmediatamente

**Casos de prueba**:
- Escenario "Normal" (pocas amenazas esperadas)
- Escenario "Bajo Ataque" (muchas amenazas esperadas)
- Escenario "Escaneo" (amenazas de tipo scan)
- Escenario "Mixto" (variedad de amenazas)

---

### 4. Test de Análisis de Archivo

**Objetivo**: Verificar procesamiento de archivos CSV

**Preparación**: Crear archivo `test_data.csv`:

```csv
PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,PC11,PC12,PC13,PC14,PC15,PC16
-0.5,0.3,-0.2,0.8,-0.1,0.4,-0.6,0.2,-0.3,0.7,-0.4,0.1,-0.2,0.5,-0.1,0.3
0.8,-0.6,0.4,-0.9,0.2,-0.7,0.5,-0.3,0.6,-0.8,0.3,-0.5,0.4,-0.6,0.2,-0.4
-0.3,0.5,-0.7,0.2,-0.5,0.6,-0.4,0.8,-0.2,0.4,-0.6,0.3,-0.5,0.7,-0.3,0.5
```

**Pasos**:
1. Navegar a "Análisis de Archivo"
2. Cargar `test_data.csv`
3. Esperar procesamiento
4. Verificar resultados:
   - Tabla de predicciones
   - Gráfico de distribución
   - Estadísticas de análisis

**Verificar**:
- Todas las muestras se procesan
- Predicciones tienen confianza > 0.5
- Gráfico muestra distribución correcta
- Botones de exportación funcionan

**Criterio de éxito**:
- Procesamiento completa sin errores
- Tiempo de procesamiento < 1s por 100 muestras
- Exportar CSV descarga archivo
- Exportar PDF genera reporte

**Casos de error a manejar**:
- Archivo con columnas incorrectas
- Archivo con valores faltantes
- Archivo vacío
- Formato no CSV

---

### 5. Test de Métricas

**Objetivo**: Verificar visualización de métricas del modelo

**Pasos**:
1. Navegar a "Métricas"
2. Seleccionar modelo: "Sintético"
3. Verificar secciones:
   - Estadísticas del dataset
   - Métricas de rendimiento
   - Matriz de confusión
   - Métricas por clase
4. Cambiar a modelo "Real"
5. Verificar que métricas cambian

**Verificar**:
- Accuracy mostrada: ~97% (sintético), ~84% (real)
- Matriz de confusión tiene 6×6 celdas
- Todas las clases aparecen
- Gráficos se renderizan correctamente

**Criterio de éxito**:
- Todas las métricas son consistentes
- Cambio de modelo actualiza todas las visualizaciones
- No hay valores NaN o infinitos

---

## Casos de Prueba Detallados

### CP-01: Carga Inicial del Sistema

| Campo | Detalle |
|-------|---------|
| **ID** | CP-01 |
| **Nombre** | Carga inicial del sistema |
| **Prioridad** | Alta |
| **Precondiciones** | Python 3.8+, dependencias instaladas |
| **Pasos** | 1. `streamlit run app.py`<br>2. Abrir navegador en localhost:8501 |
| **Resultado esperado** | Página principal carga con todos los elementos |
| **Resultado obtenido** | (A completar durante testing) |
| **Estado** | (Pasó / Falló) |

---

### CP-02: Predicción con Modelo Sintético

| Campo | Detalle |
|-------|---------|
| **ID** | CP-02 |
| **Nombre** | Predicción con modelo sintético |
| **Prioridad** | Alta |
| **Precondiciones** | Modelo sintético cargado |
| **Pasos** | 1. Navegar a Comparación<br>2. Generar 10 muestras<br>3. Verificar predicciones |
| **Resultado esperado** | 10 predicciones con confianza > 0.9 |
| **Resultado obtenido** | (A completar durante testing) |
| **Estado** | (Pasó / Falló) |

---

### CP-03: Predicción con Modelo Real

| Campo | Detalle |
|-------|---------|
| **ID** | CP-03 |
| **Nombre** | Predicción con modelo real |
| **Prioridad** | Alta |
| **Precondiciones** | Modelo real cargado |
| **Pasos** | 1. Seleccionar modelo "Real"<br>2. Navegar a Comparación<br>3. Generar 10 muestras |
| **Resultado esperado** | 10 predicciones con confianza > 0.7 |
| **Resultado obtenido** | (A completar durante testing) |
| **Estado** | (Pasó / Falló) |

---

### CP-04: Simulación Tiempo Real - Escenario Normal

| Campo | Detalle |
|-------|---------|
| **ID** | CP-04 |
| **Nombre** | Simulación tiempo real escenario normal |
| **Prioridad** | Media |
| **Precondiciones** | Sistema funcionando |
| **Pasos** | 1. Tiempo Real<br>2. Escenario "Normal"<br>3. Iniciar por 30s<br>4. Detener |
| **Resultado esperado** | < 10% amenazas detectadas |
| **Resultado obtenido** | (A completar durante testing) |
| **Estado** | (Pasó / Falló) |

---

### CP-05: Simulación Tiempo Real - Bajo Ataque

| Campo | Detalle |
|-------|---------|
| **ID** | CP-05 |
| **Nombre** | Simulación tiempo real bajo ataque DDoS |
| **Prioridad** | Media |
| **Precondiciones** | Sistema funcionando |
| **Pasos** | 1. Tiempo Real<br>2. Escenario "Bajo Ataque"<br>3. Iniciar por 30s<br>4. Detener |
| **Resultado esperado** | > 70% amenazas tipo DDoS detectadas |
| **Resultado obtenido** | (A completar durante testing) |
| **Estado** | (Pasó / Falló) |

---

### CP-06: Análisis de Archivo CSV Válido

| Campo | Detalle |
|-------|---------|
| **ID** | CP-06 |
| **Nombre** | Análisis de archivo CSV con formato correcto |
| **Prioridad** | Alta |
| **Precondiciones** | Archivo test_data.csv creado |
| **Pasos** | 1. Análisis de Archivo<br>2. Upload test_data.csv<br>3. Verificar resultados |
| **Resultado esperado** | Todas las filas procesadas, resultados mostrados |
| **Resultado obtenido** | (A completar durante testing) |
| **Estado** | (Pasó / Falló) |

---

### CP-07: Análisis de Archivo CSV Inválido

| Campo | Detalle |
|-------|---------|
| **ID** | CP-07 |
| **Nombre** | Manejo de archivo CSV con formato incorrecto |
| **Prioridad** | Media |
| **Precondiciones** | Archivo inválido creado |
| **Pasos** | 1. Análisis de Archivo<br>2. Upload archivo con 10 columnas<br>3. Verificar error |
| **Resultado esperado** | Mensaje de error claro, no crash |
| **Resultado obtenido** | (A completar durante testing) |
| **Estado** | (Pasó / Falló) |

---

### CP-08: Exportación de Resultados CSV

| Campo | Detalle |
|-------|---------|
| **ID** | CP-08 |
| **Nombre** | Exportar resultados a CSV |
| **Prioridad** | Baja |
| **Precondiciones** | Resultados de análisis disponibles |
| **Pasos** | 1. Análisis completado<br>2. Click "Descargar CSV"<br>3. Verificar archivo |
| **Resultado esperado** | Archivo CSV descargado con datos correctos |
| **Resultado obtenido** | (A completar durante testing) |
| **Estado** | (Pasó / Falló) |

---

### CP-09: Exportación de Reporte PDF

| Campo | Detalle |
|-------|---------|
| **ID** | CP-09 |
| **Nombre** | Generar reporte PDF |
| **Prioridad** | Media |
| **Precondiciones** | Resultados de análisis disponibles |
| **Pasos** | 1. Análisis completado<br>2. Click "Generar Reporte PDF"<br>3. Verificar archivo |
| **Resultado esperado** | PDF generado con gráficos y tablas |
| **Resultado obtenido** | (A completar durante testing) |
| **Estado** | (Pasó / Falló) |

---

### CP-10: Visualización de Matriz de Confusión

| Campo | Detalle |
|-------|---------|
| **ID** | CP-10 |
| **Nombre** | Mostrar matriz de confusión |
| **Prioridad** | Media |
| **Precondiciones** | Modelo cargado |
| **Pasos** | 1. Navegar a Métricas<br>2. Scroll a matriz de confusión<br>3. Verificar visualización |
| **Resultado esperado** | Matriz 6×6 con colores y valores correctos |
| **Resultado obtenido** | (A completar durante testing) |
| **Estado** | (Pasó / Falló) |

---

## Criterios de Aceptación

### Funcionalidad

- [ ] Todas las páginas cargan sin errores
- [ ] Ambos modelos (sintético y real) se cargan correctamente
- [ ] Predicciones se generan con confianza razonable (>0.5)
- [ ] Simulación en tiempo real funciona sin interrupciones
- [ ] Archivos CSV se procesan correctamente
- [ ] Exportación de resultados funciona (CSV y PDF)
- [ ] Todas las visualizaciones se renderizan

### Rendimiento

- [ ] Página principal carga en < 10 segundos
- [ ] Predicción de 100 muestras en < 5 segundos
- [ ] Simulación tiempo real mantiene intervalo configurado
- [ ] Análisis de 1000 filas en < 10 segundos
- [ ] Generación de PDF en < 5 segundos

### Usabilidad

- [ ] Interfaz intuitiva y clara
- [ ] Mensajes de error informativos
- [ ] Loading indicators visibles durante procesamientos
- [ ] Botones claramente etiquetados
- [ ] Sidebar siempre accesible

### Robustez

- [ ] Sistema maneja archivos inválidos sin crashear
- [ ] Errores de modelo muestran mensaje al usuario
- [ ] Sistema recupera de errores temporales
- [ ] No hay memory leaks en simulación prolongada

### Compatibilidad

- [ ] Funciona en Chrome/Firefox/Edge
- [ ] Funciona en Windows/Linux/macOS
- [ ] Responsive en diferentes resoluciones
- [ ] Compatible con Python 3.8-3.11

---

## Reporte de Bugs

Si encuentras un error durante testing, documentar:

```markdown
### Bug ID: BUG-XXX

**Severidad**: (Crítico / Alto / Medio / Bajo)

**Descripción**: Breve descripción del error

**Pasos para reproducir**:
1. Paso 1
2. Paso 2
3. ...

**Resultado esperado**: Lo que debería pasar

**Resultado obtenido**: Lo que realmente pasó

**Entorno**:
- OS: Windows 10 / Linux / macOS
- Python: 3.x.x
- Navegador: Chrome/Firefox/etc

**Screenshots**: (Si aplica)

**Logs**: (Si aplica)
```

---

## Checklist de Testing Completo

### Tests Automatizados
- [ ] `test_system.py` ejecutado
- [ ] Todos los tests pasan

### Tests Manuales
- [ ] CP-01: Carga inicial
- [ ] CP-02: Predicción modelo sintético
- [ ] CP-03: Predicción modelo real
- [ ] CP-04: Simulación normal
- [ ] CP-05: Simulación ataque
- [ ] CP-06: Análisis CSV válido
- [ ] CP-07: Análisis CSV inválido
- [ ] CP-08: Exportación CSV
- [ ] CP-09: Exportación PDF
- [ ] CP-10: Matriz de confusión

### Criterios de Aceptación
- [ ] Funcionalidad (7/7 checks)
- [ ] Rendimiento (5/5 checks)
- [ ] Usabilidad (5/5 checks)
- [ ] Robustez (4/4 checks)
- [ ] Compatibilidad (4/4 checks)

### Documentación
- [ ] Bugs documentados
- [ ] Resultados de tests registrados
- [ ] Recomendaciones para mejoras

---

**Última actualización**: 2025-01-20
