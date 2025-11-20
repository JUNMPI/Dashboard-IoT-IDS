"""
Script simplificado de testing para Dashboard IoT-IDS
Version compatible con Windows
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test 1: Verificar importaciones"""
    print("\n" + "="*60)
    print("TEST 1: Importacion de Dependencias")
    print("="*60 + "\n")

    required_modules = [
        ('tensorflow', 'TensorFlow'),
        ('keras', 'Keras'),
        ('sklearn', 'Scikit-learn'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('streamlit', 'Streamlit'),
        ('plotly', 'Plotly'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('reportlab', 'ReportLab'),
        ('PIL', 'Pillow')
    ]

    failed = []
    for module, name in required_modules:
        try:
            __import__(module)
            print(f"[OK] {name}")
        except ImportError as e:
            print(f"[FAIL] {name}: {e}")
            failed.append(name)

    return len(failed) == 0

def test_structure():
    """Test 2: Estructura del proyecto"""
    print("\n" + "="*60)
    print("TEST 2: Estructura del Proyecto")
    print("="*60 + "\n")

    required_files = [
        'app.py',
        'requirements.txt',
        'Dockerfile',
        'docker-compose.yml',
        '.dockerignore',
        'README.md',
        'LICENSE',
        'ARCHITECTURE.md',
        'VISUAL_GUIDE.md',
        'TESTING.md'
    ]

    required_dirs = [
        'models/synthetic',
        'models/real',
        'pages',
        'utils',
        'docs/screenshots',
        'docs/gifs'
    ]

    missing = []

    for file in required_files:
        if Path(file).exists():
            print(f"[OK] {file}")
        else:
            print(f"[FAIL] {file} - No encontrado")
            missing.append(file)

    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"[OK] {dir_path}/")
        else:
            print(f"[FAIL] {dir_path}/ - No encontrado")
            missing.append(dir_path)

    return len(missing) == 0

def test_model_files():
    """Test 3: Archivos de modelos"""
    print("\n" + "="*60)
    print("TEST 3: Archivos de Modelos")
    print("="*60 + "\n")

    models = {
        'synthetic': [
            'modelo_ae_fnn_iot_synthetic.h5',
            'scaler_synthetic.pkl',
            'label_encoder_synthetic.pkl',
            'class_names_synthetic.npy',
            'model_metadata_synthetic.json'
        ],
        'real': [
            'modelo_ae_fnn_iot_REAL.h5',
            'scaler_REAL.pkl',
            'label_encoder_REAL.pkl',
            'class_names_REAL.npy',
            'model_metadata_REAL.json'
        ]
    }

    missing = []

    for model_type, files in models.items():
        print(f"\nModelo {model_type.upper()}:")
        for file in files:
            filepath = Path(f'models/{model_type}/{file}')
            if filepath.exists():
                size = filepath.stat().st_size / (1024 * 1024)
                print(f"  [OK] {file} ({size:.2f} MB)")
            else:
                print(f"  [FAIL] {file} - No encontrado")
                missing.append(f"{model_type}/{file}")

    return len(missing) == 0

def test_model_loading():
    """Test 4: Carga de modelos"""
    print("\n" + "="*60)
    print("TEST 4: Carga de Modelos")
    print("="*60 + "\n")

    try:
        from utils.model_loader import load_model
        print("[OK] Importar load_model")
    except ImportError as e:
        print(f"[FAIL] Importar load_model: {e}")
        return False

    success = True

    # Test modelo sintético
    print("\nModelo SINTETICO:")
    try:
        model, scaler, label_encoder, class_names, metadata = load_model('synthetic')
        print("  [OK] Cargar modelo .h5")
        print("  [OK] Cargar scaler")
        print("  [OK] Cargar label_encoder")
        print(f"  [OK] Cargar class_names ({len(class_names)} clases)")
        print("  [OK] Cargar metadata")
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        success = False

    # Test modelo real
    print("\nModelo REAL:")
    try:
        model, scaler, label_encoder, class_names, metadata = load_model('real')
        print("  [OK] Cargar modelo .h5")
        print("  [OK] Cargar scaler")
        print("  [OK] Cargar label_encoder")
        print(f"  [OK] Cargar class_names ({len(class_names)} clases)")
        print("  [OK] Cargar metadata")
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        success = False

    return success

def test_predictions():
    """Test 5: Predicciones"""
    print("\n" + "="*60)
    print("TEST 5: Predicciones del Modelo")
    print("="*60 + "\n")

    try:
        import numpy as np
        from utils.model_loader import load_model
        from utils.data_simulator import generate_traffic_sample

        model, scaler, label_encoder, class_names, metadata = load_model('synthetic')

        # Generar datos de prueba (10 muestras)
        X_test = []
        labels = []
        for _ in range(10):
            sample, label = generate_traffic_sample()
            X_test.append(sample)
            labels.append(label)

        X_test = np.array(X_test)
        print(f"[OK] Generar datos de prueba ({X_test.shape[0]} muestras)")

        # Normalizar
        X_scaled = scaler.transform(X_test)

        # Hacer predicción
        reconstruction, classification = model.predict(X_scaled, verbose=0)
        predicted_indices = np.argmax(classification, axis=1)
        predictions = label_encoder.inverse_transform(predicted_indices)
        confidences = np.max(classification, axis=1)

        if len(predictions) == 10:
            print(f"[OK] Predicciones ({len(predictions)} predicciones)")
        else:
            print(f"[FAIL] Predicciones (esperado: 10, obtenido: {len(predictions)})")
            return False

        if len(confidences) == 10:
            print(f"[OK] Confianzas (rango: [{confidences.min():.3f}, {confidences.max():.3f}])")
        else:
            print(f"[FAIL] Confianzas")
            return False

        # Verificar clases válidas
        valid_classes = set(class_names)
        invalid = set(predictions) - valid_classes

        if len(invalid) == 0:
            print(f"[OK] Clases validas (todas en {list(valid_classes)[:3]}...)")
        else:
            print(f"[FAIL] Clases invalidas: {invalid}")
            return False

        return True

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pages():
    """Test 6: Páginas del dashboard"""
    print("\n" + "="*60)
    print("TEST 6: Paginas del Dashboard")
    print("="*60 + "\n")

    pages = [
        'pages/1_Comparacion_Modelos.py',
        'pages/2_Tiempo_Real.py',
        'pages/3_Analisis_Archivo.py',
        'pages/4_Metricas.py'
    ]

    success = True

    for page in pages:
        if Path(page).exists():
            size = Path(page).stat().st_size
            if size > 0:
                print(f"[OK] {Path(page).name} ({size} bytes)")
            else:
                print(f"[FAIL] {Path(page).name} - Archivo vacio")
                success = False
        else:
            print(f"[FAIL] {Path(page).name} - No encontrado")
            success = False

    return success

def main():
    """Ejecutar todos los tests"""
    print("\n" + "="*60)
    print("     DASHBOARD IoT-IDS - SISTEMA DE TESTS")
    print("     Sistema de Deteccion de Intrusiones para IoT")
    print("="*60)

    tests = [
        ("Importacion de Dependencias", test_imports),
        ("Estructura del Proyecto", test_structure),
        ("Archivos de Modelos", test_model_files),
        ("Carga de Modelos", test_model_loading),
        ("Predicciones del Modelo", test_predictions),
        ("Paginas del Dashboard", test_pages)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[FAIL] {test_name}: Error inesperado - {str(e)}")
            results.append((test_name, False))

    # Resumen
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS")
    print("="*60 + "\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print(f"\nTests Pasados: {passed}/{total}")

    if passed == total:
        print("\nTODOS LOS TESTS PASARON EXITOSAMENTE\n")
        return 0
    else:
        print("\nALGUNOS TESTS FALLARON\n")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
