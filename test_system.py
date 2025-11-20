"""
Script de Testing para Dashboard IoT-IDS
Sistema de Detección de Intrusiones para IoT

Este script verifica todas las funcionalidades del sistema:
- Carga de modelos
- Generación de datos sintéticos
- Predicciones
- Visualizaciones
- Generación de reportes
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Colores para terminal
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Imprime un encabezado formateado"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")

def print_test(test_name, status, message=""):
    """Imprime el resultado de un test"""
    if status == "OK":
        symbol = f"{Colors.GREEN}[OK]{Colors.RESET}"
    elif status == "FAIL":
        symbol = f"{Colors.RED}[FAIL]{Colors.RESET}"
    elif status == "WARN":
        symbol = f"{Colors.YELLOW}[WARN]{Colors.RESET}"
    else:
        symbol = f"{Colors.BLUE}[INFO]{Colors.RESET}"

    print(f"{symbol} {test_name}")
    if message:
        print(f"    {message}")

def test_imports():
    """Verifica que todas las dependencias se puedan importar"""
    print_header("TEST 1: Importación de Dependencias")

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
            print_test(f"Importar {name}", "OK")
        except ImportError as e:
            print_test(f"Importar {name}", "FAIL", str(e))
            failed.append(name)

    return len(failed) == 0

def test_project_structure():
    """Verifica la estructura de archivos del proyecto"""
    print_header("TEST 2: Estructura del Proyecto")

    required_files = {
        'app.py': 'Aplicación principal',
        'requirements.txt': 'Dependencias',
        'Dockerfile': 'Docker container',
        'docker-compose.yml': 'Docker Compose',
        '.dockerignore': 'Docker exclusions',
        'README.md': 'Documentación',
        'LICENSE': 'Licencia'
    }

    required_dirs = {
        'models/synthetic': 'Modelos sintéticos',
        'models/real': 'Modelos reales',
        'pages': 'Páginas del dashboard',
        'utils': 'Utilidades'
    }

    missing = []

    for file, desc in required_files.items():
        if Path(file).exists():
            print_test(f"Archivo {file}", "OK", desc)
        else:
            print_test(f"Archivo {file}", "FAIL", f"No encontrado: {desc}")
            missing.append(file)

    for dir_path, desc in required_dirs.items():
        if Path(dir_path).exists():
            print_test(f"Directorio {dir_path}/", "OK", desc)
        else:
            print_test(f"Directorio {dir_path}/", "FAIL", f"No encontrado: {desc}")
            missing.append(dir_path)

    return len(missing) == 0

def test_model_files():
    """Verifica que existan todos los archivos de modelos"""
    print_header("TEST 3: Archivos de Modelos")

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
        print(f"\n{Colors.BOLD}Modelo {model_type.upper()}:{Colors.RESET}")
        for file in files:
            filepath = Path(f'models/{model_type}/{file}')
            if filepath.exists():
                size = filepath.stat().st_size
                size_mb = size / (1024 * 1024)
                print_test(f"  {file}", "OK", f"{size_mb:.2f} MB")
            else:
                print_test(f"  {file}", "FAIL", "Archivo no encontrado")
                missing.append(f"{model_type}/{file}")

    return len(missing) == 0

def test_model_loading():
    """Verifica que los modelos se puedan cargar correctamente"""
    print_header("TEST 4: Carga de Modelos")

    try:
        from utils.model_loader import ModelLoader
        print_test("Importar ModelLoader", "OK")
    except ImportError as e:
        print_test("Importar ModelLoader", "FAIL", str(e))
        return False

    success = True

    # Test modelo sintético
    try:
        print(f"\n{Colors.BOLD}Cargando modelo SINTÉTICO:{Colors.RESET}")
        loader_synthetic = ModelLoader(model_type='synthetic')
        print_test("  Cargar modelo .h5", "OK")
        print_test("  Cargar scaler", "OK")
        print_test("  Cargar label_encoder", "OK")
        print_test("  Cargar class_names", "OK", f"{len(loader_synthetic.class_names)} clases")
        print_test("  Cargar metadata", "OK")
    except Exception as e:
        print_test("  Cargar modelo sintético", "FAIL", str(e))
        success = False

    # Test modelo real
    try:
        print(f"\n{Colors.BOLD}Cargando modelo REAL:{Colors.RESET}")
        loader_real = ModelLoader(model_type='real')
        print_test("  Cargar modelo .h5", "OK")
        print_test("  Cargar scaler", "OK")
        print_test("  Cargar label_encoder", "OK")
        print_test("  Cargar class_names", "OK", f"{len(loader_real.class_names)} clases")
        print_test("  Cargar metadata", "OK")
    except Exception as e:
        print_test("  Cargar modelo real", "FAIL", str(e))
        success = False

    return success

def test_data_simulation():
    """Verifica la generación de datos sintéticos"""
    print_header("TEST 5: Simulación de Datos")

    try:
        from utils.data_simulator import DataSimulator
        print_test("Importar DataSimulator", "OK")

        simulator = DataSimulator()
        print_test("Crear instancia DataSimulator", "OK")

        # Test generar muestra única
        sample = simulator.generate_sample(threat_type='normal')
        if len(sample) == 16:
            print_test("Generar muestra única", "OK", f"Shape: {sample.shape}")
        else:
            print_test("Generar muestra única", "FAIL", f"Shape incorrecta: {sample.shape}")
            return False

        # Test generar batch
        batch = simulator.generate_batch(n_samples=100, threat_type='ddos')
        if batch.shape == (100, 16):
            print_test("Generar batch de 100 muestras", "OK", f"Shape: {batch.shape}")
        else:
            print_test("Generar batch", "FAIL", f"Shape incorrecta: {batch.shape}")
            return False

        # Test escenarios
        for scenario in ['normal', 'attack', 'scan', 'mixed']:
            try:
                samples = simulator.generate_scenario(scenario=scenario, n_samples=50)
                print_test(f"Escenario '{scenario}'", "OK", f"{len(samples)} muestras")
            except Exception as e:
                print_test(f"Escenario '{scenario}'", "FAIL", str(e))
                return False

        return True

    except Exception as e:
        print_test("DataSimulator", "FAIL", str(e))
        return False

def test_predictions():
    """Verifica que las predicciones funcionen correctamente"""
    print_header("TEST 6: Predicciones del Modelo")

    try:
        from utils.model_loader import ModelLoader
        from utils.data_simulator import DataSimulator

        loader = ModelLoader(model_type='synthetic')
        simulator = DataSimulator()

        # Generar datos de prueba
        X_test = simulator.generate_batch(n_samples=10, threat_type='normal')
        print_test("Generar datos de prueba", "OK", f"{X_test.shape[0]} muestras")

        # Hacer predicción
        predictions, confidences = loader.predict(X_test)

        if len(predictions) == 10:
            print_test("Predicciones", "OK", f"{len(predictions)} predicciones")
        else:
            print_test("Predicciones", "FAIL", f"Número incorrecto: {len(predictions)}")
            return False

        if len(confidences) == 10:
            print_test("Confianzas", "OK", f"Rango: [{confidences.min():.3f}, {confidences.max():.3f}]")
        else:
            print_test("Confianzas", "FAIL", f"Número incorrecto: {len(confidences)}")
            return False

        # Verificar que las predicciones estén en las clases válidas
        valid_classes = set(loader.class_names)
        invalid = set(predictions) - valid_classes

        if len(invalid) == 0:
            print_test("Clases válidas", "OK", f"Todas las predicciones en {list(valid_classes)}")
        else:
            print_test("Clases válidas", "FAIL", f"Clases inválidas: {invalid}")
            return False

        return True

    except Exception as e:
        print_test("Predicciones", "FAIL", str(e))
        return False

def test_visualizations():
    """Verifica que las visualizaciones se puedan generar"""
    print_header("TEST 7: Visualizaciones")

    try:
        from utils.visualizations import create_confusion_matrix, create_metrics_chart
        print_test("Importar módulo visualizations", "OK")

        # Test matriz de confusión
        try:
            y_true = np.random.randint(0, 6, size=100)
            y_pred = np.random.randint(0, 6, size=100)
            classes = ['normal', 'bruteforce', 'ddos', 'mitm', 'scan', 'spoofing']

            fig = create_confusion_matrix(y_true, y_pred, classes)
            print_test("Crear matriz de confusión", "OK")
        except Exception as e:
            print_test("Crear matriz de confusión", "FAIL", str(e))
            return False

        # Test gráfico de métricas
        try:
            metrics = {
                'accuracy': 0.95,
                'precision': 0.93,
                'recall': 0.94,
                'f1': 0.93
            }
            fig = create_metrics_chart(metrics)
            print_test("Crear gráfico de métricas", "OK")
        except Exception as e:
            print_test("Crear gráfico de métricas", "FAIL", str(e))
            return False

        return True

    except Exception as e:
        print_test("Visualizaciones", "FAIL", str(e))
        return False

def test_report_generation():
    """Verifica la generación de reportes PDF"""
    print_header("TEST 8: Generación de Reportes")

    try:
        from utils.report_generator import ReportGenerator
        print_test("Importar ReportGenerator", "OK")

        # Crear datos de prueba
        results = pd.DataFrame({
            'sample_id': range(10),
            'prediction': ['normal'] * 5 + ['ddos'] * 5,
            'confidence': np.random.uniform(0.7, 0.99, 10),
            'is_threat': [False] * 5 + [True] * 5
        })

        report_gen = ReportGenerator()
        print_test("Crear instancia ReportGenerator", "OK")

        # Test generar reporte
        try:
            output_path = "test_report.pdf"
            report_gen.generate_report(
                results=results,
                model_type='synthetic',
                output_path=output_path
            )

            if Path(output_path).exists():
                size = Path(output_path).stat().st_size / 1024
                print_test("Generar reporte PDF", "OK", f"Tamaño: {size:.1f} KB")
                # Limpiar archivo de prueba
                Path(output_path).unlink()
            else:
                print_test("Generar reporte PDF", "FAIL", "Archivo no creado")
                return False

        except Exception as e:
            print_test("Generar reporte PDF", "FAIL", str(e))
            return False

        return True

    except Exception as e:
        print_test("ReportGenerator", "FAIL", str(e))
        return False

def test_page_imports():
    """Verifica que todas las páginas se puedan importar"""
    print_header("TEST 9: Páginas del Dashboard")

    pages = [
        'pages/1_Comparacion_Modelos.py',
        'pages/2_Tiempo_Real.py',
        'pages/3_Analisis_Archivo.py',
        'pages/4_Metricas.py'
    ]

    success = True

    for page in pages:
        if Path(page).exists():
            # Verificar que el archivo no esté vacío
            size = Path(page).stat().st_size
            if size > 0:
                print_test(f"Página {Path(page).name}", "OK", f"{size} bytes")
            else:
                print_test(f"Página {Path(page).name}", "FAIL", "Archivo vacío")
                success = False
        else:
            print_test(f"Página {Path(page).name}", "FAIL", "No encontrada")
            success = False

    return success

def run_all_tests():
    """Ejecuta todos los tests"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("=" * 65)
    print("      DASHBOARD IoT-IDS - SISTEMA DE TESTS                  ")
    print("      Sistema de Deteccion de Intrusiones para IoT          ")
    print("=" * 65)
    print(f"{Colors.RESET}\n")

    tests = [
        ("Importación de Dependencias", test_imports),
        ("Estructura del Proyecto", test_project_structure),
        ("Archivos de Modelos", test_model_files),
        ("Carga de Modelos", test_model_loading),
        ("Simulación de Datos", test_data_simulation),
        ("Predicciones del Modelo", test_predictions),
        ("Visualizaciones", test_visualizations),
        ("Generación de Reportes", test_report_generation),
        ("Páginas del Dashboard", test_page_imports)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_test(test_name, "FAIL", f"Error inesperado: {str(e)}")
            results.append((test_name, False))

    # Resumen final
    print_header("RESUMEN DE RESULTADOS")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "OK" if result else "FAIL"
        print_test(test_name, status)

    print(f"\n{Colors.BOLD}Tests Pasados: {passed}/{total}{Colors.RESET}")

    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}TODOS LOS TESTS PASARON EXITOSAMENTE{Colors.RESET}\n")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}ALGUNOS TESTS FALLARON{Colors.RESET}\n")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
