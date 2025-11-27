# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an IoT Intrusion Detection System (IoT-IDS) demo application built for an undergraduate thesis at Universidad Señor de Sipán. It implements an Autoencoder-FNN (AE-FNN) multi-task deep learning model to detect threats in IoT network traffic.

**Important**: The same AE-FNN model architecture is used in both cases. What differs is the dataset used for training:
- **Synthetic Dataset**: Model trained on balanced synthetic PCA data (97.24% accuracy)
- **Real Dataset**: Model trained on CICIoT2023 real-world data (84.48% accuracy)

**Detected threat types (6 classes):** normal, brute_force, ddos, mitm, scan, spoofing

## Development Commands

**Note**: This project was primarily developed on Windows. Paths use backslashes (`\`) in Windows but forward slashes (`/`) work in the Python code across all platforms.

### Running the Application

**Local Development:**
```bash
# Activate virtual environment (if not already activated)
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Run the Streamlit application
streamlit run app.py

# Run on specific port
streamlit run app.py --server.port 8502
```

**Docker (Recommended for Production):**
```bash
# Using Docker Compose (easiest)
docker-compose up -d

# Manual Docker build and run
docker build -t iot-ids-dashboard .
docker run -d -p 8501:8501 --name iot-ids iot-ids-dashboard

# View logs
docker-compose logs -f
# or
docker logs -f iot-ids

# Stop and remove
docker-compose down
# or
docker stop iot-ids && docker rm iot-ids
```

### Installing Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# Upgrade pip first (recommended)
pip install --upgrade pip
```

### Testing (if implemented)
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage report
pytest --cov=utils tests/

# Run specific test file
pytest tests/test_model_loader.py
```

## Architecture

### Multi-Task Autoencoder-FNN Model

The core model combines two architectures:

1. **Autoencoder (AE)**: Compresses 16 PCA components → 6 latent dimensions → 16 reconstructed
   - Encoder: 16 → 12 (LeakyReLU 0.3) → 8 (LeakyReLU 0.3) → 6 (LeakyReLU 0.3)
   - Decoder: 6 → 8 (LeakyReLU 0.3) → 12 (LeakyReLU 0.3) → 16 (Linear)

2. **Feedforward Classifier (FNN)**: Classifies from latent space
   - From latent: 6 → 64 (LeakyReLU 0.3) → 32 (LeakyReLU 0.3) → 6 (Softmax)

**Combined Loss:** `Total Loss = 0.3 × MSE_reconstruction + 0.7 × CrossEntropy_classification`

This multi-task approach:
- Forces learning of relevant features through reconstruction
- Acts as regularizer to reduce overfitting
- Enables anomaly detection via reconstruction error

### Inference Pipeline

```
Input (16 PCA components, raw)
  → StandardScaler.transform()
  → Model.predict()
  → Extract classification output [1]
  → argmax for class index
  → LabelEncoder.inverse_transform()
  → Return (prediction, probabilities, confidence)
```

**Critical:** Input must be 16 PCA components. Raw network features (35 dimensions) must be transformed via PCA before using the model.

### Session State Management

Streamlit shares state across pages via `st.session_state`:

**Global (set by sidebar_component.py):**
- `selected_model`: 'synthetic' or 'real' (indicates which dataset/trained model is active)
- `model_loaded`: Boolean indicating if model artifacts are loaded
- `model`, `scaler`, `label_encoder`, `class_names`, `metadata`: Model components

**Page-specific examples:**
- Real-time simulation: `simulation_running`, `threat_history` (deque), `threat_counts`
- File analysis: `analysis_results`, `uploaded_file_hash`, `has_labels`

Use `@st.cache_resource` for models (shared across users, not serialized) and `@st.cache_data` for data processing.

### Shared Sidebar Component Pattern

**Critical Architecture Detail**: All pages use a shared sidebar component (`utils/sidebar_component.py`) to ensure consistency.

**How it works:**
1. Import `from utils.sidebar_component import render_sidebar`
2. Call `render_sidebar()` at the top of each page (after `st.set_page_config()`)
3. The function handles:
   - Dataset selection UI (radio buttons for synthetic/real datasets)
   - Automatic model loading when dataset selection changes
   - Display of model metrics and info
   - Storing all artifacts in `st.session_state`
4. Returns the selected dataset type ('synthetic' or 'real')

**Note**: The UI correctly refers to "Dataset" selection, not "Model" selection, since the same
AE-FNN architecture is used for both - only the training data differs.

**Session state variables set by sidebar:**
- `model_loaded` (bool)
- `selected_model` ('synthetic' or 'real')
- `model`, `scaler`, `label_encoder`, `class_names`, `metadata`

This pattern eliminates code duplication and ensures all pages have identical sidebar UI/UX.

## Project Structure

```
Dashboard IoT-IDS/
├── app.py                    # Main entry point - home page, model selector
├── pages/                    # Streamlit multi-page modules
│   ├── 1_Comparacion_Modelos.py    # Side-by-side model comparison
│   ├── 2_Tiempo_Real.py             # Real-time threat simulation
│   ├── 3_Carga_de_Datos.py          # Batch CSV file upload and analysis
│   └── 4_Metricas.py                # Metrics dashboard
├── utils/                    # Reusable utility modules
│   ├── model_loader.py       # Model loading, prediction functions, Keras compatibility
│   ├── sidebar_component.py  # Shared sidebar component for all pages
│   ├── data_simulator.py     # Synthetic traffic generation
│   ├── visualizations.py     # Plotly/matplotlib visualizations
│   └── report_generator.py   # PDF report generation (reportlab)
├── models/                   # Trained models and preprocessing artifacts
│   ├── synthetic/            # Synthetic model artifacts (97.24% accuracy)
│   │   ├── modelo_ae_fnn_iot_synthetic.h5
│   │   ├── scaler_synthetic.pkl
│   │   ├── label_encoder_synthetic.pkl
│   │   ├── class_names_synthetic.npy
│   │   └── model_metadata_synthetic.json
│   └── real/                 # Real model artifacts (84.48% accuracy)
│       ├── modelo_ae_fnn_iot_REAL.h5
│       ├── scaler_REAL.pkl
│       ├── label_encoder_REAL.pkl
│       ├── class_names_REAL.npy
│       └── model_metadata_REAL.json
├── data/                     # Example datasets
│   ├── dataset_pca_capa3_iot_ultra_fixed_100k_dataset.csv  # Synthetic
│   └── CICIoT2023_samples.csv                               # Real
└── docs/                     # Detailed documentation
    ├── ARQUITECTURA.md       # Complete architecture details
    ├── MODELOS.md            # Technical model specifications
    ├── IMPLEMENTACION.md     # Phase-by-phase implementation guide
    └── OBJETIVOS_TESIS.md    # Thesis objectives alignment
```

## Key Implementation Details

### Model Loading Pattern

Models are cached using `@st.cache_resource` to load once and share across all sessions:

```python
@st.cache_resource
def load_model(model_type: str):
    """Load model with all artifacts.

    The model_loader.py includes compatibility handling for different Keras versions:
    1. Attempts normal load_model()
    2. Falls back to manual architecture + load_weights() if needed
    3. Uses custom objects for compatibility with older saved models
    """
    model, scaler, encoder, classes, metadata = load_model(model_type)
    return model, scaler, encoder, classes, metadata
```

**Important**: The `model_loader.py` module includes comprehensive Keras version compatibility
handling that automatically falls back to manual architecture reconstruction if normal loading
fails. This resolves issues with models saved in different Keras/TensorFlow versions.

### Model Output Structure

The AE-FNN model returns **two outputs**:
1. `predictions[0]`: Reconstruction output (16 dimensions) - for autoencoder task
2. `predictions[1]`: Classification output (6 dimensions) - for threat classification

**Always use `predictions[1]` for classification tasks.**

### Data Format Requirements

**Input CSV format for batch analysis:**
- Required: 16 columns named `PC1, PC2, ..., PC16` (PCA components)
- Optional: `label` column for ground truth (enables accuracy metrics)
- Values: Floats (result of PCA transformation)

**For manual prediction:** Array of 16 floats representing PCA-transformed network features.

### Real-Time Simulation Architecture

The real-time page uses a loop pattern with Streamlit's session state:

```python
while st.session_state.simulation_running:
    sample, true_label = generate_traffic_sample()
    prediction, probs, conf = predict_sample(model, scaler, encoder, names, sample)
    update_history(prediction, conf)
    update_visualizations()
    display_alert_if_threat(prediction, conf)
    time.sleep(1)  # 1 sample per second
```

Note: This is synchronous and blocks the UI. For production, consider async processing.

### Known Model Limitations

**Synthetic Model:**
- MITM class has low recall (68%) - confused with normal traffic
- Optimized for synthetic patterns, may underperform on real-world edge cases

**Real Model:**
- 84.48% accuracy (vs 97% synthetic) due to:
  - Real-world data complexity and noise
  - Class imbalance in CICIoT2023 dataset
  - Subtler attack patterns harder to distinguish

**Both:**
- Require exact PCA transformation (35 → 16) from training
- Cannot detect attack types not in training data (6 classes only)
- Performance may degrade with network traffic drift over time

### Critical Dependencies

- **TensorFlow 2.10-2.15**: Model uses Keras format from this version range
- **Streamlit ≥1.25**: Multi-page app features, session state
- **NumPy <2.0**: Compatibility with TensorFlow and scikit-learn versions
- **scikit-learn ≥1.2**: For StandardScaler, LabelEncoder, PCA compatibility

## Common Development Patterns

### Adding a New Visualization

1. Add function to `utils/visualizations.py`
2. Use Plotly for interactive charts (preferred) or matplotlib for static
3. Return figure object (`go.Figure` or `plt.Figure`)
4. Import and use in page modules

### Adding a New Simulation Attack Pattern

1. Edit `utils/data_simulator.py`
2. Add pattern to `ATTACK_PATTERNS` dict with multipliers
3. Update `generate_attack_pattern()` function
4. Add button trigger in `2_⚡_Tiempo_Real.py`

### Creating a New Page

1. Create file in `pages/` with format: `N_Name.py` (e.g., `5_Custom_Analysis.py`)
2. Use `st.set_page_config()` at the top
3. **Import and call `render_sidebar()`** from `utils.sidebar_component` for consistent UI
4. The sidebar automatically loads the selected model into session state
5. Check if model is loaded: `if not st.session_state.get('model_loaded'): st.error(...); st.stop()`
6. Access shared model via `st.session_state['model']`, `st.session_state['scaler']`, etc.

**Critical**: Always call `render_sidebar()` at the beginning of each page to ensure
consistent sidebar appearance and model loading across all pages.

### Working with Model Metadata

Model metadata JSON files contain:
- Architecture details (layer sizes, activations)
- Training hyperparameters (epochs, batch size, learning rate)
- Performance metrics (accuracy, precision, recall, F1)
- Loss function weights (λ₁, λ₂)

Access via: `st.session_state['metadata']` after model loading.

## Testing Considerations

If implementing tests:
- Mock model loading to avoid loading large .h5 files in every test
- Use fixtures for sample data (16-component arrays)
- Test prediction output shape: (class_name: str, probabilities: array(6), confidence: 0-100)
- Verify scaler transforms maintain correct dimensions (16,) → (1, 16) → (16,)

## Documentation Cross-References

For deeper implementation details, consult:
- `docs/ARQUITECTURA.md`: Complete system architecture, data flows, deployment
- `docs/MODELOS.md`: Model specifications, training details, performance analysis
- `docs/IMPLEMENTACION.md`: Step-by-step implementation guide by development phases
- `docs/OBJETIVOS_TESIS.md`: Thesis objectives and how project fulfills them
- `README.md`: Installation, usage, project overview
- `INSTRUCCIONES.md`: Quick start, troubleshooting, deployment options

Each `pages/README.md`, `utils/README.md`, `data/README.md`, `models/README.md` contains component-specific documentation.
