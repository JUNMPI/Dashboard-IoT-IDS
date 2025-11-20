# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an IoT Intrusion Detection System (IoT-IDS) demo application built for an undergraduate thesis at Universidad Señor de Sipán. It implements an Autoencoder-FNN (AE-FNN) multi-task deep learning model to detect threats in IoT network traffic.

**Two models are compared:**
- **Synthetic Model**: 97% accuracy on balanced synthetic PCA data
- **Real Model**: 84.48% accuracy on CICIoT2023 real-world data

**Detected threat types:** Benign, DDoS, DoS, Brute Force, Spoofing, MITM, Scan, Recon

## Development Commands

### Running the Application
```bash
# Activate virtual environment (if not already activated)
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Run the Streamlit application
streamlit run app.py

# Run on specific port
streamlit run app.py --server.port 8502
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

1. **Autoencoder (AE)**: Compresses 16 PCA components → 4 latent dimensions → 16 reconstructed
   - Encoder: 16 → 8 (ReLU) → 4 (ReLU)
   - Decoder: 4 → 8 (ReLU) → 16 (Linear)

2. **Feedforward Classifier (FNN)**: Classifies from latent space
   - From latent: 4 → 16 (ReLU, Dropout 0.3) → 8 (Softmax)

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

**Global (app.py):**
- `current_model`: 'synthetic' or 'real'
- `model`, `scaler`, `label_encoder`, `class_names`, `metadata`: Model components

**Page-specific examples:**
- Real-time simulation: `simulation_running`, `threat_history` (deque), `threat_counts`
- File analysis: `analysis_results`, `uploaded_file_hash`, `has_labels`

Use `@st.cache_resource` for models (shared across users, not serialized) and `@st.cache_data` for data processing.

## Project Structure

```
Dashboard IoT-IDS/
├── app.py                    # Main entry point - home page, model selector
├── pages/                    # Streamlit multi-page modules
│   ├── 1_🔬_Comparacion_Modelos.py    # Side-by-side model comparison
│   ├── 2_⚡_Tiempo_Real.py             # Real-time threat simulation
│   ├── 3_📊_Analisis_Archivo.py        # Batch CSV file analysis
│   └── 4_📈_Metricas.py                # Metrics dashboard
├── utils/                    # Reusable utility modules
│   ├── model_loader.py       # Model loading, prediction functions
│   ├── data_simulator.py     # Synthetic traffic generation
│   ├── visualizations.py     # Plotly/matplotlib visualizations
│   └── report_generator.py   # PDF report generation (reportlab)
├── models/                   # Trained models and preprocessing artifacts
│   ├── modelo_ae_fnn_iot_synthetic.h5, scaler_synthetic.pkl, etc.
│   └── modelo_ae_fnn_iot_real.h5, scaler_real.pkl, etc.
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
def load_synthetic_model():
    model = tf.keras.models.load_model('models/modelo_ae_fnn_iot_synthetic.h5')
    scaler = pickle.load(open('models/scaler_synthetic.pkl', 'rb'))
    label_encoder = pickle.load(open('models/label_encoder_synthetic.pkl', 'rb'))
    class_names = np.load('models/class_names_synthetic.npy')
    metadata = json.load(open('models/model_metadata_synthetic.json'))
    return model, scaler, label_encoder, class_names, metadata
```

### Model Output Structure

The AE-FNN model returns **two outputs**:
1. `predictions[0]`: Reconstruction output (16 dimensions) - for autoencoder task
2. `predictions[1]`: Classification output (8 dimensions) - for threat classification

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
- Cannot detect attack types not in training data (8 classes only)
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

1. Create file in `pages/` with format: `N_emoji_Name.py` (e.g., `5_🔍_Custom_Analysis.py`)
2. Use `st.set_page_config()` at the top
3. Check if model is loaded: `if 'model' not in st.session_state: st.error(...); st.stop()`
4. Access shared model via `st.session_state['model']`, etc.

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
- Test prediction output shape: (class_name: str, probabilities: array(8), confidence: 0-100)
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
