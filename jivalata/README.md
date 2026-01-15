# JĪVALATĀ MVP
# Floodplain Restoration & Flood Risk Reduction Dashboard

## Project Structure

```
jivalata/
├── src/
│   ├── data_loader.py      # Module 1: Data loading & feature extraction
│   ├── flood_risk_model.py # Module 2: Logistic regression model (TBD)
│   ├── simulation.py       # Module 3: Restoration simulation engine (TBD)
│   ├── priority_scoring.py # Module 4: Priority scoring system (TBD)
│   └── dashboard.py        # Module 5: Streamlit dashboard (TBD)
├── data/                   # Place GeoTIFF files here
│   ├── haridwar_merged_dem.tif
│   └── ndvi_aligned_to_dem.tif
├── requirements.txt
└── README.md
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your GeoTIFF files in the `data/` folder

3. Test the data loader:
   ```bash
   python src/data_loader.py data/haridwar_merged_dem.tif data/ndvi_aligned_to_dem.tif
   ```

## Module 1: Data Loader

The data loader module provides:

- **`load_data(dem_path, ndvi_path)`**: Main function to load all data
- **`LoadedData`**: Container with elevation, slope, ndvi arrays and feature table
- **`get_feature_matrix(loaded_data)`**: Get ML-ready feature matrix (n_pixels × 3)
- **`load_features(dem_path, ndvi_path, preprocessed_path)`**: Unified entry point for ML
- **`load_data_cached()`**: Streamlit-compatible cached loader

## Module 2: Flood Risk Model (Random Forest)

Implements a Random Forest Classifier trained on physically-based synthetic labels.

- **`FloodRiskModel` Class**:
    - `train(X)`: Generates labels on-the-fly and trains features.
    - `predict(X)`: Returns risk classes {0: Low, 1: Med, 2: High}.
    - `predict_proba(X)`: Usage for Module 4 scoring.
    - `get_feature_importance()`: Returns model weights.

### Usage Example

```python
from src.data_loader import load_features
from src.flood_risk_model import FloodRiskModel

# 1. Load Features
X = load_features(dem_path="data/haridwar_merged_dem.tif", ndvi_path="data/ndvi_aligned_to_dem.tif")

# 2. Train Model
model = FloodRiskModel()
model.train(X)

# 3. Predict Risk
risk_classes = model.predict(X)
```

## Module 5: Interactive Dashboard

Lightweight Streamlit application visualizing flood risk and restoration impact.

### Features
- **Interactive Controls**: Adjust NDVI increase (0.0 - 0.5) and run simulations.
- **Dynamic Mapping**: Side-by-side comparison of Base vs. Simulated Risk.
- **Decision Support**: Priority Heatmap and downloadable CSV of ranked zones.

### Running the Dashboard

```bash
streamlit run src/dashboard.py
```

## Status

- [x] Module 1: Data Loader
- [x] Module 2: Flood Risk Model
- [x] Module 3: Restoration Simulation Engine
- [x] Module 4: Priority Scoring
- [x] Module 5: Interactive Dashboard
