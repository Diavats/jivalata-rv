# JĪVALATĀ  
## AI Decision Engine System For Floodplain Risk Reduction

JĪVALATĀ is a spatial intelligence framework designed to support regenerative floodplain planning.  

Instead of focusing solely on flood mitigation, the system integrates terrain morphology, ecological sensitivity, vegetation health, and human exposure to prioritize high-impact restoration zones.

---

## Objective

To move beyond sustainability-driven containment models and enable data-driven regenerative decision-making in climate-vulnerable floodplains.

---

## System Architecture

The pipeline integrates multiple spatial layers and produces ranked intervention zones:

1. **Spatial Harmonization**
   - DEM-aligned raster preprocessing
   - Pixel-level feature extraction

2. **Feature Engineering**
   - Elevation
   - Slope
   - NDVI
   - Population exposure (WorldPop)
   - Wetland sensitivity

3. **Flood Risk Modeling**
   - Random Forest classifier
   - Physically-informed synthetic labeling
   - Multi-class risk output (Low / Medium / High)

4. **Restoration Simulation Engine**
   - NDVI-based intervention modeling
   - Risk reduction simulation

5. **Priority Scoring**
   - Composite regenerative score
   - Ranked intervention zones
   - Exportable decision-support CSV

---

## Project Structure


jivalata/
├── src/
│ ├── data_loader.py
│ ├── flood_risk_model.py
│ ├── simulation.py
│ ├── priority_scoring.py
│ ├── ui_components.py
│ └── dashboard.py
├── data/ # Local raster inputs (excluded from repo)
├── requirements.txt
├── packages.txt
└── README.md


---

##  Setup Instructions

###  1) Clone Repository
```bash
git clone https://github.com/Diavats/Jivalata.git
cd Jivalata
### 2) Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate
### 3) Install Dependencies
pip install -r requirements.txt
### 4) Add Required Data

Place required GeoTIFF raster layers inside the data/ directory.

Example:

haridwar_merged_dem.tif

ndvi_aligned_to_dem.tif

worldpop_2026.tif

wetland_sensitivity_gee.tif

### Run Data Pipeline

Test feature extraction:

python src/data_loader.py

###Launch Interactive Dashboard
streamlit run src/dashboard.py

### Dashboard Capabilities
Adjustable NDVI-based restoration simulation

Base vs. simulated flood risk comparison

Dynamic risk heatmaps

Priority ranking of intervention zones

Exportable CSV output for planning use

### Key Outcomes

->Pixel-level flood risk classification

->Regenerative intervention prioritization

->Exposure-aware ecological restoration modeling

->Decision-support visualization for planners

### Technology Stack

Python

Rasterio

NumPy

Pandas

Scikit-learn

Streamlit

### Status

-> Data Loader
-> Flood Risk Model
-> Restoration Simulation
-> Priority Scoring
-> Interactive Dashboard

### Author

Dia Vats