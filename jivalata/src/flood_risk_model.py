"""
JĪVALATĀ MVP - Module 2: Flood Risk Model
=========================================
Implements a Random Forest classifier for flood risk assessment.
Uses synthetic labeling based on physical heuristics since ground truth is unavailable.

Model: Random Forest Classifier
Inputs: Elevation, Slope, NDVI
Outputs: Risk Class (0: Low, 1: Medium, 2: High)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict, Optional

# Import Module 1
from src.data_loader import load_features, load_data

class FloodRiskModel:
    def __init__(self):
        """
        Initialize the Flood Risk Model MVP.
        
        Hyperparameters (Frozen for MVP):
        - n_estimators=100
        - max_depth=10 (constrained to avoid overfitting synthetic labels)
        - random_state=42
        """
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Risk Class Definitions
        self.classes = {0: "Low", 1: "Medium", 2: "High"}

    def _generate_synthetic_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Internal method: Generate synthetic training labels using physical heuristics.
        
        Heuristic:
        Risk is inversely proportional to Elevation, Slope, and Vegetation (NDVI).
        
        Score S = 0.5*(1-Elev) + 0.3*(1-Slope) + 0.2*(1-NDVI)
        
        Classes:
        - High (2): Top 20% of Score
        - Medium (1): Variable Middle 30%
        - Low (0): Bottom 50%
        
        Args:
            X: Feature matrix (n_pixels, 3) -> [Elevation, Slope, NDVI]
            
        Returns:
            y: Labels (n_pixels,)
        """
        # MinMax scale features temporarily just for scoring (0 to 1)
        # We handle NaN inputs by imputation if necessary, though Data Loader filters them.
        mm_scaler = MinMaxScaler()
        X_norm = mm_scaler.fit_transform(X)
        
        E_norm = X_norm[:, 0]  # Elevation
        S_norm = X_norm[:, 1]  # Slope
        V_norm = X_norm[:, 2]  # NDVI
        
        # Calculate Risk Index (Higher = More Risk)
        # Note: We want LOW elevation/slope/ndvi to be HIGH risk.
        risk_score = (
            0.5 * (1 - E_norm) + 
            0.3 * (1 - S_norm) + 
            0.2 * (1 - V_norm)
        )
        
        # Determine percentiles for class boundaries
        p80 = np.percentile(risk_score, 80)
        p50 = np.percentile(risk_score, 50)
        
        # Assign classes
        y = np.zeros(len(X), dtype=int)
        
        # High Risk (Class 2) > 80th percentile
        y[risk_score > p80] = 2
        
        # Medium Risk (Class 1) > 50th percentile AND <= 80th
        y[(risk_score > p50) & (risk_score <= p80)] = 1
        
        # Low Risk (Class 0) is default (<= 50th)
        
        return y

    def train(self, X: np.ndarray) -> None:
        """
        Train the model using features X.
        Generates synthetic labels internally.
        
        Args:
            X: Feature matrix (n_pixels, 3)
        """
        print("Feature Engineering: Generating synthetic labels...")
        y = self._generate_synthetic_labels(X)
        
        print("Preprocessing: Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Training Random Forest on {len(X)} pixels...")
        self.rf_model.fit(X_scaled, y)
        self.is_trained = True
        print("Training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict risk class for new/input features.
        
        Args:
            X: Feature matrix (n_pixels, 3)
            
        Returns:
            Predictions (n_pixels,) -> Values {0, 1, 2}
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train(X) first.")
            
        X_scaled = self.scaler.transform(X)
        return self.rf_model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict risk probabilities (for detailed scoring).
        
        Args:
            X: Feature matrix (n_pixels, 3)
            
        Returns:
            Probabilities (n_pixels, 3)
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train(X) first.")
            
        X_scaled = self.scaler.transform(X)
        return self.rf_model.predict_proba(X_scaled)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary of {feature_name: importance_score}
        """
        if not self.is_trained:
            return {}
            
        importances = self.rf_model.feature_importances_
        features = ["Elevation", "Slope", "NDVI"]
        return dict(zip(features, importances))


if __name__ == "__main__":
    # Test Block
    import sys
    from pathlib import Path
    
    print("JIVALATA - Module 2 Test")
    print("=========================")
    
    # Define paths
    preprocessed_path = "data/ml_features_full.npy"
    dem_path = "data/haridwar_merged_dem.tif"
    ndvi_path = "data/ndvi_aligned_to_dem.tif"
    
    # 1. Load Data
    print("\n[1] Loading Features...")
    X = None
    try:
        # Pass all potential paths to load_features; it handles priority
        X = load_features(
            dem_path=dem_path, 
            ndvi_path=ndvi_path, 
            preprocessed_path=preprocessed_path
        )
        print(f"    Loaded X shape: {X.shape}")
        
    except Exception as e:
        print(f"    Error loading data: {e}")
        print("    Please ensure data/ml_features_full.npy OR GeoTIFFs are present.")
        sys.exit(1)

    if X is not None:
        try:
            # 2. Initialize Model
            print("\n[2] Initializing FloodRiskModel...")
            model = FloodRiskModel()
            
            # 3. Train
            print("\n[3] Training Model (Random Forest)...")
            model.train(X)
            
            # 4. Verify Feature Importance
            print("\n[4] Verifying Feature Importance...")
            imp = model.get_feature_importance()
            sorted_imp = dict(sorted(imp.items(), key=lambda item: item[1], reverse=True))
            for f, score in sorted_imp.items():
                print(f"    {f}: {score:.4f}")
                
            # Sanity Check: Elevation should be high importance
            ifImp = imp.get('Elevation', 0)
            if ifImp > 0.2: 
                print("    [OK] Elevation is a key driver (Expected)")
            else:
                print(f"    ! Warning: Elevation importance low ({ifImp:.4f})")
                
            # 5. Test Prediction
            print("\n[5] Test Prediction (first 5 pixels)...")
            preds = model.predict(X[:5])
            probs = model.predict_proba(X[:5])
            print(f"    Predictions: {preds}")
            print(f"    Probabilities:\n{probs}")
            
            print("\n[OK] Module 2 Verification Complete!")
            
        except Exception as e:
            print(f"\n[ERROR] Error during test: {e}")
            import traceback
            traceback.print_exc()
