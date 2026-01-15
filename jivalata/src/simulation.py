"""
JĪVALATĀ MVP - Module 3: Restoration Simulation Engine
======================================================
Simulates restoration interventions (specifically vegetation increase)
and quantifies the resulting flood risk reduction.

Class: RestorationSimulator
Inputs: Trained FloodRiskModel, Feature Matrix X
Outputs: Simulated Risk Classes, Risk Reduction Score
"""

import numpy as np
from typing import Tuple, Dict, Any
import copy

# Import previous modules
try:
    from src.flood_risk_model import FloodRiskModel
except ImportError:
    # Fallback for direct script execution
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.flood_risk_model import FloodRiskModel

class RestorationSimulator:
    def __init__(self, model: FloodRiskModel):
        """
        Initialize the simulator with a trained FloodRiskModel.
        
        Args:
            model: Trained instance of FloodRiskModel (Module 2)
        """
        if not model.is_trained:
            raise RuntimeError("Simulator requires a trained FloodRiskModel.")
        self.model = model

    def run_simulation(
        self, 
        X: np.ndarray, 
        ndvi_change: float = 0.0,
        elevation_change: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run a "what-if" simulation by modifying NDVI and/or elevation.
        
        Logic:
        1. Calculate Baseline Risk (Probability of High Risk).
        2. Create X_sim with modified NDVI (clamped 0.0-1.0) and/or elevation.
        3. Calculate Simulated Risk.
        4. Compute Change = Baseline_Prob_High - Simulated_Prob_High.
        
        Args:
            X: Baseline feature matrix (n_pixels, 3) [Elev, Slope, NDVI]
            ndvi_change: Amount to add/subtract from NDVI (can be positive or negative)
            elevation_change: Amount to add/subtract from elevation in meters
            
        Returns:
            Tuple containing:
            - new_risk_classes (np.ndarray): Predicted classes after intervention
            - risk_change_map (np.ndarray): Risk probability change (can be + or -)
        """
        # 1. Baseline Risk (Probability of High Risk)
        probs_base = self.model.predict_proba(X)
        
        # Check how many classes exist in the model
        n_classes = probs_base.shape[1]
        
        # We assume the "Highest Risk" class is the last column
        high_risk_idx = n_classes - 1
        prob_high_base = probs_base[:, high_risk_idx]
        
        # 2. Apply Interventions
        X_sim = X.copy()
        
        # Apply NDVI change (column 2) - clamp to [0.0, 1.0]
        if ndvi_change != 0.0:
            X_sim[:, 2] += ndvi_change
            X_sim[:, 2] = np.clip(X_sim[:, 2], 0.0, 1.0)
        
        # Apply elevation change (column 0)
        if elevation_change != 0.0:
            X_sim[:, 0] += elevation_change
        
        # 3. Simulated Risk
        probs_sim = self.model.predict_proba(X_sim)
        
        # Ensure simulated output has same number of classes
        if probs_sim.shape[1] != n_classes:
            prob_high_sim = probs_sim[:, -1]  # Fallback
        else:
            prob_high_sim = probs_sim[:, high_risk_idx]
        new_risk_classes = self.model.predict(X_sim)
        
        # 4. Calculate Risk Change
        # Positive value = risk decreased (good)
        # Negative value = risk increased (bad)
        risk_change_map = prob_high_base - prob_high_sim
        
        return new_risk_classes, risk_change_map


if __name__ == "__main__":
    # Internal Verification Block
    from src.data_loader import load_features
    import sys
    
    print("JIVALATA - Module 3 Test")
    print("=========================")
    
    # 1. Load Data
    try:
        print("[1] Loading Data...")
        X = load_features(preprocessed_path="data/ml_features_full.npy")
        print(f"    Loaded X: {X.shape}")
        
        # 2. Train Base Model
        print("\n[2] Training Base Model (Module 2)...")
        model = FloodRiskModel()
        model.train(X)
        
        # 3. Initialize Simulator
        print("\n[3] Initializing Simulator...")
        sim = RestorationSimulator(model)
        
        # 4. Run Simulation
        NDVI_DELTA = 0.2
        print(f"\n[4] Running Simulation (+{NDVI_DELTA} NDVI)...")
        new_classes, reduction = sim.run_simulation(X, ndvi_increase=NDVI_DELTA)
        
        # 5. Verify Results
        print("\n[5] Analysis:")
        
        # Count high risk pixels before vs after
        # Re-predict baseline classes for comparison
        base_classes = model.predict(X)
        n_high_base = np.sum(base_classes == 2)
        n_high_sim = np.sum(new_classes == 2)
        
        print(f"    High Risk Pixels (Baseline):  {n_high_base:,}")
        print(f"    High Risk Pixels (Simulated): {n_high_sim:,}")
        print(f"    Difference:                   {n_high_sim - n_high_base:,}")
        
        # Check reduction score stats
        max_red = np.max(reduction)
        mean_red = np.mean(reduction)
        print(f"    Max Risk Reduction Score:     {max_red:.4f}")
        print(f"    Mean Risk Reduction Score:    {mean_red:.4f}")
        
        if n_high_sim < n_high_base:
            print("\n[OK] Simulation successfully reduced flood risk!")
        elif n_high_sim == n_high_base:
            print("\n[WARN] No change in risk classes (Intervention might be too small)")
        else:
            print("\n[ERROR] Risk increased? Check logic.")
            
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
