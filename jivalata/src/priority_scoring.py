"""
JĪVALATĀ MVP - Module 4: Priority Scoring
=========================================
Ranks restoration interventions by calculating a Priority Score for each pixel.

Formula: Priority = Risk Reduction * Area * Feasibility

Inputs: Risk Reduction Map, Slope Map (for feasibility)
Outputs: Priority Map, Ranked CSV
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

class PriorityScorer:
    def __init__(self, pixel_area_m2: float = 900.0):
        """
        Initialize Scorer.
        
        Args:
            pixel_area_m2: Area of one pixel in square meters (default 30m x 30m = 900)
        """
        self.pixel_area = pixel_area_m2

    def calculate_feasibility(self, slope_map: np.ndarray) -> np.ndarray:
        """
        Calculate restoration feasibility based on slope.
        Heuristic: Flatter terrain is easier/cheaper to restore.
        
        Formula: F = 1.0 - (Slope_Degrees / 45.0)
        Clipped to range [0.1, 1.0] (never fully impossible, but very hard at steep slopes)
        
        Args:
            slope_map: 1D array of slope values in degrees
            
        Returns:
            Feasibility map (values 0.1 to 1.0)
        """
        # Linear decay: 0 deg = 1.0 feasibility, 45 deg = 0.0 feasibility
        f = 1.0 - (slope_map / 45.0)
        return np.clip(f, 0.1, 1.0)

    def compute_scores(
        self, 
        risk_reduction_map: np.ndarray, 
        slope_map: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pixel-wise priority scores.
        
        Args:
            risk_reduction_map: 1D array of risk delta (0.0 to 1.0)
            slope_map: 1D array of slope (degrees)
            
        Returns:
            priority_map: Score per pixel
            feasibility_map: Calculated feasibility per pixel
        """
        feasibility_map = self.calculate_feasibility(slope_map)
        
        # Priority = (Risk Delta * Area) * Feasibility
        # Interpretation: "Square Meters of Risk Reduced, weighted by difficulty"
        priority_map = risk_reduction_map * self.pixel_area * feasibility_map
        
        return priority_map, feasibility_map

    def get_ranked_zones(
        self, 
        feature_df: pd.DataFrame, 
        risk_reduction_map: np.ndarray, 
        top_n: int = 100
    ) -> pd.DataFrame:
        """
        Generate a ranked list of top priority zones/pixels.
        
        Args:
            feature_df: Original DataFrame from Module 1 (must have 'slope', 'row', 'col', 'elevation')
            risk_reduction_map: Output from Module 3
            top_n: Number of top pixels to return
            
        Returns:
            DataFrame of top N candidates sorted by Priority
        """
        # Ensure alignment
        if len(feature_df) != len(risk_reduction_map):
            raise ValueError(f"Feature DF length ({len(feature_df)}) != Risk Map length ({len(risk_reduction_map)})")
            
        slope_values = feature_df['slope'].values
        
        # Calculate Scores
        priority_scores, feasibility = self.compute_scores(risk_reduction_map, slope_values)
        
        # Create Result DF
        # We allow modifying the input DF copy or creating new one
        result = feature_df.copy()
        result['risk_reduction'] = risk_reduction_map
        result['feasibility'] = feasibility
        result['priority_score'] = priority_scores
        
        # Filter: Only non-zero reduction
        # (Save computation/memory by ignoring pixels with 0 impact)
        # For ranking we want highest scores
        result = result.sort_values(by='priority_score', ascending=False)
        
        return result.head(top_n)


if __name__ == "__main__":
    # Test Block
    import sys
    from src.data_loader import load_features, load_data
    from src.flood_risk_model import FloodRiskModel
    from src.simulation import RestorationSimulator
    
    print("JIVALATA - Module 4 Test")
    print("=========================")
    
    try:
        # 1. Pipeline Setup
        print("[1] Loading Data...")
        # Need raw data wrapper to get DataFrame for coords/slope
        # Assuming ml_features_full.npy matches data.feature_table order (it does by design)
        # But load_features returns X only. We need metadata (slope, coords).
        # So we load raw data to get the DataFrame structure.
        raw_data = load_data("data/haridwar_merged_dem.tif", "data/ndvi_aligned_to_dem.tif")
        X = raw_data.feature_table[['elevation', 'slope', 'ndvi']].values
        feature_df = raw_data.feature_table
        print(f"    Loaded {len(feature_df)} pixels")
        
        # 2. Run Simulation Stats (Mocking Module 3 for speed or running it)
        print("[2] Running Simulation Logic...")
        model = FloodRiskModel()
        model.train(X)
        
        sim = RestorationSimulator(model)
        _, risk_red = sim.run_simulation(X, ndvi_increase=0.2)
        
        # 3. Priority Scoring
        print("[3] Calculating Priority Scores...")
        scorer = PriorityScorer()
        
        # Get Top 10
        ranked_df = scorer.get_ranked_zones(feature_df, risk_red, top_n=10)
        
        print("\n[4] Top 5 Priority Zones:")
        print(ranked_df[['row', 'col', 'slope', 'risk_reduction', 'priority_score']].to_string())
        
        # Export CSV
        out_csv = "data/ranked_zones_test.csv"
        ranked_df.to_csv(out_csv, index=False)
        print(f"\n[OK] Exported ranked list to {out_csv}")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        # import traceback
        # traceback.print_exc()
