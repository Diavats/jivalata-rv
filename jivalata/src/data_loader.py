"""
JĪVALATĀ MVP - Module 1: Data Loader
=====================================
Loads preprocessed GeoTIFF files and creates pixel-level feature table.

Inputs:
    - haridwar_merged_dem.tif (DEM)
    - ndvi_aligned_to_dem.tif (NDVI)

Outputs:
    - Feature table with columns: elevation, slope, ndvi
    - Spatial metadata (transform, CRS, shape)
"""

import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class SpatialMetadata:
    """Stores spatial reference information from raster files."""
    transform: rasterio.Affine
    crs: rasterio.crs.CRS
    shape: Tuple[int, int]
    bounds: rasterio.coords.BoundingBox


@dataclass
class LoadedData:
    """Container for all loaded and computed data."""
    elevation: np.ndarray
    slope: np.ndarray
    ndvi: np.ndarray
    feature_table: pd.DataFrame
    metadata: SpatialMetadata
    valid_mask: np.ndarray


def compute_slope(dem: np.ndarray, cell_size: float = 30.0) -> np.ndarray:
    """
    Compute slope from DEM using gradient method.
    
    Args:
        dem: 2D numpy array of elevation values
        cell_size: Pixel resolution in meters (default 30m)
    
    Returns:
        2D numpy array of slope values in degrees
    """
    # Compute gradients in x and y directions
    dy, dx = np.gradient(dem, cell_size)
    
    # Calculate slope in radians, then convert to degrees
    slope_radians = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_degrees = np.degrees(slope_radians)
    
    return slope_degrees


def load_raster(filepath: Path) -> Tuple[np.ndarray, SpatialMetadata]:
    """
    Load a GeoTIFF raster file.
    
    Args:
        filepath: Path to the GeoTIFF file
    
    Returns:
        Tuple of (data array, spatial metadata)
    """
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float32)
        metadata = SpatialMetadata(
            transform=src.transform,
            crs=src.crs,
            shape=data.shape,
            bounds=src.bounds
        )
    return data, metadata


def create_feature_table(
    elevation: np.ndarray,
    slope: np.ndarray,
    ndvi: np.ndarray,
    valid_mask: np.ndarray
) -> pd.DataFrame:
    """
    Create pixel-level feature table from raster arrays.
    
    Args:
        elevation: 2D array of elevation values
        slope: 2D array of slope values
        ndvi: 2D array of NDVI values
        valid_mask: Boolean mask of valid pixels
    
    Returns:
        DataFrame with columns: row, col, elevation, slope, ndvi
    """
    # Get row and column indices for all pixels
    rows, cols = np.indices(elevation.shape)
    
    # Flatten all arrays
    df = pd.DataFrame({
        'row': rows.ravel(),
        'col': cols.ravel(),
        'elevation': elevation.ravel(),
        'slope': slope.ravel(),
        'ndvi': ndvi.ravel(),
        'valid': valid_mask.ravel()
    })
    
    # Filter to valid pixels only
    df_valid = df[df['valid']].drop(columns=['valid']).reset_index(drop=True)
    
    return df_valid


def load_data(
    dem_path: str,
    ndvi_path: str,
    nodata_value: Optional[float] = None
) -> LoadedData:
    """
    Main function to load all data and create feature table.
    
    Args:
        dem_path: Path to DEM GeoTIFF file
        ndvi_path: Path to NDVI GeoTIFF file
        nodata_value: Value to treat as nodata (default: auto-detect)
    
    Returns:
        LoadedData object containing all arrays and feature table
    """
    dem_path = Path(dem_path)
    ndvi_path = Path(ndvi_path)
    
    # Validate file existence
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")
    if not ndvi_path.exists():
        raise FileNotFoundError(f"NDVI file not found: {ndvi_path}")
    
    # Load DEM
    elevation, metadata = load_raster(dem_path)
    
    # Load NDVI
    ndvi, _ = load_raster(ndvi_path)
    
    # Compute slope from DEM
    # Extract cell size from transform (assumes square pixels)
    cell_size = abs(metadata.transform[0])
    slope = compute_slope(elevation, cell_size)
    
    # Create valid pixel mask (exclude nodata values)
    if nodata_value is not None:
        valid_mask = (elevation != nodata_value) & (ndvi != nodata_value)
    else:
        # Auto-detect: exclude NaN, Inf, and extreme values
        valid_mask = (
            np.isfinite(elevation) & 
            np.isfinite(ndvi) & 
            np.isfinite(slope)
        )
    
    # Create feature table
    feature_table = create_feature_table(elevation, slope, ndvi, valid_mask)
    
    return LoadedData(
        elevation=elevation,
        slope=slope,
        ndvi=ndvi,
        feature_table=feature_table,
        metadata=metadata,
        valid_mask=valid_mask
    )


def get_feature_arrays(loaded_data: LoadedData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract feature arrays from loaded data for ML model input.
    
    Args:
        loaded_data: LoadedData object
    
    Returns:
        Tuple of (elevation_flat, slope_flat, ndvi_flat) for valid pixels only
    """
    df = loaded_data.feature_table
    return (
        df['elevation'].values,
        df['slope'].values,
        df['ndvi'].values
    )


def get_feature_matrix(loaded_data: LoadedData) -> np.ndarray:
    """
    Get feature matrix (X) ready for ML model input.
    
    Args:
        loaded_data: LoadedData object
    
    Returns:
        2D numpy array of shape (n_valid_pixels, 3)
        Columns: [elevation, slope, ndvi]
    """
    df = loaded_data.feature_table
    return df[['elevation', 'slope', 'ndvi']].values


# Convenience function for Streamlit caching
def load_data_cached(dem_path: str, ndvi_path: str) -> LoadedData:
    """
    Wrapper function suitable for Streamlit @st.cache_data decorator.
    
    Usage in Streamlit:
        @st.cache_data
        def get_data():
            return load_data_cached(DEM_PATH, NDVI_PATH)
    """
    return load_data(dem_path, ndvi_path)


def load_features(
    dem_path: Optional[str] = None, 
    ndvi_path: Optional[str] = None, 
    preprocessed_path: Optional[str] = None
) -> np.ndarray:
    """
     Unified entry point to get the feature matrix X (n_pixels, 3).
     
     Priority:
     1. If preprocessed_path is provided and exists, load .npy directly.
     2. Else, load raw GeoTIFFs via dem_path/ndvi_path and compute features.
     
     Args:
         dem_path: Path to DEM GeoTIFF
         ndvi_path: Path to NDVI GeoTIFF
         preprocessed_path: Path to precomputed .npy feature matrix
         
     Returns:
         X: 2D numpy array of shape (n_pixels, 3) -> [elevation, slope, ndvi]
    """
    # 1. Try loading preprocessed features
    if preprocessed_path:
        p_path = Path(preprocessed_path)
        if p_path.exists():
            print(f"Loading preprocessed features from {p_path}")
            return np.load(p_path)
        else:
            print(f"Preprocessed file not found: {p_path}. Falling back to raw data.")

    # 2. Fallback to raw data processing
    if not dem_path or not ndvi_path:
        raise ValueError("Must provide either 'preprocessed_path' OR both 'dem_path' and 'ndvi_path'.")
    
    # Load raw data using existing logic
    print("Processing raw GeoTIFFs...")
    data = load_data(dem_path, ndvi_path)
    
    # Extract feature matrix
    return get_feature_matrix(data)



if __name__ == "__main__":
    # Example usage / testing
    import sys
    
    if len(sys.argv) >= 3:
        dem_file = sys.argv[1]
        ndvi_file = sys.argv[2]
    else:
        # Default paths (update these to your actual file locations)
        dem_file = "data/haridwar_merged_dem.tif"
        ndvi_file = "data/ndvi_aligned_to_dem.tif"
    
    print("JĪVALATĀ Data Loader - Module 1")
    print("=" * 40)
    
    try:
        data = load_data(dem_file, ndvi_file)
        
        print(f"\n✓ Data loaded successfully!")
        print(f"\nSpatial Metadata:")
        print(f"  Shape: {data.metadata.shape}")
        print(f"  CRS: {data.metadata.crs}")
        print(f"  Bounds: {data.metadata.bounds}")
        
        print(f"\nFeature Statistics:")
        print(f"  Total pixels: {data.elevation.size:,}")
        print(f"  Valid pixels: {data.valid_mask.sum():,}")
        print(f"\nFeature Table Preview:")
        print(data.feature_table.describe())
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("  Please provide valid file paths.")
