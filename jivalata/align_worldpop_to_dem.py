from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

ROOT = Path(__file__).resolve().parent
DEM = ROOT / "data" / "haridwar_merged_dem.tif"
POP = ROOT / "data" / "worldpop_2026.tif"
OUT = ROOT / "data" / "worldpop_2026_aligned_to_dem.tif"

if not DEM.exists():
    raise FileNotFoundError(f"Missing DEM: {DEM}")
if not POP.exists():
    raise FileNotFoundError(f"Missing WorldPop: {POP}")

with rasterio.open(DEM) as dem:
    dem_crs = dem.crs
    dem_transform = dem.transform
    dem_width = dem.width
    dem_height = dem.height
    dem_profile = dem.profile.copy()

dest = np.zeros((dem_height, dem_width), dtype=np.float32)

with rasterio.open(POP) as src:
    src_data = src.read(1)

    reproject(
        source=src_data,
        destination=dest,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dem_transform,
        dst_crs=dem_crs,
        resampling=Resampling.bilinear,
        dst_nodata=0,
    )

dem_profile.update(
    dtype="float32",
    count=1,
    compress="lzw",
    nodata=0
)

with rasterio.open(OUT, "w", **dem_profile) as dst:
    dst.write(dest, 1)

print("DONE:", OUT)
print("Aligned shape:", dest.shape)
print("Min/Max:", float(np.nanmin(dest)), float(np.nanmax(dest)))
