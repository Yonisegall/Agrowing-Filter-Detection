import os
from pathlib import Path
import numpy as np
import rasterio

project_root = Path(__file__).resolve().parents[2]  # goes up two levels from viewing

# === Config ===
npz_file_path = project_root / r'stacks\im1_norm.npz'

# === Load data
stack = np.load(npz_file_path)['stack']
band_names = np.load(npz_file_path)['band_names']

# === Define output path ===
output_tiff_file = os.path.splitext(npz_file_path)[0] + '.tif'  # 'stacks\im1_norm.tif'

# === Define dummy geotransform and CRS ===
# (Update these if you have real geospatial info)
transform = rasterio.transform.from_origin(0, 0, 1, 1)  # top-left x/y, pixel size
crs = "EPSG:4326"  # WGS84 (or use None for no CRS)

# === Write TIFF ===
height, width, bands = stack.shape

with rasterio.open(
    output_tiff_file,
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=bands,
    dtype=stack.dtype,
    crs=crs,
    transform=transform
) as dst:
    for i in range(bands):
        dst.write(stack[:, :, i], i + 1)
        dst.set_band_description(i + 1, band_names[i])

print(f"Saved multi-band TIFF to: {output_tiff_file}")