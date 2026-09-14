import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import rasterio
import numpy as np
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt

"""
This script generates a dilated binary mask raster from an input water network GeoTIFF.
It reads a single-band water raster, converts it to a binary mask, applies square-kernel morphological dilation,
and exports the result as a GeoTIFF with full preserved geospatial metadata.
A grayscale visualization of the final mask is displayed after processing.
"""

# Input and output file paths
input_tif = rf"{ROOT}\Data\dataset\geo\Shouxi_shuixi_30.tif"
output_tif = rf"../{ROOT}\Data\dataset/geo/Shouxi_mask30m_ROI10.tif"

# Dilation radius in pixels
# Adjustable, e.g. 2, 3, 5
N = 5

# Open input GeoTIFF file
# Open input GeoTIFF file
with rasterio.open(input_tif) as src:
    data = src.read(1)  # Read first band, assumed to contain the water network raster
    # Copy full raster metadata including geospatial information
    meta = src.meta.copy()
    # Retrieve affine geotransform (coordinates and pixel size)
    transform = src.transform
    # Retrieve coordinate reference system
    crs = src.crs

# Binarize the river raster: values in (0, 1] are treated as river cells.
water_binary = np.where((data > 0) & (data <= 1), 1, 0).astype(np.uint8)

# Generate binary mask
mask = water_binary


# Define structuring element for dilation (square kernel with radius N)
# Creates a (2N+1) × (2N+1) square kernel
struct = np.ones((2*N+1, 2*N+1), dtype=np.uint8)

# Perform binary dilation
dilated_mask = binary_dilation(mask, structure=struct).astype(np.uint8)

# Save output mask - preserve full geospatial metadata
meta.update(
    dtype=rasterio.uint8,
    count=1,
    nodata=None,
    transform=transform,  # Include affine geotransform
    crs=crs               # Include coordinate reference system
)
with rasterio.open(output_tif, "w", **meta) as dst:
    dst.write(dilated_mask, 1)

# Visualization
plt.figure(figsize=(8, 8))
plt.imshow(dilated_mask, cmap="gray", vmin=0, vmax=1)  # 0 = black, 1 = white
plt.title("Mask visualization")
plt.axis("off")
plt.show()
