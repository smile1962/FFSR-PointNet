# Standard library imports
import os

# Third-party library imports
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

"""
This script resamples a single GeoTIFF raster to a specified target pixel resolution with bilinear interpolation.
It replaces low-value pixels with a NoData value (-9999) before resampling, and preserves full geospatial metadata (CRS, affine transform).
A commented utility block is included for replacing low pixel values with NoData in an existing raster without resampling.
"""

# input_path = "../../dataset/sourceData/geo/Shancha_shuixi_1_5_mask_polyfilled.tif"
# output_path = "../../dataset/sourceData/geo/Shancha_shuixi_20_mask_polyfilled.tif"
input_path = r"G:\Projects\ShouXi\paper\img\DrawingSourceData\SR-PointNet\sanjiang0300\D_ori/2019-08-20_00-30.tif"
output_path = r"G:\Projects\ShouXi\paper\img\DrawingSourceData\SR-PointNet\sanjiang0300\D_int/2019-08-20_00-30.tif"

# Target pixel resolution (in CRS units)
target_res = 1.5

with rasterio.open(input_path) as src:
    # Retrieve source raster metadata
    transform = src.transform
    crs = src.crs
    bounds = src.bounds
    data = src.read(1)
    meta = src.meta

    # Set pixels below threshold to NoData value
    data[data < 0.1] = -9999
    meta.update(dtype=rasterio.float32, nodata=-9999)

    # Calculate output raster dimensions
    new_width = int((bounds.right - bounds.left) / target_res)
    new_height = int((bounds.top - bounds.bottom) / target_res)

    # Define output affine geotransform
    new_transform = from_origin(bounds.left, bounds.top, target_res, target_res)

    # Update output raster profile
    profile = src.profile.copy()
    profile.update({
        "transform": new_transform,
        "width": new_width,
        "height": new_height,
        "dtype": "float32"
    })

    # Write resampled output raster
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data.astype(rasterio.float32), 1)
        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=crs,
            dst_transform=new_transform,
            dst_crs=crs,
            resampling=Resampling.bilinear
        )

print(f"Resampling completed: {output_path}")

# import rasterio
# import numpy as np
#
# # Input and output file paths
# input_tif  = "../../dataset/sourceData/geo/MIKE_12_5_FILLED.tif"
# output_tif = "../../dataset/sourceData/geo/MIKE_12_5_FILLED_NAN.tif"
#
# # Open source DEM raster
# with rasterio.open(input_tif) as src:
#     data = src.read(1)  # Read first band
#     meta = src.meta.copy()
#
# # Replace pixels below threshold with NoData (-9999)
# data[data < 100] = -9999
#
# # Update metadata to register NoData value for GIS software compatibility
# meta.update(dtype=rasterio.float32, nodata=-9999)
#
# # Save modified raster
# with rasterio.open(output_tif, "w", **meta) as dst:
#     dst.write(data.astype(rasterio.float32), 1)
#
# print(f"Generated output file: {output_tif}, pixels below threshold replaced with NoData (-9999)")