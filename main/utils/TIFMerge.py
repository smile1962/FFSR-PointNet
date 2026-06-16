import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np

# Input file paths
terrain_path = r"G:\Projects\ShouXi\python\SR-PointNet\resultSaving_Sanjiang\Terrain_sanjiang.tif"
dem_path = r"G:\Projects\ShouXi\python\SR-PointNet\dataset\sourceData\geo\dem15.tif"
output_path = r"G:\Projects\ShouXi\python\SR-PointNet\resultSaving_Sanjiang\Terrain_sanjiang_fixed.tif"

# Open the sub-region terrain DEM
with rasterio.open(terrain_path) as terrain_src:
    terrain_data = terrain_src.read(1)
    terrain_meta = terrain_src.meta.copy()
    nodata_val = terrain_src.nodata if terrain_src.nodata is not None else -9999

    # Open the full reference DEM
    with rasterio.open(dem_path) as dem_src:
        # Create an array matching the sub-region DEM dimensions to store reprojected DEM data
        dem_reproj = np.empty_like(terrain_data, dtype=np.float32)

        # Reproject the full DEM to the spatial reference and resolution of the sub-region terrain DEM
        reproject(
            source=rasterio.band(dem_src, 1),
            destination=dem_reproj,
            src_transform=dem_src.transform,
            src_crs=dem_src.crs,
            dst_transform=terrain_src.transform,
            dst_crs=terrain_src.crs,
            resampling=Resampling.bilinear  # Resampling.nearest is also available
        )

# Fill missing NoData values
filled_data = terrain_data.copy()
mask = (terrain_data == nodata_val)
filled_data[mask] = dem_reproj[mask]

# Update raster metadata
terrain_meta.update(dtype=rasterio.float32, nodata=nodata_val)

# Write output file
with rasterio.open(output_path, "w", **terrain_meta) as dst:
    dst.write(filled_data.astype(np.float32), 1)

print(f"Filled DEM generated successfully: {output_path}")