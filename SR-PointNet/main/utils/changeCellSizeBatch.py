# Standard library imports
import os

# Third-party library imports
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

# from Unet.utils.dataset_PKL_Gen import event_folder

"""
This script performs batch resampling of GeoTIFF raster files with precise geospatial alignment.
It supports two working modes:
1. Resample rasters to a user-specified target resolution with configurable resampling methods.
2. Align all input rasters to the exact dimensions, affine transform and CRS of a reference mask raster.
Output rasters are stored in float32 datatype, with built-in validation to ensure grid consistency across all files.
"""

# ------------------------------
# Legacy resampling workflow (by target resolution)
# ------------------------------
# # Original dataset root directory
# root_dir = r"G:\Projects\ShouXi\Sancha"
# # root_dir = r"G:\Projects\ShouXi\python\SR-PointNet\dataset\sourceData\SanJiang"
# # root_dir = r"G:\Projects\ShouXi\paper\img\DrawingSourceData\UpdateSanjiang"
# # Output root directory for resampled rasters
# output_root = r"G:\Projects\ShouXi\python\SR-PointNet\dataset\sourceData\SanJiang\aaa"
#
# # List of event subdirectories to process
# # event_folders = ["2year500mesh", "5year500mesh", "10year500mesh",
# #                  "20year500mesh", "50year500mesh", "100year500mesh", "real2_500mesh"]
#
# # event_folders = ["2year30mesh", "5year30mesh", "10year30mesh",
# #                  "20year30mesh", "50year30mesh", "100year30mesh", "realRain", "real430"]
#
# # event_folders = ["2_cons", "5_cons", "10_cons",
# #                  "20_cons", "50_cons", "100_cons", "realRainfall", "real2"]
#
# # event_folders = ["val_30"]
# event_folders = ["real2_30"]
# # event_folders = ["real420"]
#
# # Target pixel resolution (in CRS units)
# target_res = 20
#
# for event in event_folders:
#     input_folder = os.path.join(root_dir, event)
#     output_folder = os.path.join(output_root, event)
#
#     # Create output directory
#     os.makedirs(output_folder, exist_ok=True)
#
#     # Iterate over all TIFF files
#     for file in os.listdir(input_folder):
#         if file.endswith(".tif"):
#             input_path = os.path.join(input_folder, file)
#             output_path = os.path.join(output_folder, file)
#
#             with rasterio.open(input_path) as src:
#                 # Source raster metadata
#                 transform = src.transform
#                 crs = src.crs
#                 bounds = src.bounds
#
#                 # Calculate new raster dimensions
#                 new_width = int((bounds.right - bounds.left) / target_res)
#                 new_height = int((bounds.top - bounds.bottom) / target_res)
#
#                 # Define new affine transform
#                 new_transform = from_origin(bounds.left, bounds.top, target_res, target_res)
#
#                 # Update output raster profile
#                 profile = src.profile.copy()
#                 profile.update({
#                     "transform": new_transform,
#                     "width": new_width,
#                     "height": new_height,
#                     "dtype": "float32"
#                 })
#
#                 # Write resampled output raster
#                 with rasterio.open(output_path, "w", **profile) as dst:
#                     reproject(
#                         source=rasterio.band(src, 1),
#                         destination=rasterio.band(dst, 1),
#                         src_transform=src.transform,
#                         src_crs=crs,
#                         dst_transform=new_transform,
#                         dst_crs=crs,
#                         resampling=Resampling.max
#                     )
#
#             print(f"Resampling completed: {output_path}")
#
# print("All files have been resampled and saved to ../Data\dataset")


# ------------------------------
# Active workflow: resample to exact mask grid
# ------------------------------
# Original dataset root directory
root_dir = r"G:\Projects\ShouXi\paper\img\DrawingSourceData\Yuhuo"
# Output root directory for resampled rasters
output_root = r"G:\Projects\ShouXi\paper\img\DrawingSourceData\Yuhuo\aaa"

# List of subdirectories to process
event_folders = ["U_ori"]

# Target pixel resolution
target_res = 1.5

# Load reference mask raster to obtain exact grid parameters for alignment
mask_path = r"G:\Projects\ShouXi\paper\img\DrawingSourceData\Yuhuo/yuhuo_1_5_align.tif"
with rasterio.open(mask_path) as mask_src:
    mask_width = mask_src.width
    mask_height = mask_src.height
    mask_transform = mask_src.transform
    mask_bounds = mask_src.bounds
    mask_crs = mask_src.crs

print(f"Reference mask metadata: dimensions = {mask_width} × {mask_height}, bounds = {mask_bounds}")


def resample_to_exact_size(
    input_path: str,
    output_path: str,
    target_width: int,
    target_height: int,
    target_transform: rasterio.transform.Affine,
    target_crs: rasterio.crs.CRS
) -> None:
    """
    Resample a single GeoTIFF raster to match exact target dimensions, affine transform and CRS.
    Uses bilinear resampling by default and validates output dimensions after writing.

    Args:
        input_path: File path of source raster
        output_path: File path of resampled output raster
        target_width: Target number of columns
        target_height: Target number of rows
        target_transform: Target affine geotransform
        target_crs: Target coordinate reference system
    """
    with rasterio.open(input_path) as src:
        # Read source raster and its geospatial metadata
        data = src.read(1)
        src_crs = src.crs
        src_transform = src.transform

        # Configure output raster profile
        profile = src.profile.copy()
        profile.update({
            'transform': target_transform,
            'width': target_width,
            'height': target_height,
            'crs': target_crs,
            'dtype': 'float32'
        })

        # Perform reprojection and resampling
        with rasterio.open(output_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
                dst_nodata=profile.get('nodata', 0)
            )

        # Validate output raster dimensions
        with rasterio.open(output_path) as dst_check:
            actual_width = dst_check.width
            actual_height = dst_check.height
            actual_transform = dst_check.transform

            print(f"  Output dimensions: {actual_width} × {actual_height}")
            print(f"  Target dimensions: {target_width} × {target_height}")

            if actual_width == target_width and actual_height == target_height:
                print(f"  Dimension alignment verified")
            else:
                print(f"  WARNING: Dimension mismatch detected")


def batch_resample_to_mask_size(
    root_dir: str,
    output_root: str,
    event_folders: list[str],
    mask_width: int,
    mask_height: int,
    mask_transform: rasterio.transform.Affine,
    mask_crs: rasterio.crs.CRS
) -> None:
    """
    Batch resample all TIFF files in specified directories to match the reference mask grid exactly.

    Args:
        root_dir: Root directory of input datasets
        output_root: Root directory for resampled outputs
        event_folders: List of subdirectory names to process
        mask_width: Reference mask column count
        mask_height: Reference mask row count
        mask_transform: Reference mask affine transform
        mask_crs: Reference mask coordinate reference system
    """
    for event in event_folders:
        input_folder = os.path.join(root_dir, event)
        output_folder = os.path.join(output_root, event)

        # Create output directory
        os.makedirs(output_folder, exist_ok=True)

        print(f"\nProcessing directory: {event}")
        print(f"  Target grid dimensions: {mask_width} × {mask_height}")

        # Iterate over all TIFF files
        for file in os.listdir(input_folder):
            if file.endswith(".tif"):
                input_path = os.path.join(input_folder, file)
                output_path = os.path.join(output_folder, file)

                print(f"\n  Processing file: {file}")

                try:
                    resample_to_exact_size(
                        input_path, output_path,
                        mask_width, mask_height,
                        mask_transform, mask_crs
                    )
                    print(f"  Resampling completed: {output_path}")
                except Exception as e:
                    print(f"  Processing failed: {str(e)}")


def calculate_dimensions_from_bounds(
    bounds: rasterio.coords.BoundingBox,
    resolution: float
) -> tuple[int, int, rasterio.transform.Affine]:
    """
    Calculate raster column/row count and affine transform from spatial bounds and pixel resolution.
    Uses rounding to ensure alignment with reference mask grid.

    Args:
        bounds: Spatial bounding box of the study area
        resolution: Target pixel resolution (in CRS units)

    Returns:
        Tuple of (number_of_columns, number_of_rows, affine_transform)
    """
    width = bounds.right - bounds.left
    height = bounds.top - bounds.bottom

    # Apply rounding to ensure consistency with reference mask
    num_cols = int(np.round(width / resolution))
    num_rows = int(np.round(height / resolution))

    print(f"Bounding box calculation: width = {width:.4f}, height = {height:.4f}")
    print(f"Target resolution: {resolution}")
    print(f"Computed dimensions: {num_cols} × {num_rows}")

    # Generate affine transform (negative y-resolution for north-up orientation)
    transform = from_origin(
        bounds.left,
        bounds.top,
        resolution,
        resolution
    )

    return num_cols, num_rows, transform


# Execute batch resampling aligned to reference mask grid
batch_resample_to_mask_size(
    root_dir, output_root, event_folders,
    mask_width, mask_height, mask_transform, mask_crs
)

print("\nAll raster files have been resampled successfully")

# Optional: validate output rasters against reference mask dimensions
print("\nValidating output raster dimensions:")
for event in event_folders:
    output_folder = os.path.join(output_root, event)

    if os.path.exists(output_folder):
        tif_files = [f for f in os.listdir(output_folder) if f.endswith(".tif")]

        # Check first 3 files as sample validation
        for tif_file in tif_files[:3]:
            tif_path = os.path.join(output_folder, tif_file)
            with rasterio.open(tif_path) as src:
                if src.width == mask_width and src.height == mask_height:
                    print(f"PASS {tif_file}: {src.width} × {src.height} (matches mask)")
                else:
                    print(f"FAIL {tif_file}: {src.width} × {src.height} (mismatches mask)")