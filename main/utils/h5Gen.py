# Standard library imports
import os
import glob
import json
from datetime import datetime

# Third-party library imports
import h5py
import numpy as np
import rasterio

"""
This script converts GeoTIFF format hydraulic raster data into HDF5 point-cloud datasets
for flow field super-resolution model training and validation.

Main functionalities:
1. Load study area mask, DEM and slope raster files, and extract coordinates of valid pixels
2. Batch process paired depth and velocity TIFF files from multiple event folders
3. Parse timestamps from filenames, clip out-of-range values, and replace NoData placeholders
4. Align all raster data to valid mask regions and construct point-wise feature arrays
5. Save processed samples into an HDF5 file with unique keys and attached metadata
6. Output dataset statistical summaries (max depth, max velocity) and data preview
"""


# ==============================
# 1. Load Mask and Terrain Rasters
# ==============================
mask_path = r"G:\Projects\ShouXi\python\SR-PointNet\dataset\sourceData\geo\Shancha_shuixi_20_mask_polyfilled.tif"
dem_path = r"G:\Projects\ShouXi\python\SR-PointNet\dataset\sourceData\geo\dem_sanjiang_1_5.tif"
slope_path = r"G:\Projects\ShouXi\python\SR-PointNet\dataset\sourceData\geo\Sanjiang_slope_1_5.tif"

with rasterio.open(mask_path) as src:
    mask = src.read(1)  # Single-band binary mask raster
    height, width = mask.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src.transform, rows, cols)
    xs = np.array(xs).flatten()
    ys = np.array(ys).flatten()
    mask_vals = mask.flatten()
    valid_idx = mask_vals != 0  # Pixel indices inside the valid study area

# Load DEM elevation raster
with rasterio.open(dem_path) as src:
    dem = src.read(1)
    dem_values = dem.flatten()
    dem_values[dem_values == -9999] = 0  # Replace NoData value with 0

# Load terrain slope raster
with rasterio.open(slope_path) as src:
    slope = src.read(1)
    slope_values = slope.flatten()
    slope_values[slope_values == -9999] = 0  # Replace NoData value with 0


# ==============================
# 2. Batch Process Raster Folders
# ==============================
# base_dir = r"G:\Projects\ShouXi\python\SR-PointNet\dataset\sourceData"
# folders = ["2year300mesh", "5year300mesh", "10year300mesh",
#            "20year300mesh", "50year300mesh", "100year300mesh", "realRain_2_300mesh"]
#
# base_dir = r"G:\Projects\ShouXi\python\SR-PointNet\dataset\sourceData"
# folders = ["2year500mesh", "5year500mesh", "10year500mesh",
#            "20year500mesh", "50year500mesh", "100year500mesh", "real2_500mesh"]

base_dir = r"G:\Projects\ShouXi\python\SR-PointNet\dataset\sourceData\SanJiang"
# folders = ["2year30mesh", "5year30mesh", "10year30mesh",
#             "20year30mesh", "50year30mesh", "100year30mesh", "realRain", "real430"]
#
# folders = ["2_cons", "5_cons", "10_cons",
#                  "20_cons", "50_cons", "100_cons", "realRainfall", "real2"]

# folders = ["real2_30", "real4_30"]
folders = ["val_LR"]

# Containers for all processed samples and corresponding metadata
all_samples = []
all_meta = []


def make_unique_key(f: h5py.File, base_key: str) -> str:
    """
    Generate a unique dataset key for the target HDF5 file.
    If the base_key already exists, append an incremental suffix (1), (2)...
    to avoid key collision and data overwriting.

    Args:
        f: Target HDF5 file object
        base_key: Intended original dataset key name

    Returns:
        Unique key string that does not exist in the current HDF5 file
    """
    key = base_key
    i = 1
    while key in f:
        key = f"{base_key}({i})"
        i += 1
    return key


for folder in folders:
    # event = folder.replace("300mesh", "")
    folder_path = os.path.join(base_dir, folder)

    # Collect and sort depth and velocity TIFF files by filename
    depth_files = sorted(glob.glob(os.path.join(folder_path, "Depth*.tif")))
    velocity_files = sorted(glob.glob(os.path.join(folder_path, "Velocity*.tif")))
    assert len(depth_files) == len(velocity_files), \
        f"File count mismatch in folder {folder}: depth {len(depth_files)} vs velocity {len(velocity_files)}"

    for dfile, vfile in zip(depth_files, velocity_files):
        # Parse timestamp string from TIFF filename
        fname = os.path.basename(dfile)
        time_str = fname.split("(")[1].split(")")[0]
        tstamp = datetime.strptime(time_str, "%d%b%Y %H %M %S")
        time_fmt = tstamp.strftime("%Y-%m-%d %H:%M")

        # Physical value clipping thresholds for reasonableness check
        MAX_DEPTH = 500  # Upper bound of valid water depth
        MAX_VELOCITY = 60.0  # Upper bound of valid flow velocity

        # Read and preprocess depth raster
        with rasterio.open(dfile) as src:
            depth_arr = src.read(1)
            depth_arr[depth_arr == -9999] = 0  # Replace NoData placeholder
            depth_arr = np.clip(depth_arr, 0, MAX_DEPTH)  # Clip to valid range
            depth_arr = depth_arr.flatten()

        # Read and preprocess velocity raster
        with rasterio.open(vfile) as src:
            vel_arr = src.read(1)
            vel_arr[vel_arr == -9999] = 0  # Replace NoData placeholder
            vel_arr = np.clip(vel_arr, 0, MAX_VELOCITY)  # Clip to valid range
            vel_arr = vel_arr.flatten()

        # Extract values only within valid mask region
        depth_vals = depth_arr[valid_idx]
        vel_vals = vel_arr[valid_idx]
        # dem_vals = dem_values[valid_idx]
        # slope_vals = slope_values[valid_idx]
        xs_valid = xs[valid_idx]
        ys_valid = ys[valid_idx]

        # Construct point feature array [M, 4] = [x_coord, y_coord, depth, velocity]
        # Uncomment below to include DEM and slope features -> [M, 6]
        # sample = np.stack([xs_valid, ys_valid, depth_vals, vel_vals, dem_vals, slope_vals], axis=-1)
        sample = np.stack([xs_valid, ys_valid, depth_vals, vel_vals], axis=-1)
        all_samples.append(sample)

        # Record sample metadata
        all_meta.append({
            "time": time_fmt
        })

# Stack all samples into 3D array: [sample_count, point_count, feature_dim]
data_array = np.stack(all_samples, axis=0)


# ==============================
# 3. Save Dataset to HDF5 File
# ==============================
h5_path = "../../dataset/SanJiang_LR_val.h5"
meta_for_h5 = []  # Metadata list mapping HDF5 keys to original timestamps

with h5py.File(h5_path, "w") as f:
    for idx, sample in enumerate(all_samples):
        orig_time = all_meta[idx]["time"]
        # Replace HDF5-incompatible characters in dataset key
        base_key = orig_time.replace(':', '-').replace(' ', '_')
        key_name = make_unique_key(f, base_key)  # Ensure globally unique key
        f.create_dataset(key_name, data=sample)

        # # Alternative key construction workflow
        # base_key = orig_time.replace(':', '-').replace(' ', '_')
        # key_name = make_unique_key(f, base_key)
        # f.create_dataset(key_name, data=sample)

        # Record metadata entry with original timestamp and actual HDF5 key
        meta_for_h5.append({
            "time": orig_time,
            "h5_key": key_name
        })

    # Save full metadata list to file-level attributes
    f.attrs['meta'] = json.dumps(meta_for_h5)

print(f"HDF5 dataset saved successfully, total samples = {len(all_samples)}")


# ==============================
# 4. Dataset Statistics Summary
# ==============================
max_depth = np.max(data_array[:, :, 2])
max_velocity = np.max(data_array[:, :, 3])
print(f"Maximum water depth in dataset: {max_depth}")
print(f"Maximum flow velocity in dataset: {max_velocity}")


# ==============================
# 5. Sample Data Preview
# ==============================
print("\nFirst 5 rows of the first sample (x, y, depth, velocity):")
print(data_array[0, :5, :])