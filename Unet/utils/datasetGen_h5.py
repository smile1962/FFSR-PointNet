import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import glob
import h5py
import numpy as np
import rasterio
from scipy.io import savemat
from datetime import datetime
import gc  # for memory reclamation


# ===================== Configuration constants (thresholds and invalid values) =====================
THRESHOLDS = {
    "depth_upper": 500,    # depth cap: values above 500 m are clipped to 500
    "velocity_upper": 60   # velocity cap: values above 60 are clipped to 60
}
NODATA_VALUE = -9999     # TIFF invalid value
REPLACE_VALUE = 0         # replacement value for invalid data
# ==========================================================================


def read_tiff_file(file_path, target_shape=(1200, 1800)):
# def read_tiff_file(file_path, target_shape=(2048, 1152)):
    """Read a TIFF file, replace -9999 invalid values with 0, and zero-pad to the target resolution [1200, 1800]"""
    try:
        with rasterio.open(file_path) as src:
            # 1. Read the TIFF data (first band)
            data = src.read(1).astype(np.float32)  # cast to float32 to avoid integer type issues
            original_shape = data.shape

            # 2. Core: replace -9999 with 0 and count replacements (for logging)
            nodata_count = np.sum(data == NODATA_VALUE)
            if nodata_count > 0:
                print(f"  TIFF file {os.path.basename(file_path)}: found {nodata_count} values of {NODATA_VALUE}, replaced with {REPLACE_VALUE}")
                data[data == NODATA_VALUE] = REPLACE_VALUE  # in-place replacement with no extra memory

            # 3. Resolution cropping (if it exceeds the target shape)
            if data.shape[0] > target_shape[0] or data.shape[1] > target_shape[1]:
                print(f"  Warning: file {os.path.basename(file_path)} exceeds the resolution limit ({data.shape}); cropping to {target_shape}")
                data = data[:target_shape[0], :target_shape[1]]

            # 4. Compute padding (zeros on top and left, consistent with the original logic)
            pad_rows = target_shape[0] - data.shape[0]
            pad_cols = target_shape[1] - data.shape[1]

            # 5. Remove padding (only when padding was applied)
            if pad_rows > 0 or pad_cols > 0:
                padded_data = np.pad(
                    data,
                    ((pad_rows, 0), (pad_cols, 0)),  # pad pad_rows rows on top and pad_cols columns on the left
                    mode='constant',
                    constant_values=REPLACE_VALUE  # zero padding (matching the invalid-value replacement)
                )
            else:
                padded_data = data

            return padded_data, original_shape
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, None


def load_dem_slope(geo_dir):
    """Read and preprocess the DEM and Slope files (global, -9999 already handled by read_tiff_file)"""
    # dem_path = os.path.join(geo_dir, "dem_sanjiang_1_5.tif")
    # slope_path = os.path.join(geo_dir, "Sanjiang_slope_1_5.tif")
    dem_path = os.path.join(geo_dir, "dem_30.tif")
    slope_path = os.path.join(geo_dir, "slope_30.tif")

    # Read the DEM (-9999 already handled)
    dem_data, dem_shape = read_tiff_file(dem_path)
    if dem_data is None:
        raise ValueError(f"Cannot read the DEM file: {dem_path}")

    # Read the Slope (-9999 already handled)
    slope_data, slope_shape = read_tiff_file(slope_path)
    if slope_data is None:
        raise ValueError(f"Cannot read the Slope file: {slope_path}")

    # Check that DEM and Slope have the same shape
    if dem_data.shape != slope_data.shape:
        raise ValueError(f"DEM and Slope shapes differ: {dem_data.shape} vs {slope_data.shape}")

    print(f"\nSuccessfully loaded DEM and Slope data (-9999 handled), shape: {dem_data.shape}")
    return dem_data, slope_data


def count_total_samples(base_dir):
    """Count the total number of samples up front to avoid loading everything at once"""
    # high_res_folders = ["2year30mesh", "5year30mesh", "10year30mesh",
    #                     "20year30mesh", "50year30mesh", "100year30mesh",
    #                     "real1_30", "real2_30"]
    # low_res_folders = ["2_cons", "5_cons", "10_cons", "20_cons",
    #                    "50_cons", "100_cons", "real1_300", "real2_300"]

    high_res_folders = ["test_real_30"]
    low_res_folders = ["test_real_300"]

    # high_res_folders = ["val_HR"]
    # low_res_folders = ["val_LR"]
    # high_res_folders = ["real2_15mesh", "real4_15mesh"]
    # low_res_folders = ["real2_20mesh_bicbuic", "real4_20mesh_bicbuic"]

    total_hr = 0
    total_lr = 0

    # Count the high-resolution samples
    for folder in high_res_folders:
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            depth_files = glob.glob(os.path.join(folder_path, "Depth (*).Terrain.MIKE_12_5_FILLED.tif"))
            # depth_files = glob.glob(os.path.join(folder_path, "Depth (*).Terrain.sanjiang.tif"))
            total_hr += len(depth_files)

    # Count the low-resolution samples
    for folder in low_res_folders:
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            depth_files = glob.glob(os.path.join(folder_path, "Depth (*).Terrain.MIKE_12_5_FILLED.tif"))
            # depth_files = glob.glob(os.path.join(folder_path, "Depth (*).Terrain.sanjiang.tif"))
            total_lr += len(depth_files)

    print(f"Estimated total samples - high resolution: {total_hr}, low resolution: {total_lr}")
    return total_hr, total_lr


def process_all_data(base_dir, geo_dir):
    """Process data with low memory usage and write to the H5 file in batches (all data has been processed for -9999)"""
    # Load DEM and Slope (-9999 already handled)
    dem_data, slope_data = load_dem_slope(geo_dir)

    # Count the total number of samples up front
    total_hr, total_lr = count_total_samples(base_dir)
    if total_hr == 0 or total_lr == 0:
        print("No valid samples found; exiting")
        return

    # Define the high/low-resolution folder pairs
    # high_res_folders = ["2year30mesh", "5year30mesh", "10year30mesh",
    #                     "20year30mesh", "50year30mesh", "100year30mesh",
    #                     "real1_30", "real2_30"]
    # low_res_folders = ["2_cons", "5_cons", "10_cons", "20_cons",
    #                    "50_cons", "100_cons", "real1_300", "real2_300"]

    high_res_folders = ["test_real_30"]
    low_res_folders = ["test_real_300"]

    # high_res_folders = ["val_HR"]
    # low_res_folders = ["val_LR"]
    # high_res_folders = ["real2_15mesh", "real4_15mesh"]
    # low_res_folders = ["real2_20mesh_bicbuic", "real4_20mesh_bicbuic"]

    # Create the H5 files and pre-allocate space
    lr_h5 = os.path.join(base_dir, "LR_Shouxi_val.h5")
    hr_h5 = os.path.join(base_dir, "HR_Shouxi_val.h5")

    # Print the data-processing configuration
    print(f"\n=== Data processing configuration ===")
    print(f"1. Invalid value handling: {NODATA_VALUE} -> {REPLACE_VALUE}")
    print(f"2. Threshold clipping: depth <= {THRESHOLDS['depth_upper']} m, velocity <= {THRESHOLDS['velocity_upper']}")
    print(f"3. Target resolution: {target_shape}")
    print("====================\n")

    # Create the low-resolution H5 file (4 channels)  [1200, 1800]  2048, 1152
    with h5py.File(lr_h5, 'w') as f_lr:
        lr_ds = f_lr.create_dataset(
            'data',
            shape=(total_lr, 4, 1200, 1800),
            dtype=np.float32,
            compression='gzip'
        )

        # Create the high-resolution H5 file (2 channels)
        with h5py.File(hr_h5, 'w') as f_hr:
            hr_ds = f_hr.create_dataset(
                'data',
                shape=(total_hr, 2, 1200, 1800),
                dtype=np.float32,
                compression='gzip'
            )

            # Metadata and index tracking
            all_metadata = []
            lr_idx = 0  # current write position in the low-resolution data
            hr_idx = 0  # current write position in the high-resolution data

            # Iterate over each event pair
            for high_folder, low_folder in zip(high_res_folders, low_res_folders):
                event_name = high_folder.split('15mesh')[0].split('_')[0]
                print(f"Processing the {event_name} event...")

                # Process high-resolution data (2 channels: depth, velocity; -9999 already handled)
                hr_timestamps, hr_count = process_high_resolution(
                    os.path.join(base_dir, high_folder),
                    event_name,
                    hr_ds,
                    hr_idx
                )
                hr_idx += hr_count

                # Process low-resolution data (4 channels, -9999 already handled)
                lr_count = process_low_resolution(
                    os.path.join(base_dir, low_folder),
                    event_name,
                    dem_data,
                    slope_data,
                    lr_ds,
                    lr_idx
                )
                lr_idx += lr_count

                # Record metadata
                for i in range(hr_count):
                    timestamp = hr_timestamps[i] if i < len(hr_timestamps) else "unknown time"
                    all_metadata.append({
                        'event': event_name,
                        'hr_index': hr_idx - hr_count + i,
                        'lr_index': lr_idx - lr_count + i if lr_count == hr_count else -1,
                        'timestamp': timestamp
                    })

                print(f"Finished processing the {event_name} event - high resolution: {hr_count}, low resolution: {lr_count}\n")
                gc.collect()  # free memory

    # Save metadata
    mat_file = os.path.join(base_dir, "metadata_val.mat")
    savemat(mat_file, {'metadata': all_metadata})
    print(f"Metadata saved: {mat_file}, records: {len(all_metadata)}")

    print(f"=== Processing complete ===")
    print(f"Low-resolution H5: {lr_h5} ({lr_idx} samples, 4 channels, -9999 handled)")
    print(f"High-resolution H5: {hr_h5} ({hr_idx} samples, 2 channels, -9999 handled)")


def process_high_resolution(folder_path, event_name, h5_dataset, start_idx):
    """Process high-resolution data (2 channels: depth, velocity); -9999 already handled by read_tiff_file"""
    if not os.path.exists(folder_path):
        print(f"  Folder does not exist: {folder_path}; skipping")
        return [], 0

    # depth_files = sorted(glob.glob(os.path.join(folder_path, "Depth (*).Terrain.sanjiang.tif")))
    depth_files = sorted(glob.glob(os.path.join(folder_path, "Depth (*).Terrain.MIKE_12_5_FILLED.tif")))
    count = 0
    timestamps = []

    for depth_file in depth_files:
        try:
            # Extract the timestamps
            time_str = depth_file.split("Depth (")[1].split(").Terrain")[0]
            timestamp = datetime.strptime(time_str, "%d%b%Y %H %M %S").strftime("%Y-%m-%d %H:%M:%S")
            timestamps.append(timestamp)

            # Match the velocity files
            velocity_file = depth_file.replace("Depth", "Velocity")
            if not os.path.exists(velocity_file):
                print(f"  Velocity file not found: {os.path.basename(velocity_file)}; skipping this sample")
                continue

            # Read data (-9999 already handled)
            depth_data, _ = read_tiff_file(depth_file)
            velocity_data, _ = read_tiff_file(velocity_file)

            if depth_data is None or velocity_data is None:
                print(f"  Failed to read depth/velocity data; skipping {os.path.basename(depth_file)}")
                continue

            # Shape validation
            if depth_data.shape != (1200, 1800) or velocity_data.shape != (1200, 1800):
                print(f"  Unexpected data shape (1200x1800 expected); skipping {os.path.basename(depth_file)}")
                continue

            # Threshold clipping (depth and velocity)
            # Depth clipping
            depth_over = np.sum(depth_data > THRESHOLDS["depth_upper"])
            if depth_over > 0:
                print(f"  High-resolution {event_name} sample {count+1}: {depth_over} pixels exceeded {THRESHOLDS['depth_upper']} m depth and were clipped")
            depth_data = np.clip(depth_data, a_min=-np.inf, a_max=THRESHOLDS["depth_upper"])

            # Velocity clipping
            velocity_over = np.sum(velocity_data > THRESHOLDS["velocity_upper"])
            if velocity_over > 0:
                print(f"  High-resolution {event_name} sample {count+1}: {velocity_over} pixels exceeded {THRESHOLDS['velocity_upper']} velocity and were clipped")
            velocity_data = np.clip(velocity_data, a_min=-np.inf, a_max=THRESHOLDS["velocity_upper"])

            # Write to H5 (2 channels: depth, velocity)
            h5_dataset[start_idx + count] = np.stack([depth_data, velocity_data], axis=0)
            count += 1

            # Progress indicator
            if count % 10 == 0:
                print(f"  Processed {count} high-resolution {event_name} samples")

        except Exception as e:
            print(f"  Error processing {os.path.basename(depth_file)}: {e}; skipping")
            continue

    return timestamps, count


def process_low_resolution(folder_path, event_name, dem_data, slope_data, h5_dataset, start_idx):
    """Process low-resolution data (4 channels); -9999 already handled by read_tiff_file"""
    if not os.path.exists(folder_path):
        print(f"  Folder does not exist: {folder_path}; skipping")
        return 0

    # depth_files = sorted(glob.glob(os.path.join(folder_path, "Depth (*).Terrain.sanjiang.tif")))
    depth_files = sorted(glob.glob(os.path.join(folder_path, "Depth (*).Terrain.MIKE_12_5_FILLED.tif")))
    count = 0

    for depth_file in depth_files:
        try:
            # Match the velocity files
            velocity_file = depth_file.replace("Depth", "Velocity")
            if not os.path.exists(velocity_file):
                print(f"  Velocity file not found: {os.path.basename(velocity_file)}; skipping this sample")
                continue

            # Read data (-9999 already handled)
            depth_data, _ = read_tiff_file(depth_file)
            velocity_data, _ = read_tiff_file(velocity_file)

            if depth_data is None or velocity_data is None:
                print(f"  Failed to read depth/velocity data; skipping {os.path.basename(depth_file)}")
                continue

            # Shape validation (ensure it matches DEM/Slope)
            if not (depth_data.shape == velocity_data.shape == dem_data.shape == (1200, 1800)):
                print(f"  Unexpected data shape (1200x1800 expected); skipping {os.path.basename(depth_file)}")
                continue

            # Threshold clipping (depth and velocity)
            # Depth clipping
            depth_over = np.sum(depth_data > THRESHOLDS["depth_upper"])
            if depth_over > 0:
                print(f"  Low-resolution {event_name} sample {count+1}: {depth_over} pixels exceeded {THRESHOLDS['depth_upper']} m depth and were clipped")
            depth_data = np.clip(depth_data, a_min=-np.inf, a_max=THRESHOLDS["depth_upper"])

            # Velocity clipping
            velocity_over = np.sum(velocity_data > THRESHOLDS["velocity_upper"])
            if velocity_over > 0:
                print(f"  Low-resolution {event_name} sample {count+1}: {velocity_over} pixels exceeded {THRESHOLDS['velocity_upper']} velocity and were clipped")
            velocity_data = np.clip(velocity_data, a_min=-np.inf, a_max=THRESHOLDS["velocity_upper"])

            # Write to H5 (4 channels: depth, velocity, DEM, Slope)
            h5_dataset[start_idx + count] = np.stack([
                depth_data, velocity_data, dem_data, slope_data
            ], axis=0)
            count += 1

            # Progress indicator
            if count % 10 == 0:
                print(f"  Processed {count} low-resolution {event_name} samples")

        except Exception as e:
            print(f"  Error processing {os.path.basename(depth_file)}: {e}; skipping")
            continue

    return count


if __name__ == "__main__":
    # Base configuration (adjust the paths to your setup)
    base_directory = rf"{ROOT}\Data\dataset\Unet\geo_inside"
    geo_directory = rf"{ROOT}\Data\dataset\geo"
    target_shape = (1200, 1800)  # global target resolution for case 1
    # target_shape = (2048, 1152)  # global target resolution for case 2 -- SanJiang

    # Path validation
    if not os.path.exists(base_directory):
        print(f"Error: base data path does not exist: {base_directory}")
    elif not os.path.exists(geo_directory):
        print(f"Error: DEM/Slope path does not exist: {geo_directory}")
    else:
        process_all_data(base_directory, geo_directory)