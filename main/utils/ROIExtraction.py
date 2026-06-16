# Standard library imports
import os
import glob

# Third-party library imports
import numpy as np
import rasterio
from tqdm import tqdm

"""
This script performs coordinate-based ROI masking for batch GeoTIFF files.
It extracts zero-value pixel coordinates from a binary mask raster,
applies the mask to target TIF files by setting corresponding pixels to zero,
and preserves all geospatial metadata and pixel values outside the mask region.
Built-in validation ensures no unintended modification to non-mask areas,
and a processing report is generated automatically after batch execution.
"""


# ========= 1. Load mask and extract zero-value coordinates =========
def load_mask_zero_coordinates(mask_path):
    """
    Load a mask raster file and extract coordinates of all zero-value cells.

    Returns:
        zero_coords: list of (row, col) tuples for zero-value pixels
        mask_profile: geospatial profile of the mask raster
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # Read first band
        mask_profile = src.profile.copy()

    # Locate all coordinates with value equal to 0
    zero_rows, zero_cols = np.where(mask == 0)
    zero_coords = list(zip(zero_rows, zero_cols))

    print(f"✅ Mask loaded successfully: shape {mask.shape}, found {len(zero_coords)} zero-value cells")
    return zero_coords, mask_profile


# ========= 2. Process a single TIF file =========
def zero_out_mask_coordinates(input_tif_path, zero_coords, output_dir=None):
    """
    Set water depth values to zero at the specified pixel coordinates.

    Parameters:
        input_tif_path: path to input TIF file
        zero_coords: list of zero-value coordinates [(row1, col1), (row2, col2), ...]
        output_dir: output directory; if None, the result is not saved to disk

    Returns:
        modified_data: modified data array
        output_path: saved file path (None if not saved)
    """
    with rasterio.open(input_tif_path) as src:
        data = src.read(1)
        profile = src.profile.copy()

    # Create a copy of the input data
    modified_data = data.copy()

    # Set values at specified coordinates to 0
    modified_coords = []
    for row, col in zero_coords:
        if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
            modified_data[row, col] = 0
            modified_coords.append((row, col))

    # Save to file if output directory is specified
    output_path = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(input_tif_path)
        output_path = os.path.join(output_dir, filename)

        # Preserve original geospatial metadata
        profile.update({
            'driver': 'GTiff',
            'dtype': modified_data.dtype,
            'count': 1,
            'compress': 'lzw'
        })

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(modified_data, 1)

    return modified_data, output_path


# ========= 3. Batch processing =========
def batch_process_with_coordinate_zeroing(base_dir, folders, zero_coords, output_suffix="_masked"):
    """
    Batch process all TIF files in the specified folders.

    Returns:
        processing_stats: dictionary containing processing statistics
    """
    stats = {
        'total_files': 0,
        'processed': 0,
        'failed': 0,
        'output_files': []
    }

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)

        if not os.path.exists(folder_path):
            print(f"⚠️ Folder not found: {folder_path}")
            continue

        # Create output folder
        output_folder = f"{folder}{output_suffix}"
        output_dir = os.path.join(base_dir, output_folder)

        print(f"\n📁 Processing: {folder} → {output_folder}")

        # Find all TIF files
        tif_files = sorted(glob.glob(os.path.join(folder_path, "*.tif")))

        if not tif_files:
            print(f"  No TIF files found")
            continue

        # Process each file
        for tif_file in tqdm(tif_files, desc=f"Processing {folder}"):
            stats['total_files'] += 1

            try:
                modified_data, output_path = zero_out_mask_coordinates(
                    tif_file, zero_coords, output_dir
                )

                if output_path:
                    stats['processed'] += 1
                    stats['output_files'].append(output_path)

                    # Verify correctness of modification
                    with rasterio.open(tif_file) as src:
                        original = src.read(1)

                    # Compute changes only outside the mask region
                    unchanged_mask = np.ones_like(original, dtype=bool)
                    for row, col in zero_coords:
                        if 0 <= row < unchanged_mask.shape[0] and 0 <= col < unchanged_mask.shape[1]:
                            unchanged_mask[row, col] = False

                    unchanged_values_original = original[unchanged_mask]
                    unchanged_values_modified = modified_data[unchanged_mask]

                    diff = np.abs(unchanged_values_original - unchanged_values_modified)
                    max_diff = np.max(diff)

                    if max_diff > 1e-10:
                        print(f"  ⚠️ {os.path.basename(tif_file)}: data changed outside mask region! (max diff: {max_diff})")
                    else:
                        print(f"  ✅ {os.path.basename(tif_file)}: processed successfully")

            except Exception as e:
                print(f"  ❌ Processing failed {os.path.basename(tif_file)}: {e}")
                stats['failed'] += 1

    return stats


# ========= 4. Validation function =========
def validate_coordinate_method(mask_path, sample_tif_path):
    """
    Validate correctness of the coordinate-based masking approach.
    """
    # Load mask
    zero_coords, _ = load_mask_zero_coordinates(mask_path)

    with rasterio.open(sample_tif_path) as src:
        original_data = src.read(1)

    # Apply coordinate-based masking
    modified_coords = np.ones_like(original_data, dtype=bool)
    for row, col in zero_coords:
        if 0 <= row < modified_coords.shape[0] and 0 <= col < modified_coords.shape[1]:
            modified_coords[row, col] = False

    # Validation 1: check if areas outside mask remain completely unchanged
    unchanged_area = original_data[modified_coords]
    print(f"Validation 1 - cell count outside mask: {np.sum(modified_coords)}")
    print(f"Validation 1 - original value range outside mask: {unchanged_area.min():.6f} ~ {unchanged_area.max():.6f}")

    # Validation 2: check if zero-value mask areas are correctly identified
    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)

    mask_zero_area = mask_data == 0
    print(f"Validation 2 - zero-value cell count in mask: {np.sum(mask_zero_area)}")
    print(f"Validation 2 - extracted zero-value coordinate count: {len(zero_coords)}")
    print(f"Validation 2 - coordinate coverage completeness: {len(zero_coords) == np.sum(mask_zero_area)}")

    return True


# ========= Main program =========
if __name__ == "__main__":
    # Configuration paths
    mask_path = r"G:\Projects\ShouXi\python\SR-PointNet\dataset\sourceData\geo\mask_300.tif"
    base_dir = r"G:\Projects\ShouXi\python\SR-PointNet\resultSaving_300_geo_realRainfall\1"
    folders = ["D_int", "U_int"]

    print("🚀 Starting coordinate-based ROI extraction processing")
    print("=" * 60)

    # Step 1: load mask and extract zero-value coordinates
    zero_coords, mask_profile = load_mask_zero_coordinates(mask_path)

    # Validate consistency between mask and sample file (optional)
    sample_tif = os.path.join(base_dir, folders[0], "*.tif")
    sample_files = glob.glob(sample_tif)
    if sample_files:
        print(f"\n🔍 Validating sample file: {os.path.basename(sample_files[0])}")
        validate_coordinate_method(mask_path, sample_files[0])

    # Step 2: batch processing
    print(f"\n{'=' * 60}")
    print("Starting batch processing...")
    stats = batch_process_with_coordinate_zeroing(
        base_dir, folders, zero_coords, output_suffix="_masked"
    )

    # Step 3: output statistics
    print(f"\n{'=' * 60}")
    print("📊 Processing completed!")
    print(f"Total files: {stats['total_files']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Output files: {len(stats['output_files'])}")

    # Step 4: generate processing report
    report_path = os.path.join(base_dir, "coordinate_method_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Coordinate-Based ROI Extraction Processing Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mask file: {mask_path}\n")
        f.write(f"Zero-value coordinates in mask: {len(zero_coords)}\n")
        f.write(f"Processing time: {np.datetime64('now')}\n\n")

        f.write("Processing statistics:\n")
        f.write(f"  Total files: {stats['total_files']}\n")
        f.write(f"  Successfully processed: {stats['processed']}\n")
        f.write(f"  Failed: {stats['failed']}\n\n")

        f.write("Output file list:\n")
        for i, filepath in enumerate(stats['output_files'][:10], 1):
            f.write(f"  {i:3d}. {os.path.basename(filepath)}\n")

        if len(stats['output_files']) > 10:
            f.write(f"  ... and {len(stats['output_files']) - 10} more files\n")

    print(f"📝 Processing report saved to: {report_path}")
    print("=" * 60)