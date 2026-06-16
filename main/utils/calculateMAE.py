# Standard library imports
import os
from pathlib import Path

# Third-party library imports
import numpy as np
import rasterio

"""
This script computes absolute error GeoTIFF rasters between ground truth and predicted flow field data.
It supports two modes: single raster processing and batch processing for all matched TIFF pairs.
Geospatial consistency (CRS, transform, raster shape) is validated before error calculation.
Output error rasters use float32 datatype with LZW compression.
"""


def calculate_absolute_error() -> str:
    """
    Compute absolute error raster for a single matched pair of ground truth and prediction TIFF.
    Validate geospatial consistency, generate error GeoTIFF, and print raster statistical summary.

    Returns:
        str: Full file path of exported absolute error raster
    """
    # File path configuration
    gt_path = r"G:\Projects\ShouXi\paper\img\DrawingSourceData\Yuhuo\D_true\2019-08-20_04_00.tif"
    pred_path = r"G:\Projects\ShouXi\paper\img\DrawingSourceData\Yuhuo\D_int\2019-08-20_04_00.tif"
    output_path = r"G:\Projects\ShouXi\paper\img\DrawingSourceData\Yuhuo\D_error_int\2019-08-20_04_00.tif"

    # Create output directory recursively if it does not exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading Ground Truth raster: {gt_path}")
    print(f"Reading Prediction raster: {pred_path}")

    # Load ground truth raster and geospatial metadata
    with rasterio.open(gt_path) as gt_src:
        gt_data = gt_src.read()
        gt_profile = gt_src.profile.copy()
        gt_transform = gt_src.transform
        gt_crs = gt_src.crs
        gt_bounds = gt_src.bounds

        print(f"\nGround Truth raster metadata:")
        print(f"  Shape: {gt_data.shape}")
        print(f"  Datatype: {gt_data.dtype}")
        print(f"  CRS: {gt_crs}")
        print(f"  Affine transform: {gt_transform}")
        print(f"  Spatial bounds: {gt_bounds}")
        print(f"  Cell size: {gt_transform.a} x {-gt_transform.e}")

    # Load prediction raster and geospatial metadata
    with rasterio.open(pred_path) as pred_src:
        pred_data = pred_src.read()
        pred_profile = pred_src.profile.copy()
        pred_transform = pred_src.transform
        pred_crs = pred_src.crs
        pred_bounds = pred_src.bounds

        print(f"\nPrediction raster metadata:")
        print(f"  Shape: {pred_data.shape}")
        print(f"  Datatype: {pred_data.dtype}")
        print(f"  CRS: {pred_crs}")
        print(f"  Affine transform: {pred_transform}")
        print(f"  Spatial bounds: {pred_bounds}")
        print(f"  Cell size: {pred_transform.a} x {-pred_transform.e}")

    # Check full geospatial matching between two rasters
    geospatial_info_match = (
        gt_transform == pred_transform
        and gt_crs == pred_crs
        and gt_data.shape == pred_data.shape
    )

    if not geospatial_info_match:
        print("\nWARNING: Geospatial metadata or raster shape mismatch detected!")
        print("Differences between Ground Truth and Prediction:")
        print(f"  Affine transform match: {gt_transform == pred_transform}")
        print(f"  CRS match: {gt_crs == pred_crs}")
        print(f"  Raster shape match: {gt_data.shape == pred_data.shape}")

        # Handle shape inconsistency
        if gt_data.shape != pred_data.shape:
            print(f"  Ground Truth shape: {gt_data.shape}")
            print(f"  Prediction shape: {pred_data.shape}")

            # Mismatched band count
            if len(gt_data.shape) == len(pred_data.shape) and gt_data.shape[0] != pred_data.shape[0]:
                print(f"  Band count mismatch: GT={gt_data.shape[0]}, Pred={pred_data.shape[0]}")
                raise ValueError("Band count inconsistent, cannot compute absolute error")
            else:
                raise ValueError("Raster spatial shape inconsistent, cannot compute absolute error")

    # Calculate pixel-wise absolute error
    print("\nCalculating pixel-wise absolute error...")
    absolute_error = np.abs(gt_data.astype(np.float32) - pred_data.astype(np.float32))

    # Update output raster profile: float32 + LZW compression
    gt_profile.update(
        dtype=rasterio.float32,
        count=absolute_error.shape[0],
        compress="lzw"
    )

    # Export absolute error GeoTIFF
    print(f"Writing absolute error raster to: {output_path}")
    with rasterio.open(output_path, "w", **gt_profile) as dst:
        dst.write(absolute_error)

    # Load exported error raster for statistical validation
    with rasterio.open(output_path) as src:
        error_data = src.read()
        error_profile = src.profile

        print(f"\nExported absolute error raster statistics:")
        print(f"  Shape: {error_data.shape}")
        print(f"  Datatype: {error_data.dtype}")
        print(f"  Min value: {np.nanmin(error_data):.4f}")
        print(f"  Max value: {np.nanmax(error_data):.4f}")
        print(f"  Mean value: {np.nanmean(error_data):.4f}")
        print(f"  Median value: {np.nanmedian(error_data):.4f}")
        print(f"  Standard deviation: {np.nanstd(error_data):.4f}")
        print(f"  CRS: {src.crs}")
        print(f"  Affine transform: {src.transform}")
        print(f"  Spatial bounds: {src.bounds}")

    # Verify geospatial consistency of output error raster
    geo_consistent = (
        error_profile["transform"] == gt_transform
        and error_profile["crs"] == gt_crs
        and error_data.shape == gt_data.shape
    )

    if geo_consistent:
        print("\n✓ Geospatial metadata of error raster fully matches ground truth")
    else:
        print("\n⚠ WARNING: Geospatial metadata mismatch in exported error raster")

    return output_path


def batch_calculate_absolute_errors() -> None:
    """
    Batch process all matched TIFF pairs to generate absolute error rasters.
    Filenames of ground truth and prediction must be identical.
    Automatically skip missing prediction files or rasters with shape mismatch.
    """
    # Root directory configuration for batch processing
    base_dir = r"G:\Projects\ShouXi\paper\img\DrawingSourceData\unet\resulting_sanjiang_paperUsed"
    gt_dir = os.path.join(base_dir, "D_true")
    pred_dir = os.path.join(base_dir, "D_int")
    output_dir = os.path.join(base_dir, "D_int_error")

    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    # Collect all TIFF files in ground truth directory
    gt_files = [f for f in os.listdir(gt_dir) if f.lower().endswith(".tif")]
    print(f"Detected {len(gt_files)} ground truth raster files for batch processing")

    # Iterate sorted raster list
    for idx, gt_filename in enumerate(sorted(gt_files)):
        print(f"\nProcessing file {idx + 1}/{len(gt_files)}: {gt_filename}")

        # Full path of ground truth raster
        gt_path = os.path.join(gt_dir, gt_filename)
        pred_filename = gt_filename
        pred_path = os.path.join(pred_dir, pred_filename)

        # Skip if matching prediction file does not exist
        if not os.path.exists(pred_path):
            print(f"  WARNING: Corresponding prediction file missing: {pred_path}")
            continue

        output_path = os.path.join(output_dir, output_filename=gt_filename)

        try:
            # Open paired rasters and compute error
            with rasterio.open(gt_path) as gt_src, rasterio.open(pred_path) as pred_src:
                gt_data = gt_src.read()
                pred_data = pred_src.read()

                # Skip rasters with inconsistent shape
                if gt_data.shape != pred_data.shape:
                    print(f"  WARNING: Shape mismatch, GT={gt_data.shape}, Pred={pred_data.shape}, skip this file")
                    continue

                # Compute absolute error array
                absolute_error = np.abs(gt_data.astype(np.float32) - pred_data.astype(np.float32))

                # Configure output raster metadata
                gt_profile = gt_src.profile.copy()
                gt_profile.update(
                    dtype=rasterio.float32,
                    count=absolute_error.shape[0],
                    compress="lzw"
                )

                # Write error GeoTIFF
                with rasterio.open(output_path, "w", **gt_profile) as dst:
                    dst.write(absolute_error)

                # Print brief statistical summary
                mean_error = np.nanmean(absolute_error)
                max_error = np.nanmax(absolute_error)
                print(f"  Completed: {gt_filename} | Mean Absolute Error: {mean_error:.4f} | Max Absolute Error: {max_error:.4f}")

        except Exception as e:
            print(f"  ERROR occurred while processing {gt_filename}: {str(e)}")

    print(f"\nBatch processing finished! All error rasters saved to: {output_dir}")


if __name__ == "__main__":
    # Single raster processing entry
    print("=" * 60)
    print("Single File Absolute Error Calculation Pipeline")
    print("=" * 60)
    try:
        output_file_path = calculate_absolute_error()
        print(f"\nSuccessfully exported absolute error raster to: {output_file_path}")
    except Exception as e:
        print(f"Runtime error during single file processing: {str(e)}")

    # Uncomment the block below to enable batch processing mode
    # print("\n" + "=" * 60)
    # print("Batch Absolute Error Calculation Pipeline")
    # print("=" * 60)
    # batch_calculate_absolute_errors()