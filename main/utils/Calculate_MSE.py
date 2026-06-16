# Standard library imports
import os

# Third-party library imports
import numpy as np
import rasterio
import pandas as pd
from scipy.stats import pearsonr

"""
This script reads GeoTIFF files and calculates MSE and Pearson correlation coefficient metrics
between predicted results and ground truth, filtered by a study area mask.
"""

# ===================== Configuration Parameters =====================
# Root directory of experiment outputs
root_dir = r"G:\Projects\ShouXi\python\SR-PointNet\resultSaving_300_geo_realRainfall"
# Folder storing predicted velocity raster data
pred_dir = os.path.join(root_dir, "U_pred")
# Folder storing ground truth velocity raster data
true_dir = os.path.join(root_dir, "U_true")
# Path to study area binary mask raster
mask_path = r"G:\Projects\ShouXi\python\SR-PointNet\dataset\sourceData\geo\mask_30.tif"
# Output path for metric Excel table
output_excel = os.path.join(root_dir, "evaluation_metrics_U.xlsx")


# ===================== Core Utility Functions =====================
def read_tif(file_path: str) -> np.ndarray | None:
    """
    Read single-band GeoTIFF raster file and return the pixel value array.

    Args:
        file_path: Full file path of target TIFF raster

    Returns:
        2D numpy array of raster values; return None if reading fails
    """
    try:
        with rasterio.open(file_path) as src:
            # Read the first band (depth / velocity rasters are stored as single-band)
            data = src.read(1)
        return data
    except Exception as e:
        print(f"Failed to read file {file_path}: {e}")
        return None


def calculate_metrics(
    pred_data: np.ndarray,
    true_data: np.ndarray,
    mask: np.ndarray
) -> tuple[float, float]:
    """
    Compute MSE and Pearson correlation coefficient within valid mask regions.

    Args:
        pred_data: 2D array of model prediction raster
        true_data: 2D array of ground truth raster
        mask: Binary mask array (1 = valid study area, 0 = invalid background)

    Returns:
        Tuple (mse, r_value); returns np.nan for both if valid sample count is insufficient
    """
    # Step 1: Extract pixels only within mask valid region
    mask_valid = mask == 1
    pred_valid = pred_data[mask_valid]
    true_valid = true_data[mask_valid]

    # Step 2: Remove pixels containing NaN values
    non_nan_mask = ~(np.isnan(pred_valid) | np.isnan(true_valid))
    pred_clean = pred_valid[non_nan_mask]
    true_clean = true_valid[non_nan_mask]

    # Check if available valid data points meet calculation requirement
    if len(pred_clean) < 2 or len(true_clean) < 2:
        print("Warning: Insufficient valid data points, cannot compute correlation coefficient")
        return np.nan, np.nan

    # Step 3: Calculate Mean Squared Error
    mse = np.mean((pred_clean - true_clean) ** 2)

    # Step 4: Calculate Pearson correlation coefficient
    r_value, _ = pearsonr(pred_clean, true_clean)

    return mse, r_value


# ===================== Main Execution Pipeline =====================
def main() -> None:
    # Load binary study area mask
    mask = read_tif(mask_path)
    if mask is None:
        print("Mask file loading failed, terminating program")
        return

    # Collect all TIFF filenames in prediction folder, sorted alphabetically
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith(".tif")]
    pred_files.sort()
    if not pred_files:
        print("No TIFF files found in prediction directory")
        return

    # Initialize list to store metric records
    results = []

    # Iterate over paired prediction / ground truth raster files
    for file_name in pred_files:
        pred_file = os.path.join(pred_dir, file_name)
        true_file = os.path.join(true_dir, file_name)

        # Skip sample if corresponding ground truth file does not exist
        if not os.path.exists(true_file):
            print(f"Warning: Matching ground truth file for {file_name} missing, skip this sample")
            continue

        # Read raster arrays
        pred_data = read_tif(pred_file)
        true_data = read_tif(true_file)

        if pred_data is None or true_data is None:
            print(f"Warning: Failed to load raster data for {file_name}, skip this sample")
            continue

        # Validate consistent spatial dimensions across prediction, ground truth and mask
        if pred_data.shape != true_data.shape or pred_data.shape != mask.shape:
            print(f"Warning: Spatial dimension mismatch for {file_name}, skip this sample")
            continue

        # Compute quantitative evaluation metrics
        mse, r_value = calculate_metrics(pred_data, true_data, mask)

        # Append metrics record
        results.append({
            "FileName": file_name,
            "MSE": mse,
            "R_Correlation": r_value
        })
        print(f"Processed sample: {file_name} | MSE: {mse:.6f} | R_Correlation: {r_value:.6f}")

    # Export all collected metrics to Excel spreadsheet
    if results:
        df = pd.DataFrame(results)
        df.to_excel(output_excel, index=False, engine="openpyxl")
        print(f"\nAll evaluation metrics successfully exported to: {output_excel}")
    else:
        print("No valid sample metrics to save")


if __name__ == "__main__":
    main()