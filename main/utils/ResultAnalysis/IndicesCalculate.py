"""
Author: Jiahui Wang
Date: 2026-01-18
Function: Calculate evaluation metrics across different water depth and velocity intervals for generating result tables in papers
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


def calculate_metrics(y_true, y_pred):
    """Calculate multiple accuracy evaluation metrics"""
    metrics = {}

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Remove invalid values
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (~np.isinf(y_true)) & (~np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return None

    # Calculate RMSE
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))

    # Calculate R² (square of correlation coefficient)
    if len(y_true) > 1:
        correlation_matrix = np.corrcoef(y_true, y_pred)
        metrics['R'] = correlation_matrix[0, 1]
        metrics['R2'] = r2_score(y_true, y_pred)
    else:
        metrics['R'] = np.nan
        metrics['R2'] = np.nan

    # Calculate PSNR
    max_val = np.max(y_true) if np.max(y_true) > 0 else 1
    mse = mean_squared_error(y_true, y_pred)
    if mse > 0:
        metrics['PSNR'] = 10 * np.log10(max_val ** 2 / mse)
    else:
        metrics['PSNR'] = np.inf

    # Calculate classification metrics (based on 0.01 m threshold, for near-zero water depth cases)
    # A prediction is considered correct if both values fall in the same interval or the difference is below the threshold
    threshold = 0.01
    correct = np.abs(y_true - y_pred) < threshold

    # For classification, a more relaxed criterion can be applied
    # Correct if both true and predicted depth are below 0.5 m
    # Since calculations are already performed within specific intervals, focus is on numerical difference

    metrics['Accuracy'] = np.mean(correct) * 100

    # For Recall and Precision, positive and negative samples need to be defined
    # Here "correct prediction" is defined as the positive class
    # Recall = TP / (TP + FN)
    # Precision = TP / (TP + FP)
    # These metrics are not straightforward for this regression task

    # Alternative: compute the proportion of samples with absolute error below the threshold
    # Serves as an accuracy-like metric
    # Metrics under multiple thresholds can be computed

    # For water depth prediction, the following classification metrics are defined:
    # Correct prediction: |pred - true| < 0.1 m OR relative error < 10%
    abs_error = np.abs(y_true - y_pred)
    rel_error = abs_error / (y_true + 1e-10)  # Avoid division by zero

    # Condition 1: absolute error < 0.1 m
    condition1 = abs_error < 0.1

    # Condition 2: relative error < 10%
    condition2 = rel_error < 0.1

    # Combined condition
    correct_class = condition1 | condition2
    metrics['Recall'] = np.mean(correct_class) * 100
    metrics['Precision'] = np.mean(correct_class) * 100  # Recall equals Precision in this definition

    return metrics


def extract_water_depth_by_intervals(water_depth_array, bounds=[0.5, 1.0, 2.0]):
    """
    Extract coordinates and water depth values grouped by depth intervals

    Parameters:
        water_depth_array: 2D array of water depth values
        bounds: interval boundaries [0.5, 1.0, 2.0], representing <0.5, 0.5-1, 1-2, >2 m

    Returns:
        coords_intervals: list of 4 coordinate lists, one per interval
        values_intervals: list of 4 depth value lists, one per interval
    """
    # Get array dimensions
    rows, cols = water_depth_array.shape

    # Initialize storage variables
    coords_intervals = [[], [], [], []]  # Coordinates for four intervals
    values_intervals = [[], [], [], []]  # Depth values for four intervals

    # Iterate over all grid cells
    for i in range(rows):
        for j in range(cols):
            depth = water_depth_array[i, j]

            # Check for valid values (non-NaN and non-inf)
            if np.isnan(depth) or np.isinf(depth):
                continue

            # Classify samples into intervals by depth value
            if depth < bounds[0]:  # < 0.5 m
                coords_intervals[0].append([i, j])  # row index, column index
                values_intervals[0].append(depth)
            elif bounds[0] <= depth < bounds[1]:  # 0.5-1 m
                coords_intervals[1].append([i, j])
                values_intervals[1].append(depth)
            elif bounds[1] <= depth < bounds[2]:  # 1-2 m
                coords_intervals[2].append([i, j])
                values_intervals[2].append(depth)
            else:  # > 2 m
                coords_intervals[3].append([i, j])
                values_intervals[3].append(depth)

    return coords_intervals, values_intervals


def extract_values_by_coords(water_depth_array, coords_list):
    """
    Extract depth values from a 2D array based on a list of coordinates

    Parameters:
        water_depth_array: 2D array of water depth values
        coords_list: list of coordinates, each element is [row, col]

    Returns:
        List of extracted water depth values
    """
    values = []
    for coord in coords_list:
        row, col = coord
        # Ensure coordinates are within array bounds
        if 0 <= row < water_depth_array.shape[0] and 0 <= col < water_depth_array.shape[1]:
            depth = water_depth_array[row, col]
            values.append(depth)
    return values


def process_tif_files(true_folder, pred_folder, output_excel_path):
    """
    Main function: process all TIF files and compute accuracy metrics

    Parameters:
        true_folder: path to folder containing ground truth depth TIF files
        pred_folder: path to folder containing predicted depth TIF files
        output_excel_path: path to save the output Excel report
    """

    # Get all TIF files
    true_files = [f for f in os.listdir(true_folder) if f.endswith('.tif')]
    pred_files = [f for f in os.listdir(pred_folder) if f.endswith('.tif')]

    # Ensure one-to-one file correspondence
    common_files = sorted(list(set(true_files) & set(pred_files)))

    if not common_files:
        print("Error: no matching TIF files found in the two folders!")
        return

    print(f"Found {len(common_files)} matching TIF files")

    # Initialize DataFrame to store all results
    all_results = []

    # Process each file
    for idx, filename in enumerate(common_files):
        print(f"Processing file {idx + 1}/{len(common_files)}: {filename}")

        # 1. Read ground truth depth TIF file
        true_path = os.path.join(true_folder, filename)
        with rasterio.open(true_path) as src:
            true_array = src.read(1)  # Read first band
            true_transform = src.transform

        # 2. Extract four depth intervals from ground truth data
        true_coords_intervals, true_values_intervals = extract_water_depth_by_intervals(true_array)

        # 3. Read predicted depth TIF file
        pred_path = os.path.join(pred_folder, filename)
        with rasterio.open(pred_path) as src:
            pred_array = src.read(1)

        # 4. Extract predicted depth values at ground truth coordinates
        pred_values_intervals = []
        for coords in true_coords_intervals:
            pred_values = extract_values_by_coords(pred_array, coords)
            pred_values_intervals.append(pred_values)

        # 5. Calculate accuracy metrics for each interval
        for interval_idx in range(4):
            interval_names = ["<0.5m", "0.5-1m", "1-2m", ">2m"]

            true_values = true_values_intervals[interval_idx]
            pred_values = pred_values_intervals[interval_idx]

            # Ensure both lists have the same length
            min_len = min(len(true_values), len(pred_values))
            if min_len == 0:
                # Skip if no data in this interval
                continue

            true_values = true_values[:min_len]
            pred_values = pred_values[:min_len]

            # Compute metrics
            metrics = calculate_metrics(true_values, pred_values)

            if metrics:
                # Create result dictionary
                result = {
                    'File': filename,
                    'Interval': interval_names[interval_idx],
                    'Cell_Count': min_len,
                    'Recall_%': metrics.get('Recall', np.nan),
                    'Precision_%': metrics.get('Precision', np.nan),
                    'Accuracy_%': metrics.get('Accuracy', np.nan),
                    'RMSE': metrics.get('RMSE', np.nan),
                    'R': metrics.get('R', np.nan),
                    'R2': metrics.get('R2', np.nan),
                    'PSNR': metrics.get('PSNR', np.nan)
                }

                all_results.append(result)

    # Build DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)

        # Save to Excel file
        results_df.to_excel(output_excel_path, index=False)
        print(f"Results saved to: {output_excel_path}")

        # Compute average metrics per interval
        print("\nAverage accuracy metrics per depth interval:")
        avg_results = results_df.groupby('Interval').agg({
            'Cell_Count': 'sum',
            'Recall_%': 'mean',
            'Precision_%': 'mean',
            'Accuracy_%': 'mean',
            'RMSE': 'mean',
            'R': 'mean',
            'PSNR': 'mean'
        }).reset_index()

        print(avg_results.to_string(index=False))

        # Save average values to a separate sheet
        with pd.ExcelWriter(output_excel_path, mode='a', engine='openpyxl') as writer:
            avg_results.to_excel(writer, sheet_name='Summary', index=False)

    else:
        print("No valid results to save")


# Main program
if __name__ == "__main__":
    # Configure paths
    true_folder = r"G:\Projects\ShouXi\python\SR-PointNet\resultSaving_300_geo_realRainfall\D_true"
    pred_folder = r"G:\Projects\ShouXi\python\SR-PointNet\resultSaving_300_geo_realRainfall\D_pred"
    output_excel_path = r"G:\Projects\ShouXi\python\SR-PointNet\resultSaving_300_geo_realRainfall\accuracy_results.xlsx"

    # Run processing
    process_tif_files(true_folder, pred_folder, output_excel_path)