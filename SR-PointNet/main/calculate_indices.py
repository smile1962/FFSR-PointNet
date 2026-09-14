# Standard library imports
import os
import re

# Third-party library imports
import numpy as np
import pandas as pd
import rasterio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ===================== Metric Calculation Functions =====================
def calculate_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Calculate Root Mean Squared Error between prediction and ground truth."""
    return np.sqrt(np.mean((pred - true) ** 2))


def calculate_r(pred: np.ndarray, true: np.ndarray) -> float:
    """Calculate Pearson correlation coefficient between prediction and ground truth."""
    # pred_flat = pred.flatten()
    # true_flat = true.flatten()
    return np.corrcoef(pred, true)[0, 1]


def calculate_psnr(pred: np.ndarray, true: np.ndarray, data_range: float = None) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR).
    Uses ground truth value range if not explicitly provided.
    """
    if data_range is None:
        data_range = true.max() - true.min()
    return peak_signal_noise_ratio(true, pred, data_range=data_range)


def calculate_ssim(pred: np.ndarray, true: np.ndarray, data_range: float = None) -> float:
    """Calculate Structural Similarity Index Measure (SSIM).
    Uses ground truth value range if not explicitly provided.
    """
    if data_range is None:
        data_range = true.max() - true.min()
    return structural_similarity(true, pred, data_range=data_range)


def calculate_binary_metrics(
        pred: np.ndarray,
        true: np.ndarray,
        threshold: float
) -> tuple[float, float, float]:
    """Calculate binary classification metrics (Recall, Precision, Accuracy) based on threshold.
    Returns values in percentage (0-100 scale).
    """
    pred_binary = (pred > threshold).astype(np.uint8)
    true_binary = (true > threshold).astype(np.uint8)

    tp = np.sum((pred_binary == 1) & (true_binary == 1))
    fp = np.sum((pred_binary == 1) & (true_binary == 0))
    fn = np.sum((pred_binary == 0) & (true_binary == 1))
    tn = np.sum((pred_binary == 0) & (true_binary == 0))

    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) * 100 if (tp + fp + fn + tn) > 0 else 0

    return recall, precision, accuracy


# ===================== Path and Parameter Configuration =====================
base_path = r"../resultSaving_ShouXi"  # Root directory for result files

# Specify the peak sample by filename (e.g., "1.tif" or "1")
# To use index instead, set peak_name = None and uncomment the peak_index logic below
# (filename matching is used by default)
peak_name = "2019-08-20_04_00.tif"  # <-- Modify to your target sample name
# peak_index = 40  # Uncomment to use index-based selection instead of filename

# Define data groups (Depth/Velocity) with corresponding folders and binarization thresholds
groups = {
    "Depth": {
        "folders": ["D_ori", "D_pred", "D_true"],
        "threshold": 0
    },
    "Velocity": {
        "folders": ["U_ori", "U_pred", "U_true"],
        "threshold": 0
    }
}


def natural_sort_key(s: str) -> list:
    """Natural sort key function: sorts strings with numeric parts by numerical value.
    Ensures correct ordering of filenames like "10.tif" > "2.tif".
    """
    parts = re.split(r'(\d+)', s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


# ===================== Batch Metric Calculation =====================
results = {}
for group_name, info in groups.items():
    ori_folder, pred_folder, true_folder = info["folders"]
    threshold = info["threshold"]

    true_dir = os.path.join(base_path, true_folder)
    if not os.path.isdir(true_dir):
        raise FileNotFoundError(f"Directory not found: {true_dir}")

    # Get all sample filenames (ensure consistency across ori/pred/true folders)
    sample_files = [f for f in os.listdir(true_dir) if f.lower().endswith(".tif")]
    if not sample_files:
        raise ValueError(f"No .tif files found in directory: {true_dir}")

    sample_files.sort(key=natural_sort_key)  # Apply natural sort

    # Resolve peak index by sample name (supports with/without extension, partial match)
    use_index_mode = False
    if 'peak_name' in globals() and peak_name:
        peak_input = peak_name
        peak_input_base = os.path.splitext(peak_input)[0]

        # 1. Exact filename match
        if peak_input in sample_files:
            peak_file = peak_input
        # 2. Match with .tif extension appended
        elif f"{peak_input}.tif" in sample_files:
            peak_file = f"{peak_input}.tif"
        # 3. Partial match by base name
        else:
            matches = [f for f in sample_files if peak_input_base in os.path.splitext(f)[0]]
            if len(matches) == 1:
                peak_file = matches[0]
                print(f"Warning: No exact match found, using partial match: {peak_file}")
            elif len(matches) > 1:
                peak_file = matches[0]
                print(f"Warning: Multiple partial matches found, using first: {peak_file}")
                print(f"Top 10 matches: {matches[:10]}")
            else:
                sample_preview = sample_files[:20]
                raise ValueError(
                    f"No sample matching '{peak_name}' found.\n"
                    f"Available samples (first 20): {sample_preview}"
                )

        peak_index_resolved = sample_files.index(peak_file)
        print(f"Peak sample resolved: '{peak_file}' (index {peak_index_resolved} in sorted list)")

    sample_metrics = []
    for file_name in sample_files:
        ori_path = os.path.join(base_path, ori_folder, file_name)
        pred_path = os.path.join(base_path, pred_folder, file_name)
        true_path = os.path.join(base_path, true_folder, file_name)

        # Validate all required files exist
        for p in (ori_path, pred_path, true_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Missing required file: {p}")

        # Read raster data
        with rasterio.open(ori_path) as src:
            ori = src.read(1)
        with rasterio.open(pred_path) as src:
            pred = src.read(1)
        with rasterio.open(true_path) as src:
            true = src.read(1)

        # Flatten arrays for correlation calculation
        pred_flatten = pred.flatten()
        ori_flatten = ori.flatten()
        true_flatten = true.flatten()

        # Calculate all metrics
        rmse = calculate_rmse(pred, true)
        r = calculate_r(pred_flatten, true_flatten)
        psnr = calculate_psnr(pred, true)
        ssim = calculate_ssim(pred, true)
        recall, precision, accuracy = calculate_binary_metrics(pred, true, threshold)

        sample_metrics.append({
            "File": file_name,
            "Recall (%)": recall,
            "Precision (%)": precision,
            "Accuracy (%)": accuracy,
            "RMSE": rmse,
            "R": r,
            "PSNR": psnr,
            "SSIM": ssim
        })

    # Calculate "All" subset: average metrics across all samples
    all_metrics = {}
    for metric in ["Recall (%)", "Precision (%)", "Accuracy (%)", "RMSE", "R", "PSNR", "SSIM"]:
        all_metrics[metric] = np.mean([sm[metric] for sm in sample_metrics])

    # Calculate "Peak" subset: metrics for the resolved peak sample
    peak_metrics = sample_metrics[peak_index_resolved]

    results[group_name] = {
        "All": all_metrics,
        "Peak": peak_metrics
    }

# ===================== Generate Excel Report =====================
rows = []
for group_name in ["Depth", "Velocity"]:
    for subset in ["All", "Peak"]:
        metrics = results[group_name][subset]
        # Include filename in label for peak subset to enable traceability
        subset_label = subset
        if subset == "Peak":
            subset_label = f"Peak ({metrics.get('File', 'unknown')})"

        for metric_name in ["Recall (%)", "Precision (%)", "Accuracy (%)", "RMSE", "R", "PSNR", "SSIM"]:
            rows.append({
                "Performance index": metric_name,
                "Group": group_name,
                "Subset": subset_label,
                "Value": metrics[metric_name]
            })

# Create pivot table for structured output
df = pd.DataFrame(rows)
pivot_df = df.pivot_table(
    index=["Performance index", "Group"],
    columns="Subset",
    values="Value"
).reset_index()

# Export to Excel
excel_path = r"G:\Projects\ShouXi\python\flood_quantitative_results_SRPointNet.xlsx"
pivot_df.to_excel(excel_path, index=False)

print(f"Quantitative analysis results exported to: {excel_path}")