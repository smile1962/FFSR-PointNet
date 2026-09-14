import rasterio
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import re
from validation_shouxi import process_flattened_arrays

# ===================== Metric functions =====================
def calculate_rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))

def calculate_r(pred, true):
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    r = np.corrcoef(pred_flat, true_flat)[0, 1]
    return r

def calculate_psnr(pred, true, data_range=None):
    if data_range is None:
        data_range = true.max() - true.min()
    return peak_signal_noise_ratio(true, pred, data_range=data_range)

def calculate_ssim(pred, true, data_range=None):
    if data_range is None:
        data_range = true.max() - true.min()
    return structural_similarity(true, pred, data_range=data_range)

def calculate_binary_metrics(pred, true, threshold):
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

# ===================== Paths and parameters =====================
base_path = r"G:\Projects\ShouXi\python\resulting_shouxi_withoutThreshold"  # root result path

# The peak sample is now specified by sample name (e.g. "1.tif" or "1").
# To keep using an index, set peak_name = None and enable the peak_index logic below (filename is the default).
peak_name = "sample_orig23_test23.tif"  # <-- change this to the sample name you want (with or without .tif)
# peak_index = 40  # uncomment and set peak_name to None to select the peak by index instead of filename

# Define the data groups (depth/velocity), their folders and the binary thresholds
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

# Natural sort key (sorts embedded numbers numerically)
def natural_sort_key(s):
    parts = re.split(r'(\d+)', s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

# ===================== Batch metric computation =====================
results = {}
for group_name, info in groups.items():
    ori_folder, pred_folder, true_folder = info["folders"]
    threshold = info["threshold"]

    true_dir = os.path.join(base_path, true_folder)
    if not os.path.isdir(true_dir):
        raise FileNotFoundError(f"directory does not exist: {true_dir}")

    # Get the filenames of all samples (ensuring the three folders agree)
    sample_files = [f for f in os.listdir(true_dir) if f.lower().endswith(".tif")]
    if not sample_files:
        raise ValueError(f"No .tif files found in {true_dir}.")

    sample_files.sort(key=natural_sort_key)  # natural sort

    # Try to locate peak_index from the sample name (with or without extension, partial match)
    use_index_mode = False
    if 'peak_name' in globals() and peak_name:
        # Normalize the input
        peak_input = peak_name
        peak_input_base = os.path.splitext(peak_input)[0]

        # 1) Try an exact match first (case-sensitive filename comparison)
        if peak_input in sample_files:
            peak_file = peak_input
        # 2) Try matching with and without the file extension
        elif f"{peak_input}.tif" in sample_files:
            peak_file = f"{peak_input}.tif"
        else:
            # 3) Partial match: search using the base name as a substring (may match several)
            matches = [f for f in sample_files if peak_input_base in os.path.splitext(f)[0]]
            if len(matches) == 1:
                peak_file = matches[0]
                print(f"Warning: exact filename not found; using the partially matched sample: {peak_file}")
            elif len(matches) > 1:
                peak_file = matches[0]
                print(f"Warning: multiple partial matches found; using the first: {peak_file}. Matches (first 10): {matches[:10]}")
            else:
                # No match -> raise an error and list examples of available files
                sample_preview = sample_files[:20]
                raise ValueError(f"No sample file matches '{peak_name}'.\n"
                                 f"example available samples (up to 20): {sample_preview}")
        peak_index_resolved = sample_files.index(peak_file)
        print(f"Resolved peak sample: '{peak_file}' (index {peak_index_resolved} in the sorted sample list).")

    sample_metrics = []
    for file_name in sample_files:
        ori_path = os.path.join(base_path, ori_folder, file_name)
        pred_path = os.path.join(base_path, pred_folder, file_name)
        true_path = os.path.join(base_path, true_folder, file_name)

        # Check whether the file exists and raise a clear error if not
        for p in (ori_path, pred_path, true_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"missing file: {p}")

        with rasterio.open(ori_path) as src:
            ori = src.read(1)
        with rasterio.open(pred_path) as src:
            pred = src.read(1)
        with rasterio.open(true_path) as src:
            true = src.read(1)

        # flatten and preprocess
        pred_flatten = pred.flatten()
        ori_flatten = ori.flatten()
        true_flatten = true.flatten()

        rmse = calculate_rmse(pred, true)
        r = calculate_r(pred, true)
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

    # Compute "All": the average over all samples
    all_metrics = {}
    for metric in ["Recall (%)", "Precision (%)", "Accuracy (%)", "RMSE", "R", "PSNR", "SSIM"]:
        all_metrics[metric] = np.mean([sm[metric] for sm in sample_metrics])

    # Compute "Peak": using the resolved peak_index_resolved
    peak_metrics = sample_metrics[peak_index_resolved]

    results[group_name] = {
        "All": all_metrics,
        "Peak": peak_metrics
    }

# ===================== Generate the Excel table =====================
rows = []
for group_name in ["Depth", "Velocity"]:
    for subset in ["All", "Peak"]:
        metrics = results[group_name][subset]
        # For the peak case, record the filename in the table as well (for traceability)
        subset_label = subset
        if subset == "Peak":
            subset_label = f"Peak ({metrics.get('File','unknown')})"

        for metric_name in ["Recall (%)", "Precision (%)", "Accuracy (%)", "RMSE", "R", "PSNR", "SSIM"]:
            rows.append({
                "Performance index": metric_name,
                "Group": group_name,
                "Subset": subset_label,
                "Value": metrics[metric_name]
            })

df = pd.DataFrame(rows)

pivot_df = df.pivot_table(
    index=["Performance index", "Group"],
    columns="Subset",
    values="Value"
).reset_index()

excel_path = r"G:\Projects\ShouXi\python\flood_results_SRUnet.xlsx"
pivot_df.to_excel(excel_path, index=False)

print(f"Quantitative analysis results exported to: {excel_path}")
