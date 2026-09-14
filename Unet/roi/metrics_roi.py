import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from skimage.metrics import structural_similarity

import roi_data


METRIC_NAMES = [
    "Recall (%)",
    "Precision (%)",
    "Accuracy (%)",
    "RMSE",
    "R",
    "PSNR",
    "SSIM",
]


def threshold_array(arr, threshold=0.01):
    out = np.array(arr, dtype=np.float64, copy=True)
    out = np.clip(out, 0, None)
    out[out < threshold] = 0.0
    return out


def sample_metrics(pred, true, roi, peak_data_range=None):
    pred = threshold_array(pred)
    true = threshold_array(true)
    pr = pred[roi]
    tr = true[roi]

    rmse = float(np.sqrt(np.mean((pr - tr) ** 2)))
    if np.std(pr) < 1e-12 and np.std(tr) < 1e-12:
        r = 1.0
    elif np.std(pr) < 1e-12 or np.std(tr) < 1e-12:
        r = 0.0
    else:
        r = float(np.corrcoef(pr, tr)[0, 1])

    data_range = float(np.max(tr) - np.min(tr))
    if data_range < 1e-8:
        data_range = 1.0
    psnr = 10.0 * np.log10(data_range ** 2 / (rmse ** 2 + 1e-12))

    # SSIM is computed on the rectangular ROI bounding box after zeroing the
    # outside; this keeps the formula identical for both models.
    rows = np.where(roi.any(axis=1))[0]
    cols = np.where(roi.any(axis=0))[0]
    if len(rows) and len(cols):
        r0, r1 = rows[0], rows[-1] + 1
        c0, c1 = cols[0], cols[-1] + 1
        roi_box = roi[r0:r1, c0:c1]
        pred_box = np.where(roi_box, pred[r0:r1, c0:c1], 0.0)
        true_box = np.where(roi_box, true[r0:r1, c0:c1], 0.0)
        ssim_range = float(np.max(true_box) - np.min(true_box))
        if ssim_range < 1e-8:
            ssim = 1.0
        else:
            ssim = float(
                structural_similarity(
                    true_box, pred_box, data_range=ssim_range, win_size=7
                )
            )
    else:
        ssim = 1.0

    pred_bin = pr > 0
    true_bin = tr > 0
    tp = np.sum(pred_bin & true_bin)
    fp = np.sum(pred_bin & ~true_bin)
    fn = np.sum(~pred_bin & true_bin)
    tn = np.sum(~pred_bin & ~true_bin)
    recall = 100.0 * tp / (tp + fn) if (tp + fn) else 0.0
    precision = 100.0 * tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = 100.0 * (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "Recall (%)": recall,
        "Precision (%)": precision,
        "Accuracy (%)": accuracy,
        "RMSE": rmse,
        "R": r,
        "PSNR": psnr,
        "SSIM": ssim,
    }


def read_folder_tifs(folder):
    folder = Path(folder)
    files = sorted(p for p in folder.glob("*.tif"))
    return files


def read_array(path):
    with rasterio.open(str(path)) as src:
        return src.read(1)


def build_comparison_table(cfg, ffsr_dir, unet_dir, peak_name, cnn_label):
    roi = roi_data.load_roi_original(cfg)
    true_files = read_folder_tifs(Path(unet_dir) / "D_true")
    if not true_files:
        true_files = read_folder_tifs(Path(ffsr_dir) / "D_true")

    models = {"FFSR-PointNet": ffsr_dir, cnn_label: unet_dir}
    all_rows = []
    summary = {}
    for model_name, base_dir in models.items():
        base = Path(base_dir)
        summary[model_name] = {}
        for group, letter in (("Depth", "D"), ("Velocity", "U")):
            pred_files = read_folder_tifs(base / f"{letter}_pred")
            true_group_files = read_folder_tifs(base / f"{letter}_true")
            if not true_group_files:
                true_group_files = true_files
            file_map = {p.stem: p for p in pred_files}
            true_map = {p.stem: p for p in true_group_files}
            common = sorted(set(file_map) & set(true_map))
            if not common:
                raise FileNotFoundError(
                    f"No matching files in {base / letter}_pred"
                )

            per_sample = []
            for stem in common:
                pred = read_array(file_map[stem])
                true = read_array(true_map[stem])
                metrics = sample_metrics(pred, true, roi)
                metrics["File"] = stem + ".tif"
                per_sample.append(metrics)

            for metric in METRIC_NAMES:
                all_values = float(np.mean([p[metric] for p in per_sample]))
                peak_candidates = [p for p in per_sample if p["File"] == peak_name]
                peak_value = (
                    float(peak_candidates[0][metric])
                    if peak_candidates
                    else float("nan")
                )
                for subset, value in (("All", all_values), ("Peak", peak_value)):
                    all_rows.append(
                        {
                            "Model": model_name,
                            "Group": group,
                            "Subset": subset,
                            "Performance index": metric,
                            "Value": value,
                        }
                    )
                summary[model_name][(group, metric)] = all_values

    df = pd.DataFrame(all_rows)
    pivot = df.pivot_table(
        index=["Model", "Group", "Subset"],
        columns="Performance index",
        values="Value",
    ).reset_index()
    return pivot, summary


def print_summary(summary):
    for model, metrics in summary.items():
        print(f"\n{model}")
        for group in ("Depth", "Velocity"):
            line = [f"{group} All"]
            for metric in ("RMSE", "R", "Recall (%)", "Precision (%)"):
                line.append(f"{metric}={metrics[(group, metric)]:.4f}")
            print(" | ".join(line))
