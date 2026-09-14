"""Compare original and revised model predictions on the 8.20 event.

The overall metrics follow the original calculate_indices convention: metrics
are computed on full 30 m TIF grids, and wet/dry classification uses the same
binarization threshold. An interval-level summary is also emitted so that the
report is close to the paper's depth/velocity interval table.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import os
import re

import numpy as np
import pandas as pd
import rasterio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def natural_sort_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def sample_files(folder):
    names = [f for f in os.listdir(folder) if f.endswith(".tif")]
    names.sort(key=natural_sort_key)
    return names


def read_raster(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float64)


def corr(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def binary_metrics(pred, true, threshold=0.0):
    pb = pred > threshold
    tb = true > threshold
    tp = float(np.logical_and(pb, tb).sum())
    fp = float(np.logical_and(pb, ~tb).sum())
    fn = float(np.logical_and(~pb, tb).sum())
    tn = float(np.logical_and(~pb, ~tb).sum())
    recall = tp / (tp + fn) * 100 if tp + fn else float("nan")
    precision = tp / (tp + fp) * 100 if tp + fp else float("nan")
    accuracy = (tp + tn) / (tp + fp + fn + tn) * 100 if tp + fp + fn + tn else float("nan")
    return recall, precision, accuracy


def sample_overall(pred, true):
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    r = corr(pred, true)
    data_range = float(true.max() - true.min())
    psnr = (
        peak_signal_noise_ratio(true, pred, data_range=data_range)
        if data_range > 0 else float("inf")
    )
    ssim = (
        structural_similarity(true, pred, data_range=data_range)
        if data_range > 0 else 1.0
    )
    recall, precision, accuracy = binary_metrics(pred, true, threshold=0.0)
    return {
        "RMSE": rmse,
        "R": r,
        "PSNR": psnr,
        "SSIM": ssim,
        "Recall (%)": recall,
        "Precision (%)": precision,
        "Accuracy (%)": accuracy,
    }


def interval_metrics(pred, true, bounds=(0.5, 1.0, 2.0)):
    labels = ["<0.5", "0.5-1", "1-2", ">2"]
    rows = {}
    limits = [(-np.inf, bounds[0]), bounds[:2], (bounds[1], bounds[2]), (bounds[2], np.inf)]
    for label, (lo, hi) in zip(labels, limits):
        mask_true = (true > lo) & (true <= hi) if np.isfinite(lo) and np.isfinite(hi) else (
            true <= hi if np.isneginf(lo) else true > lo
        )
        mask_pred = (pred > lo) & (pred <= hi) if np.isfinite(lo) and np.isfinite(hi) else (
            pred <= hi if np.isneginf(lo) else pred > lo
        )
        if not mask_true.any() and not mask_pred.any():
            continue
        pb = mask_pred
        tb = mask_true
        tp = float(np.logical_and(pb, tb).sum())
        fp = float(np.logical_and(pb, ~tb).sum())
        fn = float(np.logical_and(~pb, tb).sum())
        tn = float(np.logical_and(~pb, ~tb).sum())
        recall = tp / (tp + fn) * 100 if tp + fn else float("nan")
        precision = tp / (tp + fp) * 100 if tp + fp else float("nan")
        accuracy = (tp + tn) / (tp + fp + fn + tn) * 100 if tp + fp + fn + tn else float("nan")
        rows[label] = {
            "Recall (%)": recall,
            "Precision (%)": precision,
            "Accuracy (%)": accuracy,
            "RMSE": float(np.sqrt(np.mean((pred[mask_true] - true[mask_true]) ** 2))) if mask_true.any() else float("nan"),
            "R": corr(pred[mask_true], true[mask_true]) if mask_true.sum() > 2 else float("nan"),
        }
    return rows


def mean_rows(rows):
    keys = ["RMSE", "R", "PSNR", "SSIM", "Recall (%)", "Precision (%)", "Accuracy (%)"]
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_original", default=rf"{ROOT}\Results\FFSR_PointNet\Shouxi_820")
    parser.add_argument("--root_v2", default=r"E:\Project\Research\National2026\Validation_820_results")
    parser.add_argument("--peak_name", default="2019-08-20_04_00.tif")
    parser.add_argument("--output", default=rf"{ROOT}\Results\FFSR_PointNet\Shouxi_820\comparison_820.xlsx")
    args = parser.parse_args()

    records = []
    interval_records = []
    models = [
        ("original", args.root_original),
        ("v2", args.root_v2),
    ]
    for model_name, root in models:
        for var, prefix in [("Depth", "D"), ("Velocity", "U")]:
            true_files = sample_files(os.path.join(root, f"{prefix}_true"))
            all_samples = []
            for fname in true_files:
                true = read_raster(os.path.join(root, f"{prefix}_true", fname))
                pred = read_raster(os.path.join(root, f"{prefix}_pred", fname))
                metrics = sample_overall(pred, true)
                metrics["File"] = fname
                all_samples.append(metrics)
                for interval_label, vals in interval_metrics(pred, true).items():
                    interval_records.append(
                        {
                            "Model": model_name,
                            "Variable": var,
                            "Interval": interval_label,
                            **vals,
                        }
                    )
            all_avg = mean_rows(all_samples)
            peak = next(x for x in all_samples if x["File"] == args.peak_name)
            for subset, value in [("All", all_avg), ("Peak", peak)]:
                row = {"Model": model_name, "Variable": var, "Subset": subset}
                row.update({k: value[k] for k in ["RMSE", "R", "PSNR", "SSIM", "Recall (%)", "Precision (%)", "Accuracy (%)"]})
                records.append(row)

    overall = pd.DataFrame(records)
    intervals = pd.DataFrame(interval_records)
    interval_summary = (
        intervals.groupby(["Model", "Variable", "Interval"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        overall.to_excel(writer, sheet_name="Overall", index=False)
        intervals.to_excel(writer, sheet_name="Intervals", index=False)
        interval_summary.to_excel(writer, sheet_name="IntervalSummary", index=False)

    print(overall.to_string(index=False))
    print()
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
