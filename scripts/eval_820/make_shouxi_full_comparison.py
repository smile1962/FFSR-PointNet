"""Build the complete Shouxi 8.20 comparison workbook.

Includes all/peak overall metrics, time-averaged interval metrics, peak-time
interval metrics, model profile/training/inference statistics and the 10-sample
TIF audit summary.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT / "scripts" / "eval_820"))

import analyze_results as ar  # noqa: E402

RESULTS = Path(rf"{ROOT}\Results")
ROI_PATH = ROOT / "Data" / "dataset" / "geo" / "shouxi_mask_30_Point.tif"
PEAK = "2019-08-20_04_00.tif"

MODELS = [
    "SR-PointNet",
    "FFSR-PointNet-Light",
    "SR-Unet",
    "FLO-SR",
    "Linear-Interpolation-Shouxi-820",
]

PROFILE = {
    "SR-PointNet": {
        "Params (M)": 265.6856,
        "GFLOPs (MACx2)": 95.9721,
        "Full-domain GFLOPs (MACx2)": 95.9721,
        "Wet patch count mean": "N/A",
        "Patch size": "N/A",
        "Patch count": "N/A",
        "Per-patch GFLOPs (MACx2)": "N/A",
        "GFLOPs basis": "1x126460x4 point cloud",
        "Training time (s)": 19534.0,
        "Training time source": "paper record",
        "Inference avg (s/sample)": 0.089428,
        "Inference 72 samples (s)": 6.438788,
    },
    "FFSR-PointNet-Light": {
        "Params (M)": 39.2003,
        "GFLOPs (MACx2)": 27.6390,
        "Full-domain GFLOPs (MACx2)": 27.6390,
        "Wet patch count mean": "N/A",
        "Patch size": "N/A",
        "Patch count": "N/A",
        "Per-patch GFLOPs (MACx2)": "N/A",
        "GFLOPs basis": "1x126460x4 point cloud",
        "Training time (s)": 3658.75,
        "Training time source": "training_time_shouxi.txt",
        "Inference avg (s/sample)": 0.058494,
        "Inference 72 samples (s)": 4.211592,
    },
    "SR-Unet": {
        "Params (M)": 8.0507,
        "GFLOPs (MACx2)": 1435.23,
        "Full-domain GFLOPs (MACx2)": 2453.48,
        "Wet patch count mean": 87.7463,
        "Patch size": "120x120",
        "Patch count": "87.75 wet / 150 full",
        "Per-patch GFLOPs (MACx2)": 16.3565568,
        "GFLOPs basis": "120x120 wet patches, mean 87.7463 per Shouxi train/val sample",
        "Training time (s)": 19402.50,
        "Training time source": "training_time_shouxi.txt",
        "Inference avg (s/sample)": 0.123084,
        "Inference 72 samples (s)": 8.862064,
    },
    "FLO-SR": {
        "Params (M)": 1.1852,
        "GFLOPs (MACx2)": 2989.82,
        "Full-domain GFLOPs (MACx2)": 5111.01,
        "Wet patch count mean": 87.7463,
        "Patch size": "120x120",
        "Patch count": "87.75 wet / 150 full",
        "Per-patch GFLOPs (MACx2)": 34.0733952,
        "GFLOPs basis": "120x120 wet patches, mean 87.7463 per Shouxi train/val sample",
        "Training time (s)": 32032.50,
        "Training time source": "training_time_shouxi.txt",
        "Inference avg (s/sample)": 0.213583,
        "Inference 72 samples (s)": 15.377954,
    },
    "Linear-Interpolation-Shouxi-820": {
        "Params (M)": None,
        "GFLOPs (MACx2)": None,
        "Full-domain GFLOPs (MACx2)": None,
        "Wet patch count mean": "N/A",
        "Patch size": "N/A",
        "Patch count": "N/A",
        "Per-patch GFLOPs (MACx2)": "N/A",
        "GFLOPs basis": "not applicable",
        "Training time (s)": None,
        "Training time source": "not applicable",
        "Inference avg (s/sample)": None,
        "Inference 72 samples (s)": None,
    },
}

LABELS = {
    "SR-PointNet": "SR-PointNet",
    "FFSR-PointNet-Light": "FFSR-PointNet",
    "SR-Unet": "SR-Unet",
    "FLO-SR": "FLO-SR",
    "Linear-Interpolation-Shouxi-820": "Linear Interpolation (Shouxi)",
}


def read_band(path):
    with rasterio.open(str(path)) as src:
        return src.read(1).astype(np.float64)


def peak_interval_rows(result_dir):
    result_dir = Path(result_dir)
    roi = read_band(ROI_PATH) > 0
    out = []
    for variable, letter in (("Depth", "D"), ("Velocity", "U")):
        pred = read_band(result_dir / f"{letter}_pred" / PEAK)
        true = read_band(result_dir / f"{letter}_true" / PEAK)
        for row in ar.interval_sample(pred, true, roi):
            out.append({"Variable": variable, "File": PEAK, **row})
    return out


def read_overall(result_dir):
    path = Path(result_dir) / "quantitative_metrics_820.xlsx"
    df = pd.read_excel(path, sheet_name="Overall")
    return df


def read_interval_summary(result_dir):
    path = Path(result_dir) / "quantitative_metrics_820.xlsx"
    df = pd.read_excel(path, sheet_name="IntervalSummary")
    return df


def audit_summary(result_dir):
    path = Path(result_dir) / "random_sample_audit.xlsx"
    if not path.exists():
        return None
    df = pd.read_excel(path, sheet_name="Summary")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(RESULTS / "shouxi_full_comparison_820.xlsx"),
    )
    args = parser.parse_args()
    overall_list = []
    all_interval_list = []
    peak_interval_list = []
    audit_list = []
    profile_rows = []
    for folder in MODELS:
        result_dir = RESULTS / folder
        label = LABELS[folder]
        ov = read_overall(result_dir).copy()
        ov.insert(0, "Model", label)
        overall_list.append(ov)

        iv = read_interval_summary(result_dir).copy()
        iv.insert(0, "Model", label)
        all_interval_list.append(iv)

        peak = pd.DataFrame(peak_interval_rows(result_dir))
        peak.insert(0, "Model", label)
        peak_interval_list.append(peak)

        aud = audit_summary(result_dir)
        if aud is not None:
            aud = aud.copy()
            if "Model" not in aud.columns:
                aud.insert(0, "Model", label)
            audit_list.append(aud)

        info = PROFILE[folder]
        profile_rows.append({"Model": label, **info})

    overall_df = pd.concat(overall_list, ignore_index=True)
    all_interval_df = pd.concat(all_interval_list, ignore_index=True)
    peak_interval_df = pd.concat(peak_interval_list, ignore_index=True)
    profile_df = pd.DataFrame(profile_rows)
    audit_df = pd.concat(audit_list, ignore_index=True) if audit_list else pd.DataFrame()

    out = Path(args.output)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        overall_df.to_excel(writer, sheet_name="AllOverall", index=False)
        all_interval_df.to_excel(
            writer, sheet_name="AllInterval_72samples", index=False
        )
        peak_interval_df.to_excel(
            writer, sheet_name="PeakInterval_0400", index=False
        )
        profile_df.to_excel(writer, sheet_name="ModelProfile", index=False)
        audit_df.to_excel(writer, sheet_name="Audit10Samples", index=False)
    print(f"Saved: {out}")
    print("Overall:")
    print(overall_df.to_string(index=False))
    print("Peak intervals:")
    print(peak_interval_df.to_string(index=False))


if __name__ == "__main__":
    main()
