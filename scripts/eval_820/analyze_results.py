"""ROI-based quantitative analysis for 820-event results.

The script reads GeoTIFF predictions and references produced by the model
validation scripts, computes per-sample overall metrics, aggregate All/Peak
metrics, and interval-level metrics using the same 0.5/1.0/2.0 m or m/s
thresholds used in the manuscript tables.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from skimage.metrics import structural_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PEAK_NAME = "2019-08-20_04_00.tif"
ROI_PATH = ROOT / "Data" / "dataset" / "geo" / "shouxi_mask_30_Point.tif"
METRIC_ORDER = [
    "Recall (%)",
    "Precision (%)",
    "Accuracy (%)",
    "RMSE",
    "R",
    "PSNR",
    "SSIM",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--roi_path", default=str(ROI_PATH))
    parser.add_argument("--peak_name", default=PEAK_NAME)
    parser.add_argument("--output_name", default="quantitative_metrics_820.xlsx")
    return parser.parse_args()


def natural_sort_key(name):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", name)]


def read_tif(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float64)


def threshold_array(arr, threshold=0.01):
    out = np.array(arr, dtype=np.float64, copy=True)
    out = np.clip(out, 0, None)
    out[out < threshold] = 0.0
    return out


def correlation(a, b):
    if np.std(a) < 1e-12 and np.std(b) < 1e-12:
        return 1.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def overall_sample(pred, true, roi):
    pred = threshold_array(pred)
    true = threshold_array(true)
    pr = pred[roi]
    tr = true[roi]
    rmse = float(np.sqrt(np.mean((pr - tr) ** 2)))
    r = correlation(pr, tr)
    data_range = float(np.max(tr) - np.min(tr))
    if data_range < 1e-8:
        data_range = 1.0
    psnr = 10.0 * np.log10(data_range**2 / (rmse**2 + 1e-12))

    rows = np.where(roi.any(axis=1))[0]
    cols = np.where(roi.any(axis=0))[0]
    if len(rows) and len(cols):
        r0, r1 = rows[0], rows[-1] + 1
        c0, c1 = cols[0], cols[-1] + 1
        roi_box = roi[r0:r1, c0:c1]
        pb = np.where(roi_box, pred[r0:r1, c0:c1], 0.0)
        tb = np.where(roi_box, true[r0:r1, c0:c1], 0.0)
        srange = float(np.max(tb) - np.min(tb))
        ssim = (
            1.0
            if srange < 1e-8
            else float(
                structural_similarity(
                    tb, pb, data_range=srange, win_size=7
                )
            )
        )
    else:
        ssim = 1.0

    pb = pr > 0
    tb = tr > 0
    tp = np.sum(pb & tb)
    fp = np.sum(pb & ~tb)
    fn = np.sum(~pb & tb)
    tn = np.sum(~pb & ~tb)
    return {
        "Recall (%)": 100.0 * tp / (tp + fn) if (tp + fn) else 0.0,
        "Precision (%)": 100.0 * tp / (tp + fp) if (tp + fp) else 0.0,
        "Accuracy (%)": 100.0 * (tp + tn) / max(tp + tn + fp + fn, 1),
        "RMSE": rmse,
        "R": r,
        "PSNR": psnr,
        "SSIM": ssim,
    }


def interval_sample(pred, true, roi, bounds=(0.5, 1.0, 2.0)):
    pred = threshold_array(pred)
    true = threshold_array(true)
    pr = pred[roi]
    tr = true[roi]
    labels = ["<0.5", "0.5-1", "1-2", ">2"]
    true_labels = np.digitize(tr, bins=bounds)
    pred_labels = np.digitize(pr, bins=bounds)
    rows = []
    for idx, label in enumerate(labels):
        true_mask = true_labels == idx
        cell_count = int(true_mask.sum())
        if cell_count == 0:
            rows.append(
                {
                    "Interval": label,
                    "Cell_Count": 0,
                    "Recall (%)": np.nan,
                    "Precision (%)": np.nan,
                    "Accuracy (%)": np.nan,
                    "RMSE": np.nan,
                    "R": np.nan,
                    "PSNR": np.nan,
                }
            )
            continue
        pv = pr[true_mask]
        tv = tr[true_mask]
        tp = np.sum((pred_labels == idx) & true_mask)
        fp = np.sum((pred_labels == idx) & ~true_mask)
        fn = np.sum((pred_labels != idx) & true_mask)
        tn = np.sum((pred_labels != idx) & ~true_mask)
        rmse = float(np.sqrt(np.mean((pv - tv) ** 2)))
        r = correlation(pv, tv) if len(pv) > 2 else 0.0
        drange = float(np.max(tv) - np.min(tv))
        if drange < 1e-8:
            drange = 1.0
        psnr = 10.0 * np.log10(drange**2 / (rmse**2 + 1e-12))
        rows.append(
            {
                "Interval": label,
                "Cell_Count": cell_count,
                "Recall (%)": 100.0 * tp / (tp + fn) if (tp + fn) else np.nan,
                "Precision (%)": 100.0 * tp / (tp + fp) if (tp + fp) else np.nan,
                "Accuracy (%)": 100.0 * (tp + tn) / max(tp + tn + fp + fn, 1),
                "RMSE": rmse,
                "R": r,
                "PSNR": psnr,
            }
        )
    return rows


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    roi = read_tif(args.roi_path) > 0
    peak_name = args.peak_name
    per_sample_all = []
    interval_all = []
    for variable, letter in (("Depth", "D"), ("Velocity", "U")):
        true_folder = result_dir / f"{letter}_true"
        true_files = sorted(
            true_folder.glob("*.tif"),
            key=lambda p: natural_sort_key(p.name),
        )
        if not true_files:
            raise FileNotFoundError(f"No {letter}_true files in {result_dir}")
        for true_path in true_files:
            name = true_path.name
            pred_path = result_dir / f"{letter}_pred" / name
            if not pred_path.exists():
                continue
            pred = read_tif(pred_path)
            true = read_tif(true_path)
            m = overall_sample(pred, true, roi)
            per_sample_all.append(
                {"Variable": variable, "File": name, **m}
            )
            for row in interval_sample(pred, true, roi):
                interval_all.append({"Variable": variable, "File": name, **row})

    df_samples = pd.DataFrame(per_sample_all)
    interval_df = pd.DataFrame(interval_all)

    overall_rows = []
    for variable in ("Depth", "Velocity"):
        sub = df_samples[df_samples["Variable"] == variable]
        all_row = {
            "Variable": variable,
            "Subset": "All",
        }
        peak_row = {"Variable": variable, "Subset": "Peak"}
        peak = sub[sub["File"] == peak_name]
        for metric in METRIC_ORDER:
            all_row[metric] = float(np.nanmean(sub[metric]))
            peak_row[metric] = (
                float(peak.iloc[0][metric]) if len(peak) else np.nan
            )
        overall_rows.append(all_row)
        overall_rows.append(peak_row)

    interval_metrics = [
        "Recall (%)",
        "Precision (%)",
        "Accuracy (%)",
        "RMSE",
        "R",
        "PSNR",
    ]
    counts = (
        interval_df.groupby(["Variable", "Interval"])
        .agg({"File": "nunique", "Cell_Count": "sum"})
        .reset_index()
        .rename(columns={"File": "Sample_Count"})
    )
    means = (
        interval_df.groupby(["Variable", "Interval"], as_index=False)[
            interval_metrics
        ]
        .mean()
    )
    summary = counts.merge(means, on=["Variable", "Interval"], how="left")

    overall_df = pd.DataFrame(overall_rows)
    output = result_dir / args.output_name
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        overall_df.to_excel(writer, sheet_name="Overall", index=False)
        df_samples.to_excel(writer, sheet_name="PerSample", index=False)
        interval_df.to_excel(writer, sheet_name="Intervals", index=False)
        summary.to_excel(writer, sheet_name="IntervalSummary", index=False)
        meta = pd.DataFrame(
            [
                {"Model": args.model_label, "PeakFile": peak_name},
                {
                    "RoiCount": int(roi.sum()),
                    "RoiPath": str(args.roi_path),
                    "ResultDir": str(result_dir),
                },
                {
                    "Definition": "All metrics computed inside the static ROI; "
                    "values below 0.01 are set to zero."
                },
            ]
        )
        meta.to_excel(writer, sheet_name="Metadata", index=False)

    print(f"Saved: {output}")
    print(overall_df.to_string(index=False))


if __name__ == "__main__":
    main()
