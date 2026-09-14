"""Random-sample audit for the saved quantitative tables.

The audit recomputes metrics from the stored GeoTIFF files with a separate,
minimal code path, then compares the values with the per-sample sheet in the
quantitative Excel produced by analyze_results.py.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from skimage.metrics import structural_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROI_PATH = ROOT / "Data" / "dataset" / "geo" / "shouxi_mask_30_Point.tif"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def read(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float64)


def recompute(pred_arr, true_arr, roi):
    pred = np.clip(pred_arr, 0, None)
    true = np.clip(true_arr, 0, None)
    pred[pred < 0.01] = 0.0
    true[true < 0.01] = 0.0
    p = pred[roi]
    t = true[roi]
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    r = 0.0
    if np.std(p) > 1e-12 and np.std(t) > 1e-12:
        r = float(np.corrcoef(p, t)[0, 1])
    elif np.std(p) < 1e-12 and np.std(t) < 1e-12:
        r = 1.0
    drange = float(t.max() - t.min())
    drange = drange if drange > 1e-8 else 1.0
    psnr = 10.0 * np.log10(drange**2 / (rmse**2 + 1e-12))
    pb = p > 0
    tb = t > 0
    tp = np.sum(pb & tb)
    fp = np.sum(pb & ~tb)
    fn = np.sum(~pb & tb)
    tn = np.sum(~pb & ~tb)
    rows_idx = np.where(roi.any(axis=1))[0]
    cols_idx = np.where(roi.any(axis=0))[0]
    if len(rows_idx) and len(cols_idx):
        r0, r1 = rows_idx[0], rows_idx[-1] + 1
        c0, c1 = cols_idx[0], cols_idx[-1] + 1
        rb = roi[r0:r1, c0:c1]
        pb2 = np.where(rb, pred[r0:r1, c0:c1], 0.0)
        tb2 = np.where(rb, true[r0:r1, c0:c1], 0.0)
        sr = float(tb2.max() - tb2.min())
        ssim = (
            1.0
            if sr < 1e-8
            else float(structural_similarity(tb2, pb2, data_range=sr, win_size=7))
        )
    else:
        ssim = 1.0
    return {
        "RMSE": rmse,
        "R": r,
        "PSNR": psnr,
        "SSIM": ssim,
        "Recall (%)": 100.0 * tp / (tp + fn) if (tp + fn) else 0.0,
        "Precision (%)": 100.0 * tp / (tp + fp) if (tp + fp) else 0.0,
        "Accuracy (%)": 100.0 * (tp + tn) / max(tp + tn + fp + fn, 1),
    }


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    roi = read(ROI_PATH) > 0
    excel_path = result_dir / "quantitative_metrics_820.xlsx"
    stored = pd.read_excel(excel_path, sheet_name="PerSample")
    files = sorted(
        set(stored.loc[stored["Variable"] == "Depth", "File"])
    )
    if len(files) < args.n:
        files = files * (args.n // len(files) + 1)
    rng = np.random.RandomState(args.seed)
    chosen = list(rng.choice(files, size=args.n, replace=False))

    rows = []
    max_diff = 0.0
    for fname in chosen:
        for variable, letter in (("Depth", "D"), ("Velocity", "U")):
            pred = read(result_dir / f"{letter}_pred" / fname)
            true = read(result_dir / f"{letter}_true" / fname)
            calc = recompute(pred, true, roi)
            saved_row = stored[
                (stored["File"] == fname) & (stored["Variable"] == variable)
            ].iloc[0]
            for metric, value in calc.items():
                saved = float(saved_row[metric])
                diff = abs(value - saved)
                max_diff = max(max_diff, diff)
                rows.append(
                    {
                        "File": fname,
                        "Variable": variable,
                        "Metric": metric,
                        "AuditRecomputed": value,
                        "TableValue": saved,
                        "AbsDiff": diff,
                    }
                )

    audit = pd.DataFrame(rows)
    audit_path = result_dir / "random_sample_audit.xlsx"
    with pd.ExcelWriter(audit_path, engine="openpyxl") as writer:
        audit.to_excel(writer, sheet_name="Audit", index=False)
        pd.DataFrame(
            [
                {
                    "Model": args.model_label,
                    "Samples": args.n,
                    "Seed": args.seed,
                    "MaxAbsDiff": max_diff,
                    "Passed": max_diff < 1e-8,
                }
            ]
        ).to_excel(writer, sheet_name="Summary", index=False)
    print(f"Audit {args.model_label}: max_abs_diff={max_diff:.3e}")
    print(f"Saved: {audit_path}")


if __name__ == "__main__":
    main()
