#!/usr/bin/env python
"""Qualitative and quantitative validation for Sanjiang-source transfer models.

Terminal example:

  E:\\Anaconda\\envs\\pytorch\\python.exe validate_transfer_sanjiang.py

Default case is yuhua/yuhuo and output root is:

  Results\\Transfer\\yuhua_light

The script writes GeoTIFF maps (HR/LR depth and velocity, absolute errors),
original-format density scatter figures, per-sample and interval metrics, and
model parameter/GFLOPs/inference-time statistics.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.transform import from_origin
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = ROOT / "FFSR-PointNet" / "main"
LIGHT_MAIN = ROOT / "FFSR-PointNet-Light" / "main"
sys.path.insert(0, str(MAIN))

from utils.datasetGen import FlowFieldDataset  # noqa: E402
from utils.plotFunctions import (  # noqa: E402
    plot_correlation_scatter_Depth,
    plot_correlation_scatter_Vel,
)


DATA_ROOT = ROOT / "Data" / "dataset"
TRANSFER_ROOT = Path(rf"{ROOT}\Transfer")
RESULTS_ROOT = Path(rf"{ROOT}\Results\Transfer")

CASE_PROFILES = {
    "yuhua": {
        "display_name": "Yuhua (files named Yuhuo)",
        "val_low_res": str(DATA_ROOT / "Yuhuo_LR_val.h5"),
        "val_high_res": str(DATA_ROOT / "Yuhuo_HR_val.h5"),
        "mask": str(DATA_ROOT / "geo" / "yuhuo" / "ROI_mask_align.tif"),
        "weight": str(TRANSFER_ROOT / "outputs" / "yuhua_light" / "best_yuhua_light_transfer.pth"),
        "mean_std": str(TRANSFER_ROOT / "outputs" / "yuhua_light" / "mean_std_yuhua_36_1.npz"),
        "save_root": str(RESULTS_ROOT / "yuhua_light"),
    },
    "laoyangcun": {
        "display_name": "LaoYangCun",
        "val_low_res": str(DATA_ROOT / "LaoYangCun_LR_val.h5"),
        "val_high_res": str(DATA_ROOT / "LaoYangCun_HR_val.h5"),
        "mask": None,
        "weight": str(TRANSFER_ROOT / "outputs" / "laoyangcun_light" / "best_laoyangcun_light_transfer.pth"),
        "mean_std": str(TRANSFER_ROOT / "outputs" / "laoyangcun_light" / "mean_std_laoyangcun_36_1.npz"),
        "save_root": str(RESULTS_ROOT / "laoyangcun_light"),
    },
}

METRIC_ORDER = [
    "Recall (%)",
    "Precision (%)",
    "Accuracy (%)",
    "RMSE",
    "R",
    "PSNR",
    "SSIM",
]


def resolve_case(case):
    case = str(case).strip().lower()
    if case == "yuhuo":
        return "yuhua"
    if case not in CASE_PROFILES:
        raise ValueError(f"Unknown case '{case}'")
    return case


def load_light_model():
    sys.path.insert(0, str(LIGHT_MAIN))
    from model.FFSRP_subsampled_compressed import SubsampledCompressed
    return SubsampledCompressed


def load_mask(path):
    with rasterio.open(path) as src:
        mask = (src.read(1) > 0).astype(np.uint8)
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width
        profile = src.profile.copy()
    dx = abs(transform.a)
    dy = abs(transform.e)
    xmin = transform.c
    ymax = transform.f
    profile.update(
        height=height,
        width=width,
        count=1,
        dtype="float32",
        nodata=0.0,
        crs=crs,
        transform=transform,
    )
    return mask, height, width, dx, dy, xmin, ymax, profile


def geometry_from_points(x, y, resolution=1.5):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dx = float(resolution)
    dy = float(resolution)
    left = float(x.min()) - dx / 2.0
    right = float(x.max()) + dx / 2.0
    top = float(y.max()) + dy / 2.0
    bottom = float(y.min()) - dy / 2.0
    width = int(np.ceil((right - left) / dx))
    height = int(np.ceil((top - bottom) / dy))
    transform = from_origin(left, top, dx, dy)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "nodata": 0.0,
        "crs": "EPSG:3857",
        "transform": transform,
    }
    return None, height, width, dx, dy, left, top, profile


def rasterize_points(x, y, values, height, width, dx, dy, xmin, ymax):
    grid = np.zeros((height, width), dtype=np.float32)
    # Point coordinates are cell centres. The half-cell offset is required
    # to map them onto the correct raster cell instead of the lower-left corner.
    col = np.floor((np.asarray(x) - xmin) / dx + 0.5).astype(np.int64)
    row = np.floor((ymax - np.asarray(y)) / dy + 0.5).astype(np.int64)
    valid = (col >= 0) & (col < width) & (row >= 0) & (row < height)
    if valid.any():
        grid[row[valid], col[valid]] = np.asarray(values)[valid]
    return grid


def save_geotif(values, path, profile):
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(values.astype(np.float32), 1)


def nearest_map(src_coords, dst_coords, src_values):
    tree = cKDTree(src_coords)
    _, idx = tree.query(dst_coords, k=1)
    return np.asarray(src_values)[idx]


def threshold_array(arr, threshold=0.01):
    out = np.clip(np.asarray(arr, dtype=np.float64), 0.0, None)
    out[out < threshold] = 0.0
    return out


def correlation(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) < 1e-12 and np.std(b) < 1e-12:
        return 1.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def sample_metrics(pred, true):
    pred = threshold_array(pred)
    true = threshold_array(true)
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    r = correlation(pred, true)
    data_range = float(np.max(true) - np.min(true))
    if data_range < 1e-8:
        data_range = 1.0
    psnr = 10.0 * np.log10(data_range**2 / (rmse**2 + 1e-12))
    pb = pred > 0
    tb = true > 0
    tp = int(np.sum(pb & tb))
    fp = int(np.sum(pb & ~tb))
    fn = int(np.sum(~pb & tb))
    tn = int(np.sum(~pb & ~tb))
    return {
        "Recall (%)": 100.0 * tp / (tp + fn) if (tp + fn) else 0.0,
        "Precision (%)": 100.0 * tp / (tp + fp) if (tp + fp) else 0.0,
        "Accuracy (%)": 100.0 * (tp + tn) / max(tp + tn + fp + fn, 1),
        "RMSE": rmse,
        "R": r,
        "PSNR": psnr,
    }


def ssim_from_grids(pred_grid, true_grid, mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return 1.0
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    mask_box = mask[r0:r1, c0:c1]
    pb = np.where(mask_box, pred_grid[r0:r1, c0:c1], 0.0)
    tb = np.where(mask_box, true_grid[r0:r1, c0:c1], 0.0)
    drange = float(np.max(tb) - np.min(tb))
    if drange < 1e-8:
        return 1.0
    return float(
        structural_similarity(tb, pb, data_range=drange, win_size=7)
    )


def interval_metrics(pred, true, bounds=(0.5, 1.0, 2.0)):
    pred = threshold_array(pred)
    true = threshold_array(true)
    labels = ["<0.5", "0.5-1", "1-2", ">2"]
    true_labels = np.digitize(true, bins=bounds)
    pred_labels = np.digitize(pred, bins=bounds)
    rows = []
    for idx, label in enumerate(labels):
        mask = true_labels == idx
        cell_count = int(mask.sum())
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
        pv = pred[mask]
        tv = true[mask]
        tp = int(np.sum((pred_labels == idx) & mask))
        fp = int(np.sum((pred_labels == idx) & ~mask))
        fn = int(np.sum((pred_labels != idx) & mask))
        tn = int(np.sum((pred_labels != idx) & ~mask))
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


def count_gflops(model, input_tensor):
    macs = 0
    original_bmm = torch.bmm

    def counting_bmm(a, b):
        nonlocal macs
        if a.dim() == 3 and b.dim() == 3:
            batch = a.size(0)
            k = a.size(1)
            n = b.size(2)
            macs += batch * k * k * n
        return original_bmm(a, b)

    def hook(module, inp, out):
        nonlocal macs
        if isinstance(module, torch.nn.Conv1d):
            x = inp[0]
            macs += (
                x.size(0)
                * module.out_channels
                * module.in_channels
                * module.kernel_size[0]
                * x.size(2)
            )
        elif isinstance(module, torch.nn.Linear):
            x = inp[0]
            macs += (
                x.numel()
                // x.size(-1)
                * module.out_features
                * module.in_features
            )

    handles = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    torch.bmm = counting_bmm
    try:
        with torch.no_grad():
            model(input_tensor)
    finally:
        torch.bmm = original_bmm
        for handle in handles:
            handle.remove()
    return macs * 2.0 / 1e9


def write_quantitative_xlsx(save_root, model_label, per_sample, intervals, profile_stats, peak_name):
    samples_df = pd.DataFrame(per_sample)
    interval_df = pd.DataFrame(intervals)
    overall_rows = []
    for variable in ("Depth", "Velocity"):
        sub = samples_df[samples_df["Variable"] == variable]
        all_row = {"Variable": variable, "Subset": "All"}
        peak_row = {"Variable": variable, "Subset": "Peak"}
        peak = sub[sub["File"] == peak_name]
        for metric in METRIC_ORDER:
            all_row[metric] = float(np.nanmean(sub[metric]))
            peak_row[metric] = (
                float(peak.iloc[0][metric]) if len(peak) else np.nan
            )
        overall_rows.append(all_row)
        overall_rows.append(peak_row)
    overall_df = pd.DataFrame(overall_rows)

    interval_metrics_list = METRIC_ORDER[:6]
    counts = (
        interval_df.groupby(["Variable", "Interval"])
        .agg({"File": "nunique", "Cell_Count": "sum"})
        .reset_index()
        .rename(columns={"File": "Sample_Count"})
    )
    means = (
        interval_df.groupby(["Variable", "Interval"], as_index=False)[
            interval_metrics_list
        ]
        .mean()
    )
    interval_summary = counts.merge(means, on=["Variable", "Interval"], how="left")
    profile_df = pd.DataFrame(
        [
            {
                "Model": model_label,
                "Params (M)": profile_stats["params_m"],
                "GFLOPs (MACx2)": profile_stats["gflops"],
                "N_high": profile_stats["n_high"],
                "rank": profile_stats["rank"],
                "context_points": profile_stats["context_points"],
                "avg_inference_s": profile_stats["avg_inference_s"],
            }
        ]
    )

    out_path = Path(save_root) / "quantitative_metrics_transfer.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        overall_df.to_excel(writer, sheet_name="Overall", index=False)
        samples_df.to_excel(writer, sheet_name="PerSample", index=False)
        interval_df.to_excel(writer, sheet_name="Intervals", index=False)
        interval_summary.to_excel(writer, sheet_name="IntervalSummary", index=False)
        profile_df.to_excel(writer, sheet_name="ModelProfile", index=False)
        pd.DataFrame(
            [
                {
                    "Model": model_label,
                    "PeakFile": peak_name,
                    "ResultDir": str(save_root),
                    "MetricDefinition": (
                        "Classification uses >0 threshold after clipping values "
                        "below 0.01; intervals are <0.5, 0.5-1, 1-2, >2."
                    ),
                }
            ]
        ).to_excel(writer, sheet_name="Metadata", index=False)
    print(f"Quantitative workbook saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="yuhua")
    parser.add_argument("--weight", default=None)
    parser.add_argument("--mean-std", default=None)
    parser.add_argument("--low-res", default=None)
    parser.add_argument("--high-res", default=None)
    parser.add_argument("--save-root", default=None)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--context-points", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--model-label", default=None)
    args = parser.parse_args()

    case = resolve_case(args.case)
    profile = CASE_PROFILES[case]
    low_res = args.low_res or profile["val_low_res"]
    high_res = args.high_res or profile["val_high_res"]
    mask_path = args.mask or profile.get("mask")
    weight = args.weight or profile["weight"]
    mean_std = args.mean_std or profile["mean_std"]
    save_root = Path(args.save_root or profile["save_root"])
    model_label = args.model_label or profile["display_name"]

    if not os.path.exists(mean_std):
        legacy_mean_std = (
            TRANSFER_ROOT
            / "outputs"
            / f"{case}_light"
            / f"mean_std_{case}_{profile.get('default_samples', 36)}.npz"
        )
        if legacy_mean_std.exists():
            mean_std = str(legacy_mean_std)
            print(f"Using legacy normalization file: {mean_std}")

    save_root.mkdir(parents=True, exist_ok=True)
    for folder in (
        "D_true",
        "D_pred",
        "D_ori",
        "D_error",
        "U_true",
        "U_pred",
        "U_ori",
        "U_error",
        "D_coeff",
        "U_coeff",
    ):
        (save_root / folder).mkdir(exist_ok=True)

    dataset = FlowFieldDataset(
        low_h5_path=low_res,
        high_h5_path=high_res,
        num_samples=None,
        return_coords=True,
        normalize=True,
        mean_std_file=mean_std,
    )
    if mask_path and os.path.exists(mask_path):
        mask, height, width, dx, dy, xmin, ymax, profile_tif = load_mask(mask_path)
    else:
        sample_coords = dataset[0][3].numpy()
        mask, height, width, dx, dy, xmin, ymax, profile_tif = geometry_from_points(
            sample_coords[:, 0], sample_coords[:, 1]
        )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    keys = dataset.high_keys
    end = len(keys) if args.limit <= 0 else min(args.start + args.limit, len(keys))

    ModelClass = load_light_model()
    model = ModelClass(
        input_dim=dataset.input_dim,
        global_feat_dim=512,
        output_dim=dataset.output_dim,
        N_high=dataset.N_high,
        rank=args.rank,
        context_points=args.context_points,
    )
    state = torch.load(weight, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"State mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    total_params = sum(p.numel() for p in model.parameters())
    fake_input = torch.zeros(1, dataset.N_high, dataset.input_dim)
    gflops = count_gflops(model.cpu(), fake_input)
    model.to(device).eval()

    print(
        f"Case={case}, samples={end - args.start}, total_keys={len(keys)}, "
        f"N_high={dataset.N_high}, params={total_params / 1e6:.2f}M, "
        f"GFLOPs={gflops:.2f}"
    )

    per_sample = []
    interval_rows = []
    depth_volumes = []
    inference_times = []

    for idx in range(args.start, end):
        sample_name = str(keys[idx]).replace(" ", "_").replace(":", "_")
        inputs, targets, coords_low, coords_high = dataset[idx]
        inputs = inputs.unsqueeze(0).to(device)
        targets = targets.unsqueeze(0).to(device)
        coords_high = coords_high.unsqueeze(0).to(device)

        with torch.no_grad():
            t0 = time.perf_counter()
            outputs = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - t0)

        pred = outputs.float().cpu().numpy()[0]
        true = targets.cpu().numpy()[0]
        low_feat = inputs.cpu().numpy()[0]
        ch = coords_high.cpu().numpy()[0]
        cl = coords_low.numpy()

        pred[:, 0] = pred[:, 0] * dataset.std_out[0] + dataset.mean_out[0]
        pred[:, 1] = pred[:, 1] * dataset.std_out[1] + dataset.mean_out[1]
        true[:, 0] = true[:, 0] * dataset.std_out[0] + dataset.mean_out[0]
        true[:, 1] = true[:, 1] * dataset.std_out[1] + dataset.mean_out[1]
        low_d = (
            low_feat[:, 0] * dataset.std_depth + dataset.mean_depth
        )
        low_u = (
            low_feat[:, 1] * dataset.std_velocity + dataset.mean_velocity
        )

        if len(cl) != len(ch) or not np.allclose(cl, ch, atol=1e-4):
            low_d = nearest_map(cl[:, :2], ch[:, :2], low_d)
            low_u = nearest_map(cl[:, :2], ch[:, :2], low_u)
            cl = ch

        Xp = ch[:, 0]
        Yp = ch[:, 1]
        Dp = np.clip(pred[:, 0], 0.0, None)
        Up = np.clip(pred[:, 1], 0.0, None)
        Dg = np.clip(true[:, 0], 0.0, None)
        Ug = np.clip(true[:, 1], 0.0, None)
        Do = np.clip(low_d, 0.0, None)
        Uo = np.clip(low_u, 0.0, None)
        D_err = np.abs(Dg - Dp)
        U_err = np.abs(Ug - Up)

        maps = {
            "D_true": (Xp, Yp, Dg),
            "D_pred": (Xp, Yp, Dp),
            "D_ori": (Xp, Yp, Do),
            "D_error": (Xp, Yp, D_err),
            "U_true": (Xp, Yp, Ug),
            "U_pred": (Xp, Yp, Up),
            "U_ori": (Xp, Yp, Uo),
            "U_error": (Xp, Yp, U_err),
        }
        grids = {}
        for folder, (xx, yy, val) in maps.items():
            grid = rasterize_points(
                xx, yy, val, height, width, dx, dy, xmin, ymax
            )
            save_geotif(
                grid, save_root / folder / f"{sample_name}.tif", profile_tif
            )
            grids[folder] = grid

        for metric in (
            ("D", "Depth", Dp, Dg),
            ("U", "Velocity", Up, Ug),
        ):
            prefix, variable, pv, tv = metric
            row = sample_metrics(pv, tv)
            ssim_mask = (
                mask > 0
                if mask is not None
                else (
                    (grids[f"{prefix}_true"] > 0)
                    | (grids[f"{prefix}_pred"] > 0)
                )
            )
            row["SSIM"] = ssim_from_grids(
                grids[f"{prefix}_pred"],
                grids[f"{prefix}_true"],
                ssim_mask,
            )
            per_sample.append({"Variable": variable, "File": sample_name, **row})
            for iv_row in interval_metrics(pv, tv):
                interval_rows.append(
                    {"Variable": variable, "File": sample_name, **iv_row}
                )

        depth_volumes.append(float(np.sum(Dg)))
        print(f"[{idx - args.start + 1}/{end - args.start}] {sample_name}")

        wet_d = (Dp > 0.01) | (Dg > 0.01)
        wet_u = (Up > 0.01) | (Ug > 0.01)
        if wet_d.sum() > 50:
            plot_correlation_scatter_Depth(
                Dp[wet_d],
                Dg[wet_d],
                str(save_root / "D_coeff" / f"{sample_name}.png"),
            )
        if wet_u.sum() > 50:
            plot_correlation_scatter_Vel(
                Up[wet_u],
                Ug[wet_u],
                str(save_root / "U_coeff" / f"{sample_name}.png"),
            )

    peak_name = str(keys[int(np.argmax(depth_volumes))]).replace(" ", "_").replace(":", "_")
    profile_stats = {
        "params_m": total_params / 1e6,
        "gflops": gflops,
        "n_high": dataset.N_high,
        "rank": args.rank,
        "context_points": args.context_points,
        "avg_inference_s": float(np.mean(inference_times)) if inference_times else 0.0,
    }
    write_quantitative_xlsx(
        save_root,
        model_label,
        per_sample,
        interval_rows,
        profile_stats,
        peak_name,
    )

    (save_root / "inference_time_report.txt").write_text(
        f"samples={len(inference_times)},"
        f"total_inference_s={sum(inference_times):.6f},"
        f"avg_s_per_sample={profile_stats['avg_inference_s']:.6f}\n",
        encoding="utf-8",
    )
    (save_root / "model_profile.json").write_text(
        json.dumps(
            {
                "model_label": model_label,
                "params_m": profile_stats["params_m"],
                "gflops": profile_stats["gflops"],
                "n_high": profile_stats["n_high"],
                "rank": profile_stats["rank"],
                "context_points": profile_stats["context_points"],
                "avg_inference_s": profile_stats["avg_inference_s"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done. Results in {save_root}")


if __name__ == "__main__":
    main()
