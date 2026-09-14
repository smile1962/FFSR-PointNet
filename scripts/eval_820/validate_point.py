"""820-event qualitative validation for point-cloud FFSR models.

Supports the full-parameter FFSR-PointNet v2 model and the FFSR-PointNet-Light
model. Outputs the same GeoTIFF map families and original density-scatter
figures as the legacy Shouxi validation scripts, but writes to an arbitrary
result root so that large outputs can be stored on another drive.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAIN = ROOT / "FFSR-PointNet" / "main"
LIGHT_MAIN = ROOT / "FFSR-PointNet-Light" / "main"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=["shouxi", "sanjiang"],
        default="shouxi",
        help="Point-cloud dataset profile to validate",
    )
    parser.add_argument(
        "--model",
        choices=["original", "full", "light"],
        required=True,
        help="original=SR-PointNet dense decoder; full=FFSR v2; light=FFSR-Light",
    )
    parser.add_argument("--weight", required=True)
    parser.add_argument("--save_root", required=True)
    parser.add_argument("--mean_std", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all samples")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--split_json", default=None)
    parser.add_argument("--write_metrics", action="store_true")
    return parser.parse_args()


def point_sample_metrics(pred, true):
    pred = np.clip(np.asarray(pred, dtype=np.float64), 0.0, None)
    true = np.clip(np.asarray(true, dtype=np.float64), 0.0, None)
    pred[pred < 0.01] = 0.0
    true[true < 0.01] = 0.0
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    if np.std(pred) < 1e-12 and np.std(true) < 1e-12:
        r = 1.0
    elif np.std(pred) < 1e-12 or np.std(true) < 1e-12:
        r = 0.0
    else:
        r = float(np.corrcoef(pred, true)[0, 1])
    drange = float(np.max(true) - np.min(true))
    if drange < 1e-8:
        drange = 1.0
    psnr = 10.0 * np.log10(drange ** 2 / (rmse ** 2 + 1e-12))
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
        "true_sum": float(true.sum()),
    }


def import_ffsr_modules(light=False):
    sys.path.insert(0, str(LIGHT_MAIN if light else MAIN))
    if light:
        sys.path.insert(0, str(MAIN))
    import config  # noqa: F401
    from utils.datasetGen import FlowFieldDataset
    from utils.plotFunctions import (
        create_tif_from_points,
        plot_correlation_scatter_Depth,
        plot_correlation_scatter_Vel,
        remove_zero_groundtruth,
    )
    return (
        FlowFieldDataset,
        create_tif_from_points,
        plot_correlation_scatter_Depth,
        plot_correlation_scatter_Vel,
        remove_zero_groundtruth,
    )


def import_light_model():
    sys.path.insert(0, str(LIGHT_MAIN))
    from model.FFSRP_subsampled_compressed import SubsampledCompressed
    return SubsampledCompressed


def import_full_model():
    sys.path.insert(0, str(MAIN))
    from model.FFSRP_v2 import FFSRPointNetV2
    return FFSRPointNetV2


def import_original_model():
    sys.path.insert(0, str(MAIN))
    from model.FFSRP import PointNetRegression
    return PointNetRegression


def make_model(model_name, dataset):
    if model_name == "original":
        cls = import_original_model()
        return cls(
            input_dim=dataset.input_dim,
            global_feat_dim=512,
            output_dim=2,
            N_high=dataset.N_high,
        )
    if model_name == "full":
        cls = import_full_model()
        return cls(
            input_dim=dataset.input_dim,
            global_feat_dim=512,
            output_dim=2,
            N_high=dataset.N_high,
            prior_dim=4,
        )
    if model_name == "light":
        cls = import_light_model()
        return cls(
            input_dim=dataset.input_dim,
            global_feat_dim=512,
            output_dim=2,
            N_high=dataset.N_high,
            rank=128,
            context_points=8192,
        )
    raise ValueError(model_name)


def raster_geometry(mask_path):
    with rasterio.open(mask_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs_wkt = src.crs.to_wkt()
    dx = abs(transform.a)
    dy = abs(transform.e)
    xmin = transform.c
    ymax = transform.f
    return height, width, dx, dy, xmin, ymax, crs_wkt


def main():
    args = parse_args()
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "D_pred",
        "D_true",
        "D_ori",
        "D_error",
        "U_pred",
        "U_true",
        "U_ori",
        "U_error",
        "D_coeff",
        "U_coeff",
    ):
        (save_root / name).mkdir(parents=True, exist_ok=True)

    (
        FlowFieldDataset,
        create_tif,
        plot_depth,
        plot_vel,
        remove_zero_groundtruth,
    ) = import_ffsr_modules(light=(args.model == "light"))

    data_root = ROOT / "Data" / "dataset"
    if args.case == "sanjiang":
        low_h5 = data_root / "SanJiang_LR.h5"
        high_h5 = data_root / "SanJiang_HR.h5"
        mask_path = data_root / "geo" / "Shancha_shuixi_1_5_mask_polyfilled.tif"
    else:
        low_h5 = data_root / "Point_LR_geo_val.h5"
        high_h5 = data_root / "Point_HR_Val.h5"
        mask_path = data_root / "geo" / "shouxi_mask_30_Point.tif"

    dataset = FlowFieldDataset(
        low_h5_path=str(low_h5),
        high_h5_path=str(high_h5),
        num_samples=None,
        return_coords=True,
        normalize=True,
        mean_std_file=args.mean_std,
        return_meta=False,
        return_index=False,
        return_hr_priors=True,
    )

    model = make_model(args.model, dataset)
    state = torch.load(args.weight, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"Missing state keys: {missing[:10]}")
    if unexpected:
        raise RuntimeError(f"Unexpected state keys: {unexpected[:10]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    rows, cols, dx, dy, xmin, ymax, crs_wkt = raster_geometry(mask_path)
    keys = dataset.high_keys
    end = len(keys) if args.limit <= 0 else min(args.start + args.limit, len(keys))
    if args.split_json and Path(args.split_json).exists():
        with open(args.split_json, "r", encoding="utf-8") as f:
            all_indices = json.load(f)
        sample_indices = all_indices[args.start:end]
    else:
        sample_indices = list(range(args.start, end))

    sanjiang_name_map = {}
    split_map_path = ROOT / "Data" / "dataset" / "sanjiang_grid_to_point_split.json"
    if args.case == "sanjiang" and split_map_path.exists():
        with open(split_map_path, "r", encoding="utf-8") as f:
            san_split = json.load(f)
        grid_test = [int(x) for x in san_split["grid_test"]]
        for gidx, pidx in san_split["grid_to_point"].items():
            gidx = int(gidx)
            pidx = int(pidx)
            if gidx in grid_test:
                sanjiang_name_map[pidx] = (
                    f"sample_orig{gidx}_test{grid_test.index(gidx)}.tif"
                )

    print(
        f"Case={args.case} Model={args.model} "
        f"samples={len(sample_indices)} total={len(keys)}"
    )
    timings = []
    metric_records = []
    for pos, i in enumerate(sample_indices):
        inputs, targets, hr_priors, coords_low, coords_high = dataset[i]
        inputs = inputs.unsqueeze(0).to(device)
        targets = targets.unsqueeze(0).to(device)
        coords_high = coords_high.unsqueeze(0).to(device)
        hr_priors = hr_priors.unsqueeze(0).to(device)

        with torch.no_grad():
            t0 = time.perf_counter()
            if args.model == "full":
                outputs = model(inputs, hr_priors)
            else:
                outputs = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - t0)

        pred = outputs.float().cpu().numpy()[0]
        true = targets.cpu().numpy()[0]
        ori = inputs.cpu().numpy()[0]
        ch = coords_high.cpu().numpy()[0]
        cl = coords_low.numpy()

        for c in range(2):
            pred[:, c] = pred[:, c] * dataset.std_out[c] + dataset.mean_out[c]
            true[:, c] = true[:, c] * dataset.std_out[c] + dataset.mean_out[c]
        ori[:, 0] = ori[:, 0] * dataset.std_depth + dataset.mean_depth
        ori[:, 1] = ori[:, 1] * dataset.std_velocity + dataset.mean_velocity

        Xp, Yp = ch[:, 0], ch[:, 1]
        Xo, Yo = cl[:, 0], cl[:, 1]
        Dp, Up = pred[:, 0], pred[:, 1]
        Dg, Ug = true[:, 0], true[:, 1]
        Do, Uo = ori[:, 0], ori[:, 1]

        sample_name = sanjiang_name_map.get(
            int(i), str(keys[i]).replace(" ", "_").replace(":", "_")
        )
        if sample_name.endswith(".tif"):
            sample_name = sample_name[:-4]
        if args.write_metrics:
            for variable, pv, tv in (
                ("Depth", Dp, Dg),
                ("Velocity", Up, Ug),
            ):
                metric_records.append(
                    {
                        "Variable": variable,
                        "File": sample_name + ".tif",
                        **point_sample_metrics(pv, tv),
                    }
                )

        D_error = np.abs(Dg - Dp)
        U_error = np.abs(Ug - Up)
        Dg[Dg < 0.01] = 0.0
        Dp[Dp < 0.01] = 0.0
        D_error[D_error < 0.015] = 0.0
        Ug[Ug < 0.01] = 0.0
        Up[Up < 0.01] = 0.0
        U_error[U_error < 0.02] = 0.0

        Xp, Yp, Dp, Up, Xg, Yg, Dg, Ug, _, removed_idx = remove_zero_groundtruth(
            Xp,
            Yp,
            Dp,
            Up,
            Xp,
            Yp,
            Dg,
            Ug,
            tol=0.0,
            mode="both",
            return_removed_index=True,
        )
        keep_error = np.ones(len(D_error), dtype=bool)
        keep_error[removed_idx] = False
        D_error = D_error[keep_error]
        U_error = U_error[keep_error]

        bounds = (
            rows,
            cols,
            dx,
            dy,
            xmin,
            ymax - dy * (rows - 1),
            xmin + dx * (cols - 1),
            ymax,
        )
        for sub, (xx, yy, val) in {
            "D_pred": (Xp, Yp, Dp),
            "D_true": (Xg, Yg, Dg),
            "D_error": (Xp, Yp, D_error),
            "U_pred": (Xp, Yp, Up),
            "U_true": (Xg, Yg, Ug),
            "U_error": (Xp, Yp, U_error),
        }.items():
            create_tif(
                *bounds,
                xx,
                yy,
                val,
                out_tif=str(save_root / sub / f"{sample_name}.tif"),
                crs_wkt=crs_wkt,
            )

        for sub, val in (("D_ori", Do), ("U_ori", Uo)):
            create_tif(
                *bounds,
                Xo,
                Yo,
                val,
                out_tif=str(save_root / sub / f"{sample_name}.tif"),
                crs_wkt=crs_wkt,
            )

        plot_depth(
            Dp,
            Dg,
            str(save_root / "D_coeff" / f"{sample_name}.tif"),
        )
        plot_vel(
            Up,
            Ug,
            str(save_root / "U_coeff" / f"{sample_name}.tif"),
        )
        print(f"[{pos + 1}/{len(sample_indices)}] {sample_name}")

    if timings:
        report = save_root / "inference_time_report.txt"
        report.write_text(
            f"samples={len(timings)},total_inference_s={sum(timings):.6f},"
            f"avg_s_per_sample={float(np.mean(timings)):.6f}\n",
            encoding="utf-8",
        )
    if args.write_metrics and metric_records:
        df = pd.DataFrame(metric_records)
        metric_cols = [
            "Recall (%)",
            "Precision (%)",
            "Accuracy (%)",
            "RMSE",
            "R",
            "PSNR",
        ]
        overall_rows = []
        for variable in ("Depth", "Velocity"):
            sub = df[df["Variable"] == variable]
            peak_idx = int(sub["true_sum"].idxmax())
            peak_file = str(sub.loc[peak_idx, "File"])
            all_row = {"Variable": variable, "Subset": "All"}
            peak_row = {"Variable": variable, "Subset": "Peak"}
            peak_sub = sub[sub["File"] == peak_file].iloc[0]
            for col in metric_cols:
                all_row[col] = float(sub[col].mean())
                peak_row[col] = float(peak_sub[col])
            all_row["PeakFile"] = peak_file
            overall_rows.append(all_row)
            overall_rows.append(peak_row)
        overall_df = pd.DataFrame(overall_rows)
        metric_path = save_root / f"quantitative_metrics_{args.case}.xlsx"
        with pd.ExcelWriter(metric_path, engine="openpyxl") as writer:
            overall_df.to_excel(writer, sheet_name="Overall", index=False)
            df.drop(columns=["true_sum"]).to_excel(
                writer, sheet_name="PerSample", index=False
            )
        print(f"Metrics written to {metric_path}")
    print(f"Saved maps to {save_root}")


if __name__ == "__main__":
    main()
