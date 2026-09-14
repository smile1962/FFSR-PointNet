import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os

import numpy as np
import pandas as pd

import metrics_roi
import roi_data


PEAK_NAME = "2019-08-20_04_00.tif"
MODEL_DIRS = {
    "FFSR-PointNet": rf"{ROOT}\Results\FFSR_PointNet\Shouxi_820",
    "SR-Unet": rf"{ROOT}\Results\Unet\srunet_roi_shouxi_820",
    "FLO-SR": (
        rf"{ROOT}\Results\Unet\flosr_roi_shouxi_820"
    ),
}
OUTPUT = (
    rf"{ROOT}\Results\Unet\excel"
    r"\roi_three_model_comparison_820.xlsx"
)


def model_rows(cfg, model_name, base_dir):
    base = base_dir
    rows = []
    summary = {}
    for group, letter in (("Depth", "D"), ("Velocity", "U")):
        pred_files = metrics_roi.read_folder_tifs(f"{base}/{letter}_pred")
        true_files = metrics_roi.read_folder_tifs(f"{base}/{letter}_true")
        true_map = {p.stem: p for p in true_files}
        per_sample = []
        for pred_path in pred_files:
            true_path = true_map.get(pred_path.stem)
            if true_path is None:
                continue
            m = metrics_roi.sample_metrics(
                metrics_roi.read_array(pred_path),
                metrics_roi.read_array(true_path),
                roi,
            )
            m["File"] = pred_path.name
            per_sample.append(m)
        for metric in metrics_roi.METRIC_NAMES:
            all_value = float(np.mean([p[metric] for p in per_sample]))
            peak_candidates = [p for p in per_sample if p["File"] == PEAK_NAME]
            peak_value = (
                float(peak_candidates[0][metric])
                if peak_candidates
                else float("nan")
            )
            for subset, value in (("All", all_value), ("Peak", peak_value)):
                rows.append(
                    {
                        "Model": model_name,
                        "Group": group,
                        "Subset": subset,
                        "Performance index": metric,
                        "Value": value,
                    }
                )
        summary[group] = {
            metric: float(np.mean([p[metric] for p in per_sample]))
            for metric in metrics_roi.METRIC_NAMES
        }
    return rows, summary


def main():
    cfg = roi_data.get_config("watershed")
    global roi
    roi = roi_data.load_roi_original(cfg)

    all_rows = []
    for model_name, base_dir in MODEL_DIRS.items():
        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f"Missing result folder: {base_dir}")
        rows, summary = model_rows(cfg, model_name, base_dir)
        all_rows.extend(rows)
        print(f"\n{model_name}")
        for group in ("Depth", "Velocity"):
            s = summary[group]
            print(
                f"{group} All | RMSE={s['RMSE']:.4f} | R={s['R']:.4f} | "
                f"Recall={s['Recall (%)']:.2f}% | "
                f"Precision={s['Precision (%)']:.2f}%"
            )

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    pivot = (
        pd.DataFrame(all_rows)
        .pivot_table(
            index=["Model", "Group", "Subset"],
            columns="Performance index",
            values="Value",
        )
        .reset_index()
    )
    pivot.to_excel(OUTPUT, index=False)
    print(f"\nThree-model table written to: {OUTPUT}")


if __name__ == "__main__":
    main()

