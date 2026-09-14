import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np

import metrics_roi
import roi_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["srunet", "flosr"],
        default="srunet",
    )
    parser.add_argument(
        "--result_dir",
        default=None,
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import os

    result_dir = args.result_dir or (
        rf"{ROOT}\Unet\roi\comparison\results_flosr_village"
        if args.model == "flosr"
        else rf"{ROOT}\Results\Unet\srunet_roi_sanjiang"
    )
    excel_dir = rf"{ROOT}\Results\Unet\excel"
    os.makedirs(excel_dir, exist_ok=True)
    output = args.output or os.path.join(
        excel_dir, f"roi_{args.model}_sanjiang_metrics.xlsx"
    )
    cfg = roi_data.get_config("village")
    roi = roi_data.load_roi_original(cfg)
    rows = []
    summary = {}
    for group, letter in (("Depth", "D"), ("Velocity", "U")):
        pred_files = metrics_roi.read_folder_tifs(
            f"{result_dir}/{letter}_pred"
        )
        true_files = metrics_roi.read_folder_tifs(
            f"{result_dir}/{letter}_true"
        )
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
            value = float(sum(p[metric] for p in per_sample) / len(per_sample))
            rows.append(
                {
                    "Case": "Sanjiang",
                    "Group": group,
                    "Subset": "All",
                    "Performance index": metric,
                    "Value": value,
                }
            )
        summary[group] = {
            metric: float(np.mean([p[metric] for p in per_sample]))
            for metric in metrics_roi.METRIC_NAMES
        }

    import pandas as pd

    pivot = (
        pd.DataFrame(rows)
        .pivot_table(
            index=["Group", "Subset"],
            columns="Performance index",
            values="Value",
        )
        .reset_index()
    )
    pivot.to_excel(output, index=False)
    for group, metrics in summary.items():
        print(
            f"{group} All | RMSE={metrics['RMSE']:.4f} | R={metrics['R']:.4f} | "
            f"Recall={metrics['Recall (%)']:.2f}% | "
            f"Precision={metrics['Precision (%)']:.2f}%"
        )
    print(f"Metrics written to: {output}")


if __name__ == "__main__":
    main()
