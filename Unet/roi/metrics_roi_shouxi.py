import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

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
        "--ffsr_dir",
        default=rf"{ROOT}\Results\FFSR_PointNet\Shouxi_820",
    )
    parser.add_argument(
        "--unet_dir",
        default=None,
    )
    parser.add_argument(
        "--peak_name",
        default="2019-08-20_04_00.tif",
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = roi_data.get_config("watershed")
    cnn_dir = (
        rf"{ROOT}\Results\Unet\flosr_roi_shouxi_820"
        if args.model == "flosr"
        else rf"{ROOT}\Results\Unet\srunet_roi_shouxi_820"
    )
    unet_dir = args.unet_dir or cnn_dir
    label = "FLO-SR" if args.model == "flosr" else "ROI-aware Patch SR-Unet"
    excel_dir = rf"{ROOT}\Results\Unet\excel"
    import os

    os.makedirs(excel_dir, exist_ok=True)
    output = args.output or os.path.join(
        excel_dir, f"roi_fair_comparison_{args.model}_820.xlsx"
    )
    pivot, summary = metrics_roi.build_comparison_table(
        cfg, args.ffsr_dir, unet_dir, args.peak_name, label
    )
    pivot.to_excel(output, index=False)
    metrics_roi.print_summary(summary)
    print(f"\nComparison table written to: {output}")


if __name__ == "__main__":
    main()
