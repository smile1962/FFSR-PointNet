"""Build one comparison workbook from the four model result folders."""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", required=True)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output workbook path.",
    )
    return parser.parse_args()


def read_sheet(folder, sheet):
    path = Path(folder) / "quantitative_metrics_820.xlsx"
    return pd.read_excel(path, sheet_name=sheet)


def main():
    args = parse_args()
    root = Path(args.results_root)
    out = Path(args.output) if args.output else root / "all_model_comparison_820.xlsx"
    models = {
        "SR-PointNet": root / "SR-PointNet",
        "FFSR-PointNet-Light": root / "FFSR-PointNet-Light",
        "SR-Unet": root / "SR-Unet",
        "FLO-SR": root / "FLO-SR",
    }
    linear_folder = root / "Linear-Interpolation-Shouxi-820"
    if linear_folder.exists() and (
        linear_folder / "quantitative_metrics_820.xlsx"
    ).exists():
        models["Linear Interpolation (Shouxi)"] = linear_folder
    overall = []
    intervals = []
    audits = []
    for label, folder in models.items():
        ov = read_sheet(folder, "Overall")
        ov.insert(0, "Model", label)
        overall.append(ov)
        iv = read_sheet(folder, "IntervalSummary")
        iv.insert(0, "Model", label)
        intervals.append(iv)
        audit_path = folder / "random_sample_audit.xlsx"
        if audit_path.exists():
            audits.append(pd.read_excel(audit_path, sheet_name="Summary"))
    profile = pd.read_csv(root / "SR-PointNet" / "model_profile.csv")

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pd.concat(overall, ignore_index=True).to_excel(
            writer, sheet_name="Overall", index=False
        )
        pd.concat(intervals, ignore_index=True).to_excel(
            writer, sheet_name="IntervalSummary", index=False
        )
        profile.to_excel(writer, sheet_name="ModelProfile", index=False)
        if audits:
            pd.concat(audits, ignore_index=True).to_excel(
                writer, sheet_name="Audit", index=False
            )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
