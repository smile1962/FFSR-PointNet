"""Combine available Sanjiang result workbooks into one comparison table."""

from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = Path(rf"{ROOT}\Results\Sanjiang\available_model_comparison_sanjiang.xlsx")

SR_POINTNET = Path(rf"{ROOT}\Results\Sanjiang\SR-PointNet\quantitative_metrics_sanjiang.xlsx")
LIGHT = Path(rf"{ROOT}\Results\Sanjiang\FFSR-PointNet-Light\quantitative_metrics_sanjiang.xlsx")
FLO = Path(rf"{ROOT}\Results\Sanjiang\FLO-SR\quantitative_metrics_sanjiang.xlsx")
LINEAR = Path(rf"{ROOT}\Results\Sanjiang\Linear-Interpolation\quantitative_metrics_sanjiang.xlsx")
SRUNET = Path(rf"{ROOT}\Results\Sanjiang\SR-Unet\quantitative_metrics_sanjiang.xlsx")

METRICS = [
    "Recall (%)",
    "Precision (%)",
    "Accuracy (%)",
    "RMSE",
    "R",
    "PSNR",
    "SSIM",
]


def wide_rows(path, model, domain):
    df = pd.read_excel(path, sheet_name="Overall")
    out = []
    for _, row in df.iterrows():
        record = {
            "Model": model,
            "Domain": domain,
            "Variable": row["Variable"],
            "Subset": row["Subset"],
        }
        for metric in METRICS:
            record[metric] = row.get(metric, float("nan"))
        out.append(record)
    return out


def main():
    rows = []
    rows += wide_rows(SR_POINTNET, "SR-PointNet", "Point-cloud")
    rows += wide_rows(LIGHT, "FFSR-PointNet-Light", "Point-cloud")
    rows += wide_rows(FLO, "FLO-SR", "Grid 1.5m")
    rows += wide_rows(LINEAR, "Linear Interpolation", "Grid 1.5m")
    rows += wide_rows(SRUNET, "SR-Unet", "Grid 1.5m")
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Overall", index=False)
        meta = pd.DataFrame(
            [
                {
                    "File": "available_model_comparison_sanjiang.xlsx",
                    "Samples": 27,
                    "MetricDomainNote": (
                        "Point-cloud models use per-sample 130,618-point arrays; "
                        "grid models use the 490,746-cell 1.5m ROI mask."
                    ),
                }
            ]
        )
        meta.to_excel(writer, sheet_name="Metadata", index=False)
    print(df.to_string(index=False))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
