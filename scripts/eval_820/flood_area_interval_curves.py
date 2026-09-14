"""Flooded-area time-series statistics by depth/velocity intervals.

For each GeoTIFF time slice under a point-cloud model result directory, cells
are assigned to:

- 0.1-0.5  (0.1 <= v < 0.5)
- 0.5-1.0  (0.5 <= v < 1.0)
- >1.0     (v >= 1.0)

Folder mapping used here:

- HR numerical simulation : D_true / U_true
- FFSR model prediction   : D_pred / U_pred
- interpolation baseline  : D_ori / U_ori (LR field stored on the HR grid)

The result is exported to Excel next to the input folder with both flooded
area in km2 and raw grid-cell counts for the depth and velocity variables.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

INTERVALS = [
    ("0.1-0.5", lambda v: (v >= 0.2) & (v < 0.5)),
    ("0.5-1", lambda v: (v >= 0.5) & (v < 1)),
    (">1", lambda v: v >= 1),
]

SOURCES = [
    ("HR", "D_true", "U_true"),
    ("FFSR", "D_pred", "U_pred"),
    ("Interp", "D_ori", "U_ori"),
]


def read_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float64)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--result_dir",
        type=Path,
        default=Path(rf"{ROOT}\Results\0\FFSR-PointNet-Light"),
        help="Directory containing D_true/D_pred/D_ori and U_* subfolders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output xlsx path (default: result_dir/flood_area_interval_curves.xlsx).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.result_dir.resolve()
    output = (args.output or root / "flood_area_interval_curves.xlsx").resolve()

    true_dir = root / "D_true"
    true_files = sorted(true_dir.glob("*.tif"))
    if not true_files:
        raise FileNotFoundError(f"No D_true files under {true_dir}")

    for _, d_sub, u_sub in SOURCES:
        d_files = sorted((root / d_sub).glob("*.tif"))
        u_files = sorted((root / u_sub).glob("*.tif"))
        if len(d_files) != len(true_files) or len(u_files) != len(true_files):
            raise FileNotFoundError(
                f"File count mismatch in {d_sub}/{u_sub}: "
                f"{len(d_files)}/{len(u_files)} vs {len(true_files)}"
            )
        for d_path, d_ref in zip(d_files, true_files):
            if d_path.name != d_ref.name:
                raise ValueError(f"Filename mismatch: {d_path.name} vs {d_ref.name}")
        for u_path, u_ref in zip(u_files, true_files):
            if u_path.name != u_ref.name:
                raise ValueError(f"Filename mismatch: {u_path.name} vs {u_ref.name}")

    with rasterio.open(true_files[0]) as src:
        dx = abs(src.transform.a)
        dy = abs(src.transform.e)
    cell_area_m2 = float(dx * dy)
    area_km2_per_cell = cell_area_m2 / 1.0e6

    rows = []
    for ref in true_files:
        name = ref.name
        timestamp = name[:-4].replace("_", " ")
        fields = {}
        for variable in ("D", "U"):
            for source_name, d_sub, u_sub in SOURCES:
                sub = d_sub if variable == "D" else u_sub
                path = root / sub / name
                fields[f"{source_name}_{variable}"] = read_band(path)

        rec = {"Time": timestamp, "Hours": 0.0}
        for variable in ("D", "U"):
            for source_name, _, _ in SOURCES:
                arr = fields[f"{source_name}_{variable}"]
                for interval_name, mask_fn in INTERVALS:
                    count = int(mask_fn(arr).sum())
                    rec[f"{source_name}_{variable}_{interval_name}_cells"] = count
                    rec[f"{source_name}_{variable}_{interval_name}_km2"] = (
                        count * area_km2_per_cell
                    )
        rows.append(rec)

    df = pd.DataFrame(rows)
    parsed_time = pd.to_datetime(df["Time"], format="%Y-%m-%d %H %M")
    df["Time"] = parsed_time.dt.strftime("%Y-%m-%d %H:%M")
    df["Hours"] = (parsed_time - parsed_time.iloc[0]).dt.total_seconds() / 3600.0

    depth_area_cols = ["Time", "Hours"]
    depth_count_cols = ["Time", "Hours"]
    vel_area_cols = ["Time", "Hours"]
    vel_count_cols = ["Time", "Hours"]
    for source_name, _, _ in SOURCES:
        for interval_name, _ in INTERVALS:
            depth_area_cols.append(f"{source_name}_D_{interval_name}_km2")
            depth_count_cols.append(f"{source_name}_D_{interval_name}_cells")
            vel_area_cols.append(f"{source_name}_U_{interval_name}_km2")
            vel_count_cols.append(f"{source_name}_U_{interval_name}_cells")

    long_records = []
    for i in range(len(df)):
        for variable in ("D", "U"):
            for source_name, _, _ in SOURCES:
                for interval_name, _ in INTERVALS:
                    long_records.append(
                        {
                            "Time": df.loc[i, "Time"],
                            "Hours": float(df.loc[i, "Hours"]),
                            "Variable": "Depth" if variable == "D" else "Velocity",
                            "Source": source_name,
                            "Interval": interval_name,
                            "Area_km2": float(
                                df.loc[
                                    i, f"{source_name}_{variable}_{interval_name}_km2"
                                ]
                            ),
                            "CellCount": int(
                                df.loc[
                                    i, f"{source_name}_{variable}_{interval_name}_cells"
                                ]
                            ),
                        }
                    )
    long_df = pd.DataFrame(long_records)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df[depth_area_cols].to_excel(writer, sheet_name="Depth_Area_km2", index=False)
        df[vel_area_cols].to_excel(writer, sheet_name="Velocity_Area_km2", index=False)
        df[depth_count_cols].to_excel(writer, sheet_name="Depth_CellCount", index=False)
        df[vel_count_cols].to_excel(writer, sheet_name="Velocity_CellCount", index=False)
        long_df.to_excel(writer, sheet_name="Long_Area_km2", index=False)

        readme = pd.DataFrame(
            [
                ["Source directory", str(root)],
                ["Cell area (m2)", cell_area_m2],
                ["Area unit", "km2"],
                ["Depth intervals", "0.1-0.5, 0.5-1.0, >=1.0 m"],
                ["Velocity intervals", "0.1-0.5, 0.5-1.0, >=1.0 m/s"],
                ["HR =", "D_true / U_true"],
                ["FFSR =", "D_pred / U_pred"],
                ["Interp =", "D_ori / U_ori"],
                ["Time samples", len(df)],
            ],
            columns=["Item", "Value"],
        )
        readme.to_excel(writer, sheet_name="Readme", index=False)

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
