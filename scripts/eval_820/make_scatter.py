"""Generate original-format density scatter figures from stored GeoTIFF maps.

Used for the grid CNN models (SR-Unet and FLO-SR), whose inference loop does
not create the D_coeff/U_coeff figures directly. Metrics remain untouched by
this script.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--roi_path", default=str(ROOT / "Data" / "dataset" / "geo" / "shouxi_mask_30_Point.tif"))
    return parser.parse_args()


def read_first(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def main():
    args = parse_args()
    sys.path.insert(0, str(ROOT / "FFSR-PointNet" / "main"))
    from utils.plotFunctions import (  # noqa: E402
        plot_correlation_scatter_Depth,
        plot_correlation_scatter_Vel,
    )

    result_dir = Path(args.result_dir)
    roi_path = Path(args.roi_path)
    roi = read_first(roi_path) > 0
    pred_names = sorted((result_dir / "D_pred").glob("*.tif"))
    if not pred_names:
        raise FileNotFoundError(f"No D_pred files under {result_dir}")
    (result_dir / "D_coeff").mkdir(exist_ok=True)
    (result_dir / "U_coeff").mkdir(exist_ok=True)

    for pred_path in pred_names:
        name = pred_path.name
        true_path = result_dir / "D_true" / name
        if not true_path.exists():
            print(f"Skipping missing D_true {name}")
            continue
        dp = read_first(pred_path)
        dt = read_first(true_path)
        up = read_first(result_dir / "U_pred" / name)
        ut = read_first(result_dir / "U_true" / name)
        dp[dp < 0.01] = 0.0
        dt[dt < 0.01] = 0.0
        up[up < 0.01] = 0.0
        ut[ut < 0.01] = 0.0
        m_d = roi & ((dp > 0) | (dt > 0))
        m_u = roi & ((up > 0) | (ut > 0))
        if m_d.sum() < 10 or m_u.sum() < 10:
            continue
        plot_correlation_scatter_Depth(
            dp[m_d], dt[m_d], str(result_dir / "D_coeff" / name)
        )
        plot_correlation_scatter_Vel(
            up[m_u], ut[m_u], str(result_dir / "U_coeff" / name)
        )
        print(f"Scatter saved for {name}")


if __name__ == "__main__":
    main()
