"""Zero out all pixels outside the static ROI in saved result GeoTIFFs.

This ensures the CNN models do not expose 500-valued sentinel cells outside
the study ROI. Metrics were already ROI-based and are not changed by this step.
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROI_PATH = ROOT / "Data" / "dataset" / "geo" / "shouxi_mask_30_Point.tif"
SUBDIRS = (
    "D_pred",
    "D_true",
    "D_ori",
    "D_error",
    "U_pred",
    "U_true",
    "U_ori",
    "U_error",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    with rasterio.open(ROI_PATH) as src:
        roi = src.read(1)
        profile = src.profile
    roi_bool = roi > 0
    for sub in SUBDIRS:
        folder = result_dir / sub
        if not folder.exists():
            continue
        for path in folder.glob("*.tif"):
            with rasterio.open(path) as src:
                data = src.read(1)
                p = src.profile.copy()
            data[~roi_bool] = 0.0
            with rasterio.open(path, "w", **p) as dst:
                dst.write(data, 1)
    print(f"Masked ROI outputs under {result_dir}")


if __name__ == "__main__":
    main()
