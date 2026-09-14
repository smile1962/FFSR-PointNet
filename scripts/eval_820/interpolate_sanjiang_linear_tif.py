"""Bilinear-upscale the raw Sanjiang LR interpolation TIFFs to the HR grid.

The original D_ori/U_ori GeoTIFFs are 20 m rasters (150 x 82) while the
D_true/U_true rasters are 1.5 m (2010 x 1094). This script reprojects the LR
rasters onto the HR grid with bilinear resampling and writes the same physical
post-processing used by the other Sanjiang baselines into D_pred/U_pred.
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULT_ROOT = Path(rf"{ROOT}\Results\Sanjiang\Linear-Interpolation")
SRC_NODATA = -9999.0


def threshold_like_model(data):
    """Match the 0.01 clipping used by model validation outputs."""
    out = np.clip(np.asarray(data, dtype=np.float32), 0.0, None)
    out[out < 0.01] = 0.0
    return out


def bilinear_upscale_to_hr(lr_path, hr_path, out_path):
    with rasterio.open(str(lr_path)) as lr_src:
        lr = lr_src.read(1)
        src_transform = lr_src.transform
        src_crs = lr_src.crs

    with rasterio.open(str(hr_path)) as hr_src:
        dst_shape = (hr_src.height, hr_src.width)
        dst_transform = hr_src.transform
        dst_crs = hr_src.crs
        profile = hr_src.profile.copy()

    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=0.0,
        compress=None,
    )

    dst = np.zeros(dst_shape, dtype=np.float32)
    reproject(
        source=lr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=SRC_NODATA,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=0.0,
        resampling=Resampling.bilinear,
    )
    dst = threshold_like_model(dst)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(out_path), "w", **profile) as dst_ds:
        dst_ds.write(dst, 1)
    print(f"Saved {out_path}")


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    for prefix, lr_folder, hr_folder, out_folder in (
        ("D", "D_ori", "D_true", "D_pred"),
        ("U", "U_ori", "U_true", "U_pred"),
    ):
        lr_dir = RESULT_ROOT / lr_folder
        hr_dir = RESULT_ROOT / hr_folder
        out_dir = RESULT_ROOT / out_folder
        out_dir.mkdir(parents=True, exist_ok=True)
        if not lr_dir.is_dir() or not hr_dir.is_dir():
            print(f"Skipping missing {prefix} folders")
            continue
        for lr_path in sorted(lr_dir.glob("*.tif")):
            hr_path = hr_dir / lr_path.name
            if not hr_path.exists():
                print(f"Missing HR counterpart: {hr_path}")
                continue
            bilinear_upscale_to_hr(lr_path, hr_path, out_dir / lr_path.name)


if __name__ == "__main__":
    main()
