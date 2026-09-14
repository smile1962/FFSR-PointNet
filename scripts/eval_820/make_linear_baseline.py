"""Create linear/bilinear interpolation baseline maps from stored LR fields.

The stored low-resolution fields (`D_ori`, `U_ori`) already represent the LR
hydrodynamic maps after bilinear downscaling/resampling onto the HR spatial
grid. This script repackages them as a `Linear-Interpolation` baseline result
folder so that maps, errors, density scatter and ROI metrics can be generated
with the same post-processing used for all learned models.
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--save_root", required=True)
    return parser.parse_args()


def read(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    return data, profile


def write(path, data, profile):
    profile.update(count=1, dtype="float32")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def threshold(data):
    out = np.clip(data, 0, None)
    out[out < 0.01] = 0.0
    return out


def main():
    args = parse_args()
    src = Path(args.source_dir)
    dst = Path(args.save_root)
    for sub in (
        "D_pred",
        "D_true",
        "D_ori",
        "D_error",
        "U_pred",
        "U_true",
        "U_ori",
        "U_error",
    ):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    true_files = sorted((src / "D_true").glob("*.tif"))
    if not true_files:
        raise FileNotFoundError(f"No D_true under {src}")
    for true_path in true_files:
        name = true_path.name
        for prefix, error_thr in (("D", 0.015), ("U", 0.02)):
            pred_raw, prof = read(src / f"{prefix}_ori" / name)
            true_arr, _ = read(src / f"{prefix}_true" / name)
            pred = threshold(pred_raw)
            true = threshold(true_arr)
            err = np.abs(pred - true)
            err[err < error_thr] = 0.0
            write(dst / f"{prefix}_ori" / name, pred_raw, prof)
            write(dst / f"{prefix}_pred" / name, pred, prof)
            write(dst / f"{prefix}_true" / name, true, prof)
            write(dst / f"{prefix}_error" / name, err, prof)
        print(name)
    print(f"Saved baseline maps to {dst}")


if __name__ == "__main__":
    main()
