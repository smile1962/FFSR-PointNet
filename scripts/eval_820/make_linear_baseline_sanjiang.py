"""Build the Sanjiang grid-domain linear interpolation baseline.

The grid LR h5 already contains the low-resolution depth/velocity fields
resampled onto the HR grid (with the same top/left padding as HR_San.h5).
This script crops that padding, applies the same 0.01 thresholding used for
model outputs, and writes the baseline maps and errors for the 27 held-out
samples.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT / "Unet" / "roi"))

import roi_data  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    return parser.parse_args()


def threshold(data, threshold=0.01):
    out = np.clip(np.asarray(data, dtype=np.float32), 0.0, None)
    out[out < threshold] = 0.0
    return out


def main():
    args = parse_args()
    cfg = roi_data.get_config("village")
    indices = roi_data.get_sample_indices(cfg)["test"]
    save_root = Path(args.result_dir)
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
        (save_root / sub).mkdir(parents=True, exist_ok=True)

    with h5py.File(str(cfg.h5_dir / cfg.lr_train_h5), "r") as f_lr, \
            h5py.File(str(cfg.h5_dir / cfg.hr_train_h5), "r") as f_hr:
        pad_h, pad_w = cfg.pad_origin
        for test_pos, sample_idx in enumerate(indices):
            lr = roi_data.read_sample(f_lr, int(sample_idx))
            hr = roi_data.read_sample(f_hr, int(sample_idx))
            lr_ori = lr[0:2, pad_h:, pad_w:]
            true = hr[0:2, pad_h:, pad_w:]
            pred = threshold(lr_ori)
            true = threshold(true)
            d_err = np.abs(pred[0] - true[0])
            u_err = np.abs(pred[1] - true[1])
            d_err[d_err < 0.015] = 0.0
            u_err[u_err < 0.02] = 0.0
            name = roi_data.output_name_for_sample(
                cfg, int(sample_idx), test_pos, None
            )
            maps = {
                "D_pred": pred[0],
                "D_true": true[0],
                "D_ori": lr_ori[0],
                "D_error": d_err,
                "U_pred": pred[1],
                "U_true": true[1],
                "U_ori": lr_ori[1],
                "U_error": u_err,
            }
            for sub, matrix in maps.items():
                roi_data.create_tif(cfg, matrix, save_root / sub / name)
            print(name)
    print(f"Saved Sanjiang interpolation baseline to {save_root}")


if __name__ == "__main__":
    main()
