# -*- coding: utf-8 -*-
"""
Visualize the water-depth field (channel 0) of one sample from HR_San.h5 / LR_San.h5 for comparison
Outputs a single PNG saved in this directory.
"""
import os
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=int, default=20)
parser.add_argument("--channel", type=int, default=0)
parser.add_argument(
    "--lr",
    default=os.path.join(HERE, "LR_Shouxi_val.h5"),
)
parser.add_argument(
    "--hr",
    default=os.path.join(HERE, "HR_Shouxi_val.h5"),
)
parser.add_argument("--out", default=None)
args = parser.parse_args()

SAMPLE_IDX = args.sample
CH = args.channel
DRY = 0.001
LR_PATH = args.lr
HR_PATH = args.hr
OUT_PNG = args.out or os.path.join(
    HERE,
    f"depth_LR_HR_sample{SAMPLE_IDX}_{Path(LR_PATH).stem}.png",
)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

# ---------------- Read data (one sample at a time to save memory) ----------------
with h5py.File(LR_PATH, "r") as f:
    lr = f["data"][SAMPLE_IDX, CH].astype(np.float32)
with h5py.File(HR_PATH, "r") as f:
    hr = f["data"][SAMPLE_IDX, CH].astype(np.float32)


def prep(a):
    """Set dry cells to NaN so they appear as a white background in the map"""
    a = a.copy()
    a[a <= DRY] = np.nan
    return a


lr_d, hr_d = prep(lr), prep(hr)

# Common colour scale: use the 99th percentile of HR so a few very deep points do not stretch the scale
vmax = float(np.nanpercentile(hr_d, 99))
vmin = 0.0

# Compute the metrics using the common valid mask where neither is dry
mask = (~np.isnan(lr_d)) & (~np.isnan(hr_d))
rmse = float(np.sqrt(np.mean((lr_d[mask] - hr_d[mask]) ** 2)))
r = float(np.corrcoef(lr_d[mask], hr_d[mask])[0, 1])

# ---------------- Plotting ----------------
cmap = plt.get_cmap("jet").copy()
cmap.set_bad("white")

fig, axes = plt.subplots(1, 2, figsize=(9, 7))

for ax, arr, title in zip(axes, [lr_d, hr_d],
                          [f"LR (input) sample #{SAMPLE_IDX}",
                           f"HR (ground truth) sample #{SAMPLE_IDX}"]):
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="lower", aspect="equal", interpolation="nearest")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("W", fontsize=11)
    ax.set_ylabel("H", fontsize=11)
    ax.tick_params(labelsize=9)

# Shared colorbar
cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
cbar.set_label("Water depth (m)", fontsize=11)
cbar.ax.tick_params(labelsize=9)

fig.suptitle(f"Water depth field: LR (smoothed low-res input) vs HR (ground truth)   |   "
             f"joint-wet pixels = {mask.sum()}   RMSE = {rmse:.3f} m   R = {r:.4f}",
             fontsize=12)

fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved:", OUT_PNG)
print(f"joint-wet pixels={mask.sum()}  vmax (p99)={vmax:.3f} m  RMSE={rmse:.4f}  R={r:.4f}")
