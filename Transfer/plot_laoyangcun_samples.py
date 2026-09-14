#!/usr/bin/env python
"""Plot one peak-like LaoYangCun sample from train and test HDF5 sets."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def best_peak_key(low_h5_path, high_h5_path):
    with h5py.File(low_h5_path, "r") as low, h5py.File(high_h5_path, "r") as high:
        keys = list(low.keys())
        volume = []
        for key in keys:
            volume.append(float(np.sum(low[key][:, 2])))
        volume = np.asarray(volume)
        order = np.argsort(volume)[::-1]
        picked = int(order[0])
        picked_key = keys[picked]
        low_arr = low[picked_key][()]
        high_arr = high[picked_key][()]
    return picked_key, low_arr, high_arr


def plot_split(low_path, high_path, split_name, out_png):
    key, low, high = best_peak_key(low_path, high_path)
    x = high[:, 0]
    y = high[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    norm = Normalize(vmin=0.0, vmax=12.0)
    cmap = plt.get_cmap("rainbow")

    plots = [
        (axes[0, 0], low[:, 2], f"{split_name} | Depth LR | {key}"),
        (axes[0, 1], high[:, 2], f"{split_name} | Depth HR | {key}"),
        (axes[1, 0], high[:, 3], f"{split_name} | Velocity HR | {key}"),
        (axes[1, 1], low[:, 3], f"{split_name} | Velocity LR | {key}"),
    ]
    for ax, values, title in plots:
        ax.scatter(
            x,
            y,
            c=values,
            cmap=cmap,
            norm=norm,
            s=0.25,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        orientation="vertical",
        fraction=0.025,
        pad=0.015,
    )
    cbar.set_label("Depth [m] / Velocity [m/s]", fontsize=13)
    fig.suptitle(
        f"{split_name} field check - sample {key}",
        fontsize=15,
        y=0.99,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {split_name} visual to {out_png}")


def main():
    base = rf"{ROOT}\Data\dataset"
    out_dir = rf"{ROOT}\Transfer\validation\LaoYangCun_visual_check"
    os.makedirs(out_dir, exist_ok=True)

    plot_split(
        os.path.join(base, "LaoYangCun_LR.h5"),
        os.path.join(base, "LaoYangCun_HR.h5"),
        "Training",
        os.path.join(out_dir, "LaoYangCun_training_sample.png"),
    )
    plot_split(
        os.path.join(base, "LaoYangCun_LR_val.h5"),
        os.path.join(base, "LaoYangCun_HR_val.h5"),
        "Testing",
        os.path.join(out_dir, "LaoYangCun_testing_sample.png"),
    )


if __name__ == "__main__":
    main()
