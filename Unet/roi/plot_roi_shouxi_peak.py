import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

import roi_data


PEAK = "2019-08-20_04_00.tif"
FFSR_DIR = rf"{ROOT}\Results\FFSR_PointNet\Shouxi_820"
UNET_DIR = rf"{ROOT}\Results\Unet\srunet_roi_shouxi_820"
OUT_PNG = rf"{ROOT}\Results\Unet\roi_peak_comparison_820.png"


def read(folder, group, name):
    with rasterio.open(f"{folder}/{group}/{name}") as src:
        return src.read(1).astype(np.float32)


def main():
    cfg = roi_data.get_config("watershed")
    roi = roi_data.load_roi_original(cfg)
    true = read(UNET_DIR, "D_true", PEAK)
    ffsr = read(FFSR_DIR, "D_pred", PEAK)
    unet = read(UNET_DIR, "D_pred", PEAK)
    for arr in (true, ffsr, unet):
        arr[~roi] = np.nan

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    titles = ["Fine-grid depth (8.20 peak)", "FFSR-PointNet", "ROI-aware Patch SR-Unet"]
    for ax, arr, title in zip(axes, (true, ffsr, unet), titles):
        im = ax.imshow(arr, cmap="turbo", vmin=0, vmax=5.0)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Depth (m)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()

