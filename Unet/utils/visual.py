import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.ndimage import zoom


# ============================================================
# Configuration
# ============================================================

LR_PATH = rf"{ROOT}\Data\dataset\Unet\LR_Shouxi.h5"
HR_PATH = rf"{ROOT}\Data\dataset\Unet\HR_Shouxi.h5"

# Index of the sample to inspect
i = 10


# ============================================================
# Read the H5 file
# ============================================================

with h5py.File(LR_PATH, "r") as f:
    lr_all_shape = f["data"].shape
    lr_data = f["data"][i]

with h5py.File(HR_PATH, "r") as f:
    hr_all_shape = f["data"].shape
    hr_data = f["data"][i]


print("=" * 70)
print("H5 dataset information")
print("=" * 70)

print(f"LR dataset shape : {lr_all_shape}")
print(f"HR dataset shape : {hr_all_shape}")

print(f"LR sample shape  : {lr_data.shape}")
print(f"HR sample shape  : {hr_data.shape}")


# ============================================================
# Check the channel count
# ============================================================

if lr_data.shape[0] != 4:
    raise ValueError(
        f"LR channel count error: expected 4, got {lr_data.shape[0]}"
    )

if hr_data.shape[0] != 2:
    raise ValueError(
        f"HR channel count error: expected 2, got {hr_data.shape[0]}"
    )


# ============================================================
# Extract the channels correctly
#
# LR:
#   0 -> Depth
#   1 -> Velocity
#   2 -> DEM
#   3 -> Slope
#
# HR:
#   0 -> Depth
#   1 -> Velocity
# ============================================================

lr_depth = lr_data[0]
lr_velocity = lr_data[1]
lr_dem = lr_data[2]
lr_slope = lr_data[3]

hr_depth = hr_data[0]
hr_velocity = hr_data[1]


# ============================================================
# Print basic statistics
# ============================================================

print("\n" + "=" * 70)
print(f"Sample {i} data statistics")
print("=" * 70)

print("\nLR:")
print(
    f"  Depth    : min={lr_depth.min():.6f}, "
    f"max={lr_depth.max():.6f}"
)
print(
    f"  Velocity : min={lr_velocity.min():.6f}, "
    f"max={lr_velocity.max():.6f}"
)
print(
    f"  DEM      : min={lr_dem.min():.6f}, "
    f"max={lr_dem.max():.6f}"
)
print(
    f"  Slope    : min={lr_slope.min():.6f}, "
    f"max={lr_slope.max():.6f}"
)

print("\nHR:")
print(
    f"  Depth    : min={hr_depth.min():.6f}, "
    f"max={hr_depth.max():.6f}"
)
print(
    f"  Velocity : min={hr_velocity.min():.6f}, "
    f"max={hr_velocity.max():.6f}"
)


# ============================================================
# Size check
# ============================================================

print("\n" + "=" * 70)
print("Spatial size")
print("=" * 70)

print(f"LR Depth    : {lr_depth.shape}")
print(f"LR Velocity : {lr_velocity.shape}")
print(f"HR Depth    : {hr_depth.shape}")
print(f"HR Velocity : {hr_velocity.shape}")


# ============================================================
# Correlation computation
#
# If HR and LR have the same shape, compare them directly.
# If the shapes differ, resize HR to the LR shape.
#
# Note:
# This scaling is only used for statistical comparison; the raw data in the H5 is unchanged.
# ============================================================

def resize_to_shape(data, target_shape):
    zoom_factor = (
        target_shape[0] / data.shape[0],
        target_shape[1] / data.shape[1]
    )

    return zoom(
        data,
        zoom_factor,
        order=1
    )


if hr_depth.shape != lr_depth.shape:

    print("\nHR and LR differ in size; HR is resized to the LR size for the correlation computation.")

    hr_depth_corr = resize_to_shape(
        hr_depth,
        lr_depth.shape
    )

    hr_velocity_corr = resize_to_shape(
        hr_velocity,
        lr_velocity.shape
    )

else:

    hr_depth_corr = hr_depth
    hr_velocity_corr = hr_velocity


# ============================================================
# Evaluation metrics
# ============================================================

def calculate_metrics(a, b):

    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

    mask = (
        np.isfinite(a)
        & np.isfinite(b)
    )

    a = a[mask]
    b = b[mask]

    # Pearson correlation
    r, p = pearsonr(a, b)

    # MAE
    mae = np.mean(
        np.abs(a - b)
    )

    # RMSE
    rmse = np.sqrt(
        np.mean((a - b) ** 2)
    )

    # R²
    ss_res = np.sum(
        (a - b) ** 2
    )

    ss_tot = np.sum(
        (a - np.mean(a)) ** 2
    )

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1 - ss_res / ss_tot

    return {
        "Pearson r": r,
        "p-value": p,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae
    }


depth_metrics = calculate_metrics(
    lr_depth,
    hr_depth_corr
)

velocity_metrics = calculate_metrics(
    lr_velocity,
    hr_velocity_corr
)


# ============================================================
# Print the correlation
# ============================================================

print("\n" + "=" * 70)
print(f"Sample {i} correlation analysis")
print("=" * 70)

print("\nDepth:")
print(
    f"  Pearson r = {depth_metrics['Pearson r']:.6f}"
)
print(
    f"  R²        = {depth_metrics['R2']:.6f}"
)
print(
    f"  RMSE      = {depth_metrics['RMSE']:.6f}"
)
print(
    f"  MAE       = {depth_metrics['MAE']:.6f}"
)

print("\nVelocity:")
print(
    f"  Pearson r = {velocity_metrics['Pearson r']:.6f}"
)
print(
    f"  R²        = {velocity_metrics['R2']:.6f}"
)
print(
    f"  RMSE      = {velocity_metrics['RMSE']:.6f}"
)
print(
    f"  MAE       = {velocity_metrics['MAE']:.6f}"
)


# ============================================================
# Visualization
#
# First row: Depth
# Second row: Velocity
#
# Left column: LR
# Right column: HR
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(15, 10)
)


# ============================================================
# Depth - LR
# ============================================================

im1 = axes[0, 0].imshow(
    lr_depth,
    cmap="viridis",
    vmin=0,
    vmax=12
)

axes[0, 0].set_title(
    f"LR Depth - Sample {i}"
)

axes[0, 0].set_xlabel("Width")
axes[0, 0].set_ylabel("Height")

cbar1 = fig.colorbar(
    im1,
    ax=axes[0, 0]
)

cbar1.set_label("Depth")


# ============================================================
# Depth - HR
# ============================================================

im2 = axes[0, 1].imshow(
    hr_depth,
    cmap="viridis",
    vmin=0,
    vmax=12
)

axes[0, 1].set_title(
    f"HR Depth - Sample {i}"
)

axes[0, 1].set_xlabel("Width")
axes[0, 1].set_ylabel("Height")

cbar2 = fig.colorbar(
    im2,
    ax=axes[0, 1]
)

cbar2.set_label("Depth")


# ============================================================
# Velocity - LR
# ============================================================

im3 = axes[1, 0].imshow(
    lr_velocity,
    cmap="viridis",
    vmin=0,
    vmax=12
)

axes[1, 0].set_title(
    f"LR Velocity - Sample {i}"
)

axes[1, 0].set_xlabel("Width")
axes[1, 0].set_ylabel("Height")

cbar3 = fig.colorbar(
    im3,
    ax=axes[1, 0]
)

cbar3.set_label("Velocity")


# ============================================================
# Velocity - HR
# ============================================================

im4 = axes[1, 1].imshow(
    hr_velocity,
    cmap="viridis",
    vmin=0,
    vmax=12
)

axes[1, 1].set_title(
    f"HR Velocity - Sample {i}"
)

axes[1, 1].set_xlabel("Width")
axes[1, 1].set_ylabel("Height")

cbar4 = fig.colorbar(
    im4,
    ax=axes[1, 1]
)

cbar4.set_label("Velocity")


# ============================================================
# Annotate the correlation on the figure
# ============================================================

depth_text = (
    f"Pearson r = {depth_metrics['Pearson r']:.4f}\n"
    f"R² = {depth_metrics['R2']:.4f}\n"
    f"RMSE = {depth_metrics['RMSE']:.4f}\n"
    f"MAE = {depth_metrics['MAE']:.4f}"
)

velocity_text = (
    f"Pearson r = {velocity_metrics['Pearson r']:.4f}\n"
    f"R² = {velocity_metrics['R2']:.4f}\n"
    f"RMSE = {velocity_metrics['RMSE']:.4f}\n"
    f"MAE = {velocity_metrics['MAE']:.4f}"
)


axes[0, 1].text(
    0.02,
    0.98,
    depth_text,
    transform=axes[0, 1].transAxes,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85
    )
)

axes[1, 1].text(
    0.02,
    0.98,
    velocity_text,
    transform=axes[1, 1].transAxes,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85
    )
)


# ============================================================
# Overall title
# ============================================================

fig.suptitle(
    f"LR vs HR Dataset Validation - Sample {i}",
    fontsize=16
)

plt.tight_layout()

plt.show()