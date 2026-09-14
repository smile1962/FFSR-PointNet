# Standard library imports
import os
import re
import timeit

# Third-party library imports
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, Subset

# Local module imports
from config import get_config
from utils.datasetGen import FlowFieldDataset
from model.FFSRP import PointNetRegression
from utils.plotFunctions import (
    plot_scatter_save, filter_by_x, plot_dual_y_curve,
    plot_relative_error_histogram, export_to_excel, plot_contour_save,
    plot_correlation_scatter_Vel, plot_correlation_scatter_Depth,
    interpolate_lowres_to_highres_flat, plot_contour_save_LR,
    create_tif_from_points, remove_zero_groundtruth, remove_zero_groundtruth_single
)

# Global matplotlib font configuration
plt.rcParams.update({
    'font.size': 24,
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix'  # Math symbol font compatible with Times New Roman
})

# Load experiment configuration
config = get_config()

# Root directory for output files
save_root = config.save_root

# List of output subdirectories to create
subdirs = [
    "U_pred",  # Predicted composite velocity
    "U_ori",  # Low-resolution composite velocity
    "U_true",  # High-resolution ground truth velocity
    "U_error",  # Absolute velocity error
    "U_coeff_bicbuic",
    "D_coeff_bicbuic",
    "U_coeff",  # Velocity relative error / correlation
    "D_coeff",  # Depth relative error / correlation
    "D_pred",  # Predicted water depth
    "D_ori",  # Low-resolution water depth
    "D_true",  # High-resolution ground truth depth
    "D_error",  # Absolute depth error
    "U_interpolated",
    "D_interpolated"
]

# Alternative TIF georeference configuration (deprecated)
# rows, cols = 1075, 1779
# cellsize_x, cellsize_y = 30.0, 30.0
# xmin, ymin, xmax, ymax = (
#     11469243.372897469,
#     3605916.6798295653,
#     11522613.372897469,
#     3638166.6798295653,
# )
#
# crs_wkt = """PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere",
# GEOGCS["GCS_WGS_1984",
#     DATUM["D_WGS_1984",
#         SPHEROID["WGS_1984",6378137.0,298.257223563]],
#     PRIMEM["Greenwich",0.0],
#     UNIT["Degree",0.0174532925199433]],
# PROJECTION["Mercator_Auxiliary_Sphere"],
# PARAMETER["False_Easting",0.0],
# PARAMETER["False_Northing",0.0],
# PARAMETER["Central_Meridian",0.0],
# PARAMETER["Standard_Parallel_1",0.0],
# PARAMETER["Auxiliary_Sphere_Type",0.0],
# UNIT["Meter",1.0]]"""

# High-resolution (HR) output TIF georeference parameters
rows, cols = 2010, 1094
cellsize_x, cellsize_y = 1.5, 1.5
xmin, ymin, xmax, ymax = (
    11503199.581122924,
    3620677.5104884803,
    11504840.581122924,
    3623692.5104884803,
)

crs_wkt = """PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere",
GEOGCS["GCS_WGS_1984",
    DATUM["D_WGS_1984",
        SPHEROID["WGS_1984",6378137.0,298.257223563]],
    PRIMEM["Greenwich",0.0],
    UNIT["Degree",0.0174532925199433]],
PROJECTION["Mercator_Auxiliary_Sphere"],
PARAMETER["False_Easting",0.0],
PARAMETER["False_Northing",0.0],
PARAMETER["Central_Meridian",0.0],
PARAMETER["Standard_Parallel_1",0.0],
PARAMETER["Auxiliary_Sphere_Type",0.0],
UNIT["Meter",1.0]]"""

# Low-resolution (LR) input TIF georeference parameters
rows_LR, cols_LR = 150, 82
cellsize_x_LR, cellsize_y_LR = 20.0, 20.0
xmin_LR, ymin_LR, xmax_LR, ymax_LR = (
    11503199.581122924,
    3620692.5104884803,
    11504839.581122924,
    3623692.5104884803,
)

# Auto-create output directory structure
os.makedirs(save_root, exist_ok=True)
for d in subdirs:
    os.makedirs(os.path.join(save_root, d), exist_ok=True)

# Set random seeds for reproducible dataset split
torch.manual_seed(1)
np.random.seed(1)

# =========================================================
# Experiment Parameter Configuration
# =========================================================
batch_size = 1  # Process samples one by one during inference
use_samples = config.use_samples  # Match sample count with training
global_feat_dim = config.global_feat_dim  # Encoder global feature dimension
# Dataset split ratios (must match training configuration)
train_ratio = config.train_ratio
test_ratio = config.test_ratio
val_ratio = config.val_ratio

# Global matplotlib style configuration
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14

# Unified colorbar ranges for consistent visualization
# Velocity colorbar range
speed_min = 0.0
speed_max = 1.0

# Water depth colorbar range
depth_min = 0.0
depth_max = 5.0

# Error map colorbar range
error_min = 0.0
error_max = 1.0

# Additional matplotlib style configuration (optional)
# mpl.rcParams['axes.linewidth'] = 1.5
# mpl.rcParams['axes.labelsize'] = 18
# mpl.rcParams['xtick.labelsize'] = 16
# mpl.rcParams['ytick.labelsize'] = 16

# =========================================================
# Dataset Path Configuration
# Ensure paths match training configuration exactly
# =========================================================
low_res_folder = config.low_res_folder
high_res_folder = config.high_res_folder
mean_std_files = config.mean_std_file

# =========================================================
# Utility Functions
# =========================================================
_ILLEGAL_CHARS = r'<>:"/\\|?*\n\r\t'


def sanitize_filename(name: str, replacement: str = '_') -> str:
    """Sanitize filename by removing illegal filesystem characters.

    Args:
        name: Raw filename string
        replacement: Character to replace illegal characters with

    Returns:
        Sanitized filename safe for filesystem use
    """
    if name is None:
        return None
    s = str(name).strip()
    for ch in _ILLEGAL_CHARS:
        s = s.replace(ch, replacement)
    s = re.sub(r'\s+', replacement, s).strip(replacement)
    return s if len(s) > 0 else "sample"


# =========================================================
# Dataset Initialization
# =========================================================
dataset = FlowFieldDataset(
    low_h5_path=low_res_folder,
    high_h5_path=high_res_folder,
    # channels=2,  # Uncomment to explicitly set channel count
    num_samples=use_samples,
    return_coords=True,
    normalize=True,
    mean_std_file=mean_std_files,
    x_range=(-9999999, 9999999999999999999),
    y_range=(-9999999, 9999999999999999999),
    return_meta=False,
    return_index=False
)

# Get sample metadata
high_key_list = dataset.high_keys
name_list = dataset.meta_list

# =========================================================
# DataLoader Initialization (NO SHUFFLE for inference)
# =========================================================
# Dataset split (use test set for inference by default)
train_size = int(use_samples * train_ratio)
test_size = int(use_samples * test_ratio)
val_size = use_samples - train_size - test_size
_, test_data, val_data = random_split(dataset, [train_size, test_size, val_size])

test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True
)
# Alternative: use full dataset for inference
# test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# =========================================================
# Model Setup & Weight Loading
# =========================================================
# Note: Input/output dimensions are feature dimensions only (X/Y coordinates excluded)
model = PointNetRegression(
    input_dim=dataset.input_dim,
    global_feat_dim=global_feat_dim,
    output_dim=dataset.output_dim,
    N_high=dataset.N_high
)

# Load best model weights from training
model_path = config.best_save_dir
assert os.path.exists(model_path), f"Model weight file not found: {model_path}"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


def predict_model(inputs: torch.Tensor) -> torch.Tensor:
    """Wrapper function for model inference with gradient disabled."""
    with torch.no_grad():
        return model(inputs)  # Output shape: (B, N_high, output_dim)


model.eval()

# =========================================================
# Inference & Visualization (Contour Comparison)
# =========================================================
# Normalization statistics for denormalization
mean_depth = dataset.mean_depth
std_depth = dataset.std_depth
mean_velocity = dataset.mean_velocity
std_velocity = dataset.std_velocity
mean_out = dataset.mean_out
std_out = dataset.std_out

# User selection mode: 0 = output all samples, 1 = output specific sample
mode = int(input("Enter 0 to output all results, or 1 to output a specific sample: "))

desired_index = None
if mode == 1:
    desired_index = int(input("Enter the sample index you want to process: "))

# Main inference loop
for i, (inputs, targets, coords_low, coords_high) in enumerate(test_loader):
    # Skip non-target samples in single-sample mode
    if mode == 1 and i != desired_index:
        continue

    with torch.no_grad():
        # Measure inference time
        timer = timeit.Timer(lambda: predict_model(inputs))
        execution_time = timer.timeit(number=1)
        print(f"Sample {i} inference time: {execution_time:.6f} seconds")
        outputs = model(inputs)

        # Get sanitized sample name for output files
        sample_name = high_key_list[i]
        sample_name = sample_name.replace(" ", "_").replace(":", "_")

    # Remove batch dimension and denormalize predictions
    pred_features = outputs.squeeze(0)
    gt_features = targets.squeeze(0)
    ori_features = inputs.squeeze(0)

    coords_high_t = coords_high.squeeze(0)
    coords_low_t = coords_low.squeeze(0)

    # Denormalize output features (depth & velocity)
    for c in range(2):
        pred_features[:, c] = pred_features[:, c] * std_out[c] + mean_out[c]
        gt_features[:, c] = gt_features[:, c] * std_out[c] + mean_out[c]

    # Denormalize input features
    ori_features[:, 1] = ori_features[:, 1] * std_depth + mean_depth
    ori_features[:, 2] = ori_features[:, 2] * std_velocity + mean_velocity

    # Combine coordinates with features for full output arrays
    pred_full = torch.cat([coords_high_t, pred_features], dim=1).numpy()
    gt_full = torch.cat([coords_high_t, gt_features], dim=1).numpy()
    ori_full = torch.cat([coords_low_t, ori_features], dim=1).numpy()

    # Extract individual variables
    Xp, Yp, Dp, Up = pred_full[:, 0], pred_full[:, 1], pred_full[:, 2], pred_full[:, 3]
    Xg, Yg, Dg, Ug = gt_full[:, 0], gt_full[:, 1], gt_full[:, 2], gt_full[:, 3]
    Xo, Yo, Do, Uo = ori_full[:, 0], ori_full[:, 1], ori_full[:, 2], ori_full[:, 3]

    # Calculate absolute errors
    D_error = np.abs(Dg - Dp)
    U_error = np.abs(Ug - Up)

    # Threshold small values to 0 for cleaner visualization
    Dg[Dg < 1.0] = 0.0
    Dp[Dp < 1.0] = 0.0
    D_error[D_error < 0.015] = 0.0
    Up[Up < 1.0] = 0.0
    Ug[Ug < 1.0] = 0.0
    U_error[U_error < 0.02] = 0.0

    # Remove zero ground truth points (dry areas)
    Xp, Yp, Dp, Up, Xg, Yg, Dg, Ug, removed_count, removed_idx = \
        remove_zero_groundtruth(
            Xp, Yp, Dp, Up, Xg, Yg, Dg, Ug,
            tol=0.0, mode="both", return_removed_index=True
        )

    # Define output file paths
    fname_pred = os.path.join(save_root, "D_pred", f"{sample_name}.tif")
    fname_Dori = os.path.join(save_root, "D_ori", f"{sample_name}.tif")
    fname_interpolated = os.path.join(save_root, "D_interpolated", f"{sample_name}.tif")
    fname_true = os.path.join(save_root, "D_true", f"{sample_name}.tif")
    fname_Derr = os.path.join(save_root, "D_error", f"{sample_name}.tif")
    fname_Upred = os.path.join(save_root, "U_pred", f"{sample_name}.tif")
    fname_U_interpolated = os.path.join(save_root, "U_interpolated", f"{sample_name}.tif")
    fname_Uori = os.path.join(save_root, "U_ori", f"{sample_name}.tif")
    fname_Utrue = os.path.join(save_root, "U_true", f"{sample_name}.tif")
    fname_Uerr = os.path.join(save_root, "U_error", f"{sample_name}.tif")
    fname_UCoeff = os.path.join(save_root, "U_coeff", f"{sample_name}.tif")
    fname_DCoeff = os.path.join(save_root, "D_coeff", f"{sample_name}.tif")

    # Export GeoTIFF files using existing utility function
    # Water depth outputs
    create_tif_from_points(
        rows, cols, cellsize_x, cellsize_y, xmin, ymin, xmax, ymax,
        Xp, Yp, Dp, out_tif=fname_pred, crs_wkt=crs_wkt
    )
    create_tif_from_points(
        rows_LR, cols_LR, cellsize_x_LR, cellsize_y_LR, xmin_LR, ymin_LR, xmax_LR, ymax_LR,
        Xo, Yo, Do, out_tif=fname_Dori, crs_wkt=crs_wkt
    )
    create_tif_from_points(
        rows, cols, cellsize_x, cellsize_y, xmin, ymin, xmax, ymax,
        Xp, Yp, Dg, out_tif=fname_true, crs_wkt=crs_wkt
    )
    create_tif_from_points(
        rows, cols, cellsize_x, cellsize_y, xmin, ymin, xmax, ymax,
        Xp, Yp, D_error, out_tif=fname_Derr, crs_wkt=crs_wkt
    )

    # Velocity outputs
    create_tif_from_points(
        rows, cols, cellsize_x, cellsize_y, xmin, ymin, xmax, ymax,
        Xp, Yp, Up, out_tif=fname_Upred, crs_wkt=crs_wkt
    )
    create_tif_from_points(
        rows_LR, cols_LR, cellsize_x_LR, cellsize_y_LR, xmin_LR, ymin_LR, xmax_LR, ymax_LR,
        Xo, Yo, Uo, out_tif=fname_Uori, crs_wkt=crs_wkt
    )
    create_tif_from_points(
        rows, cols, cellsize_x, cellsize_y, xmin, ymin, xmax, ymax,
        Xp, Yp, Ug, out_tif=fname_Utrue, crs_wkt=crs_wkt
    )
    create_tif_from_points(
        rows, cols, cellsize_x, cellsize_y, xmin, ymin, xmax, ymax,
        Xp, Yp, U_error, out_tif=fname_Uerr, crs_wkt=crs_wkt
    )

    # Generate correlation scatter plots
    plot_correlation_scatter_Vel(Up, Ug, fname_UCoeff)
    plot_correlation_scatter_Depth(Dp, Dg, fname_DCoeff)

    # Bilinear interpolation baseline comparison (optional, commented out)
    # grid_Xo_region = np.unique(Xo, return_counts=False)
    # grid_Yo_region = np.unique(Yo, return_counts=False)
    # grid_Xp_region = np.unique(Xp, return_counts=False)
    # grid_Yp_region = np.unique(Yp, return_counts=False)
    #
    # bicubic_result_U = interpolate_lowres_to_highres_flat(
    #     Xo, Yo, Uo, grid_Xo_region, grid_Yo_region, grid_Xp_region, grid_Yp_region
    # )
    # bicubic_result_D = interpolate_lowres_to_highres_flat(
    #     Xo, Yo, Do, grid_Xo_region, grid_Yo_region, grid_Xp_region, grid_Yp_region
    # )
    #
    # plot_correlation_scatter_Vel(bicubic_result_U, Ug, fname_UCoeff)
    # plot_correlation_scatter_Depth(bicubic_result_D, Dg, fname_DCoeff)