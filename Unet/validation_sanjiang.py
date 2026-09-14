import os
import numpy as np
import torch
import h5py
import scipy.io as sio
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import gc
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from scipy.stats import pearsonr
from matplotlib import cm

# Import the configuration
from config_village import config

# Import the model (assuming SuperResolutionModel is defined in model.py)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import set_seed, SuperResolutionModel


# ====================== Patch handling helper class ======================
class PatchHandler:
    @staticmethod
    def extract_patches(image, patch_size=(64, 64), stride=None):
        """Extract image patches for model input"""
        if stride is None:
            stride = patch_size
        C, H, W = image.shape
        ph, pw = patch_size
        sh, sw = stride
        num_h = (H - ph) // sh + 1
        num_w = (W - pw) // sw + 1
        patches, positions = [], []
        for i in range(num_h):
            for j in range(num_w):
                start_h, end_h = i * sh, i * sh + ph
                start_w, end_w = j * sw, j * sw + pw
                patches.append(image[:, start_h:end_h, start_w:end_w])
                positions.append((start_h, end_h, start_w, end_w))
        return np.array(patches), np.array(positions)

    @staticmethod
    def reconstruct_image(patches, positions, output_shape):
        """Reassemble image patches into a full image"""
        C, H, W = output_shape
        image = np.zeros(output_shape, dtype=np.float32)
        count = np.zeros((H, W), dtype=np.int32)
        for patch, (sh, eh, sw, ew) in zip(patches, positions):
            image[:, sh:eh, sw:ew] += patch
            count[sh:eh, sw:ew] += 1
        count[count == 0] = 1  # avoid division by zero
        for c in range(C):
            image[c] /= count
        return image


# ====================== TIFF helper functions ======================
def create_tif_from_matrix(rows, cols, cellsize_x, cellsize_y, xmin, ymax,
                           matrix, out_tif, crs_wkt):
    """
    Write a GeoTIFF directly from a regular-grid matrix (no interpolation needed)
    matrix: 2-D matrix (rows x cols, exactly matching the TIFF dimensions)
    """
    # Define the TIFF transform (top-left origin, y decreasing downwards)
    transform = from_origin(xmin, ymax, cellsize_x, cellsize_y)

    # Ensure the matrix dimensions match the TIFF
    assert matrix.shape == (rows, cols), \
        f"matrix dimensions {matrix.shape} do not match TIFF dimensions {rows}x{cols}!"

    # Write the TIFF
    with rasterio.open(
            out_tif,
            'w',
            driver='GTiff',
            height=rows,
            width=cols,
            count=1,  # single channel
            dtype=matrix.dtype,
            crs=crs_wkt,
            transform=transform,
    ) as dst:
        dst.write(matrix, 1)  # write the matrix directly (no flattening needed)
    print(f"TIFF saved: {os.path.basename(out_tif)}")


# ====================== Correlation computation and visualization ======================
def density_scatter(x, y, ax=None, bins=20, point_size=16):
    # Default bins:100
    """Plot a density scatter colored by -log10(probability density), skipping empty bins"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # 1. Count the total number of points
    total_points = len(x)

    # 2. Compute the 2-D histogram (bin counts)
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

    # 3. Probability density = bin count / total points
    prob_density = counts / total_points

    # 4. Assign each data point the probability density of its bin
    x_bin_idx = np.clip(np.digitize(x, x_edges) - 1, 0, bins - 1)
    y_bin_idx = np.clip(np.digitize(y, y_edges) - 1, 0, bins - 1)
    point_density = prob_density[x_bin_idx, y_bin_idx]

    # 5. Build a mask: keep only points with probability density > 0
    valid_mask = point_density > 0
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    density_valid = point_density[valid_mask]

    # 6. Colour value: -log10(probability density)
    # colors = -np.log10(density_valid)
    colors = density_valid

    # 7. Normalize the colour mapping (excluding extreme values)
    vmin = np.percentile(colors, 5)  # use the 5th and 95th percentiles
    vmax = np.percentile(colors, 95)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # 8. Scatter plot (valid points only)
    sc = ax.scatter(
        x_valid, y_valid,
        c=colors,
        s=point_size,
        alpha=0.6,
        cmap=cm.plasma,
        norm=norm,
        edgecolors='none'
    )

    # 9. Add the colourbar
    cbar = plt.colorbar(sc, ax=ax)
    # cbar.set_label('-Log$_{10}$(Probability Density)', fontsize=24)
    cbar.set_label('Probability Density', fontsize=24)
    cbar.ax.tick_params(labelsize=24)

    # 10. Report skipped points
    skipped_points = total_points - len(x_valid)
    if skipped_points > 0:
        print(f"Note: skipped {skipped_points} points in empty bins ({skipped_points / total_points:.1%} of all points)")

    return ax

def plot_correlation_scatter_Vel(pred, truth, save_path, figsize=(10, 5), dpi=600):
    """
    Plot and save a correlation scatter plot

    Parameters:
        pred: Array of predicted values (ndarray)
        truth: Array of ground truth values (ndarray)
        save_path: Path to save the plot (str)
        figsize: Figure size (tuple, default (10,8))
        dpi: Resolution (int, default 600)
    """
    # Initialize figure settings
    plt.rcParams.update({
        'font.size': 24,
        'font.family': 'Times New Roman',
        'mathtext.fontset': 'stix'
    })
    fig, ax = plt.subplots(figsize=figsize)

    # Draw density scatter
    density_scatter(pred, truth, ax=ax, bins=100, point_size=6)

    # Compute correlation and RMSE
    rho = np.corrcoef(pred, truth)[0, 1]
    rmse = np.sqrt(np.mean((pred - truth) ** 2))

    # Add reference line and annotation
    lim_min = min(pred.min(), truth.min())
    lim_max = max(pred.max(), truth.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=3)
    ax.text(0.05, 0.8, f'R = {rho:.3f}\nRMSE = {rmse:.3f} m/s',
            transform=ax.transAxes,
            fontdict={'family': 'Times New Roman', 'size': 24},
            bbox=dict(facecolor='white', alpha=0.8))

    # Set axis labels and appearance
    ax.set_xlabel('Predicted velocity (m/s)', fontsize=24)
    # ax.set_xlabel('Bicubic interpolation velocity (m/s)', fontsize=24)
    ax.set_ylabel('Numerical solution (m/s)', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
    # ax.set_aspect('equal')
    ax.grid(alpha=0.2)

    # Create directories and save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Correlation plot saved to: {save_path}")

def plot_correlation_scatter_Depth(pred, truth, save_path, figsize=(10, 5), dpi=600):
    """
    Plot and save a correlation scatter plot

    Parameters:
        pred: Array of predicted values (ndarray)
        truth: Array of ground truth values (ndarray)
        save_path: Path to save the plot (str)
        figsize: Figure size (tuple, default (10,8))
        dpi: Resolution (int, default 600)
    """
    # Initialize figure settings
    plt.rcParams.update({
        'font.size': 24,
        'font.family': 'Times New Roman',
        'mathtext.fontset': 'stix'
    })
    fig, ax = plt.subplots(figsize=figsize)

    # Draw density scatter
    density_scatter(pred, truth, ax=ax, bins=100, point_size=16)

    # Compute correlation and RMSE
    rho = np.corrcoef(pred, truth)[0, 1]
    rmse = np.sqrt(np.mean((pred - truth) ** 2))

    # Add reference line and annotation
    lim_min = min(pred.min(), truth.min())
    lim_max = max(pred.max(), truth.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=3)
    ax.text(0.05, 0.8, f'R = {rho:.3f}\nRMSE = {rmse:.3f} m/s',
            transform=ax.transAxes,
            fontdict={'family': 'Times New Roman', 'size': 24},
            bbox=dict(facecolor='white', alpha=0.8))

    # Set axis labels and appearance
    ax.set_xlabel('Predicted water depth (m)', fontsize=24)
    # ax.set_xlabel('Bicubic interpolation velocity (m/s)', fontsize=24)
    ax.set_ylabel('Numerical solution (m)', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
    # ax.set_aspect('equal')
    ax.grid(alpha=0.2)

    # Create directories and save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Correlation plot saved to: {save_path}")


# ====================== Data processing helper functions ======================
def pad_single_sample(sample, patch_size):
    """Pad a single sample (so that it is divisible by patch_size)"""
    C, H, W = sample.shape
    ph, pw = patch_size
    pad_h = (ph - H % ph) % ph
    pad_w = (pw - W % pw) % pw
    padded_sample = np.pad(sample, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    return padded_sample, (pad_h, pad_w)


def predict_single_sample(low_sample, model, device, patch_size):
    """Run prediction for a single low-resolution sample"""
    model.eval()
    low_padded, (pad_h, pad_w) = pad_single_sample(low_sample, patch_size)
    patches, positions = PatchHandler.extract_patches(low_padded, patch_size)
    preds_np = []

    with torch.no_grad():
        for idx in range(0, len(patches), config.patch_batch_size):
            batch_patches = patches[idx:idx + config.patch_batch_size]
            batch_tensor = torch.from_numpy(batch_patches).float().to(device)
            batch_pred = model(batch_tensor)
            preds_np.append(batch_pred.cpu().numpy())

    preds_np = np.concatenate(preds_np, axis=0)
    output_shape = (2, low_padded.shape[1], low_padded.shape[2])
    reconstructed_padded = PatchHandler.reconstruct_image(preds_np, positions, output_shape)

    # Remove padding
    if pad_h > 0 or pad_w > 0:
        reconstructed = reconstructed_padded[:, :-pad_h, :-pad_w]
    else:
        reconstructed = reconstructed_padded
    return reconstructed


def visualize_result(pred, real, time_str, save_path):
    """Visualize the prediction of a single test sample"""
    channel_names = ["Depth", "Velocity"]
    rainbow_cmap = plt.cm.rainbow
    abs_error_0 = np.abs(pred[0] - real[0])
    abs_error_1 = np.abs(pred[1] - real[1])

    # Fixed colourbar range (can be extended in the config)
    depth_range = (0, 10)
    velocity_range = (0, 1)
    error_range = (0, 1)

    # CJK font settings
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False

    # 2-row, 3-column subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Rainfall event: {time_str}', fontsize=18, fontweight='bold')

    # Depth channel (0): prediction, truth, error
    im0_0 = axes[0, 0].imshow(pred[0], cmap=rainbow_cmap, vmin=depth_range[0], vmax=depth_range[1])
    axes[0, 0].set_title(f'Predicted HR - {channel_names[0]}', fontsize=14)
    axes[0, 0].axis('off')
    plt.colorbar(im0_0, ax=axes[0, 0], fraction=0.046, pad=0.04, label=f'{channel_names[0]} value')

    im0_1 = axes[0, 1].imshow(real[0], cmap=rainbow_cmap, vmin=depth_range[0], vmax=depth_range[1])
    axes[0, 1].set_title(f'True HR - {channel_names[0]}', fontsize=14)
    axes[0, 1].axis('off')
    plt.colorbar(im0_1, ax=axes[0, 1], fraction=0.046, pad=0.04, label=f'{channel_names[0]} value')

    im0_2 = axes[0, 2].imshow(abs_error_0, cmap=rainbow_cmap, vmin=error_range[0], vmax=error_range[1])
    axes[0, 2].set_title(f'Absolute error - {channel_names[0]} (max: {np.max(abs_error_0):.2f})', fontsize=14)
    axes[0, 2].axis('off')
    plt.colorbar(im0_2, ax=axes[0, 2], fraction=0.046, pad=0.04, label=f'{channel_names[0]} error')

    # Velocity channel (1): prediction, truth, error
    im1_0 = axes[1, 0].imshow(pred[1], cmap=rainbow_cmap, vmin=velocity_range[0], vmax=velocity_range[1])
    axes[1, 0].set_title(f'Predicted HR - {channel_names[1]}', fontsize=14)
    axes[1, 0].axis('off')
    plt.colorbar(im1_0, ax=axes[1, 0], fraction=0.046, pad=0.04, label=f'{channel_names[1]} value')

    im1_1 = axes[1, 1].imshow(real[1], cmap=rainbow_cmap, vmin=velocity_range[0], vmax=velocity_range[1])
    axes[1, 1].set_title(f'True HR - {channel_names[1]}', fontsize=14)
    axes[1, 1].axis('off')
    plt.colorbar(im1_1, ax=axes[1, 1], fraction=0.046, pad=0.04, label=f'{channel_names[1]} value')

    im1_2 = axes[1, 2].imshow(abs_error_1, cmap=rainbow_cmap, vmin=error_range[0], vmax=error_range[1])
    axes[1, 2].set_title(f'Absolute error - {channel_names[1]} (max: {np.max(abs_error_1):.2f})', fontsize=14)
    axes[1, 2].axis('off')
    plt.colorbar(im1_2, ax=axes[1, 2], fraction=0.046, pad=0.04, label=f'{channel_names[1]} error')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved: {os.path.basename(save_path)}")
    plt.close()


def load_metadata(meta_path, total_samples):
    """Load metadata and extract time information"""
    try:
        meta_data = sio.loadmat(meta_path)
        time_list = []
        if 'metadata' in meta_data:
            metadata_list = meta_data['metadata'][0][:total_samples]
            for meta in metadata_list:
                if 'timestamp' in meta.dtype.names:
                    timestamp_str = meta['timestamp'][0][0]
                    time_list.append(datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S"))
                else:
                    time_list.append(f"unknown_time_{len(time_list)}")
        elif 'time' in meta_data:
            times = meta_data['time'].flatten()[:total_samples]
            time_list = [datetime.fromordinal(int(t)) + np.timedelta64(int((t - int(t)) * 24 * 3600), 's') for t in
                         times]
        else:
            print("Warning: no time field found in the metadata; using default names")
            time_list = [f"sample{i}" for i in range(total_samples)]
        return time_list
    except Exception as e:
        print(f"Error loading metadata: {e}; using default names")
        return [f"sample{i}" for i in range(total_samples)]


# ====================== Main function ======================
def main():
    # -------------------------- 1. Initialization and device setup --------------------------
    set_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute device: {device}")
    print(f"Sample split ratio: train {config.split_ratios[0]} | test {config.split_ratios[1]} | val {config.split_ratios[2]}")

    # Create the output directories
    for subdir in config.output_subdirs:
        subdir_path = os.path.join(config.result_dir_shouxi, subdir)
        os.makedirs(subdir_path, exist_ok=True)

    # Set the output paths
    save_root = config.result_dir_shouxi

    # -------------------------- 2. Load data and preprocess  --------------------------
    with h5py.File(os.path.join(config.data_dir, config.low_h5), 'r') as f_low, \
            h5py.File(os.path.join(config.data_dir, config.high_h5), 'r') as f_high:

        total_samples = min(len(f_low['data']), len(f_high['data']))
        print(f"Total samples in the full dataset: {total_samples}")

        # Reproduce the random index split used during training
        indices = np.random.permutation(total_samples)
        train_size = int(config.split_ratios[0] * total_samples)
        test_size = int(config.split_ratios[1] * total_samples)
        test_indices = indices[train_size:train_size + test_size]
        test_indices = sorted(test_indices)
        print(f"Test-set samples: {len(test_indices)}, original indices: {test_indices[:5]}...")  # show only the first five

        # Compute the normalization statistics of the low-resolution feature channels
        print("\nComputing normalization statistics for the low-resolution feature channels of the test set...")
        test_lr_all_channels = f_low['data'][test_indices]  # [test_size, C, H, W]
        test_lr_feats = test_lr_all_channels[:, config.lr_feat_channels, :, :]  # extract the feature channels

        lr_min = np.zeros(4, dtype=np.float32)
        lr_max = np.zeros(4, dtype=np.float32)
        for i, orig_c in enumerate(config.lr_feat_channels):
            lr_min[orig_c] = np.min(test_lr_feats[:, i, :, :])
            lr_max[orig_c] = np.max(test_lr_feats[:, i, :, :])
            if lr_max[orig_c] - lr_min[orig_c] < 1e-8:
                lr_max[orig_c] = lr_min[orig_c] + 1e-8

        # Non-feature channels (depth/velocity) are not normalized
        lr_min[config.lr_d_channel] = 0
        lr_max[config.lr_d_channel] = 1
        lr_min[config.lr_u_channel] = 0
        lr_max[config.lr_u_channel] = 1
        print(f"Low-resolution channel min: {lr_min.round(4)}, max: {lr_max.round(4)}")

    # -------------------------- 3. Load model and metadata  --------------------------
    print("\nLoading the trained model...")
    try:
        model = SuperResolutionModel().to(device)
        model.load_state_dict(
            torch.load(os.path.join(config.weight_dir, config.model_weight_path),
                       map_location=device)
        )
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading the model: {e}; exiting")
        return

    # Load the test-set metadata
    print("\nLoading test-set metadata...")
    time_list = load_metadata(os.path.join(config.data_dir, config.meta_path), len(test_indices))

    # -------------------------- 4. Predict on the test set  --------------------------
    print("\nStarting test-set processing ({} samples)...".format(len(test_indices)))
    with h5py.File(os.path.join(config.data_dir, config.low_h5), 'r') as f_low, \
            h5py.File(os.path.join(config.data_dir, config.high_h5), 'r') as f_high:

        # Initialize the error statistics
        total_error = 0.0
        total_pixels = 0

        for test_idx, orig_idx in enumerate(test_indices):
            print(f"\nProcessing sample {test_idx + 1}/{len(test_indices)} (original index: {orig_idx})...")

            # Read data
            low_sample = f_low['data'][orig_idx]  # [C, H_lr, W_lr]
            high_real = f_high['data'][orig_idx]  # [2, H_hr, W_hr]
            total_pixels += high_real.size  # accumulate the pixel count for statistics

            # Normalize the low-resolution feature channels
            low_sample_norm = np.copy(low_sample)
            for c in config.lr_feat_channels:
                low_sample_norm[c] = (low_sample[c] - lr_min[c]) / (lr_max[c] - lr_min[c])

            # Model prediction
            high_pred = predict_single_sample(low_sample_norm, model, device, config.patch_size)
            high_pred_denorm = high_pred  # no inverse normalization needed

            # Remove excess padding so the resolutions match
            # First compute how much padding must be removed
            rows = config.tiff_rows
            cols = config.tiff_cols
            ori_raws = high_pred_denorm.shape[1]
            ori_cols = high_pred_denorm.shape[2]
            deltaRows = abs(rows - ori_raws)
            deltaCols = abs(cols - ori_cols)
            # Clip
            high_pred_denorm = high_pred_denorm[:, deltaRows:, deltaCols:]
            low_sample = low_sample[:, deltaRows:, deltaCols:]
            high_real = high_real[:, deltaRows:, deltaCols:]

            # Extract each physical quantity (using the configured channel indices)
            D_pred = high_pred_denorm[config.hr_d_channel]  # predicted depth
            U_pred = high_pred_denorm[config.hr_u_channel]  # predicted velocity
            D_ori = low_sample[config.lr_d_channel]  # low-resolution depth (interpolated)
            U_ori = low_sample[config.lr_u_channel]  # low-resolution velocity (interpolated)
            D_true = high_real[config.hr_d_channel]  # true depth
            U_true = high_real[config.hr_u_channel]  # true velocity

            # Compute the MSE and correlation R for depth (D)
            D_mse = np.mean((D_true - D_pred) ** 2)  # mean squared error
            D_r, _ = pearsonr(D_true.flatten(), D_pred.flatten())  # correlation coefficient (R)
            D_error = np.abs(D_true - D_pred)

            # Compute the MSE and correlation R for velocity (U)
            U_mse = np.mean((U_true - U_pred) ** 2)  # mean squared error
            U_r, _ = pearsonr(U_true.flatten(), U_pred.flatten())  # correlation coefficient (R)
            U_error = np.abs(U_true - U_pred)

            # -------------------------- 5. Build geographic coordinates (grid to points)  --------------------------
            # High-resolution grid coordinates (corresponding to H_hr x W_hr of the TIFF)
            hr_rows, hr_cols = np.meshgrid(np.arange(rows), np.arange(cols))
            # Convert to geographic coordinates (x/y)
            hr_x = config.xmin + hr_cols * config.cellsize_x  # column -> x axis
            hr_y = config.ymax - hr_rows * config.cellsize_y  # row -> y axis (decreasing from top to bottom)
            hr_x_flat = hr_x.flatten()  # 1-D array
            hr_y_flat = hr_y.flatten()

            # Low-resolution grid coordinates (must match the geographic extent of the high-resolution TIFF)
            lr_rows, lr_cols = np.meshgrid(np.arange(rows), np.arange(cols))
            # Low-resolution grid = high-resolution grid (NOTE: the low-resolution data in this dataset is already bilinearly interpolated)
            lr_x = config.xmin + lr_cols * config.cellsize_x
            lr_y = config.ymax - lr_rows * config.cellsize_y
            lr_x_flat = lr_x.flatten()
            lr_y_flat = lr_y.flatten()

            # -------------------------- 6. Save TIFFs (covering all 12 subdirectories) --------------------------
            sample_name = f"sample_orig{orig_idx}_test{test_idx}"  # sample name (unique)

            # 6.1 Velocity-related TIFFs
            create_tif_from_matrix(
                rows=config.tiff_rows, cols=config.tiff_cols,
                cellsize_x=config.cellsize_x, cellsize_y=config.cellsize_y,
                xmin=config.xmin, ymax=config.ymax,
                matrix=U_pred,
                out_tif=os.path.join(save_root, "U_pred", f"{sample_name}.tif"),
                crs_wkt=config.crs_wkt
            )
            create_tif_from_matrix(
                rows=config.tiff_rows, cols=config.tiff_cols,
                cellsize_x=config.cellsize_x, cellsize_y=config.cellsize_y,
                xmin=config.xmin, ymax=config.ymax,
                matrix=U_ori,
                out_tif=os.path.join(save_root, "U_ori", f"{sample_name}.tif"),
                crs_wkt=config.crs_wkt
            )
            create_tif_from_matrix(
                rows=config.tiff_rows, cols=config.tiff_cols,
                cellsize_x=config.cellsize_x, cellsize_y=config.cellsize_y,
                xmin=config.xmin, ymax=config.ymax,
                matrix=U_true,
                out_tif=os.path.join(save_root, "U_true", f"{sample_name}.tif"),
                crs_wkt=config.crs_wkt
            )
            create_tif_from_matrix(
                rows=config.tiff_rows, cols=config.tiff_cols,
                cellsize_x=config.cellsize_x, cellsize_y=config.cellsize_y,
                xmin=config.xmin, ymax=config.ymax,
                matrix=U_error,
                out_tif=os.path.join(save_root, "U_error", f"{sample_name}.tif"),
                crs_wkt=config.crs_wkt
            )

            # 6.2 Depth-related TIFFs
            create_tif_from_matrix(
                rows=config.tiff_rows, cols=config.tiff_cols,
                cellsize_x=config.cellsize_x, cellsize_y=config.cellsize_y,
                xmin=config.xmin, ymax=config.ymax,  # note: pass ymax (top-left y coordinate)
                matrix=D_pred,  # pass the 2-D matrix directly (rows x cols)
                out_tif=os.path.join(save_root, "D_pred", f"{sample_name}.tif"),
                crs_wkt=config.crs_wkt
            )
            create_tif_from_matrix(
                rows=config.tiff_rows, cols=config.tiff_cols,
                cellsize_x=config.cellsize_x, cellsize_y=config.cellsize_y,
                xmin=config.xmin, ymax=config.ymax,  # note: pass ymax (top-left y coordinate)
                matrix=D_ori,  # pass the 2-D matrix directly (rows x cols)
                out_tif=os.path.join(save_root, "D_ori", f"{sample_name}.tif"),
                crs_wkt=config.crs_wkt
            )
            create_tif_from_matrix(
                rows=config.tiff_rows, cols=config.tiff_cols,
                cellsize_x=config.cellsize_x, cellsize_y=config.cellsize_y,
                xmin=config.xmin, ymax=config.ymax,  # note: pass ymax (top-left y coordinate)
                matrix=D_true,  # pass the 2-D matrix directly (rows x cols)
                out_tif=os.path.join(save_root, "D_true", f"{sample_name}.tif"),
                crs_wkt=config.crs_wkt
            )
            create_tif_from_matrix(
                rows=config.tiff_rows, cols=config.tiff_cols,
                cellsize_x=config.cellsize_x, cellsize_y=config.cellsize_y,
                xmin=config.xmin, ymax=config.ymax,  # note: pass ymax (top-left y coordinate)
                matrix=D_error,  # pass the 2-D matrix directly (rows x cols)
                out_tif=os.path.join(save_root, "D_error", f"{sample_name}.tif"),
                crs_wkt=config.crs_wkt
            )

            # 6.3 Correlation files (PNG in practice; adjust the savefig format for TIFF)
            # 6.3.1. Velocity (U_pred vs U_true)
            plot_correlation_scatter_Vel(
                pred=U_pred.flatten(),  # flatten to a 1-D array (as required by your function)
                truth=U_true.flatten(),
                save_path=os.path.join(save_root, "U_coeff", f"{sample_name}.png"),
                figsize=(10, 8),  # adjust the size as needed
                dpi=300
            )

            # 6.3.2. Depth (D_pred vs D_true)
            plot_correlation_scatter_Depth(
                pred=D_pred.flatten(),
                truth=D_true.flatten(),
                save_path=os.path.join(save_root, "D_coeff", f"{sample_name}.png"),
                figsize=(10, 8),
                dpi=300
            )

            # 6.3.3. Low-resolution velocity vs. true velocity (bicubic baseline)
            plot_correlation_scatter_Vel(
                pred=U_ori.flatten(),  # low-resolution raw data
                truth=U_true.flatten(),
                save_path=os.path.join(save_root, "U_coeff_bicbuic", f"{sample_name}.png"),
                figsize=(10, 8),
                dpi=300
            )

            # 6.3.4. Low-resolution depth vs. true depth (bicubic baseline)
            plot_correlation_scatter_Depth(
                pred=D_ori.flatten(),  # low-resolution raw data
                truth=D_true.flatten(),
                save_path=os.path.join(save_root, "D_coeff_bicbuic", f"{sample_name}.png"),
                figsize=(10, 8),
                dpi=300
            )

            # -------------------------- 7. Free memory (original logic) --------------------------
            del low_sample, high_real, low_sample_norm, high_pred, high_pred_denorm
            gc.collect()

            # -------------------------- 8. Print statistics (original logic) --------------------------
        print("\nAll samples processed. TIFF results saved to: {}".format(save_root))



if __name__ == "__main__":
    main()
