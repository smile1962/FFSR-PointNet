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

# Import the configuration (the new case parameters are already defined in config)
from config_watershed import config

# Import the model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import set_seed, SuperResolutionModel


# ====================== Patch handling helper class ======================
class PatchHandler:
    @staticmethod
    def extract_patches(image, patch_size=(64, 64), stride=None):
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


def process_flattened_arrays(U_pred_flatten, U_ori_flatten, U_true_flatten,
                             D_pred_flatten, D_ori_flatten, D_true_flatten):
    """
    Process six flattened 1-D arrays, removing positions where both U_true and D_true are 0

    Parameters:
        U_pred_flatten: flattened predicted velocity array
        U_ori_flatten: flattened original velocity array
        U_true_flatten: flattened true velocity array
        D_pred_flatten: flattened predicted depth array
        D_ori_flatten: flattened original depth array
        D_true_flatten: flattened true depth array

    Returns:
        The six processed arrays (in the same order as the input)
    """
    # Find the indices where both U_true and D_true are 0
    zero_mask = (U_true_flatten == 0) & (D_true_flatten == 0)

    # Get the indices to keep (negated)
    keep_indices = ~zero_mask

    # Apply the same index filtering to all arrays
    U_pred_processed = U_pred_flatten[keep_indices]
    U_ori_processed = U_ori_flatten[keep_indices]
    U_true_processed = U_true_flatten[keep_indices]
    D_pred_processed = D_pred_flatten[keep_indices]
    D_ori_processed = D_ori_flatten[keep_indices]
    D_true_processed = D_true_flatten[keep_indices]

    return U_pred_processed, U_ori_processed, U_true_processed, D_pred_processed, D_ori_processed, D_true_processed

# ====================== TIFF helper functions (unchanged) ======================
def create_tif_from_matrix(rows, cols, cellsize_x, cellsize_y, xmin, ymax,
                           matrix, out_tif, crs_wkt):
    """Write a GeoTIFF directly from a regular-grid matrix (no interpolation needed)"""
    transform = from_origin(xmin, ymax, cellsize_x, cellsize_y)

    # Ensure the matrix dimensions match the TIFF
    assert matrix.shape == (rows, cols), \
        f"matrix dimensions {matrix.shape} do not match TIFF dimensions {rows}x{cols}!"

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
        dst.write(matrix, 1)
    print(f"TIFF saved: {os.path.basename(out_tif)}")


# ====================== Correlation computation and visualization (unchanged) ======================
def density_scatter(x, y, ax=None, bins=20, point_size=16):
    """Plot a density scatter"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    total_points = len(x)
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    prob_density = counts / total_points

    x_bin_idx = np.clip(np.digitize(x, x_edges) - 1, 0, bins - 1)
    y_bin_idx = np.clip(np.digitize(y, y_edges) - 1, 0, bins - 1)
    point_density = prob_density[x_bin_idx, y_bin_idx]

    valid_mask = point_density > 0
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    density_valid = point_density[valid_mask]

    colors = density_valid
    vmin = np.percentile(colors, 5)
    vmax = np.percentile(colors, 95)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    sc = ax.scatter(
        x_valid, y_valid,
        c=colors,
        s=point_size,
        alpha=0.6,
        cmap=cm.plasma,
        norm=norm,
        edgecolors='none'
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Probability Density', fontsize=24)
    cbar.ax.tick_params(labelsize=24)

    skipped_points = total_points - len(x_valid)
    if skipped_points > 0:
        print(f"Note: skipped {skipped_points} points in empty bins ({skipped_points / total_points:.1%} of all points)")

    return ax

def plot_correlation_scatter_Vel(pred, truth, save_path, figsize=(10, 5), dpi=600):
    """Velocity correlation scatter plot"""
    plt.rcParams.update({
        'font.size': 24,
        'font.family': 'Times New Roman',
        'mathtext.fontset': 'stix'
    })
    fig, ax = plt.subplots(figsize=figsize)

    density_scatter(pred, truth, ax=ax, bins=100, point_size=6)

    rho = np.corrcoef(pred, truth)[0, 1]
    rmse = np.sqrt(np.mean((pred - truth) ** 2))

    lim_min = min(pred.min(), truth.min())
    lim_max = max(pred.max(), truth.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=3)
    ax.text(0.05, 0.8, f'R = {rho:.3f}\nRMSE = {rmse:.3f} m/s',
            transform=ax.transAxes,
            fontdict={'family': 'Times New Roman', 'size': 24},
            bbox=dict(facecolor='white', alpha=0.8))

    ax.set_xlabel('Predicted velocity (m/s)', fontsize=24)
    ax.set_ylabel('Numerical solution (m/s)', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.grid(alpha=0.2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Correlation plot saved to: {save_path}")

def plot_correlation_scatter_Depth(pred, truth, save_path, figsize=(10, 5), dpi=600):
    """Depth correlation scatter plot"""
    plt.rcParams.update({
        'font.size': 24,
        'font.family': 'Times New Roman',
        'mathtext.fontset': 'stix'
    })
    fig, ax = plt.subplots(figsize=figsize)

    density_scatter(pred, truth, ax=ax, bins=100, point_size=16)

    rho = np.corrcoef(pred, truth)[0, 1]
    rmse = np.sqrt(np.mean((pred - truth) ** 2))

    lim_min = min(pred.min(), truth.min())
    lim_max = max(pred.max(), truth.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=3)
    ax.text(0.05, 0.8, f'R = {rho:.3f}\nRMSE = {rmse:.3f} m',
            transform=ax.transAxes,
            fontdict={'family': 'Times New Roman', 'size': 24},
            bbox=dict(facecolor='white', alpha=0.8))

    ax.set_xlabel('Predicted water depth (m)', fontsize=24)
    ax.set_ylabel('Numerical solution (m)', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.grid(alpha=0.2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Correlation plot saved to: {save_path}")


# ====================== Data processing helper functions (unchanged) ======================
def pad_single_sample(sample, patch_size):
    """Pad a single sample"""
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

    # Colour scale range (configurable in the config)
    depth_range = (0, 10)
    velocity_range = (0, 1)
    error_range = (0, 1)

    # CJK font settings
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Rainfall event: {time_str}', fontsize=18, fontweight='bold')

    # Depth channel
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

    # Velocity channel
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


# ====================== Main function (only the split logic changed; original dimension handling kept) ======================
def main():
    # -------------------------- 1. Initialization and device setup --------------------------
    set_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute device: {device}")

    # Create the output directories
    for subdir in config.output_subdirs:
        subdir_path = os.path.join(config.result_dir_shouxi, subdir)
        os.makedirs(subdir_path, exist_ok=True)

    # Set the output paths
    save_root = config.result_dir_shouxi

    # -------------------------- 2. Load data and preprocess (key change: no split, test on all samples) --------------------------
    with h5py.File(os.path.join(config.data_dir, config.low_h5), 'r') as f_low, \
            h5py.File(os.path.join(config.data_dir, config.high_h5), 'r') as f_high:

        # Key change: use the whole dataset as the test set, in the original order (no shuffling)
        total_samples = min(len(f_low['data']), len(f_high['data']))
        test_indices = np.arange(total_samples)  # use indices 0..total_samples-1 directly (original order)
        print(f"Total samples in the independent test set: {len(test_indices)}, processed in original order")

        # Compute the normalization statistics of the low-resolution feature channels (using the full test set)
        print("\nComputing normalization statistics for the low-resolution feature channels of the test set...")
        test_lr_all_channels = f_low['data'][test_indices]  # low-resolution data for the whole test set
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

    # -------------------------- 3. Load model and metadata (unchanged) --------------------------
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

    # Load the test-set metadata (using the full test-set sample count)
    print("\nLoading test-set metadata...")
    time_list = load_metadata(os.path.join(config.data_dir, config.meta_path), len(test_indices))

    # -------------------------- 4. Test on all samples (original dimension handling kept) --------------------------
    print("\nStarting full-sample testing ({} samples, in original order)...".format(len(test_indices)))
    with h5py.File(os.path.join(config.data_dir, config.low_h5), 'r') as f_low, \
            h5py.File(os.path.join(config.data_dir, config.high_h5), 'r') as f_high:

        # Initialize the error statistics
        total_error = 0.0
        total_pixels = 0

        # Iterate over the whole test set (in original index order)
        for test_idx, orig_idx in enumerate(test_indices):
            print(f"\nProcessing sample {test_idx + 1}/{len(test_indices)} (original index: {orig_idx})...")

            # Read data
            low_sample = f_low['data'][orig_idx]  # [C, H_lr, W_lr]
            high_real = f_high['data'][orig_idx]  # [2, H_hr, W_hr]
            total_pixels += high_real.size  # accumulate the pixel count

            # Normalize the low-resolution feature channels
            low_sample_norm = np.copy(low_sample)
            for c in config.lr_feat_channels:
                low_sample_norm[c] = (low_sample[c] - lr_min[c]) / (lr_max[c] - lr_min[c])

            # Model prediction
            high_pred = predict_single_sample(low_sample_norm, model, device, config.patch_size)
            high_pred_denorm = high_pred  # no inverse normalization needed

            # [Original dimension handling kept] Remove excess padding so resolutions match
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

            # Extract physical quantities
            D_pred = high_pred_denorm[config.hr_d_channel]  # predicted depth
            U_pred = high_pred_denorm[config.hr_u_channel]  # predicted velocity
            D_ori = low_sample[config.lr_d_channel]         # low-resolution depth
            U_ori = low_sample[config.lr_u_channel]         # low-resolution velocity
            D_true = high_real[config.hr_d_channel]         # true depth
            U_true = high_real[config.hr_u_channel]         # true velocity

            D_min = D_ori.min()
            D_max = D_ori.max()
            U_min = U_ori.min()
            U_max = U_ori.max()

            # Preprocess: set depth/velocity values below the threshold to 0
            D_pred[D_pred < config.threshold_D] = 0
            D_ori[D_ori < config.threshold_D] = 0
            D_true[D_true < config.threshold_D] = 0
            U_pred[U_pred < config.threshold_U] = 0
            U_ori[U_ori < config.threshold_U] = 0
            U_true[U_true < config.threshold_U] = 0

            # Compute the error metrics
            D_mse = np.mean((D_true - D_pred) ** 2)
            D_r = np.corrcoef(D_true.flatten(), D_pred.flatten())[0, 1]
            D_error = np.abs(D_true - D_pred)

            U_mse = np.mean((U_true - U_pred) ** 2)
            U_r = np.corrcoef(U_true.flatten(), U_pred.flatten())[0, 1]
            U_error = np.abs(U_true - U_pred)

            # Cumulative error
            total_error += np.sum(D_error) + np.sum(U_error)

            # -------------------------- 5. Save TIFFs and correlation figures (unchanged) --------------------------
            sample_name = f"sample_orig{orig_idx}_test{test_idx}"  # sample name (including the original index)

            # 5.1 Velocity-related TIFFs
            create_tif_from_matrix(
                rows=rows, cols=cols,
                cellsize_x=config.cellsize_x, cellsize_y=config.cellsize_y,
                xmin=config.xmin, ymax=config.ymax,
                matrix=U_pred,
                out_tif=os.path.join(save_root, "U_pred", f"{sample_name}.tif"),
                crs_wkt=config.crs_wkt
            )
            create_tif_from_matrix(
                rows=rows, cols=cols,
                cellsize_x=config.cellsize_x, cellsize_y=config.cellsize_y,
                xmin=config.xmin, ymax=config.ymax,
                matrix=U_ori,
                out_tif=os.path.join(save_root, "U_ori", f"{sample_name}.tif"),
                crs_wkt=config.crs_wkt
            )
            create_tif_from_matrix(
                rows=rows, cols=cols,
                cellsize_x=config.cellsize_x, cellsize_y=config.cellsize_y,
                xmin=config.xmin, ymax=config.ymax,
                matrix=U_true,
                out_tif=os.path.join(save_root, "U_true", f"{sample_name}.tif"),
                crs_wkt=config.crs_wkt
            )
            create_tif_from_matrix(
                rows=rows, cols=cols,
                cellsize_x=config.cellsize_x, cellsize_y=config.cellsize_y,
                xmin=config.xmin, ymax=config.ymax,
                matrix=U_error,
                out_tif=os.path.join(save_root, "U_error", f"{sample_name}.tif"),
                crs_wkt=config.crs_wkt
            )

            # 5.2 Depth-related TIFFs
            create_tif_from_matrix(
                rows=rows, cols=cols,
                cellsize_x=config.cellsize_x, cellsize_y=config.cellsize_y,
                xmin=config.xmin, ymax=config.ymax,
                matrix=D_pred,
                out_tif=os.path.join(save_root, "D_pred", f"{sample_name}.tif"),
                crs_wkt=config.crs_wkt
            )
            create_tif_from_matrix(
                rows=rows, cols=cols,
                cellsize_x=config.cellsize_x, cellsize_y=config.cellsize_y,
                xmin=config.xmin, ymax=config.ymax,
                matrix=D_ori,
                out_tif=os.path.join(save_root, "D_ori", f"{sample_name}.tif"),
                crs_wkt=config.crs_wkt
            )
            create_tif_from_matrix(
                rows=rows, cols=cols,
                cellsize_x=config.cellsize_x, cellsize_y=config.cellsize_y,
                xmin=config.xmin, ymax=config.ymax,
                matrix=D_true,
                out_tif=os.path.join(save_root, "D_true", f"{sample_name}.tif"),
                crs_wkt=config.crs_wkt
            )
            create_tif_from_matrix(
                rows=rows, cols=cols,
                cellsize_x=config.cellsize_x, cellsize_y=config.cellsize_y,
                xmin=config.xmin, ymax=config.ymax,
                matrix=D_error,
                out_tif=os.path.join(save_root, "D_error", f"{sample_name}.tif"),
                crs_wkt=config.crs_wkt
            )

            # 5.3 Correlation figures
            # Preprocessing
            U_pred_flatten = U_pred.flatten()
            U_ori_flatten = U_ori.flatten()
            U_true_flatten = U_true.flatten()

            D_pred_flatten = D_pred.flatten()
            D_ori_flatten = D_ori.flatten()
            D_true_flatten = D_true.flatten()

            # U_pred_processed, U_ori_processed, U_true_processed, D_pred_processed, D_ori_processed, D_true_processed = process_flattened_arrays(
            #     U_pred_flatten, U_ori_flatten, U_true_flatten,
            #     D_pred_flatten, D_ori_flatten, D_true_flatten
            # )
            U_pred_processed, U_ori_processed, U_true_processed, D_pred_processed, D_ori_processed, D_true_processed = U_pred_flatten, U_ori_flatten, U_true_flatten, D_pred_flatten, D_ori_flatten, D_true_flatten

            plot_correlation_scatter_Vel(
                pred=U_pred_processed,
                truth=U_true_processed,
                save_path=os.path.join(save_root, "U_coeff", f"{sample_name}.png"),
                figsize=(10, 8),
                dpi=300
            )
            coer = np.corrcoef(U_pred_processed, U_true_processed)[0, 1]
            print(coer)
            plot_correlation_scatter_Depth(
                pred=D_pred_processed,
                truth=D_true_processed,
                save_path=os.path.join(save_root, "D_coeff", f"{sample_name}.png"),
                figsize=(10, 8),
                dpi=300
            )
            plot_correlation_scatter_Vel(
                pred=U_ori_processed,
                truth=U_true_processed,
                save_path=os.path.join(save_root, "U_coeff_bicbuic", f"{sample_name}.png"),
                figsize=(10, 8),
                dpi=300
            )
            plot_correlation_scatter_Depth(
                pred=D_ori_processed,
                truth=D_true_processed,
                save_path=os.path.join(save_root, "D_coeff_bicbuic", f"{sample_name}.png"),
                figsize=(10, 8),
                dpi=300
            )

            # (Optional) save the visualization
            if hasattr(config, 'save_fig') and config.save_fig:
                vis_path = os.path.join(save_root, f"visual_sample_{orig_idx}_test{test_idx}.png")
                time_str = time_list[test_idx]
                if isinstance(time_str, datetime):
                    time_str = time_str.strftime("%Y-%m-%d %H:%M:%S")
                visualize_result(high_pred_denorm, high_real, time_str, vis_path)

            # Free memory
            del low_sample, high_real, low_sample_norm, high_pred, high_pred_denorm
            gc.collect()

        # Print the error statistics for the whole test set
        if total_pixels > 0:
            avg_error = total_error / total_pixels
            print("\n" + "=" * 60)
            print(f"Error statistics over the whole test set ({len(test_indices)} samples)")
            print(f"Mean absolute error (MAE): {avg_error:.6f}")
            print("=" * 60)

    print("\nAll samples tested. Results saved to: {}".format(save_root))


if __name__ == "__main__":
    main()
