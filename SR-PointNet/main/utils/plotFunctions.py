import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from matplotlib import cm
import cv2
import rasterio
from rasterio.transform import from_origin
import imageio.v2 as imageio
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
import h5py

"""
This module provides a comprehensive set of plotting and post-processing utilities for flow field super-resolution research.
Included functionalities:
- Contour and scatter visualization of interpolated flow fields (HR and LR versions)
- Quantitative error metric calculation (RMSE, MAE, relative error histogram)
- Cross-section profile comparison with dual y-axis layout
- Training / validation loss curve plotting
- Excel export of point-wise flow data
- Density scatter correlation plots for velocity and depth
- Bicubic baseline interpolation from low-resolution to high-resolution grid
- GeoTIFF raster generation from scattered point data
- DEM hillshade overlay with water depth classification
- Side-by-side comparison frame generation and MP4 video synthesis
- Zero-value ground truth filtering utilities
- HDF5 dataset statistical summary extraction and formatted output
"""

# Global font configuration
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14


def plot_contour_save(X, Y, values,
                    title, xlabel, ylabel, colorbar_label,
                    vmin, vmax, save_path,
                    grid_x=1000, grid_y=400,
                    cmap='jet'):
    """
    Plot a cloud (image-style) map via imshow, align the colorbar height with the main axes, and save the figure.

    Parameters:
        X (array-like): 1D array of X coordinates.
        Y (array-like): 1D array of Y coordinates.
        values (array-like): Data values at each (X, Y) point.
        title (str): Title of the plot.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        colorbar_label (str): Label for the colorbar.
        vmin (float): Minimum value for the colormap.
        vmax (float): Maximum value for the colormap.
        save_path (str): Path where the figure will be saved.
        grid_x (int, optional): Number of grid points in X direction. Defaults to 1000.
        grid_y (int, optional): Number of grid points in Y direction. Defaults to 400.
        cmap (str, optional): Colormap name. Defaults to 'jet'.
    """
    # 1) generate uniform grid in X/Y
    # xi = np.linspace(np.min(X), np.max(X), grid_x)
    xi = np.linspace(np.min(X), 3.46750, grid_x)
    yi = np.linspace(0.1, 0.4, grid_y)
    Xi, Yi = np.meshgrid(xi, yi)

    # 2) interpolate scattered data onto grid
    Zi = griddata((X, Y), values, (Xi, Yi), method='linear')

    # 3) prepare imshow options
    plot_options = {
        'cmap':   cmap,
        'origin': 'lower',
        'extent': [X.min(), X.max(), Y.min(), Y.max()],
        'vmin':   vmin,
        'vmax':   vmax,
        'aspect': 'equal'
    }

    # 4) create figure and main axes
    fig, ax = plt.subplots()

    # 5) plot image
    im = ax.imshow(Zi, **plot_options)

    # 6) set titles and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='both', which='major', direction='in', length=1, width=0.5)

    # 7) add colorbar with same height as main axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(colorbar_label)
    cbar.ax.tick_params(axis='y', direction='in', length=1, width=0.5)

    # 8) save and close
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def plot_contour_save_LR(X, Y, values,
                    title, xlabel, ylabel, colorbar_label,
                    vmin, vmax, save_path,
                    grid_x=125, grid_y=50,
                    cmap='jet'):
    """
    Plot a cloud (image-style) map via imshow, align the colorbar height with the main axes, and save the figure.

    Parameters:
        X (array-like): 1D array of X coordinates.
        Y (array-like): 1D array of Y coordinates.
        values (array-like): Data values at each (X, Y) point.
        title (str): Title of the plot.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        colorbar_label (str): Label for the colorbar.
        vmin (float): Minimum value for the colormap.
        vmax (float): Maximum value for the colormap.
        save_path (str): Path where the figure will be saved.
        grid_x (int, optional): Number of grid points in X direction. Defaults to 1000.
        grid_y (int, optional): Number of grid points in Y direction. Defaults to 400.
        cmap (str, optional): Colormap name. Defaults to 'jet'.
    """
    # 1) generate uniform grid in X/Y
    # xi = np.linspace(np.min(X), np.max(X), grid_x)
    # yi = np.linspace(np.min(Y), np.max(Y), grid_y)
    xi = np.linspace(np.min(X), 3.45, grid_x)
    yi = np.linspace(np.min(Y), np.max(Y), grid_y)
    Xi, Yi = np.meshgrid(xi, yi)

    # 2) interpolate scattered data onto grid
    Zi = griddata((X, Y), values, (Xi, Yi), method='linear')

    # 3) prepare imshow options
    plot_options = {
        'cmap':   cmap,
        'origin': 'lower',
        'extent': [X.min(), X.max(), Y.min(), Y.max()],
        'vmin':   vmin,
        'vmax':   vmax,
        'aspect': 'equal'
    }

    # 4) create figure and main axes
    fig, ax = plt.subplots()

    # 5) plot image
    im = ax.imshow(Zi, **plot_options)

    # 6) set titles and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='both', which='major', direction='in', length=1, width=0.5)

    # 7) add colorbar with same height as main axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(colorbar_label)
    cbar.ax.tick_params(axis='y', direction='in', length=1, width=0.5)

    # 8) save and close
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def plot_scatter_save(X, Y, values, title, xlabel, ylabel, colorbar_label,
                      vmin, vmax, scatter_size, save_path, aspect='equal'):
    """
    Plot a scatter plot and save the figure to a file.

    Parameters:
        X (array-like): 1D array of X coordinates.
        Y (array-like): 1D array of Y coordinates.
        values (array-like): Data values used for coloring the scatter plot.
        title (str): Title of the plot.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        colorbar_label (str): Label for the colorbar.
        vmin (float): Minimum value for the colormap.
        vmax (float): Maximum value for the colormap.
        scatter_size (float): Size of the scatter points.
        save_path (str): Path where the figure will be saved.
        aspect (str, optional): Aspect ratio of the plot. Defaults to 'equal'.
        cbar_height (float, optional): Fixed height of the colorbar. Defaults to 0.8.
    """
    # Create a new figure and get the current axes
    fig, ax = plt.subplots()

    sc = ax.scatter(X, Y, c=values, cmap='jet', s=scatter_size, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='both', which='major', direction='in', length=1, width=0.5)

    ax.set_aspect(aspect, adjustable='box')
    ax.set_xlim(np.min(X), np.max(X))
    ax.set_ylim(np.min(Y), np.max(Y))
    # ax.set_xlim(X.min(), X.max())
    # ax.set_ylim(Y.min(), Y.max())

    # === Colorbar with same height as main axis ===
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label(colorbar_label)
    cbar.ax.tick_params(axis='y', direction='in', length=1, width=0.5)

    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def calculate_rmse_mae(y_true, y_pred):
    """
    Calculate RMSE and MAE between true and predicted values.

    Parameters:
        y_true (array-like): Ground truth values, shape (n_samples,)
        y_pred (array-like): Predicted values, shape (n_samples,)

    Returns:
        rmse (float): Root Mean Square Error
        mae (float): Mean Absolute Error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Calculate MAE
    mae = np.mean(np.abs(y_true - y_pred))

    return rmse, mae

def filter_by_x(pred_full, gt_full, a, tol=1e-5):
    """
    Filter the rows of the predicted and ground truth data where the X coordinate equals a.

    Both pred_full and gt_full are expected to be numpy arrays with shape [N, 4], where:
        - Column 0: X coordinates
        - Column 1: Y coordinates
        - Column 2: U component (velocity in X direction)
        - Column 3: V component (velocity in Y direction)

    Parameters:
        pred_full (np.ndarray): Array containing predicted data.
        gt_full (np.ndarray): Array containing ground truth data.
        a (float): The X coordinate value to filter by.
        tol (float, optional): Tolerance for floating-point comparisons. Defaults to 1e-5.

    Returns:
        filtered_pred (np.ndarray): Filtered predicted data with X == a.
        filtered_gt (np.ndarray): Filtered ground truth data with X == a.
    """
    # Use np.isclose to account for floating point inaccuracies when comparing X values
    mask_pred = np.isclose(pred_full[:, 0], a, atol=tol)
    mask_gt = np.isclose(gt_full[:, 0], a, atol=tol)

    filtered_pred = pred_full[mask_pred]
    filtered_gt = gt_full[mask_gt]

    return filtered_pred, filtered_gt


def sort_data_for_plot(x_values_pred, speed_pred):
    """
    Sort x_values_pred in ascending order and reorder speed_pred correspondingly
    to maintain point-wise correspondence for cross-section plotting.

    Parameters:
        x_values_pred (ndarray): x-axis coordinate values, shape (401,)
        speed_pred (ndarray): corresponding y-axis speed values matched with x_values_pred, shape (401,)

    Returns:
        sorted_x (ndarray): sorted x-axis values in ascending order
        sorted_speed (ndarray): corresponding speed values reordered to match sorted x coordinates
    """
    # Get indices that sort x_values_pred in ascending order
    sorted_indices = np.argsort(x_values_pred)
    # Reorder both arrays according to the sorted indices
    sorted_x = x_values_pred[sorted_indices]
    sorted_speed = speed_pred[sorted_indices]

    return sorted_x, sorted_speed

def plot_dual_y_curve(filtered_pred, filtered_gt, title, xlabel, ylabel,
                      pred_label, gt_label, save_path):
    """
    Plot a dual-axis curve to compare the resultant speeds of the predicted and ground truth data
    along a specific cross-section.

    The input arrays (filtered_pred and filtered_gt) should have shape [M, 4]. Both arrays contain:
        - Column 0: X coordinate (identical for the filtered data)
        - Column 1: Y coordinate (used as the X-axis for the curve)
        - Column 2: U component (velocity in X direction)
        - Column 3: V component (velocity in Y direction)

    The resultant speed is computed as the square root of (U^2 + V^2).

    Parameters:
        filtered_pred (np.ndarray): Filtered predicted data.
        filtered_gt (np.ndarray): Filtered ground truth data.
        title (str): Title of the plot.
        xlabel (str): Label for the X-axis.
        pred_label (str): Label for the predicted speed curve (left Y-axis).
        gt_label (str): Label for the ground truth speed curve (right Y-axis).
        save_path (str): File path to save the plot.
    """
    # Compute resultant speeds for predicted and ground truth data
    speed_pred = np.sqrt(filtered_pred[:, 2] ** 2 + filtered_pred[:, 3] ** 2)
    speed_gt = np.sqrt(filtered_gt[:, 2] ** 2 + filtered_gt[:, 3] ** 2)

    # Use the second column (Y coordinates) as the X-axis for the curves
    x_values_pred = filtered_pred[:, 1]
    x_values_gt = filtered_gt[:, 1]

    # sort the data
    x_values_pred_sort, speed_pred_sort = sort_data_for_plot(x_values_pred, speed_pred)
    x_values_gt_sort, speed_gt_sort = sort_data_for_plot(x_values_gt, speed_gt)

    # calculate RMSE and MAE
    rmes, mae = calculate_rmse_mae(speed_gt_sort, speed_pred_sort)
    print(f"The RMSE and MAE values are {rmes} m/s and {mae} m/s.")

    # Create a new figure and primary axis
    fig, ax1 = plt.subplots()
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel, color='black')
    l1, = ax1.plot(x_values_pred_sort, speed_pred_sort, '-', color='tab:blue', label=pred_label)
    ax1.tick_params(axis='y', labelcolor='black', direction='in')

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel(gt_label, color='black')
    l2, = ax2.plot(x_values_gt_sort, speed_gt_sort, '-', color='tab:red', label=gt_label)
    ax2.tick_params(axis='y', labelcolor='black', direction='in')

    # Set plot title and layout
    plt.title(title)
    fig.tight_layout()

    # Set the same y-axis limits for both axes
    min_speed = min(np.min(speed_pred), np.min(speed_gt))
    max_speed = max(np.max(speed_pred), np.max(speed_gt))
    ax1.set_ylim(min_speed, max_speed)
    ax2.set_ylim(min_speed, max_speed)

    # Hide the right y-axis labels
    ax2.yaxis.set_visible(False)

    # Add legend
    lines = [l1, l2]
    labels = [pred_label, gt_label]
    ax1.legend(lines, labels, loc='upper right')

    # Ensure the output directory exists and save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def plot_relative_error_histogram(Up, Ug, save_path, x_range=(0, 1)):
    """
    Plots a histogram of relative errors and saves it to the specified path.

    Parameters:
    Up -- Model predicted values, shape (n_samples,)
    Ug -- Ground truth values, shape (n_samples,)
    save_path -- Path to save the histogram, default is "../relative_error"
    """
    # Ensure inputs are NumPy arrays
    Up = np.asarray(Up)
    Ug = np.asarray(Ug)

    # Data preprocessing
    valid_indices = Ug >= 0.05
    Ug_filtered = Ug[valid_indices]
    Up_filtered = Up[valid_indices]

    # Calculate relative error
    relative_error = np.abs(Up_filtered - Ug_filtered) / (np.abs(Ug_filtered))

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(relative_error, bins=200, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Prediction of resultant velocity')
    plt.xlabel('Relative Error')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set X-axis range
    # plt.xlim(x_range)

    # Save the plot
    plt.savefig(save_path)
    plt.close()

def plot_loss_curves(file_path, save_path="../../resultSaving"):
    """
    Plots training and validation loss curves from a log file and saves the plot.

    Parameters:
    file_path -- Path to the loss log file
    save_path -- Path to save the plot, default is "../../resultSaving"
    """
    # Read the log file
    data = pd.read_csv(file_path, header=0)

    # Extract data
    epochs = data.iloc[:, 0].values
    train_loss = data.iloc[:, 1].values
    val_loss = data.iloc[:, 2].values

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Set plot parameters
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    plt.figure(figsize=(10, 6))

    # Plot training and validation loss
    plt.plot(epochs, train_loss, label='Training Loss', color='blue', linewidth=2)
    plt.plot(epochs, val_loss, label='Validation Loss', color='orange', linewidth=2)

    # Set axis ticks inward
    plt.tick_params(axis='both', direction='in')

    # Add legend, title, and labels
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curves saved to: {save_path}")

def export_to_excel(pred_full, gt_full, ori_full, save_path):
    """
    Export three ndarray variables to an Excel file with each variable in a separate sheet.

    Parameters:
        pred_full (ndarray): Predicted data, shape (N, 4)
        gt_full (ndarray): Ground truth data, shape (M, 4)
        ori_full (ndarray): Original data, shape (K, 4)
        save_path (str): Path to save the Excel file
    """
    # Ensure inputs are NumPy arrays
    pred_full = np.asarray(pred_full)
    gt_full = np.asarray(gt_full)
    ori_full = np.asarray(ori_full)

    # Construct DataFrames for each dataset
    df_pred = pd.DataFrame(pred_full, columns=['X', 'Y', 'U', 'V'])
    df_gt = pd.DataFrame(gt_full, columns=['X', 'Y', 'U', 'V'])
    df_ori = pd.DataFrame(ori_full, columns=['X', 'Y', 'U', 'V'])

    # Write all datasets to an Excel file with separate sheets via ExcelWriter
    with pd.ExcelWriter(save_path) as writer:
        df_pred.to_excel(writer, sheet_name='Predicted', index=False)
        df_gt.to_excel(writer, sheet_name='Ground Truth', index=False)
        df_ori.to_excel(writer, sheet_name='Original', index=False)

    print(f"Data successfully exported to {save_path}")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def density_scatter(x, y, ax=None, bins=20, point_size=16):
    # Default bins:100
    """Generate a density scatter plot with color mapped to probability density, skipping bins with zero density"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # 1. Count total number of points
    total_points = len(x)

    # 2. Compute 2D histogram (point count per bin)
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

    # 3. Calculate probability density = points per bin / total points
    prob_density = counts / total_points

    # 4. Assign probability density of each bin to corresponding data points
    x_bin_idx = np.clip(np.digitize(x, x_edges) - 1, 0, bins - 1)
    y_bin_idx = np.clip(np.digitize(y, y_edges) - 1, 0, bins - 1)
    point_density = prob_density[x_bin_idx, y_bin_idx]

    # 5. Create mask: keep only points with probability density > 0
    valid_mask = point_density > 0
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    density_valid = point_density[valid_mask]

    # 6. Compute color values from probability density
    # colors = -np.log10(density_valid)
    colors = density_valid

    # 7. Normalize colormap (exclude extreme values)
    vmin = np.percentile(colors, 5)  # Use 5th and 95th percentiles
    vmax = np.percentile(colors, 95)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # 8. Draw scatter plot (valid points only)
    sc = ax.scatter(
        x_valid, y_valid,
        c=colors,
        s=point_size,
        alpha=0.6,
        cmap=cm.plasma,
        norm=norm,
        edgecolors='none'
    )

    # 9. Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    # cbar.set_label('-Log$_{10}$(Probability Density)', fontsize=24)
    cbar.set_label('Probability Density', fontsize=24)
    cbar.ax.tick_params(labelsize=24)

    # 10. Print information about skipped points
    skipped_points = total_points - len(x_valid)
    if skipped_points > 0:
        print(f"Note: skipped {skipped_points} points in empty bins ({skipped_points / total_points:.1%} of total)")

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

def interpolate_lowres_to_highres_flat(
    Xo_mask,
    Yo_mask,
    speed_ori_in_region,
    grid_Xo,
    grid_Yo,
    grid_Xp,
    grid_Yp
):
    """
    Interpolate low-resolution scattered data onto a regular grid,
    upscale to high-resolution using bicubic interpolation,
    and return the result as a flattened 1D array.

    Parameters:
        Xo_mask: 1D array of X coordinates of low-res data points
        Yo_mask: 1D array of Y coordinates of low-res data points
        speed_ori_in_region: 1D array of speed values at low-res points
        grid_Xo: 1D array defining X axis of low-resolution grid
        grid_Yo: 1D array defining Y axis of low-resolution grid
        grid_Xp: 1D array defining X axis of high-resolution grid
        grid_Yp: 1D array defining Y axis of high-resolution grid

    Returns:
        1D array of interpolated high-resolution values, flattened
    """
    # Create meshgrid for low-resolution grid
    grid_xo_mesh, grid_yo_mesh = np.meshgrid(grid_Xo, grid_Yo)

    # Interpolate scattered low-res data to low-res grid
    lowres_grid = griddata(
        points=(Xo_mask, Yo_mask),
        values=speed_ori_in_region,
        xi=(grid_xo_mesh, grid_yo_mesh),
        method='linear',
        fill_value=0.0
    )

    # Resize to high-resolution shape using bicubic interpolation
    upsampled = cv2.resize(
        lowres_grid,
        dsize=(grid_Xp.size, grid_Yp.size),
        interpolation=cv2.INTER_CUBIC
    )

    # Flatten and return
    return upsampled.flatten()

# def interpolate_lowres_to_highres_flat(
#         Xo_mask,
#         Yo_mask,
#         speed_ori_in_region,
#         grid_Xo,
#         grid_Yo,
#         grid_Xp,
#         grid_Yp,
#         visualize=False,
#         save_path=None,
#         dpi=300
# ):
#     """
#     Interpolate low-resolution scattered data onto a regular grid,
#     upscale to high-resolution using bicubic interpolation,
#     and return the result as a flattened 1D array.
#
#     Parameters:
#         Xo_mask: 1D array of X coordinates of low-res data points
#         Yo_mask: 1D array of Y coordinates of low-res data points
#         speed_ori_in_region: 1D array of speed values at low-res points
#         grid_Xo: 1D array defining X axis of low-resolution grid
#         grid_Yo: 1D array defining Y axis of low-resolution grid
#         grid_Xp: 1D array defining X axis of high-resolution grid
#         grid_Yp: 1D array defining Y axis of high-resolution grid
#         visualize: bool, whether to generate validation plots
#         save_path: str, directory to save visualization results
#         dpi: int, output image resolution
#
#     Returns:
#         1D array of interpolated high-resolution values, flattened
#     """
#     # Create meshgrid for low-resolution grid
#     grid_xo_mesh, grid_yo_mesh = np.meshgrid(grid_Xo, grid_Yo)
#
#     # Interpolate scattered low-res data to low-res grid
#     lowres_grid = griddata(
#         points=(Xo_mask, Yo_mask),
#         values=speed_ori_in_region,
#         xi=(grid_xo_mesh, grid_yo_mesh),
#         method='linear',
#         fill_value=0.0
#     )
#
#     # Visualization: Low-res interpolation result
#     if visualize:
#         plt.figure(figsize=(10, 6))
#         plt.pcolormesh(grid_xo_mesh, grid_yo_mesh, lowres_grid, shading='auto', cmap='jet')
#         plt.colorbar(label='Velocity (m/s)')
#         plt.title('Low-resolution Interpolation Result')
#         plt.xlabel('X Coordinate')
#         plt.ylabel('Y Coordinate')
#         if save_path:
#             plt.savefig(f"{save_path}_lowres.png", dpi=dpi, bbox_inches='tight')
#         plt.close()
#
#     # Resize to high-resolution shape using bicubic interpolation
#     upsampled = cv2.resize(
#         lowres_grid,
#         dsize=(grid_Xp.size, grid_Yp.size),
#         interpolation=cv2.INTER_CUBIC
#     )
#
#     # Create high-res meshgrid for visualization
#     grid_xp_mesh, grid_yp_mesh = np.meshgrid(grid_Xp, grid_Yp)
#
#     # Visualization: High-res upsampled result
#     if visualize:
#         plt.figure(figsize=(12, 8))
#         plt.pcolormesh(grid_xp_mesh, grid_yp_mesh, upsampled, shading='auto', cmap='jet')
#         plt.colorbar(label='Velocity (m/s)')
#         plt.title('High-resolution Upsampled Result')
#         plt.xlabel('X Coordinate')
#         plt.ylabel('Y Coordinate')
#         if save_path:
#             plt.savefig(f"{save_path}_highres.png", dpi=dpi, bbox_inches='tight')
#         plt.close()
#
#     return upsampled.flatten()

def create_tif_from_points(rows, cols, cellsize_x, cellsize_y,
                           xmin, ymin, xmax, ymax,
                           Xp, Yp, Dp,
                           out_tif,
                           crs_wkt=None,
                           nodata_value=0):
    """
    Create and save a GeoTIFF raster from scattered point data (Xp, Yp, Dp).

    Parameters:
        rows, cols           : number of rows and columns of the output raster
        cellsize_x, cellsize_y : pixel size in CRS units
        xmin, ymin, xmax, ymax : spatial bounding box
        Xp, Yp               : point coordinates (ndarray, N,)
        Dp                   : water depth values at each point (ndarray, N,)
        out_tif              : output GeoTIFF file path
        crs_wkt              : coordinate reference system in WKT format (optional)
        nodata_value         : NoData value for empty pixels, default 0
    """
    # Affine geotransform
    transform = from_origin(xmin, ymax, cellsize_x, cellsize_y)

    # Initialize raster matrix filled with NoData value
    D_matrix = np.full((rows, cols), nodata_value, dtype=np.float32)

    # Helper function: convert geographic coordinates to raster indices
    def coords_to_index(x, y, xmin, ymax, dx, dy):
        col = int((x - xmin) / dx)
        row = int((ymax - y) / dy)
        return row, col

    # Rasterize point values into the grid
    for x, y, d in zip(Xp, Yp, Dp):
        row, col = coords_to_index(x, y, xmin, ymax, cellsize_x, cellsize_y)
        if 0 <= row < rows and 0 <= col < cols:
            D_matrix[row, col] = d

    # Write output GeoTIFF
    with rasterio.open(
            out_tif,
            "w",
            driver="GTiff",
            height=rows,
            width=cols,
            count=1,
            dtype=rasterio.float32,
            crs=crs_wkt,
            transform=transform,
            nodata=nodata_value,
    ) as dst:
        dst.write(D_matrix, 1)

    print(f"[OK] GeoTIFF saved to {out_tif}")
    return D_matrix


def plot_water_overlay(ax, dem_data, water_data, name, title=None):
    """
    Draw DEM hillshade with overlaid classified water depth on a given matplotlib axes.
    """
    # Define classification color palette
    # light_blue = mcolors.to_rgba('#9ecae1')
    # middle_blue = mcolors.to_rgba('#4292c6')
    # deep_blue = mcolors.to_rgba('#084594')

    light_blue = mcolors.to_rgba('#fdd49e')
    middle_blue = mcolors.to_rgba('#fc8c59')
    deep_blue = mcolors.to_rgba('#7f0000')

    # Construct RGBA water depth classification layer
    rows, cols = dem_data.shape
    water_rgba = np.zeros((rows, cols, 4), dtype=float)
    mask1 = (water_data >= 0.1) & (water_data < 0.5)
    mask2 = (water_data >= 0.5) & (water_data < 1.0)
    mask3 = (water_data >= 1.0)
    water_rgba[mask1] = light_blue
    water_rgba[mask2] = middle_blue
    water_rgba[mask3] = deep_blue

    # Render visualization layers
    ls = LightSource(azdeg=315, altdeg=45)  # Light source azimuth and altitude angles
    dem_hillshade = ls.shade(dem_data, cmap=plt.cm.gray, blend_mode='overlay')

    ax.imshow(dem_hillshade)  # Render shaded DEM base layer
    ax.imshow(water_rgba, interpolation='none')  # Overlay classified water layer
    ax.axis('off')

    # Place legend at lower right corner
    legend_elements = [
        Patch(facecolor=light_blue, edgecolor='k', label='0.1–0.5'),
        Patch(facecolor=middle_blue, edgecolor='k', label='0.5–1.0'),
        Patch(facecolor=deep_blue, edgecolor='k', label='>1.0')
    ]
    ax.legend(handles=legend_elements, title="Vel. Mag. (m/s)",
              loc='lower right', frameon=True, framealpha=1,
              fontsize=8, title_fontsize=8).get_frame().set_edgecolor('black')
    # ax.legend(handles=legend_elements, title="Water Depth (m)",
    #           loc='lower right', frameon=True, framealpha=1,
    #           fontsize=8, title_fontsize=8).get_frame().set_edgecolor('black')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    # Add timestamp label at upper left corner
    time_str = name.replace("_", " ")
    # "2020-08-17 14 30" → "2020-08-17 14:30"
    time_str = time_str[:16] + ":" + time_str[17:]

    ax.text(
        0.02, 0.95,
        f"t = {time_str}",
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=8, fontweight='bold'
    )


def plot_water_overlay_Sanjiang(ax, dem_data, water_data, name, title=None):
    """
    Draw DEM hillshade with continuous overlaid water depth on a given matplotlib axes.
    Features:
    1. Automatically render -9999 NoData regions as transparent.
    2. Apply enhanced vertical exaggeration to highlight subtle terrain relief in flat plain areas.
    3. Use continuous rainbow colormap for water depth ranging from 0 to 15 m.
    """

    # --- 1. Handle NoData values and outliers in DEM data ---
    # Mask values less than or equal to -999 as invalid NoData pixels
    dem_masked = np.ma.masked_where(dem_data <= -999, dem_data)

    # Extract valid elevation values to determine colormap range (vmin, vmax)
    # Ensures terrain colormap maps to actual land elevation, not skewed by -9999 values
    valid_dem = dem_masked.compressed()
    if len(valid_dem) > 0:
        dem_vmin = np.percentile(valid_dem, 2)  # Exclude bottom 2% of noise values
        dem_vmax = np.percentile(valid_dem, 98)  # Exclude top 2% of noise values
    else:
        dem_vmin, dem_vmax = 0, 100

    # --- 2. Render terrain base layer with hillshade ---
    # azdeg / altdeg: azimuth and altitude angles of the light source
    ls = LightSource(azdeg=315, altdeg=45)

    # vert_exag: vertical exaggeration factor. Values of 15-30 significantly enhance relief perception for flat plains
    # blend_mode: 'soft' or 'hsv' generally preserves terrain color better than 'overlay'
    dem_hillshade = ls.shade(
        dem_masked,
        cmap=plt.cm.gray,
        blend_mode='soft',
        vert_exag=20,
        vmin=dem_vmin,
        vmax=dem_vmax
    )

    ax.imshow(dem_hillshade)

    # --- 3. Render continuous water depth layer ---
    # Mask both zero water pixels and DEM NoData pixels
    water_masked = np.ma.masked_where((water_data <= 0) | (dem_data <= -999), water_data)

    # Use rainbow colormap with fixed 0-10 m range
    # Alpha around 0.7 allows underlying hillshade to show through, creating the visual effect of water over riverbed
    im = ax.imshow(water_masked,
                   cmap=plt.cm.rainbow,
                   norm=mcolors.Normalize(vmin=0, vmax=10),
                   interpolation='bilinear',
                   alpha=0.7)

    ax.axis('off')

    # --- 4. Add right-side colorbar ---
    # Use make_axes_locatable to ensure colorbar auto-scales with subplot size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Water Depth (m)', fontsize=9, fontweight='bold')
    cbar.ax.tick_params(labelsize=8)

    # --- 5. Title and timestamp label ---
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Format timestamp string for display
    time_str = name.replace("_", " ")
    if len(time_str) >= 19:
        # Format "YYYY-MM-DD HH MM SS" to "YYYY-MM-DD HH:MM"
        time_str = f"{time_str[:10]} {time_str[11:13]}:{time_str[14:16]}"

    ax.text(
        0.03, 0.96,
        f"t = {time_str}",
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=9, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3')
    )

    return im  # Return im object for potential external operations


def process_all_frames(dem_path, pred_dir, true_dir, out_dir,
                       video_name="comparison.mp4", fps=2, dpi=600):
    """
    Iterate over prediction and ground truth raster folders, generate side-by-side comparison frames, and synthesize them into an MP4 video.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load DEM raster
    with rasterio.open(dem_path) as dem_ds:
        dem_data = dem_ds.read(1)

    # Retrieve timestamp list (ensure consistency between two folders)
    files_pred = sorted([f for f in os.listdir(pred_dir) if f.endswith(".tif")])
    files_true = sorted([f for f in os.listdir(true_dir) if f.endswith(".tif")])
    timestamps = [f.replace(".tif", "") for f in files_pred if f in files_true]

    frame_paths = []
    for i, ts in enumerate(timestamps):
        pred_path = os.path.join(pred_dir, ts + ".tif")
        true_path = os.path.join(true_dir, ts + ".tif")

        # Load water depth data
        with rasterio.open(pred_path) as ds:
            pred_data = ds.read(1)
        with rasterio.open(true_path) as ds:
            true_data = ds.read(1)

        # Draw side-by-side subplot comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plot_water_overlay(axes[0], dem_data, pred_data, title="Pred", name=ts)
        plot_water_overlay(axes[1], dem_data, true_data, title="HR", name=ts)

        # Save frame image
        frame_path = os.path.join(out_dir, f"frame_{i:03d}.png")
        plt.savefig(frame_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)

    # Synthesize frames into MP4 video
    video_path = os.path.join(out_dir, video_name)
    with imageio.get_writer(video_path, fps=fps, codec='libx264', quality=8) as writer:
        for frame in frame_paths:
            img = imageio.imread(frame)
            writer.append_data(img)

    print(f"Video generated: {video_path}")


def remove_zero_groundtruth(Xp, Yp, Dp, Up, Xg, Yg, Dg, Ug,
                            tol=0.0, mode='either', return_removed_index=False):
    """
    Remove sample rows where ground truth depth Dg or velocity Ug is zero from multiple 1D numpy arrays.

    Parameters
    ----
    Xp, Yp, Dp, Up, Xg, Yg, Dg, Ug : array-like (all must have the same length)
        Input 1D arrays for predictions and ground truth; converted to numpy.ndarray internally.
    tol : float (default 0.0)
        Tolerance for zero detection. If tol>0, values with |value| < tol are treated as zero (for floating-point near-zero cases).
    mode : {'either', 'both'} (default 'either')
        'either' - remove when Dg==0 OR Ug==0 (logical OR).
        'both' - remove only when Dg==0 AND Ug==0 (logical AND).
    return_removed_index : bool (default False)
        Whether to also return the array of indices of removed samples.

    Returns
    ----
    (Xp_f, Yp_f, Dp_f, Up_f, Xg_f, Yg_f, Dg_f, Ug_f, removed_count, removed_idx)  or
    (Xp_f, Yp_f, Dp_f, Up_f, Xg_f, Yg_f, Dg_f, Ug_f, removed_count)
        Filtered arrays (all numpy.ndarray), plus count of removed samples and optionally removed indices.

    Notes
    ----
    - All input arrays must have identical length, otherwise ValueError is raised.
    - Default uses strict equality to 0; use tol parameter for floating-point near-zero cases, e.g. tol=1e-8.
    """
    # Convert input arrays to numpy arrays
    arrs = [np.asarray(a) for a in (Xp, Yp, Dp, Up, Xg, Yg, Dg, Ug)]
    lengths = [a.size for a in arrs]
    if len(set(lengths)) != 1:
        raise ValueError(f'Input array length mismatch: {lengths}')
    N = lengths[0]

    Xp_a, Yp_a, Dp_a, Up_a, Xg_a, Yg_a, Dg_a, Ug_a = arrs

    # Build zero-value mask (True indicates the sample should be removed)
    if tol > 0:
        zeroD = np.abs(Dg_a) < tol
        zeroU = np.abs(Ug_a) < tol
    else:
        zeroD = Dg_a == 0
        zeroU = Ug_a == 0

    if mode == 'either':
        remove_mask = zeroD | zeroU
    elif mode == 'both':
        remove_mask = zeroD & zeroU
    else:
        raise ValueError("mode must be 'either' or 'both'")

    removed_idx = np.nonzero(remove_mask)[0]  # Indices of removed samples
    removed_count = removed_idx.size

    keep_mask = ~remove_mask
    # Filter each array and return
    filtered = tuple(a[keep_mask] for a in (Xp_a, Yp_a, Dp_a, Up_a, Xg_a, Yg_a, Dg_a, Ug_a))

    if return_removed_index:
        return (*filtered, removed_count, removed_idx)
    else:
        return (*filtered, removed_count)


def remove_zero_groundtruth_single(Xp, Yp, Dp, Xg, Yg, Dg,
                            tol=0.0, return_removed_index=False):
    """
    Remove sample rows where ground truth depth Dg is zero from multiple 1D numpy arrays (depth-only version).

    Parameters
    ----
    Xp, Yp, Dp, Xg, Yg, Dg : array-like (all must have the same length)
        Input 1D arrays for predictions and ground truth; converted to numpy.ndarray internally.
    tol : float (default 0.0)
        Tolerance for zero detection. If tol>0, values with |value| < tol are treated as zero.
    return_removed_index : bool (default False)
        Whether to also return the array of indices of removed samples.

    Returns
    ----
    Filtered arrays plus count of removed samples, and optionally removed indices.
    """
    # Convert input arrays to numpy arrays
    arrs = [np.asarray(a) for a in (Xp, Yp, Dp, Xg, Yg, Dg)]
    lengths = [a.size for a in arrs]
    if len(set(lengths)) != 1:
        raise ValueError(f'Input array length mismatch: {lengths}')
    N = lengths[0]

    Xp_a, Yp_a, Dp_a, Xg_a, Yg_a, Dg_a = arrs

    # Build zero-value mask (True indicates the sample should be removed)
    if tol > 0:
        zeroD = np.abs(Dg_a) < tol
    else:
        zeroD = Dg_a == 0

    remove_mask = zeroD

    removed_idx = np.nonzero(remove_mask)[0]  # Indices of removed samples
    removed_count = removed_idx.size

    keep_mask = ~remove_mask
    # Filter each array and return
    filtered = tuple(a[keep_mask] for a in (Xp_a, Yp_a, Dp_a, Xg_a, Yg_a, Dg_a))

    if return_removed_index:
        return (*filtered, removed_count, removed_idx)
    else:
        return (*filtered, removed_count)


def process_all_frames_Sanjiang(dem_path, pred_dir, true_dir, out_dir,
                       video_name="comparison.mp4", fps=2, dpi=600):
    """
    Iterate over prediction and ground truth folders, generate side-by-side comparison frames, and synthesize them into an MP4 video.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load DEM raster
    with rasterio.open(dem_path) as dem_ds:
        dem_data = dem_ds.read(1)

    # Retrieve timestamp list (ensure consistency between two folders)
    files_pred = sorted([f for f in os.listdir(pred_dir) if f.endswith(".tif")])
    files_true = sorted([f for f in os.listdir(true_dir) if f.endswith(".tif")])
    timestamps = [f.replace(".tif", "") for f in files_pred if f in files_true]

    frame_paths = []
    for i, ts in enumerate(timestamps):
        pred_path = os.path.join(pred_dir, ts + ".tif")
        true_path = os.path.join(true_dir, ts + ".tif")

        # Load water depth data
        with rasterio.open(pred_path) as ds:
            pred_data = ds.read(1)
        with rasterio.open(true_path) as ds:
            true_data = ds.read(1)

        # Draw side-by-side subplot comparison
        fig, axes = plt.subplots(1, 2, figsize=(6, 12))
        plot_water_overlay_Sanjiang(axes[0], dem_data, pred_data, title="Pred", name=ts)
        plot_water_overlay_Sanjiang(axes[1], dem_data, true_data, title="HR", name=ts)

        # Save frame image
        frame_path = os.path.join(out_dir, f"frame_{i:03d}.png")
        plt.savefig(frame_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)

    # Synthesize frames into MP4 video
    video_path = os.path.join(out_dir, video_name)
    with imageio.get_writer(video_path, fps=fps, codec='libx264', quality=8) as writer:
        for frame in frame_paths:
            img = imageio.imread(frame)
            writer.append_data(img)

    print(f"Video generated: {video_path}")


def extract_depth_velocity_stats(low_res_path, high_res_path):
    """
    Extract water depth and flow velocity statistics from the first sample of
    low-resolution and high-resolution HDF5 datasets.

    Parameters:
        low_res_path: Path to the low-resolution HDF5 file (shape: [N, 6])
        high_res_path: Path to the high-resolution HDF5 file (shape: [N, 4])

    Returns:
        stats_dict: Dictionary containing full statistical information
    """

    def get_first_sample_stats(file_path, expected_cols):
        """
        Compute statistical metrics for a target sample from an HDF5 file.

        Parameters:
            file_path: Path to the HDF5 file
            expected_cols: Expected number of feature columns (6 or 4)

        Returns:
            dict: Dictionary containing depth and velocity statistics
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with h5py.File(file_path, 'r') as f:
            # Retrieve all dataset keys
            keys = list(f.keys())
            if not keys:
                raise ValueError(f"No datasets found in file: {file_path}")

            # Retrieve the dataset corresponding to the 20th key
            first_key = keys[19]
            data = f[first_key][:]

            # Validate data dimensions
            if len(data.shape) != 2:
                raise ValueError(f"Dimension mismatch: expected 2D array, got {len(data.shape)}D")

            if data.shape[1] != expected_cols:
                raise ValueError(f"Column count mismatch: expected {expected_cols} columns, got {data.shape[1]} columns")

            # Extract depth and velocity data
            # Data format reference:
            # Low-resolution: [X, Y, depth, velocity, dem, slope] -> depth at index 2, velocity at index 3
            # High-resolution: [X, Y, depth, velocity] -> depth at index 2, velocity at index 3
            depth_data = data[:, 2]  # Third column: water depth
            velocity_data = data[:, 3]  # Fourth column: flow velocity

            # Filter out invalid NoData values if present
            valid_depth = depth_data[depth_data != -9999]
            valid_velocity = velocity_data[velocity_data != -9999]

            # Fall back to raw data if no valid values remain
            if len(valid_depth) == 0:
                valid_depth = depth_data
            if len(valid_velocity) == 0:
                valid_velocity = velocity_data

            # Compute statistical metrics
            depth_min = np.min(valid_depth)
            depth_max = np.max(valid_depth)
            velocity_min = np.min(valid_velocity)
            velocity_max = np.max(valid_velocity)

            # Additional statistics: mean, standard deviation
            depth_mean = np.mean(valid_depth)
            depth_std = np.std(valid_depth)
            velocity_mean = np.mean(valid_velocity)
            velocity_std = np.std(valid_velocity)

            # Calculate ratio of non-zero data points
            depth_nonzero_ratio = np.sum(valid_depth > 0) / len(valid_depth) * 100
            velocity_nonzero_ratio = np.sum(valid_velocity > 0) / len(valid_velocity) * 100

            return {
                'key': first_key,
                'shape': data.shape,
                'depth_min': float(depth_min),
                'depth_max': float(depth_max),
                'depth_mean': float(depth_mean),
                'depth_std': float(depth_std),
                'velocity_min': float(velocity_min),
                'velocity_max': float(velocity_max),
                'velocity_mean': float(velocity_mean),
                'velocity_std': float(velocity_std),
                'depth_nonzero_ratio': float(depth_nonzero_ratio),
                'velocity_nonzero_ratio': float(velocity_nonzero_ratio),
                'sample_count': len(valid_depth)
            }

    try:
        # Compute statistics for low-resolution dataset
        low_res_stats = get_first_sample_stats(low_res_path, expected_cols=6)

        # Compute statistics for high-resolution dataset
        high_res_stats = get_first_sample_stats(high_res_path, expected_cols=4)

        # Build output dictionary
        stats_dict = {
            'low_resolution': low_res_stats,
            'high_resolution': high_res_stats,
            'comparison': {
                'depth_range_ratio': high_res_stats['depth_max'] / low_res_stats['depth_max'] if low_res_stats[
                                                                                                     'depth_max'] > 0 else float(
                    'inf'),
                'velocity_range_ratio': high_res_stats['velocity_max'] / low_res_stats['velocity_max'] if low_res_stats[
                                                                                                              'velocity_max'] > 0 else float(
                    'inf'),
                'depth_mean_ratio': high_res_stats['depth_mean'] / low_res_stats['depth_mean'] if low_res_stats[
                                                                                                      'depth_mean'] > 0 else float(
                    'inf'),
                'velocity_mean_ratio': high_res_stats['velocity_mean'] / low_res_stats['velocity_mean'] if
                low_res_stats['velocity_mean'] > 0 else float('inf')
            }
        }

        return stats_dict

    except Exception as e:
        print(f"Error occurred during processing: {e}")
        raise


def print_stats_summary(stats_dict):
    """
    Print formatted statistical summary of the dataset.
    """
    print("=" * 80)
    print("                  Dataset Statistical Summary")
    print("=" * 80)

    # Low-resolution dataset statistics
    print("\nLow-resolution data (LR):")
    print("-" * 40)
    lr = stats_dict['low_resolution']
    print(f"  Dataset key: {lr['key']}")
    print(f"  Data shape: {lr['shape'][0]} points × {lr['shape'][1]} columns")
    print(f"  Water depth statistics:")
    print(f"     Min: {lr['depth_min']:.6f}")
    print(f"     Max: {lr['depth_max']:.6f}")
    print(f"     Mean: {lr['depth_mean']:.6f} ± {lr['depth_std']:.6f}")
    print(f"     Non-zero ratio: {lr['depth_nonzero_ratio']:.2f}%")
    print(f"  Flow velocity statistics:")
    print(f"     Min: {lr['velocity_min']:.6f}")
    print(f"     Max: {lr['velocity_max']:.6f}")
    print(f"     Mean: {lr['velocity_mean']:.6f} ± {lr['velocity_std']:.6f}")
    print(f"     Non-zero ratio: {lr['velocity_nonzero_ratio']:.2f}%")

    # High-resolution dataset statistics
    print("\nHigh-resolution data (HR):")
    print("-" * 40)
    hr = stats_dict['high_resolution']
    print(f"  Dataset key: {hr['key']}")
    print(f"  Data shape: {hr['shape'][0]} points × {hr['shape'][1]} columns")
    print(f"  Water depth statistics:")
    print(f"     Min: {hr['depth_min']:.6f}")
    print(f"     Max: {hr['depth_max']:.6f}")
    print(f"     Mean: {hr['depth_mean']:.6f} ± {hr['depth_std']:.6f}")
    print(f"     Non-zero ratio: {hr['depth_nonzero_ratio']:.2f}%")
    print(f"  Flow velocity statistics:")
    print(f"     Min: {hr['velocity_min']:.6f}")
    print(f"     Max: {hr['velocity_max']:.6f}")
    print(f"     Mean: {hr['velocity_mean']:.6f} ± {hr['velocity_std']:.6f}")
    print(f"     Non-zero ratio: {hr['velocity_nonzero_ratio']:.2f}%")

    # Cross-resolution comparison metrics
    print("\nCross-resolution comparison:")
    print("-" * 40)
    comp = stats_dict['comparison']
    print(f"  Max depth ratio (HR/LR): {comp['depth_range_ratio']:.4f}")
    print(f"  Max velocity ratio (HR/LR): {comp['velocity_range_ratio']:.4f}")
    print(f"  Mean depth ratio (HR/LR): {comp['depth_mean_ratio']:.4f}")
    print(f"  Mean velocity ratio (HR/LR): {comp['velocity_mean_ratio']:.4f}")

    print("=" * 80)


def save_stats_to_file(stats_dict, output_file="dataset_stats.json"):
    """
    Save dataset statistics to a JSON file.
    """
    import json

    # Convert numpy data types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # Build JSON-serializable dictionary
    serializable_dict = {}
    for key, value in stats_dict.items():
        if isinstance(value, dict):
            serializable_dict[key] = {k: convert_to_serializable(v) for k, v in value.items()}
        else:
            serializable_dict[key] = convert_to_serializable(value)

    # Write statistics to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_dict, f, indent=2, ensure_ascii=False)

    print(f"Statistics saved to: {output_file}")