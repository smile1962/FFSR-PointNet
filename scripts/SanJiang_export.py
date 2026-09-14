import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio
import geopandas as gpd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import warnings


# ============================================================
# Global settings
# ============================================================
warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 18

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


# ============================================================
# Read the ROI mask
# ============================================================
def load_roi_mask(filepath):

    with rasterio.open(filepath) as src:

        roi_mask = src.read(1)

    return roi_mask > 0


# ============================================================
# Match the ROI mask to a data grid by nearest-neighbour sampling
# ============================================================
def match_roi_mask(roi_mask, target_shape):

    if roi_mask.shape == target_shape:
        return roi_mask

    row_idx = np.floor(
        np.linspace(
            0,
            roi_mask.shape[0] - 1,
            target_shape[0]
        )
    ).astype(int)

    col_idx = np.floor(
        np.linspace(
            0,
            roi_mask.shape[1] - 1,
            target_shape[1]
        )
    ).astype(int)

    return roi_mask[np.ix_(row_idx, col_idx)]


# ============================================================
# Read a plain TIFF
# ============================================================
def load_tif(filepath, roi_mask=None, threshold=0.01):
    """
    Read the Depth / Velocity TIFFs

    roi_mask:
        Keep data inside the ROI only; everything outside is set to NaN.

    threshold:
        Values below this are set to NaN. 0.01 is used for both depth and velocity.
    """

    with rasterio.open(filepath) as src:

        data = src.read(1).astype(float)

        nodata = src.nodata

        if nodata is not None:
            data = np.where(
                data == nodata,
                np.nan,
                data
            )

        if threshold is not None:
            data = np.where(
                data < threshold,
                np.nan,
                data
            )

        if roi_mask is not None:

            matched_roi = match_roi_mask(
                roi_mask,
                data.shape
            )

            data = np.where(
                matched_roi,
                data,
                np.nan
            )

        return (
            data,
            src.profile,
            src.bounds
        )


# ============================================================
# Read the hillshade separately
# ============================================================
def load_hillshade(filepath):

    with rasterio.open(filepath) as src:

        data = src.read(1).astype(float)

        nodata = src.nodata

        if nodata is not None:
            data = np.where(
                data == nodata,
                np.nan,
                data
            )

        # Treat zero-valued hillshade cells as transparent background.
        data[data == 0] = np.nan

        return (
            data,
            src.profile,
            src.bounds
        )


# ============================================================
# Obtain the correct extent
# ============================================================
def get_extent(bounds):

    if hasattr(bounds, 'left'):

        return [
            bounds.left,
            bounds.right,
            bounds.bottom,
            bounds.top
        ]

    else:

        return [
            bounds[0],
            bounds[2],
            bounds[1],
            bounds[3]
        ]


# ============================================================
# Data statistics
# ============================================================
def print_statistics(var_type, data_dict):

    print("\n" + "=" * 75)
    print(f"{var_type} DATA STATISTICS")
    print("=" * 75)

    for name, data in data_dict.items():

        valid = data[np.isfinite(data)]

        total = data.size
        valid_num = valid.size
        nan_num = total - valid_num

        if valid_num > 0:

            print(
                f"{name:>8s} | "
                f"shape = {data.shape} | "
                f"valid = {valid_num:,} | "
                f"NaN = {nan_num:,} | "
                f"min = {valid.min():.6f} | "
                f"max = {valid.max():.6f} | "
                f"mean = {valid.mean():.6f}"
            )

        else:

            print(
                f"{name:>8s} | "
                f"shape = {data.shape} | "
                f"NO VALID DATA"
            )

    print("=" * 75)


# ============================================================
# Six-panel figure
# ============================================================
def render_6panel_figure(
        var_type,
        base_path,
        filename,
        shp_path,
        hillshade_path=None,
        roi_path=None):

    print("\n")
    print("#" * 75)
    print(f"Rendering: {var_type}")
    print("#" * 75)

    # ========================================================
    # 1. Folder prefix
    # ========================================================
    prefix = (
        'D'
        if var_type == 'Depth'
        else 'U'
    )

    path_ori = os.path.join(
        base_path,
        f'{prefix}_ori',
        filename
    )

    path_true = os.path.join(
        base_path,
        f'{prefix}_true',
        filename
    )

    path_pred = os.path.join(
        base_path,
        f'{prefix}_pred',
        filename
    )

    path_int = os.path.join(
        base_path,
        f'{prefix}_Int',
        filename
    )

    # ========================================================
    # 2. Data thresholds and ROI
    # ========================================================
    threshold = 0.01

    if roi_path is None or not os.path.exists(roi_path):
        raise FileNotFoundError(
            f"ROI mask not found: {roi_path}"
        )

    roi_mask = load_roi_mask(roi_path)

    # ========================================================
    # 3. Read data
    # ========================================================
    try:

        d_ori, profile, bounds = load_tif(
            path_ori,
            roi_mask,
            threshold
        )

        d_true, _, _ = load_tif(
            path_true,
            roi_mask,
            threshold
        )

        d_pred, _, _ = load_tif(
            path_pred,
            roi_mask,
            threshold
        )

        d_int, _, _ = load_tif(
            path_int,
            roi_mask,
            threshold
        )

    except FileNotFoundError as e:

        print(
            f"[WARN] file not found: {e.filename}"
        )

        print(
            "Using mock data to test the layout."
        )

        d_ori = np.random.rand(
            1000,
            600
        ) * (
            5
            if var_type == 'Depth'
            else 3
        )

        d_true = d_ori.copy()
        d_pred = d_ori.copy()
        d_int = d_ori.copy()

        bounds = (
            0,
            6000,
            0,
            10000
        )

    # ========================================================
    # 4. Data statistics
    # ========================================================
    print_statistics(
        var_type,
        {
            "ori": d_ori,
            "true": d_true,
            "pred": d_pred,
            "int": d_int
        }
    )

    # ========================================================
    # 5. Compute the error
    # ========================================================
    err_pred = np.abs(
        d_pred - d_true
    )

    err_int = np.abs(
        d_int - d_true
    )

    # ========================================================
    # 6. Color scheme
    # ========================================================
    if var_type == 'Depth':

        colors_main = [
            '#00008b',
            '#00bfff',
            '#32cd32',
            '#ffd700',
            '#ff0000'
        ]

        bounds_main = [
            0,
            1,
            2,
            3,
            4,
            5
        ]

        ticks_main = [
            0,
            1,
            2,
            3,
            4,
            5
        ]

        unit = "Depth (m)"

    else:

        colors_main = [
            '#2166ac',
            '#92c5de',
            '#f7f7f7',
            '#f4a582',
            '#b2182b'
        ]

        bounds_main = [
            0,
            1,
            2,
            3,
            4,
            5
        ]

        ticks_main = [
            0,
            1,
            2,
            3,
            4
        ]

        unit = "Vel. Mag. (m/s)"

    cmap_main = ListedColormap(
        colors_main
    )

    norm_main = BoundaryNorm(
        bounds_main,
        cmap_main.N
    )

    # ========================================================
    # 7. Error colorbar
    # ========================================================
    cmap_err = ListedColormap(
        colors_main
    )

    bounds_err = [
        0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0
    ]

    norm_err = BoundaryNorm(
        bounds_err,
        cmap_err.N
    )

    ticks_err = bounds_err

    # ========================================================
    # 8. Shapefile
    # ========================================================
    try:

        gdf_bound = gpd.read_file(
            shp_path
        )

        print(
            f"[OK] Boundary loaded: "
            f"{len(gdf_bound)} geometries"
        )

    except Exception as e:

        print(
            f"[WARN] Boundary loading failed: {e}"
        )

        gdf_bound = None

    # ========================================================
    # 9. Hillshade
    # ========================================================
    hs_data = None
    hs_extent = None

    if (
        hillshade_path is not None
        and os.path.exists(hillshade_path)
    ):

        hs_data, _, hs_bounds = (
            load_hillshade(
                hillshade_path
            )
        )

        hs_extent = get_extent(
            hs_bounds
        )

        print(
            f"[OK] Hillshade loaded: "
            f"{hs_data.shape}"
        )

    else:

        print(
            "[WARN] Hillshade not found."
        )

    # ========================================================
    # 10. Data extent
    # ========================================================
    data_extent = get_extent(
        bounds
    )

    xmin, xmax, ymin, ymax = (
        data_extent
    )

    spatial_width = xmax - xmin
    spatial_height = ymax - ymin

    # ========================================================
    # 11. Compute the true spatial aspect ratio
    #
    # Note:
    # array.shape cannot be used directly.
    # The actual geographic extent should be used.
    # ========================================================
    map_aspect = (
        spatial_height
        / spatial_width
    )

    print(
        f"Data extent: {data_extent}"
    )

    print(
        f"Spatial width : {spatial_width:.3f}"
    )

    print(
        f"Spatial height: {spatial_height:.3f}"
    )

    print(
        f"Spatial aspect: {map_aspect:.4f}"
    )

    # ========================================================
    # 12. Create a 2x3 canvas
    # ========================================================
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13, 13),
        dpi=800
    )

    axes = axes.flatten()

    # ========================================================
    # 13. Layout
    # ========================================================
    fig.subplots_adjust(
        left=0.09,
        right=0.91,
        top=0.950,
        bottom=0.065,
        wspace=0.0,
        hspace=0.10
    )

    # ========================================================
    # 14. Panel information
    # ========================================================
    panels = [

        {
            "data": d_ori,
            "title": f"{unit} - Coarse (CFD)",
            "label": "Input",
            "cmap": cmap_main,
            "norm": norm_main,
            "ticks": ticks_main
        },

        {
            "data": d_true,
            "title": f"{unit} - Fine (CFD)",
            "label": "Ground Truth",
            "cmap": cmap_main,
            "norm": norm_main,
            "ticks": ticks_main
        },

        {
            "data": d_pred,
            "title": f"{unit} - Fine (AI)",
            "label": "AI",
            "cmap": cmap_main,
            "norm": norm_main,
            "ticks": ticks_main
        },

        {
            "data": d_int,
            "title": f"{unit} - Fine (Interp.)",
            "label": "Interpolation",
            "cmap": cmap_main,
            "norm": norm_main,
            "ticks": ticks_main
        },

        {
            "data": err_pred,
            "title": f"{unit} - Error",
            "label": "Error - AI",
            "cmap": cmap_err,
            "norm": norm_err,
            "ticks": ticks_err
        },

        {
            "data": err_int,
            "title": f"{unit} - Error",
            "label": "Error - Interpolation",
            "cmap": cmap_err,
            "norm": norm_err,
            "ticks": ticks_err
        }
    ]

    # ========================================================
    # 15. Plotting
    # ========================================================
    # Keep the colorbar and its title on the same vertical guide.
    # More negative values move both further left, outside the map panel.
    panel_title_x = -0.10
    colorbar_x = -0.10

    for i, (ax, p) in enumerate(
        zip(axes, panels)
    ):

        # ====================================================
        # Axes settings
        # ====================================================
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        # ====================================================
        # Most important:
        #
        # Set the axes aspect from the true spatial extent
        #
        # Do not use set_aspect('equal')
        # ====================================================
        ax.set_box_aspect(
            map_aspect
        )

        # ====================================================
        # 1. Hillshade
        #
        # Explicit aspect='auto'
        # Avoid conflicting with box_aspect
        # ====================================================
        if hs_data is not None:

            ax.imshow(
                hs_data,
                cmap='gray',
                vmin=0,
                vmax=255,
                extent=hs_extent,
                interpolation='bilinear',
                aspect='auto',
                zorder=1
            )

        else:

            ax.set_facecolor(
                '#e0e0e0'
            )

        # ====================================================
        # 2. White mask
        # ====================================================
        if hs_data is not None:

            white_mask = np.ones_like(
                hs_data,
                dtype=float
            )

            ax.imshow(
                white_mask,
                cmap=ListedColormap(
                    ['white']
                ),
                vmin=0,
                vmax=1,
                alpha=0.85,
                extent=hs_extent,
                interpolation='nearest',
                aspect='auto',
                zorder=2
            )

        # ====================================================
        # 3. Core physical fields
        #
        # aspect='auto' is an important fix here
        # ====================================================
        im = ax.imshow(
            p['data'],
            cmap=p['cmap'],
            norm=p['norm'],
            extent=data_extent,
            interpolation='none',
            aspect='auto',
            zorder=3
        )

        # ====================================================
        # 4. Force the actual display extent
        # ====================================================
        ax.set_xlim(
            xmin,
            xmax
        )

        ax.set_ylim(
            ymin,
            ymax
        )

        # ====================================================
        # 5. Boundary
        # ====================================================
        if gdf_bound is not None:

            gdf_bound.plot(
                ax=ax,
                facecolor='none',
                edgecolor='#e1e1e1',
                linewidth=2.5,
                zorder=10
            )

        # ====================================================
        # 6. Title
        # ====================================================
        ax.text(
            panel_title_x,
            1.000,
            p['title'],
            transform=ax.transAxes,
            color='black',
            ha='left',
            va='bottom',
            fontsize=18,
            fontweight='bold',
            zorder=20
        )

        # ====================================================
        # 7. Colorbar
        #
        # Place below the title
        # ====================================================
        cbaxes = inset_axes(
            ax,
            width="4%",
            height="28%",
            loc='upper left',
            bbox_to_anchor=(
                colorbar_x,
                0.0,
                1.0,
                0.94
            ),
            bbox_transform=ax.transAxes,
            borderpad=0
        )

        cbar = plt.colorbar(
            im,
            cax=cbaxes,
            orientation='vertical',
            ticks=p['ticks']
        )

        cbar.ax.tick_params(
            labelsize=16,
            direction='out',
            length=4,
            width=1.5,
            pad=3
        )

        cbar.outline.set_linewidth(
            1.5
        )

        # ====================================================
        # 8. x-y direction
        # ====================================================
        ax.plot(
            [0.86, 0.86, 0.93],
            [0.87, 0.80, 0.80],
            transform=ax.transAxes,
            color='black',
            lw=1.2,
            zorder=20
        )

        ax.text(
            0.86,
            0.885,
            'y',
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=16,
            fontstyle='italic',
            zorder=20
        )

        ax.text(
            0.94,
            0.80,
            'x',
            transform=ax.transAxes,
            ha='left',
            va='center',
            fontsize=16,
            fontstyle='italic',
            zorder=20
        )

        # ====================================================
        # 9. Red label
        # ====================================================
        ax.text(
            0.03,
            0.035,
            p['label'],
            transform=ax.transAxes,
            color='#cc0000',
            ha='left',
            va='bottom',
            fontsize=20,
            fontweight='bold',
            zorder=20
        )

        # ====================================================
        # 10. (a)-(f)
        # ====================================================
        ax.set_xlabel(
            f"({chr(97 + i)})",
            fontsize=20,
            labelpad=2,
            fontweight='bold'
        )

    # ========================================================
    # 16. Save
    # ========================================================
    out_file = os.path.join(
        base_path,
        f'Figure_{var_type}.png'
    )

    plt.savefig(
        out_file,
        dpi=600,
        bbox_inches='tight',
        pad_inches=0.10,
        transparent=False
    )

    print(
        f"\n[OK] {var_type} image saved:"
    )

    print(
        out_file
    )

    plt.close(fig)


# ============================================================
# Main program
# ============================================================
if __name__ == "__main__":

    # ========================================================
    # Sanjiang
    # ========================================================
    BASE_DIR = (
        rf"{ROOT}"
        r"\Results\Sanjiang"
        r"\FLO-SR"
    )

    FILENAME = (
        "2019-08-20_09-20(1).tif"
    )

    SHP_PATH = (
        rf"{ROOT}"
        r"\Data\dataset\geo\sanjiang"
        r"\Watershed_Boundary.shp"
    )

    HILLSHADE_PATH = (
        rf"{ROOT}"
        r"\Data\dataset\geo\sanjiang"
        r"\HillSha_sanjiang.tif"
    )

    ROI_PATH = (
        rf"{ROOT}"
        r"\Data\dataset\geo"
        r"\Shancha_shuixi_1_5_mask_polyfilled.tif"
    )

    # ========================================================
    # Path check
    # ========================================================
    print("=" * 75)
    print("PATH CHECK")
    print("=" * 75)

    print(
        "BASE_DIR      :",
        os.path.exists(BASE_DIR)
    )

    print(
        "SHP_PATH      :",
        os.path.exists(SHP_PATH)
    )

    print(
        "HILLSHADE     :",
        os.path.exists(HILLSHADE_PATH)
    )

    print(
        "ROI MASK      :",
        os.path.exists(ROI_PATH)
    )

    print("=" * 75)

    # ========================================================
    # Depth + Velocity
    # ========================================================
    for v_type in [
        'Depth',
        'Velocity'
    ]:

        render_6panel_figure(
            var_type=v_type,
            base_path=BASE_DIR,
            filename=FILENAME,
            shp_path=SHP_PATH,
            hillshade_path=HILLSHADE_PATH,
            roi_path=ROI_PATH
        )

    print(
        "\nAll figures finished."
    )
