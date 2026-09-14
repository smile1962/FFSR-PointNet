import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio
import geopandas as gpd
import warnings
from rasterio.warp import reproject, Resampling


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
# Layout parameters
#
# All positions are given as axes fractions (0-1),
# To fine-tune the layout just change the numbers here; the plotting logic below need not be touched.
# ============================================================

# -------- Canvas --------
FIG_WIDTH = 13.0          # canvas width (inches)

GRID_LEFT = 0.010
GRID_RIGHT = 0.990
GRID_TOP = 0.970
GRID_BOTTOM = 0.030

WSPACE = 0.02             # column spacing
HSPACE = 0.02             # row spacing

# Whitespace reserved above and below each row axis for the title / (a)-(f), in inches.
# The canvas height is derived from the true map aspect ratio; the smaller this value,
# The smaller the gap between the first and second rows (0.30 is about half the original).
ROW_TEXT_PAD = 0.26

# -------- Colorbar (top-left) --------
CBAR_X = 0.025            # left edge: larger means further right
CBAR_TOP = 0.900          # top edge: smaller means lower
CBAR_WIDTH_FRAC = 0.040   # width (fraction of the panel width)
CBAR_HEIGHT_FRAC = 0.263  # height (fraction of the panel height)

# -------- x-y direction indicator (top-right) --------
DIR_X = 0.860             # x of the vertical arrow
DIR_Y = 0.400             # origin y of the direction indicator: smaller means lower
DIR_LEN = 0.070           # arrow length
DIR_TEXT_GAP = 0.012      # gap between the text and the arrow tip

# -------- Red variable label (bottom-right) --------
LABEL_X = 0.970
LABEL_Y = 0.035


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
def load_tif(filepath, roi_path=None, threshold=0.01):
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
        profile = src.profile
        bounds = src.bounds
        
        # 1. Handle NoData
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)

        # 2. Apply thresholds
        if threshold is not None:
            data = np.where(data < threshold, np.nan, data)

        # 3. Match the ROI by resampling in the geographic projection
        if roi_path is not None and os.path.exists(roi_path):
            with rasterio.open(roi_path) as roi_src:
                roi_data = roi_src.read(1)
                matched_roi = np.zeros(data.shape, dtype=roi_data.dtype)
                
                # Resample the ROI strictly to the coordinates and resolution of the current TIFF
                reproject(
                    source=roi_data,
                    destination=matched_roi,
                    src_transform=roi_src.transform,
                    src_crs=roi_src.crs,
                    dst_transform=src.transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest
                )
                
                # Cells greater than 0 are treated as inside the mask
                matched_roi_mask = matched_roi > 0
                data = np.where(matched_roi_mask, data, np.nan)

        return data, profile, bounds


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
        roi_path=None,
        out_dir=None,
        out_name=None):

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
    threshold = 0.05

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
            2,
            4,
            6,
            8,
            10
        ]

        ticks_main = [
            0,
            2,
            4,
            6,
            8,
            10
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
            2.5,
            5,
            7.5,
            10,
            12.5
        ]

        ticks_main = [
            0,
            2.5,
            5,
            7.5,
            10,
            12.5
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
    #
    # The canvas height is no longer hard-coded:
    # First compute the height of a single panel from the true spatial aspect ratio,
    # Plus the whitespace needed for the row titles / (a)-(f) (ROW_TEXT_PAD),
    # Derive the total canvas height.
    # This avoids leaving a large gap between the two rows of maps.
    # ========================================================
    n_rows, n_cols = 2, 3

    cell_w = (
        FIG_WIDTH
        * (GRID_RIGHT - GRID_LEFT)
        / (n_cols + WSPACE * (n_cols - 1))
    )

    axes_h = map_aspect * cell_w

    cell_h = (
        axes_h
        + 2.0 * ROW_TEXT_PAD
    )

    fig_h = (
        cell_h
        * (n_rows + HSPACE * (n_rows - 1))
        / (GRID_TOP - GRID_BOTTOM)
    )

    print(
        f"Panel size  : "
        f"{cell_w:.3f} x {axes_h:.3f} in"
    )

    print(
        f"Figure size : "
        f"{FIG_WIDTH:.3f} x {fig_h:.3f} in"
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(FIG_WIDTH, fig_h),
        dpi=800
    )

    axes = axes.flatten()

    # ========================================================
    # 13. Layout
    # ========================================================
    fig.subplots_adjust(
        left=GRID_LEFT,
        right=GRID_RIGHT,
        top=GRID_TOP,
        bottom=GRID_BOTTOM,
        wspace=WSPACE,
        hspace=HSPACE
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
    # The left edge of the title aligns with the left edge of the panel.
    panel_title_x = 0

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
        # Positions are given directly as axes fractions: [left, bottom, width, height]
        # CBAR_X controls the horizontal position, CBAR_TOP the vertical position.
        # ====================================================
        cbaxes = ax.inset_axes(
            [
                CBAR_X,
                CBAR_TOP - CBAR_HEIGHT_FRAC,
                CBAR_WIDTH_FRAC,
                CBAR_HEIGHT_FRAC
            ],
            transform=ax.transAxes,
            zorder=30
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
        # 8. x-y direction indicator (top-right; DIR_Y controls the vertical position)
        # ====================================================
        ax.plot(
            [DIR_X, DIR_X, DIR_X + DIR_LEN],
            [DIR_Y + DIR_LEN, DIR_Y, DIR_Y],
            transform=ax.transAxes,
            color='black',
            lw=1.2,
            zorder=20
        )

        ax.text(
            DIR_X,
            DIR_Y + DIR_LEN + DIR_TEXT_GAP,
            'y',
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=16,
            fontstyle='italic',
            zorder=20
        )

        ax.text(
            DIR_X + DIR_LEN + DIR_TEXT_GAP,
            DIR_Y,
            'x',
            transform=ax.transAxes,
            ha='left',
            va='center',
            fontsize=16,
            fontstyle='italic',
            zorder=20
        )

        # ====================================================
        # 9. Red label (bottom-right)
        # ====================================================
        ax.text(
            LABEL_X,
            LABEL_Y,
            p['label'],
            transform=ax.transAxes,
            color='#cc0000',
            ha='right',
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
    #
    # out_dir / out_name are used by the batch script:
    # When run standalone these two arguments are not passed and the behaviour is unchanged
    # (saved as base_path/Figure_{var_type}.png).
    # ========================================================
    if out_dir is None:
        out_dir = base_path

    if out_name is None:
        out_name = f'Figure_{var_type}.png'

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    out_file = os.path.join(
        out_dir,
        out_name
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
        r"\Results\Transfer"
        r"\laoyangcun_light"
    )

    FILENAME = (
        "2026-09-07_06-00.tif"
    )

    SHP_PATH = (
        rf"{ROOT}"
        r"\Data\dataset\geo\laoyangcun"
        r"\boundary.shp"
    )

    HILLSHADE_PATH = (
        rf"{ROOT}"
        r"\Data\dataset\geo\laoyangcun"
        r"\HillShade.tif"
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
