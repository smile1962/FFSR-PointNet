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

# Ignore non-fatal warnings such as CRS mismatch
warnings.filterwarnings("ignore")

# ==========================================
# [Requirement 3] Global font and publication-grade formatting
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 18       # global base font size (adjust freely)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def load_tif(filepath):
    """Read a TIF file, handle NoData and return a numpy array"""
    with rasterio.open(filepath) as src:
        data = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)
        # Set dry cells (tiny values) to NaN so the base map shows through
        data = np.where(data < 0.01, np.nan, data)
        return data, src.profile, src.bounds

def render_6panel_figure(var_type, base_path, filename, shp_path, hillshade_path=None):
    """
    Render the six-panel comparison figure
    :param var_type: 'Depth' or 'Velocity'
    """
    # [Requirement 4] Velocity folder prefix changed to 'U'
    prefix = 'D' if var_type == 'Depth' else 'U'
    
    # Build the input paths
    path_ori = os.path.join(base_path, f'{prefix}_ori', filename)
    path_true = os.path.join(base_path, f'{prefix}_true', filename)
    path_pred = os.path.join(base_path, f'{prefix}_pred', filename)
    path_int = os.path.join(base_path, f'{prefix}_Int', filename)
    
    # Read the core data
    try:
        d_ori, profile, bounds = load_tif(path_ori)
        d_true, _, _ = load_tif(path_true)
        d_pred, _, _ = load_tif(path_pred)
        d_int, _, _ = load_tif(path_int)
    except FileNotFoundError as e:
        print(f"[WARN] file not found: {e.filename}. Using mock data to demonstrate the layout...")
        d_ori, d_true, d_pred, d_int = [np.random.rand(500, 500) * (5 if var_type == 'Depth' else 3) for _ in range(4)]
        bounds = (0, 0, 100, 100)
    
    # Compute the absolute error
    err_pred = np.abs(d_pred - d_true)
    err_int = np.abs(d_int - d_true)
    
    # Define the colour scheme
    if var_type == 'Depth':
        colors_main = ['#00008b', '#00bfff', '#32cd32', '#ffd700', '#ff0000']
        bounds_main = [0, 1, 2, 3, 4, 5]
        ticks_main = bounds_main
        unit = "Depth (m)"
    else:
        # [Requirement 7] The velocity field uses the 5-band ParaView (RdBu_r) style palette
        colors_main = ['#2166ac', '#92c5de', '#f7f7f7', '#f4a582', '#b2182b']
        # Set suitable bounds here to match the colourbar in the attachment (0-3 and above)
        bounds_main = [0, 0.75, 1.5, 2.25, 3.0, 3.75]
        ticks_main = [0, 1, 2, 3] # show only 0,1,2,3 on the colourbar
        unit = "Vel. Mag. (m/s)"
        
    cmap_main = ListedColormap(colors_main)
    norm_main = BoundaryNorm(bounds_main, cmap_main.N)
    
    # [Requirement 6] Force the min and max of both absolute-error colorbars to 0-1
    cmap_err = ListedColormap(colors_main) 
    bounds_err = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    norm_err = BoundaryNorm(bounds_err, cmap_err.N)
    ticks_err = bounds_err

    # Read the shapefile (boundary)
    try:
        gdf_bound = gpd.read_file(shp_path)
    except Exception:
        gdf_bound = None

    # Initialize the canvas
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), dpi=800)
    fig.subplots_adjust(hspace=0.16, wspace=0.005, left=0.005, right=0.995, top=0.995, bottom=0.055)
    axes = axes.flatten()
    
    panels = [
        {"data": d_ori,  "title": f"{unit} - Coarse (CFD)", "label": "Input", "cmap": cmap_main, "norm": norm_main, "ticks": ticks_main},
        {"data": d_true, "title": f"{unit} - Fine (CFD)",    "label": "Ground Truth", "cmap": cmap_main, "norm": norm_main, "ticks": ticks_main},
        {"data": d_pred, "title": f"{unit} - Fine (AI)",     "label": "AI", "cmap": cmap_main, "norm": norm_main, "ticks": ticks_main},
        {"data": d_int,  "title": f"{unit} - Fine (Interp.)", "label": "Interpolation", "cmap": cmap_main, "norm": norm_main, "ticks": ticks_main},
        {"data": err_pred, "title": f"{unit} - Error",        "label": "Error - AI", "cmap": cmap_err, "norm": norm_err, "ticks": ticks_err},
        {"data": err_int,  "title": f"{unit} - Error",        "label": "Error - Interpolation", "cmap": cmap_err, "norm": norm_err, "ticks": ticks_err}
    ]

    extent = [bounds[0], bounds[2], bounds[1], bounds[3]] if type(bounds) != tuple else (0, 100, 0, 100)

    for i, (ax, p) in enumerate(zip(axes, panels)):
        # Do not use ax.axis('off'); hide the frame and ticks instead, keeping the axes so (a)-(f) can be added below
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        # 1. Draw the hillshade base map
        if hillshade_path and os.path.exists(hillshade_path):
            hs_data, _, _ = load_tif(hillshade_path)
            ax.imshow(hs_data, cmap='gray', vmin=0, vmax=255, extent=extent)
        else:
            ax.set_facecolor('#e0e0e0')
            
        # 2. Draw a pure-white semi-transparent mask
        # [Requirement 1] alpha changed to 0.2 here
        white_mask = np.ones_like(hs_data, dtype=float)
        ax.imshow(
            white_mask,
            cmap=ListedColormap(['white']),
            vmin=0,
            vmax=1,
            alpha=0.85,       # recommended 0.25-0.45
            extent=extent,
            zorder=2
        )
        
        # 3. Draw the core physical fields
        im = ax.imshow(p['data'], cmap=p['cmap'], norm=p['norm'], extent=extent, zorder=3, interpolation='none')
        
        # 4. Overlay the boundary shapefile
        # [Requirement 2] How to change boundaries: edgecolor sets the colour ('#404040' dark grey, 'black' pure black), linewidth sets the thickness
        if gdf_bound is not None:
            gdf_bound.plot(ax=ax, facecolor='none', edgecolor='#e1e1e1', linewidth=2.5, zorder=4)

        # 5. Top-left: title text
        ax.text(0.02, 0.98, p['title'], transform=ax.transAxes, color='black', 
                ha='left', va='top', fontsize=18, fontweight='bold', zorder=5)

        # 6. Top-left: inset colorbar
        cbaxes = inset_axes(ax, width="4%", height="32%", loc=2, 
                    bbox_to_anchor=(0.02, -0.18, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cbar = plt.colorbar(im, cax=cbaxes, orientation='vertical', ticks=p['ticks'])
        cbar.ax.tick_params(labelsize=16, direction='out', length=4, width=1.5)
        cbar.outline.set_linewidth(1.5)

        # 7. Top-right: custom L-shaped north arrow / coordinate axes
        ax.plot([0.88, 0.88, 0.94], [0.94, 0.88, 0.88], transform=ax.transAxes, color='black', lw=1.2, zorder=5)
        ax.text(0.88, 0.95, 'y', transform=ax.transAxes, ha='center', va='bottom', fontsize=16, fontstyle='italic', zorder=5)
        ax.text(0.95, 0.88, 'x', transform=ax.transAxes, ha='left', va='center', fontsize=16, fontstyle='italic', zorder=5)

        # 8. Bottom-right: red category label
        ax.text(0.98, 0.05, p['label'], transform=ax.transAxes, color='#cc0000', 
                ha='right', va='bottom', fontsize=20, fontweight='bold', zorder=5)
                
        # 9. Append (a), (b), ... directly below each subplot
        # [Requirement 5] Use xlabel to place the labels directly below, with a set spacing (labelpad)
        ax.set_xlabel(f"({chr(97+i)})", fontsize=20, labelpad=10, fontweight='bold')

    # Save the high-resolution image
    out_file = os.path.join(base_path, f'Figure_{var_type}.png')
    plt.savefig(out_file, dpi=600, bbox_inches='tight', transparent=False)
    print(f"[OK] publication-grade {var_type} image exported: {out_file}")
    plt.close()

# ==========================================
# Run the configuration block
# ==========================================
if __name__ == "__main__":
    BASE_DIR = rf"{ROOT}\Results\SR-Unet"
    FILENAME = "2019-08-20_04_00.tif"
    SHP_PATH = rf"{ROOT}\Data\dataset\geo\shouxi\watershedBoundary.shp"
    HILLSHADE_PATH = rf"{ROOT}\Data\dataset\geo\shouxi\HillSha_dem_1_masked.tif" 
    
    # Detect automatically and render
    for v_type in ['Depth', 'Velocity']:
        prefix = 'D' if v_type == 'Depth' else 'U'
        target_dir = os.path.join(BASE_DIR, f'{prefix}_ori')
        # if os.path.exists(target_dir):  # uncomment to run only when real data exists
        render_6panel_figure(v_type, BASE_DIR, FILENAME, SHP_PATH, HILLSHADE_PATH)