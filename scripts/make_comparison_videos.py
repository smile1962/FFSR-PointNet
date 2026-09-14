# -*- coding: utf-8 -*-
"""
================================================================================
FFSR-PointNet comparison videos - one-click generation script
================================================================================

Features
----
Read the depth (D) / velocity (U) time-series tifs (HR truth / Pred prediction / Int interpolation / Error),
Draw a 2x2 comparison map per time step (DEM base map + watershed boundary + discrete bands + timestamp), then compose an mp4.
A single run outputs both the water-depth and the velocity videos.

Usage
----
1. Only modify the paths in the configuration block below;
2. Run directly:
       "E:/Anaconda/envs/pytorch/python.exe" make_comparison_videos.py
   Quick self-check (render only 2 frames per field, no video):
       ... make_comparison_videos.py --test
   Run only selected fields / limit the frame count:
       ... make_comparison_videos.py --fields D
       ... make_comparison_videos.py --limit 10

Output
----
    <OUT_ROOT>/comparison_frames_D/frame_000.png ...  +  <OUT_ROOT>/Compare_Depth_<CASE>.mp4
    <OUT_ROOT>/comparison_frames_U/frame_000.png ...  +  <OUT_ROOT>/Compare_Velocity_<CASE>.mp4

Directory conventions
--------
    RESULT_ROOT/{D,U}_true     ground truth
    RESULT_ROOT/{D,U}_pred     prediction
    RESULT_ROOT/{D,U}_error    error
    INT_ROOT   /{D,U}_interpolated   interpolation
    (If the interpolation data also lives under RESULT_ROOT, simply set INT_ROOT equal to RESULT_ROOT.)

Dependencies
----
    numpy, matplotlib, rasterio, geopandas, opencv-python
    Recommended environment: E:\\Anaconda\\envs\\pytorch
================================================================================
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import os
import sys
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless backend, suitable for servers and batch jobs
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import array_bounds
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
import cv2


# ================================ Configuration ================================
# ---- 1. Data paths (edit these) ----
RESULT_ROOT = rf"{ROOT}\Results\FFSR-PointNet-Light"
INT_ROOT    = r"E:\Project\Shouxi\SR-PointNet\shouxi"
DEM_PATH    = r"E:\Project\Shouxi\dataset\sourceData\geo\shouxi\HillSha_dem_1_masked.tif"
SHP_PATH    = r"E:\Project\Shouxi\dataset\sourceData\geo\shouxi\watershedBoundary.shp"

# ---- 2. Output ----
OUT_ROOT  = None        # None -> write to RESULT_ROOT
CASE_NAME = "Shouxi"    # video filename suffix: Compare_Depth_<CASE_NAME>.mp4

# ---- 3. Which fields to generate ----
FIELDS = ["D", "U"]     # 'D' water depth, 'U' velocity

# ---- 4. Video / rendering parameters ----
FPS       = 4           # frame rate
DPI       = 600         # frame resolution (submission grade)
CLIP_MAX  = 100.0       # values above this are treated as invalid and set to 0 (removes interpolation fill values); None disables
FONT_SIZE = 24          # subplot title / colorbar font size
RESUME    = True        # True: skip frames that already exist so an interrupted run can be resumed

# ---- 5. Colour band levels ----
BIN_EDGES       = [0.5, 1.0, 1.5, 2.0]      # main panels HR / Pred / Int
ERROR_BIN_EDGES = [0.1, 0.2, 0.3, 0.4]      # Error panels
# =======================================================================


# ---------------------------- Style constants ----------------------------
PALETTES = {
    "D": ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#3182bd"],   # blue palette
    "U": ["#fff5eb", "#fee6ce", "#fdae6b", "#f16913", "#a63603"],   # orange palette
}
ERROR_COLORS = ["#f7fcfd", "#e0ecf4", "#c7e9c0", "#fee391", "#fdbb84"]
FIELD_META = {
    "D": dict(tag="Depth",    cbar_label="Water Depth (m)"),
    "U": dict(tag="Velocity", cbar_label="Vel. Mag. (m/s)"),
}
# Panel order: title -> data source key
PANELS = [("HR", "true"), ("Pred", "pred"), ("Int", "int"), ("Error", "error")]

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"


# ---------------------------- Helper functions ----------------------------
def read_tif(path, clip_max=None):
    """Read a single-band tif as a float array; clip_max removes anomalous fill values."""
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float64)
    if clip_max is not None:
        arr = np.where(arr > clip_max, 0.0, arr)
    return arr


def fmt_time(name):
    """'2019-08-20_10_00' -> '2019-08-20 10:00'"""
    s = str(name).replace("_", " ")
    if len(s) >= 19:
        s = f"{s[:10]} {s[11:13]}:{s[14:16]}"
    return s


def plot_panel(ax, dem_nan, dem_vmin, dem_vmax, extent, gdf,
               water, ts, title, bin_edges, colors, cbar_label,
               fontname="Times New Roman", fontsize=24,
               alpha_bg=0.45, alpha_boundary=0.80, alpha_band=0.95):
    """
    Draw one panel on the given ax: DEM base map -> watershed boundary -> discrete-band depth/velocity layer -> colorbar -> title/timestamp.
    (Adapted from the original plot_water_overlay_Sanjiang, removing the per-frame shapefile reload.)

    dem_nan : plain float array with NaN at invalid positions (rendered transparent by matplotlib)
    """
    # 1) DEM base map
    ax.imshow(dem_nan, cmap="gray_r", vmin=dem_vmin, vmax=dem_vmax,
              alpha=alpha_bg, origin="upper", extent=extent, zorder=1)

    # 2) Watershed boundary (gdf is read once by the caller and reused)
    if gdf is not None:
        gdf.plot(ax=ax, facecolor=(1.0, 1.0, 1.0, alpha_boundary),
                 edgecolor=(0.82, 0.82, 0.82, 1.0), linewidth=2, zorder=2)

    # 3) Water mask: skip cells that are <=0 or where the DEM is NaN
    water = np.asarray(water, dtype=float)
    water_masked = np.ma.masked_where((water <= 0) | np.isnan(dem_nan), water)
    depth = np.full(water.shape, np.nan, dtype=float)
    if np.ma.is_masked(water_masked):
        depth[~water_masked.mask] = water_masked[~water_masked.mask]
    else:
        depth = water_masked.astype(float)

    # 4) Discrete colour bands: edges = [0] + bin_edges + [dynamic upper bound]
    user_edges = np.asarray(bin_edges, dtype=float)
    min_edge = 0.0
    last_lower = float(user_edges[-1])
    finite = np.isfinite(depth)
    data_max = float(np.nanmax(depth[finite])) if finite.any() else last_lower + 1.0
    upper = max(data_max * 1.05, last_lower + 10)
    edges = np.concatenate(([min_edge], user_edges, [upper]))

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(edges, ncolors=cmap.N, clip=False)

    im = ax.imshow(depth, cmap=cmap, norm=norm, interpolation="nearest",
                   alpha=alpha_band, origin="upper", extent=extent, zorder=3)

    # 5) Colourbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, boundaries=edges, spacing="uniform")
    ticks = np.concatenate(([min_edge], user_edges))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:g}" for t in ticks])
    cbar.set_label(cbar_label, fontsize=fontsize, fontname=fontname)
    cbar.ax.tick_params(labelsize=int(fontsize * 0.8))
    for tl in cbar.ax.get_yticklabels():
        tl.set_fontname(fontname)
        tl.set_fontsize(int(fontsize * 0.8))

    # 6) Decoration: title and top-left timestamp
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fontsize, fontweight="bold", fontname=fontname)
    ax.text(0.03, 0.96, f"t = {fmt_time(ts)}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=fontsize * 0.6, fontweight="bold", fontname=fontname,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none",
                      boxstyle="round,pad=0.3"))
    return im


def frames_to_video(frame_paths, video_path, fps=4, retries=3):
    """
    Compose the PNG frame sequence into an mp4 with OpenCV (mp4v codec).
    Verify after writing that the file exists and is non-empty (on Windows antivirus or indexing can occasionally make writes disappear), retrying if necessary.
    """
    if not frame_paths:
        raise ValueError("frame_paths is empty; cannot compose the video")
    first = cv2.imread(frame_paths[0])
    if first is None:
        raise IOError(f"cannot read frame: {frame_paths[0]}")
    h, w = first.shape[:2]

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except OSError:
            pass

        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            last_err = IOError(f"VideoWriter failed to open: {video_path}")
            time.sleep(1.0)
            continue
        written = 0
        for p in frame_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)
            written += 1
        writer.release()
        del writer

        # Verify: the file exists and has a reasonable size (>10 KB)
        if os.path.exists(video_path) and os.path.getsize(video_path) > 10 * 1024:
            return video_path
        last_err = IOError(
            f"video write verification failed (attempt {attempt}): {video_path} "
            f"exists={os.path.exists(video_path)} "
            f"size={os.path.getsize(video_path) if os.path.exists(video_path) else 0} "
            f"frames_written={written}")
        time.sleep(1.5)

    raise last_err


# ---------------------------- Main workflow ----------------------------
def process_field(field, args):
    """Render all frames for one field (D or U) and compose the video."""
    field = field.upper()
    if field not in FIELD_META:
        raise ValueError(f"unknown field type: {field} (only 'D' and 'U' are supported)")
    meta = FIELD_META[field]
    out_root = OUT_ROOT or RESULT_ROOT
    frame_dir = os.path.join(out_root, f"comparison_frames_{field}")
    video_path = os.path.join(out_root, f"Compare_{meta['tag']}_{CASE_NAME}.mp4")

    print(f"\n{'=' * 66}\n  {meta['tag']}  (field={field})   →  {frame_dir}\n{'=' * 66}")

    # ---- Directories ----
    src = {
        "true":  os.path.join(RESULT_ROOT, f"{field}_true"),
        "pred":  os.path.join(RESULT_ROOT, f"{field}_pred"),
        "error": os.path.join(RESULT_ROOT, f"{field}_error"),
        "int":   os.path.join(INT_ROOT,    f"{field}_interpolated"),
    }
    for k, d in src.items():
        if not os.path.isdir(d):
            raise FileNotFoundError(f"[{field}] {k} directory does not exist: {d}")
    os.makedirs(frame_dir, exist_ok=True)

    # ---- DEM / watershed boundary: read once ----
    with rasterio.open(DEM_PATH) as ds:
        dem_data = ds.read(1)
        dem_transform = ds.transform
        dem_crs = ds.crs
    dem_float = np.asarray(dem_data, dtype=float)
    dem_nan = np.where(np.isfinite(dem_float), dem_float, np.nan)
    valid_dem = dem_nan[np.isfinite(dem_nan)]
    if valid_dem.size:
        dem_vmin, dem_vmax = np.percentile(valid_dem, 2), np.percentile(valid_dem, 98)
    else:
        dem_vmin, dem_vmax = 0, 100
    left, bottom, right, top = array_bounds(dem_data.shape[0], dem_data.shape[1], dem_transform)
    extent = [left, right, bottom, top]

    gdf = None
    if SHP_PATH and os.path.exists(SHP_PATH):
        gdf = gpd.read_file(SHP_PATH)
        if gdf.crs is not None and dem_crs is not None and gdf.crs != dem_crs:
            gdf = gdf.to_crs(dem_crs)
    else:
        print(f"  Note: watershed boundary shp not found ({SHP_PATH}); skipping the boundary overlay")

    # ---- Time steps (intersection of pred and true) ----
    ts_list = sorted(
        f[:-4] for f in os.listdir(src["pred"]) if f.endswith(".tif")
    )
    ts_list = [t for t in ts_list if os.path.exists(os.path.join(src["true"], t + ".tif"))]
    if args.limit:
        ts_list = ts_list[:args.limit]
    if not ts_list:
        raise RuntimeError(f"[{field}] no usable time steps (check that the tif filenames match one-to-one)")
    print(f"  Time steps: {len(ts_list)}  ({ts_list[0]} -> {ts_list[-1]})")

    # ---- Render frame by frame ----
    frame_paths, t0, done = [], time.time(), 0
    for i, ts in enumerate(ts_list):
        fp = os.path.join(frame_dir, f"frame_{i:03d}.png")
        frame_paths.append(fp)
        if RESUME and os.path.exists(fp) and os.path.getsize(fp) > 0:
            done += 1
            continue

        data = {k: read_tif(os.path.join(d, ts + ".tif"), CLIP_MAX) for k, d in src.items()}

        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        for ax, (title, key) in zip(axes.ravel(), PANELS):
            edges = ERROR_BIN_EDGES if key == "error" else BIN_EDGES
            colors = ERROR_COLORS if key == "error" else PALETTES[field]
            plot_panel(ax, dem_nan, dem_vmin, dem_vmax, extent, gdf,
                       data[key], ts, title, edges, colors,
                       meta["cbar_label"], fontsize=FONT_SIZE)
        plt.subplots_adjust(wspace=0.05, hspace=0.1)
        plt.savefig(fp, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        del data

        done += 1
        el = time.time() - t0
        eta = el / max(done, 1) * (len(ts_list) - i - 1)
        print(f"  [{i + 1}/{len(ts_list)}] {ts}   "
              f"elapsed {el / 60:.1f} min, estimated remaining {eta / 60:.1f} min")

    # ---- Compose video ----
    if args.test:
        print(f"  [--test] skipping video composition. Frames written to: {frame_dir}")
        return None
    frames_to_video(frame_paths, video_path, fps=FPS)
    size_mb = os.path.getsize(video_path) / 1024 / 1024
    print(f"  [OK] video saved: {video_path}  ({len(frame_paths)} frames @ {FPS} fps, {size_mb:.1f} MB)")
    return video_path


def main():
    ap = argparse.ArgumentParser(description="FFSR-PointNet comparison video one-click generator")
    ap.add_argument("--fields", nargs="+", default=FIELDS,
                    help="fields to generate, e.g. --fields D U (defaults to FIELDS in the configuration block)")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N time steps per field (0 = all)")
    ap.add_argument("--test", action="store_true",
                    help="self-test mode: do not compose the video (use with --limit 2 for a quick check)")
    args = ap.parse_args()

    print("=" * 66)
    print("  FFSR-PointNet comparison video generation")
    print(f"  Result dir : {RESULT_ROOT}")
    print(f"  Interpolation dir : {INT_ROOT}")
    print(f"  Output dir : {OUT_ROOT or RESULT_ROOT}")
    print(f"  Fields     : {args.fields}   |  fps={FPS}  dpi={DPI}")
    print("=" * 66)

    t_all = time.time()
    produced = []
    for fld in args.fields:
        produced.append(process_field(fld, args))

    print(f"\n{'=' * 66}")
    print(f"  All done in {(time.time() - t_all) / 60:.1f} min")
    for p in produced:
        if not p:
            continue
        if os.path.exists(p) and os.path.getsize(p) > 10 * 1024:
            print(f"    ✅ {p}  ({os.path.getsize(p) / 1024 / 1024:.1f} MB)")
        else:
            print(f"    [FAIL] output missing or invalid: {p}")
    print("=" * 66)


if __name__ == "__main__":
    main()
