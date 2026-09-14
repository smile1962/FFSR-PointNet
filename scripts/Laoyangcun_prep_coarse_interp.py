"""
Laoyangcun data preparation: coarsening (20 m) + linear-interpolation baseline
==========================================================

Step 1
    D_ori / U_ori currently hold the 1.5 m fine-grid results.
    This script re-grids it to 20 m:
      * Area-weighted average (nodata / 0 excluded from the average)
      * The projection (EPSG:3857), top-left origin and coverage are unchanged
      * The original 1.5 m files are first backed up to _backup_ori_1p5m/
    20 m grid size:
      1485 x 1.5 m = 2227.5 m  ->  ceil(2227.5 / 20) = 112 columns
      1361 x 1.5 m = 2041.5 m  ->  ceil(2041.5 / 20) = 103 rows
    i.e. 112 x 103, with the top-left corner identical to the original and full coverage of the original extent.

Step 2
    Interpolate the 20 m results back to the 1.5 m fine grid using bilinear (linear) interpolation,
    Writes to the newly created D_Int / U_Int folders using the same filenames as the samples.
    The fine-grid geometry is taken from the like-named D_true / U_true (the fine grid used for plotting).
    Interpolation uses only valid neighbours and normalizes by weight (equivalent to ArcGIS bilinear +
    edge extrapolation), so no ring of holes appears along the channel margin.

Usage:
    python Laoyangcun_prep_coarse_interp.py
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import os
import shutil
import sys

import numpy as np
import rasterio
from rasterio.transform import Affine


# ============================================================
# Configuration
# ============================================================
BASE_DIR = (
    rf"{ROOT}"
    r"\Results\Transfer\laoyangcun_light"
)

COARSE_RES = 20.0          # target cell size for step 1 (m)

# True: the coarse grid uses the mean of valid cells (mean depth over wet cells)
# False: normalize by the total overlap area (partially wet cells are diluted by dry cells)
NORMALIZE_BY_VALID = True

BACKUP_DIRNAME = "_backup_ori_1p5m"

# (coarse-grid folder, fine-grid reference folder, linear-interpolation output folder)
JOBS = [
    ("D_ori", "D_true", "D_Int"),
    ("U_ori", "U_true", "U_Int"),
]


# ============================================================
# 1-D overlap-length matrix: overlap length of each dst cell with each src cell
#
# The origin passed in already has the reference point (fine-grid origin) subtracted,
# Avoid floating-point cancellation when subtracting large coordinates such as 1.16e7.
# ============================================================
def overlap_matrix(n_dst, dst_res, dst_origin,
                   n_src, src_res, src_origin):

    W = np.zeros((n_dst, n_src), dtype=np.float64)

    src_lo = src_origin + np.arange(n_src) * src_res
    src_hi = src_lo + src_res

    for i in range(n_dst):

        lo = dst_origin + i * dst_res
        hi = lo + dst_res

        W[i] = np.clip(
            np.minimum(hi, src_hi) - np.maximum(lo, src_lo),
            0.0,
            None
        )

    return W


# ============================================================
# Step 1: area-weighted averaging to 20 m
# ============================================================
def regrid_coarse(data, valid, transform, shape, coarse_res):

    res_x = transform.a
    res_y = -transform.e

    left = transform.c
    top = transform.f

    n_rows, n_cols = shape

    width = n_cols * res_x
    height = n_rows * res_y

    c_cols = int(np.ceil(width / coarse_res))
    c_rows = int(np.ceil(height / coarse_res))

    # ---- Column direction: origin at the left edge (referenced to the fine-grid left edge) ----
    Wx = overlap_matrix(
        c_cols, coarse_res, left - left,
        n_cols, res_x, 0.0
    )

    # ---- Row direction: computed bottom-up, then flipped to top-down ----
    fine_bottom = top - height
    coarse_bottom = top - c_rows * coarse_res

    Wy_bottom = overlap_matrix(
        c_rows, coarse_res, coarse_bottom - fine_bottom,
        n_rows, res_y, 0.0
    )

    Wy = Wy_bottom[::-1, ::-1]

    # ---- Weighted average ----
    values = np.where(valid, data, 0.0)
    w_valid = Wy @ valid.astype(np.float64) @ Wx.T

    if NORMALIZE_BY_VALID:
        den = w_valid
    else:
        den = Wy @ np.ones_like(values) @ Wx.T

    num = Wy @ values @ Wx.T

    ok = (w_valid > 0) & (den > 0)

    coarse = np.zeros_like(num)
    np.divide(num, den, out=coarse, where=ok)

    coarse_transform = Affine(
        coarse_res, 0.0, left,
        0.0, -coarse_res, top
    )

    return coarse, ok, coarse_transform, (c_rows, c_cols)


# ============================================================
# 1-D linear interpolation indices (along the axis)
# ============================================================
def axis_indices(n_dst, dst_res, dst_origin,
                 n_src, src_res, src_origin):

    centers = dst_origin + (np.arange(n_dst) + 0.5) * dst_res

    v = (centers - src_origin) / src_res - 0.5

    i0 = np.floor(v).astype(int)
    t = v - i0
    i1 = i0 + 1

    w0 = 1.0 - t
    w1 = t

    w0 = np.where((i0 < 0) | (i0 > n_src - 1), 0.0, w0)
    w1 = np.where((i1 < 0) | (i1 > n_src - 1), 0.0, w1)

    return np.clip(i0, 0, n_src - 1), np.clip(i1, 0, n_src - 1), w0, w1


# ============================================================
# Linear interpolation along axis 0 (nodata-aware with weight normalization)
# ============================================================
def interp_axis(values, valid, idx0, idx1, w0, w1):

    take0 = values[idx0]
    take1 = values[idx1]

    m0 = valid[idx0] & (w0 > 0.0)[:, None]
    m1 = valid[idx1] & (w1 > 0.0)[:, None]

    W0 = w0[:, None]
    W1 = w1[:, None]

    num = (
        W0 * np.where(m0, take0, 0.0)
        + W1 * np.where(m1, take1, 0.0)
    )

    den = W0 * m0 + W1 * m1

    ok = den > 0

    out = np.zeros(num.shape, dtype=np.float64)
    np.divide(num, den, out=out, where=ok)

    return out, ok


# ============================================================
# Step 2: bilinear interpolation back to the fine grid
# ============================================================
def bilinear_to_fine(coarse, coarse_valid,
                     coarse_transform, coarse_shape,
                     fine_transform, fine_shape):

    c_rows, c_cols = coarse_shape

    # ---- Row direction (converted to "distance from the top edge", origin at the fine-grid top edge) ----
    #   fine row i centre: d = (i + 0.5) * fine_res
    #   coarse row j centre: d = (fine_top - coarse_top) + (j + 0.5) * coarse_res
    r0, r1, rw0, rw1 = axis_indices(
        fine_shape[0], -fine_transform.e, 0.0,
        c_rows, -coarse_transform.e,
        fine_transform.f - coarse_transform.f
    )

    rows_vals, rows_ok = interp_axis(
        coarse, coarse_valid, r0, r1, rw0, rw1
    )

    # ---- Column direction (same treatment after transpose, relative to the fine-grid left edge) ----
    c0, c1, cw0, cw1 = axis_indices(
        fine_shape[1], fine_transform.a, 0.0,
        c_cols, coarse_transform.a,
        coarse_transform.c - fine_transform.c
    )

    cols_vals, cols_ok = interp_axis(
        rows_vals.T, rows_ok.T, c0, c1, cw0, cw1
    )

    return cols_vals.T, cols_ok.T


# ============================================================
# Single sample
# ============================================================
def process_sample(ori_dir, ref_dir, int_dir, fname, verbose):

    ori_path = os.path.join(ori_dir, fname)
    ref_path = os.path.join(ref_dir, fname)
    int_path = os.path.join(int_dir, fname)

    if not os.path.exists(ref_path):
        print(f"  [SKIP] missing fine-grid reference file: {ref_path}")
        return None

    # ---- Fine-grid geometry taken from *_true ----
    with rasterio.open(ref_path) as ref:
        fine_transform = ref.transform
        fine_shape = (ref.height, ref.width)
        fine_profile = ref.profile.copy()
        fine_crs = ref.crs
        nodata = ref.nodata

    # ---- Read the coarse-grid sources to be coarsened ----
    with rasterio.open(ori_path) as src:
        data = src.read(1).astype(np.float64)
        transform = src.transform
        shape = (src.height, src.width)
        src_profile = src.profile.copy()
        src_crs = src.crs

    if src_crs != fine_crs:
        print(f"  [WARN] projection mismatch: {src_crs} vs {fine_crs}")

    valid = np.isfinite(data) & (data != nodata)

    already_coarse = abs(
        transform.a - COARSE_RES
    ) < 1e-6

    # ========================================================
    # Step 1
    # ========================================================
    if already_coarse:

        coarse = data
        coarse_valid = valid
        coarse_transform = transform
        coarse_shape = shape

        if verbose:
            print("  [STEP1] already 20 m, skipping resampling")

    else:

        # ---- Back up the original 1.5 m files (only once) ----
        backup_path = os.path.join(
            BASE_DIR,
            BACKUP_DIRNAME,
            os.path.basename(ori_dir),
            fname
        )

        if not os.path.exists(backup_path):
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            shutil.copy2(ori_path, backup_path)

        coarse, coarse_valid, coarse_transform, coarse_shape = (
            regrid_coarse(
                data, valid, transform, shape, COARSE_RES
            )
        )

        # ---- Write back D_ori / U_ori ----
        out_profile = src_profile.copy()
        out_profile.update(
            height=coarse_shape[0],
            width=coarse_shape[1],
            transform=coarse_transform,
            dtype='float32',
            nodata=nodata
        )

        write_raster(ori_path, coarse, out_profile, nodata)

        if verbose:
            print(
                "  [STEP1] %s -> %s  res %.3f -> %.3f m  "
                "valid %s -> %s"
                % (
                    shape, coarse_shape,
                    transform.a, coarse_transform.a,
                    int(valid.sum()), int(coarse_valid.sum())
                )
            )

    # ========================================================
    # Step 2
    # ========================================================
    interp, interp_ok = bilinear_to_fine(
        coarse, coarse_valid,
        coarse_transform, coarse_shape,
        fine_transform, fine_shape
    )

    out_profile = fine_profile.copy()
    out_profile.update(
        height=fine_shape[0],
        width=fine_shape[1],
        transform=fine_transform,
        dtype='float32',
        nodata=nodata
    )

    write_raster(int_path, interp, out_profile, nodata)

    if verbose:
        c_min = coarse[coarse_valid].min() if coarse_valid.any() else 0.0
        c_max = coarse[coarse_valid].max() if coarse_valid.any() else 0.0
        i_vals = interp[interp_ok]
        print(
            "  [STEP2] %s  valid=%s  min=%.4f max=%.4f mean=%.4f   "
            "(coarse min=%.4f max=%.4f mean=%.4f)"
            % (
                int_dir, int(interp_ok.sum()),
                i_vals.min() if i_vals.size else 0.0,
                i_vals.max() if i_vals.size else 0.0,
                i_vals.mean() if i_vals.size else 0.0,
                c_min, c_max,
                coarse[coarse_valid].mean() if coarse_valid.any() else 0.0
            )
        )

    return {
        "coarse_shape": coarse_shape,
        "coarse_res": coarse_transform.a,
        "coarse_bounds": (
            coarse_transform.c,
            coarse_transform.f + coarse_transform.e * coarse_shape[0],
            coarse_transform.c + coarse_transform.a * coarse_shape[1],
            coarse_transform.f
        ),
        "coarse_valid": int(coarse_valid.sum()),
        "interp_valid": int(interp_ok.sum()),
    }


# ============================================================
# Write the GeoTIFF (keeping a profile consistent with the source file)
# ============================================================
def write_raster(path, data, profile, nodata):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    out = data.astype(np.float32)

    if nodata is not None:
        out = np.where(np.isfinite(out), out, nodata)

    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(out, 1)


# ============================================================
# Main program
# ============================================================
if __name__ == "__main__":

    # Optional: pass a number on the command line to process only the first N samples (for testing; all by default)
    LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None

    backup_root = os.path.join(BASE_DIR, BACKUP_DIRNAME)

    print("=" * 78)
    print("BASE_DIR   : %s" % BASE_DIR)
    print("COARSE_RES : %.1f m" % COARSE_RES)
    print("BACKUP     : %s" % backup_root)
    print("NORMALIZE_BY_VALID : %s" % NORMALIZE_BY_VALID)
    print("LIMIT      : %s" % LIMIT)
    print("=" * 78)

    for ori_name, ref_name, int_name in JOBS:

        ori_dir = os.path.join(BASE_DIR, ori_name)
        ref_dir = os.path.join(BASE_DIR, ref_name)
        int_dir = os.path.join(BASE_DIR, int_name)

        samples = sorted(
            f for f in os.listdir(ori_dir)
            if f.lower().endswith('.tif')
        )

        if LIMIT is not None:
            samples = samples[:LIMIT]

        print("\n" + "#" * 78)
        print("# %s  ->  %s   (%d samples)" % (ori_name, int_name, len(samples)))
        print("#" * 78)

        os.makedirs(int_dir, exist_ok=True)

        ok_count = 0
        first_info = None

        for k, fname in enumerate(samples, 1):

            verbose = (k == 1)

            if verbose:
                print("\n[sample] %s" % fname)

            info = process_sample(
                ori_dir, ref_dir, int_dir, fname, verbose
            )

            if info is None:
                continue

            ok_count += 1

            if first_info is None:
                first_info = (fname, info)

            if not verbose and k % 10 == 0:
                print("  ... %d / %d" % (k, len(samples)))

        print("\n[%s] completed %d / %d samples" % (int_name, ok_count, len(samples)))

        if first_info is not None:
            fname, info = first_info
            b = info["coarse_bounds"]
            print(
                "  First sample %s : coarse %s  res=%.1f m  "
                "bounds=(%.2f, %.2f, %.2f, %.2f)  "
                "valid %s -> interp valid %s"
                % (
                    fname, info["coarse_shape"], info["coarse_res"],
                    b[0], b[1], b[2], b[3],
                    info["coarse_valid"], info["interp_valid"]
                )
            )

    print("\nAll done.")
