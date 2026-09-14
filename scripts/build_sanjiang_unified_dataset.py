"""Rebuild Sanjiang grid datasets from the current point-cloud h5 files.

Plan B: point-cloud fields are scattered 3m samples. This script creates
nearest-neighbour upsampled 1.5m rasters on the shared 2010x1094 geometry,
exports the raw Depth/Velocity TIFF folders, then assembles the padded
[2048,1152] SanJiang LR/HR h5 datasets used by the Unet/FLO-SR workflow.
"""

import argparse
import glob
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = ROOT / "Data" / "dataset"
OUT_RAW = DS / "Unet" / "SanJiang"
OUT_H5 = DS / "Unet"

DEM_PATH = DS / "geo" / "dem_sanjiang_1_5.tif"
SLOPE_PATH = DS / "geo" / "Sanjiang_slope_1_5.tif"
MASK_PATH = DS / "geo" / "Shancha_shuixi_1_5_mask_polyfilled.tif"

ROWS, COLS = 2010, 1094
PAD_H, PAD_W = 38, 58
PADDED_H, PADDED_W = 2048, 1152
DX = DY = 1.5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--skip_tif", action="store_true", help="Skip TIFF regeneration"
    )
    parser.add_argument(
        "--skip_h5", action="store_true", help="Skip h5 regeneration"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing SanJiang output folder before regeneration",
    )
    return parser.parse_args()


def raster_profile():
    with rasterio.open(str(MASK_PATH)) as src:
        profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        height=ROWS,
        width=COLS,
        count=1,
        dtype="float32",
        nodata=0.0,
        compress="lzw",
    )
    return profile


def load_meta(path):
    with h5py.File(path, "r") as f:
        raw = f.attrs["meta"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        meta = json.loads(raw)
    return meta


def event_for_key(key):
    return "real4" if "(1)" in key else "real2"


def key_time(key):
    meta = key.replace("_", " ").replace("(1)", "")
    # stored keys are like 2019-08-19_19-10
    dt = datetime.strptime(meta, "%Y-%m-%d %H-%M")
    return dt


def tif_stem(key):
    dt = key_time(key)
    return f"{dt:%d%b%Y %H %M %S}".upper()


def write_tif(path, grid, profile):
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(grid.astype(np.float32), 1)


def load_mask_and_centers():
    with rasterio.open(str(MASK_PATH)) as src:
        mask = src.read(1).astype(np.uint8)
    roi = mask > 0
    rows, cols = np.where(roi)
    x = 11503199.581122924 + (cols + 0.5) * DX
    y = 3623692.5104884803 - (rows + 0.5) * DY
    return roi, np.column_stack([x, y]), rows, cols


def rasterize_direct(xy, values, rows, cols, roi, shape=(ROWS, COLS)):
    col = np.floor((xy[:, 0] - 11503199.581122924) / DX).astype(int)
    row = np.floor((3623692.5104884803 - xy[:, 1]) / DY).astype(int)
    valid = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
    grid = np.zeros(shape, dtype=np.float32)
    grid[row[valid], col[valid]] = values[valid]
    return grid


def nearest_grid(xy, values, target_xy, rows, cols, roi, tree_cache):
    tree = tree_cache.get((xy.shape[0], round(float(xy[:, 0].min()), 1)))
    if tree is None:
        tree = cKDTree(xy)
        tree_cache[(xy.shape[0], round(float(xy[:, 0].min()), 1))] = tree
    _, idx = tree.query(target_xy)
    grid = np.zeros((ROWS, COLS), dtype=np.float32)
    grid[roi] = np.asarray(values)[idx]
    return grid


def generate_train_tiffs(limit=0):
    hr_meta = load_meta(DS / "SanJiang_HR.h5")
    lr_meta = load_meta(DS / "SanJiang_LR.h5")
    profile = raster_profile()
    roi, target_xy, rows, cols = load_mask_and_centers()
    tree_cache = {}
    counters = {"real2": 0, "real4": 0}
    with h5py.File(str(DS / "SanJiang_HR.h5"), "r") as fh, \
            h5py.File(str(DS / "SanJiang_LR.h5"), "r") as fl:
        for rec, lr_rec in zip(hr_meta, lr_meta):
            key = rec["h5_key"]
            event = event_for_key(key)
            if limit and counters[event] >= limit:
                continue
            hr_dir = OUT_RAW / f"{event}_HR"
            lr_dir = OUT_RAW / f"{event}_LR"
            hr_dir.mkdir(parents=True, exist_ok=True)
            lr_dir.mkdir(parents=True, exist_ok=True)
            hr = fh[key][()]
            lr = fl[lr_rec["h5_key"]][()]
            stem = tif_stem(key)
            hr_d = nearest_grid(
                hr[:, :2], hr[:, 2], target_xy, rows, cols, roi, tree_cache
            )
            hr_u = nearest_grid(
                hr[:, :2], hr[:, 3], target_xy, rows, cols, roi, tree_cache
            )
            lr_d = nearest_grid(
                lr[:, :2], lr[:, 2], target_xy, rows, cols, roi, tree_cache
            )
            lr_u = nearest_grid(
                lr[:, :2], lr[:, 3], target_xy, rows, cols, roi, tree_cache
            )
            write_tif(hr_dir / f"Depth ({stem}).Terrain.sanjiang.tif", hr_d, profile)
            write_tif(hr_dir / f"Velocity ({stem}).Terrain.sanjiang.tif", hr_u, profile)
            write_tif(lr_dir / f"Depth ({stem}).Terrain.sanjiang.tif", lr_d, profile)
            write_tif(lr_dir / f"Velocity ({stem}).Terrain.sanjiang.tif", lr_u, profile)
            counters[event] += 1
            if (counters[event] + sum(counters.values()) - counters[event]) % 5 == 0:
                print(event, key, "written")
    print("train counters", counters)


def generate_val_tiffs():
    profile = raster_profile()
    roi, target_xy, rows, cols = load_mask_and_centers()
    hr_dir = OUT_RAW / "sanjiang_HR_val"
    lr_dir = OUT_RAW / "sanjiang_LR_val"
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    for prefix, folder, mode in [
        ("HR", hr_dir, "direct"),
        ("LR", lr_dir, "nearest"),
    ]:
        p = DS / f"SanJiang_{prefix}_val.h5"
        meta = load_meta(p)
        with h5py.File(str(p), "r") as f:
            for rec in meta:
                key = rec["h5_key"]
                d = f[key][()]
                stem = tif_stem(key)
                if mode == "direct":
                    depth = rasterize_direct(d[:, :2], d[:, 2], rows, cols, roi)
                    vel = rasterize_direct(d[:, :2], d[:, 3], rows, cols, roi)
                else:
                    depth = nearest_grid(d[:, :2], d[:, 2], target_xy, rows, cols, roi, {})
                    vel = nearest_grid(d[:, :2], d[:, 3], target_xy, rows, cols, roi, {})
                write_tif(folder / f"Depth ({stem}).Terrain.sanjiang.tif", depth, profile)
                write_tif(folder / f"Velocity ({stem}).Terrain.sanjiang.tif", vel, profile)
                print("val", prefix, key)


def pad_raster(arr):
    return np.pad(arr, ((PAD_H, 0), (PAD_W, 0)), mode="constant", constant_values=0)


def load_dem_slope():
    def read(path):
        with rasterio.open(path) as src:
            a = src.read(1).astype(np.float32)
        a[a == -9999] = 0
        if a.shape != (ROWS, COLS):
            raise ValueError(f"{path} shape {a.shape}")
        return pad_raster(a)
    return read(DEM_PATH), read(SLOPE_PATH)


def read_folder_depth_vel(folder):
    depth_files = sorted(glob.glob(os.path.join(folder, "Depth (*).Terrain.sanjiang.tif")))
    records = []
    for dp in depth_files:
        vp = dp.replace("Depth", "Velocity")
        name = os.path.basename(dp)
        ts = name.split("Depth (")[1].split(").Terrain")[0]
        with rasterio.open(dp) as src:
            d = src.read(1).astype(np.float32)
        with rasterio.open(vp) as src:
            v = src.read(1).astype(np.float32)
        records.append((ts, d, v))
    return records


def write_h5_group(path, samples, dem=None, slope=None, lr=True, compress=False):
    n = len(samples)
    if lr:
        shape = (n, 4, PADDED_H, PADDED_W)
    else:
        shape = (n, 2, PADDED_H, PADDED_W)
    comp = "gzip" if compress else None
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset(
            "data", shape=shape, dtype=np.float32, compression=comp
        )
        for i, (ts, d, v) in enumerate(samples):
            d = pad_raster(d)
            v = pad_raster(v)
            if lr:
                ds[i] = np.stack([d, v, dem, slope])
            else:
                ds[i] = np.stack([d, v])
    print("saved", path, shape)


def generate_h5():
    dem, slope = load_dem_slope()
    real2_hr = read_folder_depth_vel(OUT_RAW / "real2_HR")
    real2_lr = read_folder_depth_vel(OUT_RAW / "real2_LR")
    real4_hr = read_folder_depth_vel(OUT_RAW / "real4_HR")
    real4_lr = read_folder_depth_vel(OUT_RAW / "real4_LR")
    hr_train = sorted(real2_hr + real4_hr, key=lambda x: x[0])
    lr_train = sorted(real2_lr + real4_lr, key=lambda x: x[0])
    hr_val = read_folder_depth_vel(OUT_RAW / "sanjiang_HR_val")
    lr_val = read_folder_depth_vel(OUT_RAW / "sanjiang_LR_val")
    write_h5_group(OUT_H5 / "SanJiang_HR.h5", hr_train, lr=False)
    write_h5_group(OUT_H5 / "SanJiang_LR.h5", lr_train, dem=dem, slope=slope)
    write_h5_group(OUT_H5 / "SanJiang_HR_val.h5", hr_val, lr=False)
    write_h5_group(OUT_H5 / "SanJiang_LR_val.h5", lr_val, dem=dem, slope=slope)


def main():
    args = parse_args()
    if not args.skip_tif:
        if OUT_RAW.exists() and any(OUT_RAW.iterdir()) and not args.overwrite:
            raise FileExistsError(
                f"{OUT_RAW} is not empty. Pass --overwrite to replace it."
            )
        if OUT_RAW.exists():
            shutil.rmtree(OUT_RAW)
        OUT_RAW.mkdir(parents=True, exist_ok=True)
        generate_train_tiffs(args.limit)
        generate_val_tiffs()
    if not args.skip_h5:
        generate_h5()


if __name__ == "__main__":
    main()
