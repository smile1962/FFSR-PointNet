#!/usr/bin/env python
"""Build LaoYangCun HR/LR point-cloud HDF5 datasets for FFSR transfer work.

The high-resolution target is the HEC-RAS 1.5 m fine grid exported as CD in
LaoYangCun2D.g01.hdf. Coarse/fine depth and velocity GeoTIFFs are exported by
HEC-RAS on the 0.5 m terrain grid in EPSG:3857. This script samples those
rasters at the fine-grid cell centres inside the CD computational domain, so
the ROI point set is aligned with the 1.5 m numerical mesh instead of sampling
the same cell multiple times on the terrain raster.

DEM and slope are first reprojected to the exact GeoTIFF result grid and then
sampled at the same fine cell centres.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import argparse
import gc
import json
import os
import re
from datetime import datetime

import h5py
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


MAX_DEPTH = 500.0
MAX_VELOCITY = 60.0
NODATA = -9999.0


def parse_time_from_path(path):
    match = re.search(r"\((\d{2}[A-Za-z]{3}\d{4} \d{2} \d{2} \d{2})\)", path)
    if match is None:
        raise ValueError(f"Could not parse timestamp from {os.path.basename(path)}")
    ts = datetime.strptime(match.group(1), "%d%b%Y %H %M %S")
    return ts.strftime("%Y-%m-%d_%H-%M")


def flow_file_pairs(folder):
    depth_files = sorted(
        f for f in os.listdir(folder)
        if f.startswith("Depth ") and f.endswith(".tif")
    )
    pairs = []
    for name in depth_files:
        velocity_name = "Velocity" + name[len("Depth"):]
        velocity_path = os.path.join(folder, velocity_name)
        if not os.path.exists(velocity_path):
            raise FileNotFoundError(velocity_path)
        pairs.append((os.path.join(folder, name), velocity_path))
    return pairs


def nearest_sample(grid, coords, transform, width, height):
    """Sample an aligned raster at point coordinates with nearest neighbour."""
    col = np.floor((coords[:, 0] - transform.c) / transform.a + 0.5).astype(np.int64)
    row = np.floor((transform.f - coords[:, 1]) / (-transform.e) + 0.5).astype(np.int64)
    valid = (
        (col >= 0)
        & (col < width)
        & (row >= 0)
        & (row < height)
    )
    out = np.zeros(len(coords), dtype=np.float32)
    if valid.any():
        out[valid] = grid[row[valid], col[valid]]
    return out


def clean_flow(values, upper):
    values = values.astype(np.float32, copy=False)
    values = np.where(np.isfinite(values), values, 0.0)
    values = np.where(values == NODATA, 0.0, values)
    values = np.clip(values, 0.0, upper)
    return values.astype(np.float32, copy=False)


def clean_terrain(values):
    values = values.astype(np.float32, copy=False)
    values = np.where(np.isfinite(values), values, 0.0)
    values = np.where(values == NODATA, 0.0, values)
    values = np.where(values < 0, 0.0, values)
    return values.astype(np.float32, copy=False)


def reproject_to_result_grid(src_path, dst_profile):
    with rasterio.open(src_path) as src:
        src_nodata = src.nodata
        source = src.read(1)
        dst = np.empty(
            (dst_profile["height"], dst_profile["width"]), dtype=np.float32
        )
        reproject(
            source,
            dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dst_profile["transform"],
            dst_crs=dst_profile["crs"],
            dst_nodata=NODATA,
            resampling=Resampling.bilinear,
        )
        return dst


def make_reference_profile(reference_tif):
    with rasterio.open(reference_tif) as src:
        profile = src.profile.copy()
        profile.update(
            dtype="float32",
            count=1,
            nodata=NODATA,
            compress=None,
        )
        return profile, src.transform, src.width, src.height, src.bounds


def aligned_terrain(dem_path, slope_path, profile, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    geo_arrays = {}
    for label, src_path in [("dem", dem_path), ("slope", slope_path)]:
        array = reproject_to_result_grid(src_path, profile)
        out_path = os.path.join(out_dir, f"{label}_aligned_epsg3857.tif")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(array, 1)
        geo_arrays[label] = clean_terrain(array)
        print(f"Aligned {label}: {array.shape}, written to {out_path}")
        gc.collect()
    return geo_arrays


def load_hecras_centres(hdf_path):
    dataset = (
        "Geometry/2D Flow Areas/CD/Cells Center Coordinate"
    )
    with h5py.File(hdf_path, "r") as f:
        centres = f[dataset][()]
    return centres.astype(np.float64)


def write_split_h5(
    split_name,
    cases,
    centres,
    terrain,
    profile,
    src_root,
    out_root,
    low_name,
    high_name,
    max_frames=0,
):
    low_path = os.path.join(out_root, low_name)
    high_path = os.path.join(out_root, high_name)
    meta = []

    with h5py.File(low_path, "w") as low_f, h5py.File(high_path, "w") as high_f:
        for case_name in cases:
            low_dir = os.path.join(src_root, f"CoarseGrid_{case_name}")
            high_dir = os.path.join(src_root, f"FineGrid_{case_name}")
            pairs = flow_file_pairs(low_dir)
            high_pairs = flow_file_pairs(high_dir)
            if len(pairs) != len(high_pairs):
                raise RuntimeError(
                    f"{case_name}: coarse {len(pairs)} frames, "
                    f"fine {len(high_pairs)} frames"
                )
            if max_frames > 0:
                pairs = pairs[:max_frames]
                high_pairs = high_pairs[:max_frames]

            print(
                f"[{split_name}] {case_name}: {len(pairs)} paired frames"
            )
            dem_grid = terrain["dem"]
            slope_grid = terrain["slope"]
            width = profile["width"]
            height = profile["height"]
            transform = profile["transform"]

            for (depth_low, vel_low), (depth_high, vel_high) in zip(pairs, high_pairs):
                key = parse_time_from_path(depth_low)
                key_high = parse_time_from_path(depth_high)
                if key != key_high:
                    raise RuntimeError(
                        f"Coarse/fine frame mismatch in {case_name}: "
                        f"{key} vs {key_high}"
                    )
                meta.append(
                    {
                        "h5_key": key,
                        "case": case_name,
                        "time": key,
                        "event": case_name,
                        "coarse_dir": os.path.basename(low_dir),
                        "fine_dir": os.path.basename(high_dir),
                    }
                )

                with rasterio.open(depth_high) as src_high_d, \
                        rasterio.open(vel_high) as src_high_v, \
                        rasterio.open(depth_low) as src_low_d, \
                        rasterio.open(vel_low) as src_low_v:
                    high_d = clean_flow(
                        nearest_sample(
                            src_high_d.read(1),
                            centres,
                            transform,
                            width,
                            height,
                        ),
                        MAX_DEPTH,
                    )
                    high_v = clean_flow(
                        nearest_sample(
                            src_high_v.read(1),
                            centres,
                            transform,
                            width,
                            height,
                        ),
                        MAX_VELOCITY,
                    )
                    low_d = clean_flow(
                        nearest_sample(
                            src_low_d.read(1),
                            centres,
                            transform,
                            width,
                            height,
                        ),
                        MAX_DEPTH,
                    )
                    low_v = clean_flow(
                        nearest_sample(
                            src_low_v.read(1),
                            centres,
                            transform,
                            width,
                            height,
                        ),
                        MAX_VELOCITY,
                    )

                dem_vals = clean_terrain(
                    nearest_sample(
                        dem_grid, centres, transform, width, height
                    )
                )
                slope_vals = clean_terrain(
                    nearest_sample(
                        slope_grid, centres, transform, width, height
                    )
                )

                high_sample = np.column_stack(
                    [centres[:, 0], centres[:, 1], high_d, high_v]
                ).astype(np.float64)
                low_sample = np.column_stack(
                    [
                        centres[:, 0],
                        centres[:, 1],
                        low_d,
                        low_v,
                        dem_vals,
                        slope_vals,
                    ]
                ).astype(np.float64)
                high_f.create_dataset(key, data=high_sample)
                low_f.create_dataset(key, data=low_sample)

                del high_d, high_v, low_d, low_v, high_sample, low_sample
                gc.collect()

        low_f.attrs["meta"] = json.dumps(meta)
        high_f.attrs["meta"] = json.dumps(meta)
        low_f.attrs["region"] = "LaoYangCun"
        high_f.attrs["region"] = "LaoYangCun"
        low_f.attrs["points"] = len(centres)
        high_f.attrs["points"] = len(centres)

    print(
        f"[{split_name}] wrote {len(meta)} samples: "
        f"{os.path.basename(low_path)} and {os.path.basename(high_path)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-root",
        default=rf"{ROOT}\Transfer_adding_experiments",
    )
    parser.add_argument(
        "--out-root",
        default=rf"{ROOT}\Transfer\datasets_laoyangcun",
    )
    parser.add_argument(
        "--geometry-hdf",
        default=None,
        help="Path to LaoYangCun2D.g01.hdf containing fine-grid CD centres",
    )
    parser.add_argument(
        "--dem",
        default=None,
        help="Source DEM GeoTIFF path",
    )
    parser.add_argument(
        "--slope",
        default=None,
        help="Source slope GeoTIFF path",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional smoke-test limit on frames per case",
    )
    args = parser.parse_args()

    src_root = args.src_root
    out_root = args.out_root
    dem_path = args.dem or os.path.join(
        src_root,
        "dem",
        "Terrain (2).Dubahe_laoyangcun_dem_50cm_v3_BridgeRemoved.tif",
    )
    slope_path = args.slope or os.path.join(
        src_root, "dem", "Slope_LaoYangCun.tif"
    )
    geometry_hdf = args.geometry_hdf or os.path.join(
        src_root, "LaoYangCun2D.g01.hdf"
    )
    reference_tif = os.path.join(
        src_root,
        "CoarseGrid_001",
        "Depth (02SEP2026 00 15 00).Terrain.Terrain (2)."
        "Dubahe_laoyangcun_dem_50cm_v3_BridgeRemoved.tif",
    )

    if not os.path.exists(dem_path):
        raise FileNotFoundError(dem_path)
    if not os.path.exists(slope_path):
        raise FileNotFoundError(slope_path)
    if not os.path.exists(geometry_hdf):
        raise FileNotFoundError(geometry_hdf)
    if not os.path.exists(reference_tif):
        raise FileNotFoundError(reference_tif)

    os.makedirs(out_root, exist_ok=True)
    aligned_dir = os.path.join(out_root, "aligned_geo")
    profile, _, _, _, _ = make_reference_profile(reference_tif)

    print("Reprojecting DEM and slope to HEC-RAS result grid...")
    terrain = aligned_terrain(dem_path, slope_path, profile, aligned_dir)
    print("Loading fine-grid CD cell centres from HEC-RAS geometry...")
    centres = load_hecras_centres(geometry_hdf)
    print(f"ROI points: {len(centres)}")

    write_split_h5(
        "train",
        ["001", "003"],
        centres,
        terrain,
        profile,
        src_root,
        out_root,
        "LaoYangCun_LR.h5",
        "LaoYangCun_HR.h5",
        args.max_frames,
    )
    write_split_h5(
        "test",
        ["002"],
        centres,
        terrain,
        profile,
        src_root,
        out_root,
        "LaoYangCun_LR_val.h5",
        "LaoYangCun_HR_val.h5",
        args.max_frames,
    )

    if args.max_frames > 0:
        return

    summary = {
        "region": "LaoYangCun",
        "points": int(len(centres)),
        "train_samples": 44 + 66,
        "test_samples": 66,
        "train_files": [
            os.path.join(out_root, "LaoYangCun_LR.h5"),
            os.path.join(out_root, "LaoYangCun_HR.h5"),
        ],
        "test_files": [
            os.path.join(out_root, "LaoYangCun_LR_val.h5"),
            os.path.join(out_root, "LaoYangCun_HR_val.h5"),
        ],
    }
    with open(os.path.join(out_root, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Done. Summary written to", os.path.join(out_root, "dataset_summary.json"))


if __name__ == "__main__":
    main()
