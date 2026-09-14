import importlib
import re
import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import rasterio
import scipy.io as sio
from rasterio.transform import from_origin

_ROI_CACHE = {}


def get_config(case_name):
    """Load one of the ROI config modules by case name."""
    module_name = {
        "watershed": "config_roi_watershed",
        "village": "config_roi_village",
        "sanjiang": "config_roi_village",
        "shouxi": "config_roi_watershed",
    }.get(str(case_name).lower())
    if module_name is None:
        raise ValueError("case must be one of: watershed, village, sanjiang, shouxi")
    return importlib.import_module(module_name).Config


def model_tag(model_name):
    name = str(model_name).lower()
    if name in ("flosr", "flo-sr", "flo_sr", "edsr"):
        return "flosr"
    return "srunet"


def configure_model_paths(cfg, model_name):
    """Route FLO-SR artifacts to comparison/; SR-Unet keeps existing paths."""
    tag = model_tag(model_name)
    if tag == "srunet":
        return tag

    case = str(cfg.case_name).lower()
    if case == "shouxi":
        cfg.weight_dir = cfg.project_root / "Weights" / "Unet" / "flosr_watershed"
        cfg.result_dir = (
            cfg.project_root / "Results" / "Unet" / "flosr_roi_shouxi_820"
        )
    else:
        cfg.weight_dir = cfg.project_root / "Weights" / "Unet" / "flosr_village"
        cfg.result_dir = (
            cfg.project_root / "Results" / "Unet" / "flosr_roi_sanjiang"
        )
    cfg.loss_log_path = cfg.weight_dir / f"roi_loss_log_{tag}_{case}.txt"
    cfg.checkpoint_path = cfg.weight_dir / f"checkpoint_last_{tag}_{case}.ckpt"
    cfg.best_weight_path = cfg.weight_dir / f"best_roi_{tag}_{case}.pth"
    return tag


def expand_ranges(ranges):
    """Turn [start, end) ranges into a sorted list of indices."""
    if ranges is None:
        return None
    indices = []
    for start, end in ranges:
        indices.extend(range(int(start), int(end)))
    return sorted(set(indices))


def count_h5_samples(path):
    with h5py.File(str(path), "r") as f:
        return len(f["data"])


def get_sample_indices(cfg):
    """Return dict(train, val, test) according to cfg.split_mode."""
    if cfg.split_mode == "events":
        train = expand_ranges(getattr(cfg, "train_event_ranges", None))
        val = expand_ranges(getattr(cfg, "val_event_ranges", None))
        test = expand_ranges(getattr(cfg, "test_event_ranges", None))
        if test is None:
            test = list(range(count_h5_samples(cfg.h5_dir / cfg.lr_test_h5)))
        if not train or not val or not test:
            raise ValueError("Event split produced an empty subset")
        return {"train": train, "val": val, "test": test}

    if cfg.split_mode == "sample":
        total = count_h5_samples(cfg.h5_dir / cfg.lr_train_h5)
        perm = np.random.RandomState(cfg.sample_split_seed).permutation(total)
        train_ratio, test_ratio, val_ratio = cfg.split_ratios
        train_n = int(train_ratio * total)
        test_n = int(test_ratio * total)
        val_n = total - train_n - test_n
        if min(train_n, test_n, val_n) <= 0:
            raise ValueError("Sample split produced an empty subset")
        train = sorted(perm[:train_n].tolist())
        test = sorted(perm[train_n:train_n + test_n].tolist())
        val = sorted(perm[train_n + test_n:].tolist())
        if getattr(cfg, "separate_test_h5", False):
            test = list(range(count_h5_samples(cfg.h5_dir / cfg.lr_test_h5)))
        return {"train": train, "val": val, "test": test}

    raise ValueError(f"Unsupported split_mode: {cfg.split_mode}")


def read_sample(h5_file, idx):
    """Read one [C, H, W] sample without keeping the file handle."""
    return h5_file["data"][idx]


def load_roi_padded(cfg):
    """Load original ROI mask, place it at the bottom-right of the padded grid."""
    cache_key = (str(cfg.roi_path), tuple(cfg.original_shape), tuple(cfg.padded_shape))
    if cache_key in _ROI_CACHE:
        return _ROI_CACHE[cache_key]
    with rasterio.open(str(cfg.roi_path)) as src:
        raw = src.read(1)
    if raw.shape != tuple(cfg.original_shape):
        raise ValueError(
            f"ROI shape {raw.shape} does not match original_shape {cfg.original_shape}"
        )
    roi = raw > 0
    if roi.sum() != cfg.expected_roi_count:
        raise ValueError(
            f"ROI positive count {int(roi.sum())} != expected {cfg.expected_roi_count}"
        )

    pad_h, pad_w = cfg.pad_origin
    padded = np.zeros(cfg.padded_shape, dtype=np.bool_)
    padded[pad_h:, pad_w:] = roi
    _ROI_CACHE[cache_key] = (roi, padded)
    return roi, padded


def load_roi_original(cfg):
    """Return the unpadded binary ROI mask."""
    roi, _ = load_roi_padded(cfg)
    return roi


def natural_key(value):
    """Natural sort used by the point-cloud dataset for sample keys."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def load_h5_sample_names(h5_path, sample_indices):
    """Return point-cloud h5 key stems for the requested numeric indices."""
    sample_indices = [int(i) for i in sample_indices]
    with h5py.File(str(h5_path), "r") as f:
        keys = sorted(f.keys(), key=natural_key)
    return [str(keys[int(i)]) for i in sample_indices]


def extract_patches(image, patch_size):
    """Cut an image into a non-overlapping patch grid.

    image has shape [C, H, W]. The spatial size is expected to be divisible
    by patch_size; if not, the image is padded on the bottom/right first and
    the caller must later remove that padding.
    """
    image = np.asarray(image, dtype=np.float32)
    C, H, W = image.shape
    ph, pw = patch_size
    pad_h = (ph - H % ph) % ph
    pad_w = (pw - W % pw) % pw
    if pad_h or pad_w:
        image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
    H, W = image.shape[1], image.shape[2]
    nh, nw = H // ph, W // pw
    image = image.reshape(C, nh, ph, nw, pw)
    image = image.transpose(1, 3, 0, 2, 4).reshape(nh * nw, C, ph, pw)
    positions = []
    for i in range(nh):
        for j in range(nw):
            positions.append((i * ph, (i + 1) * ph, j * pw, (j + 1) * pw))
    return np.ascontiguousarray(image), np.array(positions, dtype=np.int32)


def extract_roi_patches(cfg, lr_sample, hr_sample):
    """Return (lr_patches, hr_patches, roi_masks, kept_positions).

    Patches with no water in the HR depth channel, or with no ROI pixel, are
    skipped. All arrays correspond to the already padded grid used by the h5.
    """
    _, padded_roi = load_roi_padded(cfg)
    lr_patches, positions = extract_patches(lr_sample, cfg.patch_size)
    hr_patches, _ = extract_patches(hr_sample, cfg.patch_size)

    roi_all, _ = extract_patches(
        padded_roi.astype(np.float32)[None, ...], cfg.patch_size
    )
    roi_all = roi_all[:, 0] > 0.5

    has_water = hr_patches[:, 0].max(axis=(1, 2)) > cfg.zero_threshold
    has_roi = roi_all.reshape(len(roi_all), -1).any(axis=1)
    keep = has_water & has_roi
    if not np.any(keep):
        return (
            np.empty((0, lr_patches.shape[1], *cfg.patch_size), dtype=np.float32),
            np.empty((0, hr_patches.shape[1], *cfg.patch_size), dtype=np.float32),
            np.empty((0, *cfg.patch_size), dtype=np.bool_),
            np.empty((0, 4), dtype=np.int32),
        )
    return (
        np.ascontiguousarray(lr_patches[keep]),
        np.ascontiguousarray(hr_patches[keep]),
        np.ascontiguousarray(roi_all[keep]),
        np.ascontiguousarray(positions[keep]),
    )


def compute_train_norm(cfg, train_indices):
    """Compute min/max of LR channels 2 and 3 from training samples only."""
    lr_path = cfg.h5_dir / cfg.lr_train_h5
    with h5py.File(str(lr_path), "r") as f:
        lr_min = np.full(4, np.inf, dtype=np.float32)
        lr_max = np.full(4, -np.inf, dtype=np.float32)
        for idx in train_indices:
            sample = read_sample(f, idx)
            for c in (2, 3):
                lr_min[c] = min(lr_min[c], float(np.min(sample[c])))
                lr_max[c] = max(lr_max[c], float(np.max(sample[c])))
        # Depth and velocity channels are intentionally kept in physical units.
        lr_min[0:2] = 0.0
        lr_max[0:2] = 1.0
        if np.any(lr_min[2:4] == np.inf) or np.any(lr_max[2:4] == -np.inf):
            raise ValueError("Failed to compute normalization statistics")
        return lr_min, lr_max


def ensure_norm_file(cfg, train_indices):
    """Create or load the saved normalization npz."""
    cfg.weight_dir.mkdir(parents=True, exist_ok=True)
    if cfg.norm_path.exists():
        data = np.load(str(cfg.norm_path))
        return data["lr_min"].astype(np.float32), data["lr_max"].astype(np.float32)
    lr_min, lr_max = compute_train_norm(cfg, train_indices)
    np.savez(str(cfg.norm_path), lr_min=lr_min, lr_max=lr_max)
    return lr_min, lr_max


def normalize_lr_sample(cfg, sample, lr_min, lr_max):
    """Normalize LR channels 2 and 3 using training-derived min/max."""
    out = np.array(sample, dtype=np.float32, copy=True)
    for c in (2, 3):
        span = float(lr_max[c] - lr_min[c])
        if span < 1e-8:
            span = 1e-8
        out[c] = (out[c].astype(np.float32) - float(lr_min[c])) / span
    return out


def make_output_dirs(cfg):
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "D_pred",
        "D_true",
        "D_ori",
        "D_error",
        "U_pred",
        "U_true",
        "U_ori",
        "U_error",
    ):
        (cfg.result_dir / name).mkdir(parents=True, exist_ok=True)


def create_tif(cfg, matrix, out_tif):
    """Write a matrix matching cfg.tiff_rows x cfg.tiff_cols as a GeoTIFF."""
    rows, cols = cfg.tiff_rows, cfg.tiff_cols
    if matrix.shape != (rows, cols):
        raise ValueError(f"matrix {matrix.shape} != expected {(rows, cols)}")
    transform = from_origin(cfg.xmin, cfg.ymax, cfg.cellsize_x, cfg.cellsize_y)
    with rasterio.open(
        str(out_tif),
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs=cfg.crs_wkt,
        transform=transform,
    ) as dst:
        dst.write(matrix.astype(np.float32), 1)


def load_metadata_timestamps(cfg):
    """Load test timestamps for Shouxi so output filenames match previous runs."""
    if cfg.meta_path is None or not Path(cfg.meta_path).exists():
        return None
    meta = sio.loadmat(str(cfg.meta_path))
    records = meta.get("metadata", np.array([]))[0]
    names = []
    for record in records:
        if "timestamp" not in record.dtype.names:
            continue
        raw = record["timestamp"][0][0]
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        dt = datetime.strptime(str(raw), "%Y-%m-%d %H:%M:%S")
        names.append(f"{dt:%Y-%m-%d_%H_%M}")
    return names


def output_name_for_sample(cfg, sample_idx, test_pos, names=None):
    if names is not None and test_pos < len(names):
        return names[test_pos] + ".tif"
    return f"sample_orig{sample_idx}_test{test_pos}.tif"
