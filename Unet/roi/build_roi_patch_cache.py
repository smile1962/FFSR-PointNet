import argparse
import time
from pathlib import Path

import h5py
import numpy as np

import roi_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=["watershed", "village", "sanjiang", "shouxi"],
        default="watershed",
    )
    return parser.parse_args()


def create_resizable(group, name, shape, chunks, dtype):
    return group.create_dataset(
        name,
        shape=(0, *shape[1:]),
        maxshape=(None, *shape[1:]),
        chunks=chunks,
        dtype=dtype,
    )


def append_rows(dataset, data):
    n = len(data)
    if n == 0:
        return
    current = dataset.shape[0]
    dataset.resize(current + n, axis=0)
    dataset[current:current + n] = data


def build_group(f, group_name, sample_indices, cfg, lr_min, lr_max):
    print(f"Building cache group: {group_name} ({len(sample_indices)} samples)")
    group = f.create_group(group_name)
    lr_ds = create_resizable(
        group, "lr", (0, 4, *cfg.patch_size), (16, 4, *cfg.patch_size), "f4"
    )
    hr_ds = create_resizable(
        group, "hr", (0, 2, *cfg.patch_size), (16, 2, *cfg.patch_size), "f4"
    )
    mask_ds = create_resizable(
        group, "mask", (0, 1, *cfg.patch_size), (16, 1, *cfg.patch_size), "u1"
    )
    idx_ds = create_resizable(group, "sample_idx", (0,), (4096,), "i4")

    total_patches = 0
    start = time.perf_counter()
    with h5py.File(str(cfg.h5_dir / cfg.lr_train_h5), "r") as f_lr, \
            h5py.File(str(cfg.h5_dir / cfg.hr_train_h5), "r") as f_hr:
        for pos, sample_idx in enumerate(sample_indices):
            lr_sample = roi_data.read_sample(f_lr, int(sample_idx))
            hr_sample = roi_data.read_sample(f_hr, int(sample_idx))
            lr_norm = roi_data.normalize_lr_sample(cfg, lr_sample, lr_min, lr_max)
            lr_p, hr_p, mask_p, _ = roi_data.extract_roi_patches(
                cfg, lr_norm, hr_sample
            )
            if len(lr_p) == 0:
                continue
            append_rows(lr_ds, lr_p)
            append_rows(hr_ds, hr_p)
            append_rows(mask_ds, mask_p[:, None, :, :])
            append_rows(idx_ds, np.full(len(lr_p), int(sample_idx), dtype=np.int32))
            total_patches += len(lr_p)
            if (pos + 1) % 25 == 0:
                print(
                    f"  {pos + 1}/{len(sample_indices)} samples, "
                    f"{total_patches} patches, {time.perf_counter() - start:.1f}s"
                )
    print(f"{group_name}: {total_patches} patches")
    return total_patches


def main():
    args = parse_args()
    cfg = roi_data.get_config(args.case)
    indices = roi_data.get_sample_indices(cfg)
    cfg.weight_dir.mkdir(parents=True, exist_ok=True)
    lr_min, lr_max = roi_data.ensure_norm_file(cfg, indices["train"])

    if cfg.cache_path.exists():
        print(f"Existing cache removed and rebuilt: {cfg.cache_path}")
        cfg.cache_path.unlink()

    start = time.perf_counter()
    with h5py.File(str(cfg.cache_path), "w") as f:
        f.attrs["case"] = cfg.case_name
        train_n = build_group(f, "train", indices["train"], cfg, lr_min, lr_max)
        val_n = build_group(f, "val", indices["val"], cfg, lr_min, lr_max)
    print(
        f"Cache written to {cfg.cache_path}; train patches {train_n}, "
        f"val patches {val_n}, total time {time.perf_counter() - start:.1f}s"
    )


if __name__ == "__main__":
    main()

