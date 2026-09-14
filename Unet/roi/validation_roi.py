import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

import roi_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=["watershed", "village", "sanjiang", "shouxi"],
        default="watershed",
    )
    parser.add_argument(
        "--model",
        choices=["srunet", "flosr"],
        default="srunet",
    )
    parser.add_argument("--weight_path", type=str, default=None)
    parser.add_argument("--patch_batch_size", type=int, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 means all samples")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--index_json",
        default=None,
        help="JSON list of sample indices to validate (overrides the case test h5).",
    )
    parser.add_argument(
        "--name_source_h5",
        default=None,
        help="Point-cloud h5 whose sorted keys provide matching output filenames.",
    )
    return parser.parse_args()


def make_batches(arrays, batch_size, device):
    for start in range(0, len(arrays), batch_size):
        batch = arrays[start:start + batch_size]
        yield torch.from_numpy(np.ascontiguousarray(batch)).float().to(device)


@torch.no_grad()
def predict_padded_image(model, lr_norm, cfg, device):
    C, H, W = lr_norm.shape
    ph, pw = cfg.patch_size
    pad_h = (ph - H % ph) % ph
    pad_w = (pw - W % pw) % pw
    if pad_h or pad_w:
        lr_padded = np.pad(
            lr_norm, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant"
        )
    else:
        lr_padded = lr_norm

    patches, positions = roi_data.extract_patches(lr_padded, cfg.patch_size)
    pred_patches = []
    model.eval()
    for batch in make_batches(patches, cfg.patch_batch_size, device):
        pred_patches.append(model(batch).cpu().numpy())
    pred_patches = np.concatenate(pred_patches, axis=0)

    out_h, out_w = lr_padded.shape[1], lr_padded.shape[2]
    image = np.zeros((2, out_h, out_w), dtype=np.float32)
    count = np.zeros((out_h, out_w), dtype=np.int32)
    for patch, (sh, eh, sw, ew) in zip(pred_patches, positions):
        image[:, sh:eh, sw:ew] += patch
        count[sh:eh, sw:ew] += 1
    count[count == 0] = 1
    image /= count[None, :, :]
    if pad_h or pad_w:
        image = image[:, :-pad_h, :-pad_w]
    return image


def main():
    args = parse_args()
    cfg = roi_data.get_config(args.case)
    if args.patch_batch_size is not None:
        cfg.patch_batch_size = args.patch_batch_size
    roi_data.configure_model_paths(cfg, args.model)
    if args.result_dir:
        cfg.result_dir = Path(args.result_dir)

    if not cfg.norm_path.exists():
        raise FileNotFoundError(f"Normalization file not found: {cfg.norm_path}")
    if not cfg.best_weight_path.exists():
        raise FileNotFoundError(f"Weight file not found: {cfg.best_weight_path}")
    if args.weight_path:
        weight_path = args.weight_path
    else:
        weight_path = str(cfg.best_weight_path)

    norm = np.load(str(cfg.norm_path))
    lr_min = norm["lr_min"]
    lr_max = norm["lr_max"]

    aligned_names = None
    if args.index_json and Path(args.index_json).exists():
        with open(args.index_json, "r", encoding="utf-8") as f:
            test_indices = [int(i) for i in json.load(f)]
        if args.limit > 0:
            test_indices = test_indices[args.start : args.start + args.limit]
        lr_h5 = cfg.h5_dir / cfg.lr_train_h5
        hr_h5 = cfg.h5_dir / cfg.hr_train_h5
        if args.name_source_h5:
            aligned_names = roi_data.load_h5_sample_names(
                args.name_source_h5, test_indices
            )
    else:
        indices = roi_data.get_sample_indices(cfg)
        test_indices = indices["test"]
        lr_h5 = cfg.h5_dir / cfg.lr_test_h5
        hr_h5 = cfg.h5_dir / cfg.hr_test_h5
    if args.limit > 0:
        test_indices = test_indices[args.start : args.start + args.limit]

    sys.path.insert(0, str(cfg.unet_root))
    from model.model_factory import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, input_channels=4, output_channels=2).to(device)
    state = torch.load(weight_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    roi_data.make_output_dirs(cfg)
    names = (
        aligned_names
        if aligned_names is not None
        else roi_data.load_metadata_timestamps(cfg)
    )
    print(f"Validation case: {cfg.case_name}, samples: {len(test_indices)}")

    with h5py.File(str(lr_h5), "r") as f_lr, h5py.File(str(hr_h5), "r") as f_hr:
        inference_times = []
        for test_pos, sample_idx in enumerate(test_indices):
            lr_sample = roi_data.read_sample(f_lr, int(sample_idx))
            hr_sample = roi_data.read_sample(f_hr, int(sample_idx))
            lr_norm = roi_data.normalize_lr_sample(cfg, lr_sample, lr_min, lr_max)
            t0 = time.perf_counter()
            pred_padded = predict_padded_image(model, lr_norm, cfg, device)
            inference_times.append(time.perf_counter() - t0)

            pad_h, pad_w = cfg.pad_origin
            pred = pred_padded[:, pad_h:, pad_w:]
            true = hr_sample[:, pad_h:, pad_w:]
            pred = np.clip(pred, 0, None)
            true = np.clip(true, 0, None)
            lr_ori = np.array(
                lr_sample[0:2, pad_h:, pad_w:], dtype=np.float32, copy=True
            )
            lr_ori = np.clip(lr_ori, 0, None)

            # Match the post-processing used by the original FFSR validation.
            pred[pred < cfg.value_threshold] = 0.0
            true[true < cfg.value_threshold] = 0.0
            lr_ori[lr_ori < cfg.value_threshold] = 0.0
            d_err = np.abs(pred[0] - true[0])
            u_err = np.abs(pred[1] - true[1])
            d_err[d_err < cfg.error_threshold] = 0.0
            u_err[u_err < cfg.error_threshold] = 0.0

            name = roi_data.output_name_for_sample(
                cfg, int(sample_idx), test_pos + args.start, names
            )
            out_map = {
                "D_pred": pred[0],
                "D_true": true[0],
                "D_ori": lr_ori[0],
                "D_error": d_err,
                "U_pred": pred[1],
                "U_true": true[1],
                "U_ori": lr_ori[1],
                "U_error": u_err,
            }
            for subdir, matrix in out_map.items():
                roi_data.create_tif(
                    cfg, matrix, cfg.result_dir / subdir / name
                )
            print(f"[{test_pos + 1}/{len(test_indices)}] {name} saved")

    if inference_times:
        total_inf = sum(inference_times)
        avg_inf = total_inf / len(inference_times)
        report = cfg.result_dir / "inference_time_report.txt"
        with report.open("w") as f:
            f.write(
                f"samples={len(inference_times)},"
                f"total_inference_s={total_inf:.6f},"
                f"avg_s_per_sample={avg_inf:.6f}\n"
            )
            for name_time in zip(
                (names or [f"sample_{i}" for i in range(len(inference_times))]),
                inference_times,
            ):
                f.write(f"{name_time[0]},{name_time[1]:.6f}\n")
        print(
            f"Inference timing: {len(inference_times)} samples, "
            f"total {total_inf:.3f}s, avg {avg_inf * 1000:.1f} ms/sample"
        )
        print(f"Saved to: {report}")

    print(f"All validation results saved to: {cfg.result_dir}")


if __name__ == "__main__":
    main()
