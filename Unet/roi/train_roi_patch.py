import argparse
import contextlib
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
        help="CNN backbone: srunet (existing U-Net) or flosr (FLO-SR/EDSR)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument(
        "--use_amp", action="store_true", help="Use bfloat16 autocast on CUDA"
    )
    parser.add_argument(
        "--disable_early_stop",
        action="store_true",
        help="Disable early stopping so --num_epochs is fully honored",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Optional run tag; routes weights/logs to Weights/Unet/<run_tag>",
    )
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def roi_loss_terms(pred, target, roi_mask):
    """Return (masked squared-error sum, ROI pixel count)."""
    diff = pred - target
    mask = roi_mask.unsqueeze(1).float()
    se = (diff * diff) * mask
    count = mask.sum()
    return se.sum(), count


def to_torch_batches(lr_patches, hr_patches, mask_patches, batch_size, device):
    n = len(lr_patches)
    batches = []
    for start in range(0, n, batch_size):
        lr = torch.from_numpy(np.ascontiguousarray(lr_patches[start:start + batch_size])).float()
        hr = torch.from_numpy(np.ascontiguousarray(hr_patches[start:start + batch_size])).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask_patches[start:start + batch_size])).bool()
        batches.append((lr.to(device), hr.to(device), mask.to(device)))
    return batches


def train_epoch(
    model, optimizer, cfg, train_indices, lr_min, lr_max, device, epoch, use_amp
):
    model.train()
    se_sum = 0.0
    mask_sum = 0.0
    batches_done = 0
    skipped = 0
    rng = np.random.RandomState(cfg.seed * 1000 + epoch)
    epoch_order = rng.permutation(train_indices)
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp
        else contextlib.nullcontext()
    )

    with h5py.File(str(cfg.h5_dir / cfg.lr_train_h5), "r") as f_lr, \
            h5py.File(str(cfg.h5_dir / cfg.hr_train_h5), "r") as f_hr:
        for sample_idx in epoch_order:
            lr_sample = roi_data.read_sample(f_lr, int(sample_idx))
            hr_sample = roi_data.read_sample(f_hr, int(sample_idx))
            lr_norm = roi_data.normalize_lr_sample(cfg, lr_sample, lr_min, lr_max)
            lr_p, hr_p, mask_p, _ = roi_data.extract_roi_patches(cfg, lr_norm, hr_sample)
            if len(lr_p) == 0:
                skipped += 1
                continue
            for lr_b, hr_b, mask_b in to_torch_batches(
                lr_p, hr_p, mask_p, cfg.batch_size, device
            ):
                if mask_b.sum() == 0:
                    continue
                optimizer.zero_grad()
                with amp_ctx:
                    pred = model(lr_b)
                    se, cnt = roi_loss_terms(pred, hr_b, mask_b)
                    loss = se / (2.0 * cnt + 1e-8)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                se_sum += float(se.detach())
                mask_sum += float(cnt.detach())
                batches_done += 1
    avg = se_sum / (2.0 * mask_sum + 1e-8)
    return avg, batches_done, skipped


@torch.no_grad()
def validate_epoch(model, cfg, val_indices, lr_min, lr_max, device):
    model.eval()
    se_sum = 0.0
    mask_sum = 0.0
    with h5py.File(str(cfg.h5_dir / cfg.lr_train_h5), "r") as f_lr, \
            h5py.File(str(cfg.h5_dir / cfg.hr_train_h5), "r") as f_hr:
        for sample_idx in val_indices:
            lr_sample = roi_data.read_sample(f_lr, int(sample_idx))
            hr_sample = roi_data.read_sample(f_hr, int(sample_idx))
            lr_norm = roi_data.normalize_lr_sample(cfg, lr_sample, lr_min, lr_max)
            lr_p, hr_p, mask_p, _ = roi_data.extract_roi_patches(cfg, lr_norm, hr_sample)
            if len(lr_p) == 0:
                continue
            for lr_b, hr_b, mask_b in to_torch_batches(
                lr_p, hr_p, mask_p, cfg.batch_size, device
            ):
                if mask_b.sum() == 0:
                    continue
                pred = model(lr_b)
                se, cnt = roi_loss_terms(pred, hr_b, mask_b)
                se_sum += float(se)
                mask_sum += float(cnt)
    return se_sum / (2.0 * mask_sum + 1e-8)


def cached_group_info(group):
    sample_ids = group["sample_idx"][:]
    if len(sample_ids) == 0:
        return np.empty((0, 3), dtype=np.int64)
    changes = np.flatnonzero(np.diff(sample_ids) != 0)
    starts = np.concatenate(([0], changes + 1))
    ends = np.concatenate((changes + 1, [len(sample_ids)]))
    return np.stack(
        [starts, ends, sample_ids[starts]], axis=1
    ).astype(np.int64)


def train_epoch_cached(
    model, optimizer, cfg, group, device, epoch, use_amp
):
    model.train()
    info = cached_group_info(group)
    rng = np.random.RandomState(cfg.seed * 1000 + epoch)
    order = rng.permutation(len(info))
    se_sum = 0.0
    mask_sum = 0.0
    batches_done = 0
    skipped = 0
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp
        else contextlib.nullcontext()
    )
    for pos in order:
        start_row, end_row, _ = info[pos]
        lr_p = group["lr"][start_row:end_row]
        hr_p = group["hr"][start_row:end_row]
        mask_p = group["mask"][start_row:end_row, 0] > 0.5
        for lr_b, hr_b, mask_b in to_torch_batches(
            lr_p, hr_p, mask_p, cfg.batch_size, device
        ):
            if mask_b.sum() == 0:
                skipped += 1
                continue
            optimizer.zero_grad()
            with amp_ctx:
                pred = model(lr_b)
                se, cnt = roi_loss_terms(pred, hr_b, mask_b)
                loss = se / (2.0 * cnt + 1e-8)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            se_sum += float(se.detach())
            mask_sum += float(cnt.detach())
            batches_done += 1
    avg = se_sum / (2.0 * mask_sum + 1e-8)
    return avg, batches_done, skipped


@torch.no_grad()
def validate_epoch_cached(model, cfg, group, device):
    model.eval()
    info = cached_group_info(group)
    se_sum = 0.0
    mask_sum = 0.0
    for start_row, end_row, _ in info:
        lr_p = group["lr"][start_row:end_row]
        hr_p = group["hr"][start_row:end_row]
        mask_p = group["mask"][start_row:end_row, 0] > 0.5
        for lr_b, hr_b, mask_b in to_torch_batches(
            lr_p, hr_p, mask_p, cfg.batch_size, device
        ):
            if mask_b.sum() == 0:
                continue
            pred = model(lr_b)
            se, cnt = roi_loss_terms(pred, hr_b, mask_b)
            se_sum += float(se)
            mask_sum += float(cnt)
    return se_sum / (2.0 * mask_sum + 1e-8)


def load_model(cfg, device, model_name):
    sys.path.insert(0, str(cfg.unet_root))
    from model.model_factory import build_model

    return build_model(model_name, input_channels=4, output_channels=2).to(device)


def main():
    args = parse_args()
    cfg = roi_data.get_config(args.case)
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.disable_early_stop:
        cfg.early_stop_patience = 10 ** 9
    model_tag = roi_data.configure_model_paths(cfg, args.model)
    if args.run_tag:
        cfg.weight_dir = (
            cfg.project_root / "Weights" / "Unet" / args.run_tag
        )
        cfg.result_dir = (
            cfg.project_root / "Results" / "Unet" / args.run_tag
        )
        cfg.loss_log_path = cfg.weight_dir / f"roi_loss_log_{args.run_tag}.txt"
        cfg.checkpoint_path = cfg.weight_dir / f"checkpoint_last_{args.run_tag}.ckpt"
        cfg.best_weight_path = cfg.weight_dir / f"best_roi_{args.run_tag}.pth"
    print(f"Model: {args.model} (tag={model_tag})")

    cfg.weight_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    use_amp = args.use_amp and device.type == "cuda"
    if use_amp:
        print("Mixed precision (bfloat16) enabled")
    print(f"Device: {device}")

    indices = roi_data.get_sample_indices(cfg)
    print(
        f"Split: train={len(indices['train'])} val={len(indices['val'])} "
        f"test={len(indices['test'])}"
    )
    lr_min, lr_max = roi_data.ensure_norm_file(cfg, indices["train"])
    print(f"Normalization saved at: {cfg.norm_path}")

    cache_file = None
    if cfg.cache_path.exists():
        cache_file = h5py.File(str(cfg.cache_path), "r")
        print(f"Patch cache loaded: {cfg.cache_path}")

    model = load_model(cfg, device, args.model)
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    start_epoch = 1
    best_val = float("inf")
    halved = False
    if args.resume and cfg.checkpoint_path.exists():
        ckpt = torch.load(str(cfg.checkpoint_path), map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt["best_val"])
        halved = bool(ckpt.get("halved", False))
        print(f"Resumed from epoch {start_epoch - 1} (best val {best_val:.6g})")

    if cfg.loss_log_path.exists() and not args.resume:
        cfg.loss_log_path.unlink()
    if not cfg.loss_log_path.exists():
        cfg.loss_log_path.write_text(
            "Epoch,Train ROI MSE,Val ROI MSE,Batches,Elapsed(s)\n"
        )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg.lr_patience, factor=0.5
    )
    no_improve = 0
    total_start = time.perf_counter()

    epoch = start_epoch
    while epoch <= cfg.num_epochs:
        epoch_start = time.perf_counter()
        try:
            if cache_file is not None:
                train_loss, n_batches, skipped = train_epoch_cached(
                    model, optimizer, cfg, cache_file["train"],
                    device, epoch, use_amp,
                )
                val_loss = validate_epoch_cached(
                    model, cfg, cache_file["val"], device
                )
            else:
                train_loss, n_batches, skipped = train_epoch(
                    model, optimizer, cfg, indices["train"],
                    lr_min, lr_max, device, epoch, use_amp,
                )
                val_loss = validate_epoch(
                    model, cfg, indices["val"], lr_min, lr_max, device
                )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and not halved:
                halved = True
                cfg.batch_size = max(8, cfg.batch_size // 2)
                print(f"OOM detected; batch size reduced to {cfg.batch_size}")
                torch.cuda.empty_cache()
                continue
            raise

        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise RuntimeError(f"Non-finite loss: train={train_loss}, val={val_loss}")

        scheduler.step(val_loss)
        elapsed = time.perf_counter() - epoch_start
        improved = val_loss < best_val - 1e-8
        if improved:
            best_val = val_loss
            torch.save(model.state_dict(), str(cfg.best_weight_path))
            no_improve = 0
        else:
            no_improve += 1

        with cfg.loss_log_path.open("a") as f:
            f.write(
                f"{epoch},{train_loss:.8g},{val_loss:.8g},"
                f"{n_batches},{elapsed:.1f}\n"
            )
        print(
            f"Epoch {epoch}/{cfg.num_epochs} | train {train_loss:.6g} | "
            f"val {val_loss:.6g} | best {best_val:.6g} | skipped {skipped}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val": best_val,
                "halved": halved,
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            str(cfg.checkpoint_path),
        )
        epoch += 1

        if no_improve >= cfg.early_stop_patience:
            print("Early stopping triggered.")
            break

    print(
        f"Training finished in {time.perf_counter() - total_start:.1f}s; "
        f"best ROI val MSE {best_val:.8g}"
    )
    total_seconds = 0.0
    completed_epochs = 0
    if cfg.loss_log_path.exists():
        with cfg.loss_log_path.open() as f:
            next(f, None)
            for line in f:
                fields = line.strip().split(",")
                if len(fields) >= 5 and fields[4].strip():
                    total_seconds += float(fields[4])
                    completed_epochs += 1
    if completed_epochs:
        avg_seconds = total_seconds / completed_epochs
        summary = cfg.weight_dir / f"training_time_{cfg.case_name.lower()}.txt"
        with summary.open("a") as f:
            f.write(
                f"epochs={completed_epochs},total_s={total_seconds:.2f},"
                f"avg_s_per_epoch={avg_seconds:.2f}\n"
            )
        print(
            f"Training time summary: {completed_epochs} epochs, "
            f"{total_seconds:.1f}s total, {avg_seconds:.2f}s/epoch"
        )
        print(f"Saved to: {summary}")
    if cache_file is not None:
        cache_file.close()


if __name__ == "__main__":
    main()
