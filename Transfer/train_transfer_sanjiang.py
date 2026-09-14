#!/usr/bin/env python
"""Unified lightweight-FFSR transfer trainer with Sanjiang source weights.

Target cases are selected on the command line or through a JSON config:

  python train_transfer_sanjiang.py --case yuhua
  python train_transfer_sanjiang.py --case laoyangcun

The target model is the official lightweight FFSR architecture
`SubsampledCompressed`: subsampled T-Net context points and a rank-limited
factorized decoder. The teacher is the original dense Sanjiang SR-PointNet
checkpoint. Source-compatible layers are copied, while the target-specific
decoder expansion layer is rebuilt for the target HR point count.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import argparse
import h5py
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

LIGHT_MAIN = rf"{ROOT}\FFSR-PointNet-Light\main"
sys.path.insert(0, LIGHT_MAIN)

from model.FFSRP_subsampled_compressed import SubsampledCompressed  # noqa: E402
from utils.datasetGen import FlowFieldDataset  # noqa: E402


DATA_ROOT = rf"{ROOT}\Data\dataset"
TRANSFER_ROOT = rf"{ROOT}\Transfer"
PRETRAINED_PATH = (
    rf"{ROOT}\Weights\FFSR_PointNet"
    r"\best_sanjiang_pretrained.pth"
)

CASE_PROFILES = {
    "yuhua": {
        "display_name": "Yuhua (files named Yuhuo)",
        "low_res": os.path.join(DATA_ROOT, "Yuhuo_LR.h5"),
        "high_res": os.path.join(DATA_ROOT, "Yuhuo_HR.h5"),
        "val_low_res": os.path.join(DATA_ROOT, "Yuhuo_LR_val.h5"),
        "val_high_res": os.path.join(DATA_ROOT, "Yuhuo_HR_val.h5"),
        "out_dir": (
            os.path.join(TRANSFER_ROOT, "outputs", "yuhua_light")
        ),
        "default_samples": 36,
        "num_epochs": 200,
    },
    "laoyangcun": {
        "display_name": "LaoYangCun",
        "low_res": os.path.join(DATA_ROOT, "LaoYangCun_LR.h5"),
        "high_res": os.path.join(DATA_ROOT, "LaoYangCun_HR.h5"),
        "val_low_res": os.path.join(DATA_ROOT, "LaoYangCun_LR_val.h5"),
        "val_high_res": os.path.join(DATA_ROOT, "LaoYangCun_HR_val.h5"),
        "out_dir": (
            os.path.join(TRANSFER_ROOT, "outputs", "laoyangcun_light")
        ),
        "default_samples": 36,
        "num_epochs": 200,
    },
}


def resolve_case(case):
    case = str(case).strip().lower()
    if case == "yuhuo":
        return "yuhua"
    if case not in CASE_PROFILES:
        raise ValueError(f"Unknown case '{case}'")
    return case


def load_profile(case, config_path=None):
    profile = dict(CASE_PROFILES[case])
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        if "case" in user_cfg and resolve_case(user_cfg["case"]) != case:
            raise ValueError("--case and config-file case mismatch")
        profile.update(user_cfg)
    return profile


def build_light_transfer_model(
    teacher_state,
    input_dim,
    global_feat_dim,
    output_dim,
    N_high,
    rank,
    context_points,
):
    model = SubsampledCompressed(
        input_dim=input_dim,
        global_feat_dim=global_feat_dim,
        output_dim=output_dim,
        N_high=N_high,
        rank=rank,
        context_points=context_points,
    )

    # Copy source-compatible encoder.
    with torch.no_grad():
        for key, target in model.state_dict().items():
            if not key.startswith("transform."):
                continue
            if key not in teacher_state:
                raise KeyError(f"Missing source encoder key: {key}")
            target.copy_(teacher_state[key])

        # Copy the shared decoder hidden layer from source `decoder.fc.0`.
        model.decoder[0].load_state_dict(
            {
                "weight": teacher_state["decoder.fc.0.weight"],
                "bias": teacher_state["decoder.fc.0.bias"],
            }
        )

        # SVD of the source final dense decoder gives the transferable input
        # subspace (right singular vectors). The target-specific expansion
        # rows cannot be copied because output rows correspond to target points.
        weight = teacher_state["decoder.fc.2.weight"]
        _, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
        proj = model.decoder[2][0]
        expand = model.decoder[2][1]
        proj.weight.copy_(vh[:rank])

        # Scale-free random target rows; the singular values are re-learned
        # through the factorized decoder during fine-tuning.
        nn.init.kaiming_uniform_(expand.weight, a=5 ** 0.5)
        if expand.bias is not None:
            nn.init.zeros_(expand.bias)

    del teacher_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        "Light FFSR transfer init done: encoder copied, decoder projection "
        f"initialised from top-{rank} singular vectors, target expansion "
        "layer rebuilt."
    )
    return model


def count_parameters(model):
    total = trainable = frozen = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        else:
            frozen += n
    return total, trainable, frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        default=None,
        help="Target case: yuhua (alias yuhuo) or laoyangcun",
    )
    parser.add_argument("--config", default=None, help="Optional JSON config")
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--low-res", default=None)
    parser.add_argument("--high-res", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--use-samples",
        type=int,
        default=None,
        help="Number of randomly selected target samples; <=0 uses all samples.",
    )
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--context-points", type=int, default=None)
    parser.add_argument(
        "--freeze-encoder", dest="freeze_encoder", action="store_true"
    )
    parser.add_argument(
        "--no-freeze-encoder", dest="freeze_encoder", action="store_false"
    )
    parser.set_defaults(freeze_encoder=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    user_cfg = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
    cfg_case = user_cfg.get("case") if isinstance(user_cfg, dict) else None
    case = resolve_case(args.case or cfg_case or "yuhua")
    profile = load_profile(case, args.config)

    low_res = args.low_res or profile["low_res"]
    high_res = args.high_res or profile["high_res"]
    out_dir = args.out_dir or profile["out_dir"]
    use_samples = (
        args.use_samples
        if args.use_samples is not None
        else profile.get("use_samples", profile.get("default_samples", 36))
    )
    if use_samples is not None and use_samples <= 0:
        use_samples = None
    num_epochs = (
        args.num_epochs
        if args.num_epochs is not None
        else profile.get("num_epochs", 200)
    )
    pretrained = args.pretrained or profile.get("pretrained", PRETRAINED_PATH)
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else profile.get("batch_size", 8)
    )
    lr = (
        args.lr
        if args.lr is not None
        else profile.get("learning_rate", 1e-4)
    )
    rank = (
        args.rank
        if args.rank is not None
        else profile.get("rank", 128)
    )
    context_points = (
        args.context_points
        if args.context_points is not None
        else profile.get("context_points", 8192)
    )
    freeze_encoder = bool(
        profile.get("freeze_encoder", args.freeze_encoder)
    )
    seed = int(profile.get("seed", args.seed))

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    sample_tag = str(use_samples) if use_samples is not None else "all"
    mean_std_path = os.path.join(
        out_dir, f"mean_std_{case}_{sample_tag}_{seed}.npz"
    )

    print(f"Case: {case} ({profile['display_name']})")
    print(f"Output directory: {out_dir}")

    sample_indices = None
    if use_samples is not None:
        with h5py.File(high_res, "r") as f:
            n_total = len(f.keys())
        if use_samples >= n_total:
            use_samples = None
        else:
            rng = np.random.RandomState(seed)
            sample_indices = rng.choice(
                n_total, size=use_samples, replace=False
            ).astype(int).tolist()
            print(
                f"Randomly selected {len(sample_indices)} of {n_total} "
                f"target samples (seed={seed})."
            )

    dataset = FlowFieldDataset(
        low_h5_path=low_res,
        high_h5_path=high_res,
        num_samples=use_samples,
        sample_indices=sample_indices,
        normalize=True,
        mean_std_file=mean_std_path,
    )
    n_train = int(len(dataset) * 0.8)
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    print(
        f"Dataset: samples={len(dataset)}, N_low={dataset.N_low}, "
        f"N_high={dataset.N_high}, input_dim={dataset.input_dim}, "
        f"output_dim={dataset.output_dim}"
    )

    print("Loading Sanjiang teacher state...")
    teacher_state = torch.load(pretrained, map_location="cpu")
    model = build_light_transfer_model(
        teacher_state=teacher_state,
        input_dim=dataset.input_dim,
        global_feat_dim=512,
        output_dim=dataset.output_dim,
        N_high=dataset.N_high,
        rank=rank,
        context_points=context_points,
    )
    if freeze_encoder:
        for p in model.transform.parameters():
            p.requires_grad = False
        print("Frozen lightweight encoder/transform.")

    total_params, trainable, frozen = count_parameters(model)
    print(
        f"Parameters: total={total_params/1e6:.2f}M, "
        f"trainable={trainable/1e6:.2f}M, frozen={frozen/1e6:.2f}M"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_path = os.path.join(out_dir, f"best_{case}_light_transfer.pth")
    final_path = os.path.join(out_dir, f"final_{case}_light_transfer.pth")
    log_path = os.path.join(out_dir, f"loss_{case}_light.csv")
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write("epoch,train_loss,val_loss,epoch_s\n")

    best_val = float("inf")
    start_time = time.time()
    epoch_times = []
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        if freeze_encoder:
            model.transform.eval()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            del inputs, targets, outputs, loss

        avg_train = running_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
                del inputs, targets, outputs
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)
        epoch_s = time.time() - epoch_start
        epoch_times.append(epoch_s)
        log_file.write(f"{epoch+1},{avg_train:.8f},{avg_val:.8f},{epoch_s:.2f}\n")
        log_file.flush()
        print(
            f"Epoch {epoch+1}/{num_epochs} train={avg_train:.8f} "
            f"val={avg_val:.8f} time={epoch_s:.1f}s"
        )
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), best_path)

    torch.save(model.state_dict(), final_path)
    log_file.close()
    total_s = time.time() - start_time
    stats = {
        "case": case,
        "model": "SubsampledCompressed",
        "rank": rank,
        "context_points": context_points,
        "total_params": total_params,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "total_time_s": total_s,
        "avg_epoch_time_s": float(np.mean(epoch_times)),
        "best_val_loss": float(best_val),
        "final_train_loss": float(avg_train),
        "final_val_loss": float(avg_val),
    }
    with open(os.path.join(out_dir, f"stats_{case}_light.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"Saved: {best_path}\nSaved: {final_path}")


if __name__ == "__main__":
    main()
