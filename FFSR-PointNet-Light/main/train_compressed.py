"""Fine-tune a factorized low-rank version of the original model."""

import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from config_light import get_config
from model.FFSRP_original import PointNetRegression as OriginalModel
from model.FFSRP_subsampled_compressed import SubsampledCompressed
from utils.datasetGen import FlowFieldDataset


class PreloadedDataset(Dataset):
    """Dataset that keeps normalized samples in CPU memory."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    config = get_config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    dataset = FlowFieldDataset(
        low_h5_path=config.low_res_folder,
        high_h5_path=config.high_res_folder,
        num_samples=config.use_samples,
        normalize=True,
        mean_std_file=config.mean_std_file,
        return_hr_priors=True,
    )
    n = len(dataset)
    if config.case_name == "sanjiang":
        split_file = getattr(config, "split_map_file", None)
        if split_file and os.path.exists(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                split_map = json.load(f)
            train_idx = sorted(int(i) for i in split_map["point_train"])
            test_idx = sorted(int(i) for i in split_map["point_test"])
            val_idx = sorted(int(i) for i in split_map["point_val"])
            print(
                "Sanjiang split from shared grid/point map "
                f"(train={len(train_idx)}, test={len(test_idx)}, "
                f"val={len(val_idx)})"
            )
        else:
            rng = np.random.RandomState(42)
            perm = rng.permutation(n)
            train_n = int(n * 0.7)
            test_n = int(n * 0.1)
            train_idx = sorted(perm[:train_n].tolist())
            test_idx = sorted(perm[train_n : train_n + test_n].tolist())
            val_idx = sorted(perm[train_n + test_n :].tolist())
            print(
                "Shared grid/point split map not found; using numeric fallback "
                "split. Generate it with make_sanjiang_split_map.py first."
            )
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        os.makedirs(config.save_dir, exist_ok=True)
        test_path = os.path.join(config.save_dir, "sanjiang_test_indices.json")
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_idx, f)
        print(
            f"Sanjiang 70/10/20 split: train={len(train_idx)}, "
            f"test={len(test_idx)}, val={len(val_idx)}"
        )
    else:
        train_ds, val_ds = random_split(dataset, [int(n * 0.8), n - int(n * 0.8)])
        print(f"Shouxi 80/20 split: train={len(train_ds)}, val={len(val_ds)}")
    print("Preloading train/validation tensors into CPU memory...")
    train_data = PreloadedDataset(
        [dataset[idx] for idx in train_ds.indices]
    )
    val_data = PreloadedDataset(
        [dataset[idx] for idx in val_ds.indices]
    )
    print(
        f"Preloaded train={len(train_data)} val={len(val_data)}"
    )
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        pin_memory=torch.cuda.is_available(),
    )

    state = torch.load(config.teacher_path, map_location="cpu")
    if config.model_type == "subsampled":
        model = SubsampledCompressed.from_teacher_state(
            state,
            input_dim=dataset.input_dim,
            global_feat_dim=512,
            output_dim=2,
            N_high=dataset.N_high,
            rank=config.rank,
            context_points=config.context_points,
        )
    else:
        raise ValueError(
            "Only model_type='subsampled' is supported; other Light variants "
            "were moved to Archive_UnusedModels."
        )
    if config.freeze_encoder:
        for param in model.transform.parameters():
            param.requires_grad = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    teacher = None
    if config.teacher_weight > 0.0:
        teacher = OriginalModel(
            input_dim=dataset.input_dim,
            global_feat_dim=512,
            output_dim=2,
            N_high=dataset.N_high,
        )
        teacher.load_state_dict(state)
        teacher.to(device).eval()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    os.makedirs(config.save_dir, exist_ok=True)
    resume_path = os.path.join(config.save_dir, "resume_compressed.ckpt")

    log_path = os.path.join(config.save_dir, "loss_compressed.csv")
    start_epoch = 0
    best_val = float("inf")
    resumed = False
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        best_val = ckpt.get("best_val", float("inf"))
        resumed = True
        print(f"Resumed from epoch {start_epoch}")

    log_file = open(log_path, "a" if resumed and os.path.exists(log_path) else "w")
    if not resumed or not os.path.exists(log_path):
        log_file.write("epoch,train_loss,val_loss,elapsed_s\n")

    total_start = time.perf_counter()
    for epoch in range(start_epoch, config.num_epochs):
        epoch_start = time.perf_counter()
        model.train()
        total = 0.0
        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if teacher is not None:
                with torch.no_grad():
                    teacher_out = teacher(inputs)
                loss = loss + config.teacher_weight * criterion(
                    outputs, teacher_out
                )
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_loss = total / len(train_loader)

        model.eval()
        total = 0.0
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if teacher is not None:
                    teacher_out = teacher(inputs)
                    loss = loss + config.teacher_weight * criterion(
                        outputs, teacher_out
                    )
                total += loss.item()
        val_loss = total / len(val_loader)
        scheduler.step(val_loss)
        elapsed = time.perf_counter() - epoch_start
        log_file.write(
            f"{epoch+1},{train_loss:.8f},{val_loss:.8f},{elapsed:.3f}\n"
        )
        log_file.flush()
        print(f"Epoch {epoch+1}/{config.num_epochs} train={train_loss:.6f} val={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(config.save_dir, "compressed_best.pth"),
            )
        if (
            (epoch + 1) % config.resume_checkpoint_interval == 0
            or epoch + 1 == config.num_epochs
        ):
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "best_val": best_val,
                },
                resume_path,
            )

    torch.save(model.state_dict(), os.path.join(config.save_dir, "compressed_final.pth"))
    log_file.close()

    total_s = time.perf_counter() - total_start
    total_epochs = len(
        [1 for line in open(log_path, "r").read().splitlines()[1:] if line.strip()]
    )
    avg_s = total_s / max(total_epochs, 1)
    time_path = os.path.join(config.save_dir, config.training_time_file)
    with open(time_path, "w") as f:
        f.write(
            f"epochs={total_epochs},total_s={total_s:.2f},"
            f"avg_s_per_epoch={avg_s:.2f}\n"
        )
    print(
        f"Training time: {total_epochs} epochs, "
        f"total {total_s:.1f}s, avg {avg_s:.2f}s/epoch"
    )
    print(f"Saved to {time_path}")


if __name__ == "__main__":
    main()
