import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from config import get_config
from model.FFSRP_v2 import FFSRPointNetV2
from utils.datasetGen import FlowFieldDataset, split_by_events


def _print_event_counts(dataset, train_indices, val_indices):
    train_events = Counter(dataset.meta_list[idx].get("event", "?") for idx in train_indices)
    val_events = Counter(dataset.meta_list[idx].get("event", "?") for idx in val_indices)
    print("Training event counts:", dict(train_events))
    print("Validation event counts:", dict(val_events))


def train_model():
    config = get_config()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.benchmark = True

    if not os.path.exists(config.mean_std_file):
        os.makedirs(
            os.path.dirname(os.path.abspath(config.mean_std_file)),
            exist_ok=True,
        )
        stats_dataset = FlowFieldDataset(
            low_h5_path=config.low_res_folder,
            high_h5_path=config.high_res_folder,
            num_samples=config.use_samples,
            return_coords=False,
            normalize=True,
            mean_std_file=config.mean_std_file,
            return_meta=True,
            stat_events=config.train_events,
        )
        print(
            f"Computed event-safe normalization statistics from "
            f"{len(config.train_events)} training events."
        )
        del stats_dataset

    dataset = FlowFieldDataset(
        low_h5_path=config.low_res_folder,
        high_h5_path=config.high_res_folder,
        num_samples=config.use_samples,
        return_coords=False,
        normalize=True,
        mean_std_file=config.mean_std_file,
        x_range=(-9999999, 9999999999999999999),
        y_range=(-9999999, 9999999999999999999),
        return_meta=True,
        return_index=False,
        return_hr_priors=True,
    )

    if config.use_event_split:
        train_data, val_data = split_by_events(
            dataset, config.train_events, config.val_events
        )
        train_indices = train_data.indices
        val_indices = val_data.indices
    else:
        train_size = int(len(dataset) * config.train_ratio)
        test_size = int(len(dataset) * config.test_ratio)
        val_size = len(dataset) - train_size - test_size
        train_data, _, val_data = random_split(
            dataset, [train_size, test_size, val_size]
        )
        train_indices = train_data.indices
        val_indices = val_data.indices

    _print_event_counts(dataset, train_indices, val_indices)

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=config.num_workers,
    )

    print("Number of points per sample:", dataset.N_high)
    print("Dataset input feature dimension:", dataset.input_dim)
    print("Target feature dimension:", dataset.output_dim)
    print("Total samples loaded:", len(dataset))
    print("Training samples:", len(train_data))
    print("Validation samples:", len(val_data))

    model = FFSRPointNetV2(
        input_dim=config.model_input_dim,
        global_feat_dim=config.global_feat_dim,
        output_dim=config.output_dim,
        N_high=dataset.N_high,
        prior_dim=config.prior_dim,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    amp_dtype = (
        torch.bfloat16
        if config.amp_dtype == "bfloat16"
        else torch.float16
    )
    use_amp = config.use_amp and device.type == "cuda"
    autocast_context = torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=use_amp,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    os.makedirs(config.save_dir, exist_ok=True)
    best_model_path = config.best_save_dir
    resume_path = config.resume_checkpoint
    os.makedirs(os.path.dirname(os.path.abspath(resume_path)), exist_ok=True)
    best_loss = float("inf")
    start_epoch = 0
    resumed = False

    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint.get("epoch", 0)
        best_loss = checkpoint.get("best_loss", best_loss)
        resumed = True
        print(
            f"Resuming from {resume_path}, next_epoch={start_epoch + 1}, "
            f"best_loss={best_loss:.6f}"
        )
    elif os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            best_loss = checkpoint.get("best_loss", best_loss)
        else:
            model.load_state_dict(checkpoint)
        print(
            f"Legacy resume from best weights {best_model_path}, "
            f"start_epoch={start_epoch}, best_loss={best_loss:.6f}"
        )
    elif hasattr(model, "initialize_output_head"):
        model.initialize_output_head(dataset.mean_out, dataset.std_out)

    if resumed and start_epoch >= config.num_epochs:
        print(
            f"Checkpoint is already at epoch {start_epoch}. "
            f"Pass a larger --num_epochs to continue."
        )
        return

    os.makedirs(os.path.dirname(os.path.abspath(config.loss_log_path)), exist_ok=True)
    loss_file = open(
        config.loss_log_path,
        "a" if resumed and os.path.exists(config.loss_log_path) else "w",
        encoding="utf-8",
    )
    if not resumed or not os.path.exists(config.loss_log_path):
        loss_file.write("epoch,train_loss,val_loss,lr,elapsed_s\n")

    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(start_epoch, config.num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for inputs, targets, hr_priors, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            hr_priors = hr_priors.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context:
                outputs = model(inputs, hr_priors)
                loss = criterion(outputs, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch + 1}: {loss.item()}"
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, hr_priors, _ in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                hr_priors = hr_priors.to(device, non_blocking=True)
                with autocast_context:
                    outputs = model(inputs, hr_priors)
                    val_loss += criterion(outputs, targets).item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_elapsed = time.time() - epoch_start

        loss_file.write(
            f"{epoch + 1},{avg_train_loss:.8f},{avg_val_loss:.8f},"
            f"{current_lr:.8f},{epoch_elapsed:.3f}\n"
        )
        loss_file.flush()

        print(
            f"Epoch [{epoch + 1}/{config.num_epochs}] "
            f"train={avg_train_loss:.6f} val={avg_val_loss:.6f} lr={current_lr:.2e}"
        )

        is_best = avg_val_loss < best_loss
        if is_best:
            best_loss = avg_val_loss
            epochs_without_improvement = 0
            best_checkpoint = {
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_loss": best_loss,
            }
            torch.save(best_checkpoint, best_model_path)
        else:
            epochs_without_improvement += 1

        save_resume = (
            (epoch + 1) % config.resume_checkpoint_interval == 0
            or epoch + 1 == config.num_epochs
        )
        if save_resume:
            resume_checkpoint = {
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_loss": best_loss,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }
            if config.save_optimizer_state:
                resume_checkpoint["optimizer"] = optimizer.state_dict()
                resume_checkpoint["scheduler"] = scheduler.state_dict()
                resume_checkpoint["scaler"] = scaler.state_dict()
            torch.save(resume_checkpoint, resume_path)

        if not is_best:
            if epochs_without_improvement >= config.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch + 1}; "
                    f"no validation improvement for "
                    f"{config.early_stop_patience} epochs."
                )
                break

    torch.save(model.state_dict(), config.final_save_dir)
    loss_file.close()
    elapsed = time.time() - start_time
    print(f"Training finished. Elapsed: {elapsed:.1f} s")

    total_epochs = 0
    total_s = 0.0
    with open(config.loss_log_path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            fields = line.strip().split(",")
            if len(fields) >= 5 and fields[4].strip():
                total_s += float(fields[4])
                total_epochs += 1
    avg_s = total_s / max(total_epochs, 1)
    time_path = os.path.join(config.save_dir, "training_time_shouxi.txt")
    with open(time_path, "w", encoding="utf-8") as f:
        f.write(
            f"epochs={total_epochs},total_s={total_s:.2f},"
            f"avg_s_per_epoch={avg_s:.2f}\n"
        )
    print(
        f"Training time: {total_epochs} epochs, total {total_s:.1f}s, "
        f"avg {avg_s:.2f}s/epoch"
    )
    print(f"Saved to {time_path}")
