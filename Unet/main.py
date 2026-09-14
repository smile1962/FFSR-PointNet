import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from model.SRUnet import SuperResolutionModel  # replace with your actual model import path
import time


# Set the random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Custom dataset class
class SRDataset(Dataset):
    def __init__(self, lr_patches, hr_patches):
        self.lr_patches = lr_patches
        self.hr_patches = hr_patches
        assert len(self.lr_patches) == len(self.hr_patches), "low-resolution and high-resolution sample counts do not match"

    def __len__(self):
        return len(self.lr_patches)

    def __getitem__(self, idx):
        lr_patch = torch.from_numpy(self.lr_patches[idx]).float()
        hr_patch = torch.from_numpy(self.hr_patches[idx]).float()
        return lr_patch, hr_patch


# Patch splitting and reconstruction helper class
class PatchHandler:
    @staticmethod
    def extract_patches(image, patch_size=(60, 60), stride=None):
        if stride is None:
            stride = patch_size
        C, H, W = image.shape
        ph, pw = patch_size
        sh, sw = stride

        num_h = (H - ph) // sh + 1
        num_w = (W - pw) // sw + 1

        patches = []
        positions = []
        for i in range(num_h):
            for j in range(num_w):
                start_h = i * sh
                end_h = start_h + ph
                start_w = j * sw
                end_w = start_w + pw
                patch = image[:, start_h:end_h, start_w:end_w]
                patches.append(patch)
                positions.append((start_h, end_h, start_w, end_w))
        return np.array(patches), np.array(positions)

    @staticmethod
    def reconstruct_image(patches, positions, output_shape):
        C, H, W = output_shape
        image = np.zeros(output_shape, dtype=np.float32)
        count = np.zeros((H, W), dtype=np.int32)
        for patch, (start_h, end_h, start_w, end_w) in zip(patches, positions):
            image[:, start_h:end_h, start_w:end_w] += patch
            count[start_h:end_h, start_w:end_w] += 1
        count[count == 0] = 1
        for c in range(C):
            image[c] /= count
        return image


# Data normalization utilities
class Normalizer:
    @staticmethod
    def normalize_per_channel(data):
        N, C, H, W = data.shape
        normalized_data = np.zeros_like(data, dtype=np.float32)
        norms = np.zeros((C, 2), dtype=np.float32)
        for c in range(C):
            channel_data = data[:, c, :, :]
            min_val = np.min(channel_data)
            max_val = np.max(channel_data)
            if max_val - min_val < 1e-8:
                max_val = min_val + 1e-8
            normalized_data[:, c, :, :] = (channel_data - min_val) / (max_val - min_val)
            norms[c] = [min_val, max_val]
        return normalized_data, norms

    @staticmethod
    def denormalize_per_channel(normalized_data, norms):
        if normalized_data.ndim == 3:
            normalized_data = normalized_data[np.newaxis, ...]
        N, C, H, W = normalized_data.shape
        data = np.zeros_like(normalized_data, dtype=np.float32)
        for c in range(C):
            min_val, max_val = norms[c]
            data[:, c, :, :] = normalized_data[:, c, :, :] * (max_val - min_val) + min_val
        return data[0] if N == 1 else data


def main():
    # Configuration parameters (zero_threshold added to control all-zero patch filtering)
    config = {
        "patch_size": (120, 120),
        "batch_size": 64,
        "num_epochs": 500,
        "lr": 1e-4,
        "data_dir": r"G:\Projects\ShouXi\python\SR-PointNet\dataset\sourceData\ParedNew_300mesh\geo_inside",
        "save_dir": r"../resulting",
        "lr_h5": rf"{ROOT}\Data\dataset/LR.h5",
        "hr_h5": rf"{ROOT}\Data\dataset/HR.h5",
        "norm_save_path": "normalization_params.npz",
        "positions_save_path": "patch_positions.npz",
        "model_save_path": "sr_model.pth",
        "split_ratios": [0.7, 0.1, 0.2],
        "sample_limit": 271,
        "zero_threshold": 1e-6  # threshold for detecting all-zero patches (adjust as needed)
    }

    # Device and seed
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute device: {device}")

    # Load the dataset
    print("Loading the dataset...")
    with h5py.File(config["lr_h5"], 'r') as f_lr, h5py.File(config["hr_h5"], 'r') as f_hr:
        lr_data = f_lr['data'][:]
        hr_data = f_hr['data'][:]

    # Take the first N samples (for testing)
    sample_limit = config["sample_limit"]
    if sample_limit is not None and sample_limit > 0:
        lr_data = lr_data[:sample_limit]
        hr_data = hr_data[:sample_limit]
        print(f"Took the first {sample_limit} samples")
    print(f"Low-resolution data shape: {lr_data.shape}")
    print(f"High-resolution data shape: {hr_data.shape}")

    # Split and filter out all-zero high-resolution patches
    print("Splitting and filtering patches...")
    patch_handler = PatchHandler()
    all_lr_patches = []
    all_hr_patches = []
    all_positions = []
    zero_threshold = config["zero_threshold"]

    for i in tqdm(range(len(lr_data)), desc="split + filter"):
        # Split low-resolution patches
        lr_patches, positions = patch_handler.extract_patches(lr_data[i], config["patch_size"])
        # Split high-resolution patches (same locations)
        hr_patches, _ = patch_handler.extract_patches(hr_data[i], config["patch_size"])

        # Filter: keep only patches whose high-resolution depth channel (channel 0) contains valid values
        valid_indices = []
        for j, hr_patch in enumerate(hr_patches):
            if np.max(hr_patch[0]) > zero_threshold:  # assume channel 0 is water depth
                valid_indices.append(j)

        # Keep valid patches
        valid_lr = lr_patches[valid_indices]
        valid_hr = hr_patches[valid_indices]
        valid_pos = [positions[j] for j in valid_indices]

        all_lr_patches.extend(valid_lr)
        all_hr_patches.extend(valid_hr)
        all_positions.extend([(i, *pos) for pos in valid_pos])  # record the source image index and position

    all_lr_patches = np.array(all_lr_patches)
    all_hr_patches = np.array(all_hr_patches)
    all_positions = np.array(all_positions)

    print(f"Filtered low-resolution patch shape: {all_lr_patches.shape}")
    print(f"Filtered high-resolution patch shape: {all_hr_patches.shape}")
    print(f"Total valid patches: {len(all_lr_patches)}")
    if len(all_lr_patches) == 0:
        print("Warning: no valid patches! Adjust zero_threshold or check the data")
        return

    # Save position information
    np.savez(os.path.join(config["save_dir"], config["positions_save_path"]), positions=all_positions)

    # Whether to normalize
    user_input = input("Normalize the data? (y/n): ").strip().lower()
    do_normalize = user_input == 'y'

    if do_normalize:
        print("Applying partial channel normalization (channels 3 and 4 of the low-resolution data only)...")
        lr_normed = np.copy(all_lr_patches)
        hr_normed = all_hr_patches  # high-resolution data is not normalized

        # Apply min-max normalization to channels 3 and 4 (indices 2 and 3)
        for c in [2, 3]:
            channel_data = lr_normed[:, c, :, :]
            min_val = np.min(channel_data)
            max_val = np.max(channel_data)
            if max_val - min_val < 1e-8:
                max_val = min_val + 1e-8
            lr_normed[:, c, :, :] = (channel_data - min_val) / (max_val - min_val)
        print("Normalization complete.")
    else:
        print("Skipping normalization; using the raw data.")
        lr_normed, hr_normed = all_lr_patches, all_hr_patches



    # Split the dataset
    print("Splitting the dataset...")
    total = len(lr_normed)
    train_size = int(config["split_ratios"][0] * total)
    test_size = int(config["split_ratios"][1] * total)
    val_size = total - train_size - test_size

    indices = np.random.permutation(total)
    train_idx, test_idx, val_idx = indices[:train_size], indices[train_size:train_size+test_size], indices[train_size+test_size:]

    train_lr, train_hr = lr_normed[train_idx], hr_normed[train_idx]
    test_lr, test_hr = lr_normed[test_idx], hr_normed[test_idx]
    val_lr, val_hr = lr_normed[val_idx], hr_normed[val_idx]

    print(f"Train/test/val sizes: {len(train_lr)}/{len(test_lr)}/{len(val_lr)}")

    # Data loader
    train_ds = SRDataset(train_lr, train_hr)
    test_ds = SRDataset(test_lr, test_hr)
    val_ds = SRDataset(val_lr, val_hr)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, pin_memory=True)

    # Initialize the model
    model = SuperResolutionModel().to(device)  # replace with your model class

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Load pretrained weights (if available)
    model_path = os.path.join(config["save_dir"], config["model_save_path"])
    if os.path.exists(model_path):
        print(f"Loading existing weights: {model_path}")
        model.load_state_dict(torch.load(model_path))

    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    start_time = time.perf_counter()

    # Loss log file
    loss_log = os.path.join(config["save_dir"], "loss_records.txt")
    with open(loss_log, 'w') as f:
        f.write("Epoch,Training Loss,Validation Loss\n")

    for epoch in range(config["num_epochs"]):
        # Training stage
        model.train()
        train_loss = 0.0
        for lr_p, hr_p in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            lr_p, hr_p = lr_p.to(device), hr_p.to(device)
            optimizer.zero_grad()
            out = model(lr_p)
            loss = criterion(out, hr_p)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * lr_p.size(0)
        avg_train = train_loss / len(train_ds)
        train_losses.append(avg_train)

        # Validation stage
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_p, hr_p in val_loader:
                lr_p, hr_p = lr_p.to(device), hr_p.to(device)
                out = model(lr_p)
                loss = criterion(out, hr_p)
                val_loss += loss.item() * lr_p.size(0)
        avg_val = val_loss / len(val_ds)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        # Record the loss
        with open(loss_log, 'a') as f:
            f.write(f"{epoch+1},{avg_train:.6f},{avg_val:.6f}\n")

        print(f"Epoch {epoch+1} | train loss: {avg_train:.6f} | val loss: {avg_val:.6f}")

        # Save the best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), model_path)
            print(f"Saving the best model (val loss: {best_val_loss:.6f})")

    # Test-set evaluation
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for lr_p, hr_p in test_loader:
            lr_p, hr_p = lr_p.to(device), hr_p.to(device)
            out = model(lr_p)
            loss = criterion(out, hr_p)
            test_loss += loss.item() * lr_p.size(0)
    print(f"Test-set loss: {test_loss / len(test_ds):.6f}")

    # Training time logging
    end_time = time.perf_counter()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    with open(os.path.join(config["data_dir"], "training_time.txt"), 'w') as f:
        f.write(time_str)
    print(f"Total training time: {time_str}")


if __name__ == "__main__":
    main()