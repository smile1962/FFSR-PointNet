# Standard library imports
import os
import gc
import time

# Third-party library imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Local module imports
from config_yuhuo import get_config
from model.FFSRP import PointNetRegression
from utils.datasetGen import FlowFieldDataset
from utils.datasetGen_OneChannel import FlowFieldDataset_OneChannel


# Note: The following classes are assumed to be defined in imported modules
# (TNet, Transform, PointNetDecoder, PointNetRegression)
# No need to redefine here; ensure they are available in runtime context


def load_transfer_weights(
        model: nn.Module,
        pretrained_path: str,
        device: torch.device
) -> None:
    """
    Core transfer learning weight loading function.
    Performs weight loading, dimension-aware slicing, and memory cleanup.

    Args:
        model: Target PyTorch model to load weights into
        pretrained_path: File path to pretrained weight checkpoint
        device: Target device for model deployment
    """
    print(f"Loading pretrained weights from: {pretrained_path}")

    # 1. Load pretrained weights to CPU (avoids GPU memory occupation)
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    model_dict = model.state_dict()
    new_state_dict = {}

    # 2. Process weights layer by layer with dimension matching logic
    for k, v in pretrained_dict.items():
        if k not in model_dict:
            continue

        current_shape = model_dict[k].shape
        pretrained_shape = v.shape

        # Case A: Exact shape match - load directly
        if current_shape == pretrained_shape:
            new_state_dict[k] = v

        # Case B: Input layer convolution kernel (channel reduction: 4 → 2)
        # Affected layers:
        # - transform.input_transform.conv1.weight (TNet input layer)
        # - transform.conv1.weight (backbone input layer)
        # Kernel shape format: [out_channels, in_channels, kernel_size]
        elif (('input_transform.conv1.weight' in k) or ('transform.conv1.weight' in k)) \
                and len(pretrained_shape) == 3 \
                and pretrained_shape[1] == 4 and current_shape[1] == 2:

            print(f"Transferring {k}: Slicing input channels from 4 to 2")
            # Keep first 2 channels of pretrained weights
            new_state_dict[k] = v[:, :2, :]

        # Case C: TNet output fully connected layer (dimension mismatch)
        # Layer: transform.input_transform.fc3 (outputs k×k transformation matrix)
        # Original: 4×4=16 dimensions, New: 2×2=4 dimensions
        # Skip this layer (reinitialize) due to geometric meaning mismatch
        elif 'input_transform.fc3' in k:
            print(
                f"Skipping {k}: Dimension mismatch {pretrained_shape} → {current_shape}. "
                "Layer will be reinitialized."
            )
            continue

        # Case D: Final decoder output layer (point count mismatch)
        # Layer: decoder.fc.2.weight (outputs N_high × output_dim)
        # Skip this layer (reinitialize) due to complete size change from different resolution
        elif 'decoder.fc' in k and current_shape != pretrained_shape:
            print(
                f"Skipping {k}: Output dimension mismatch (point count change). "
                "Layer will be reinitialized."
            )
            continue

        # Case E: Other unhandled dimension mismatches
        else:
            print(f"Warning: Layer {k} shape mismatch {pretrained_shape} vs {current_shape}. Skipped.")

    # 3. Load processed weights into model (strict=False allows partial initialization)
    model.load_state_dict(new_state_dict, strict=False)

    # 4. Critical memory cleanup: delete temporary variables and trigger garbage collection
    del pretrained_dict
    del new_state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Transfer learning weight loading complete. Memory cleaned.")


def train_model_transfer() -> None:
    """Main training function with transfer learning support."""
    # Load experiment configuration
    config = get_config()

    # Set random seeds for full reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)

    # Initialize dataset (Dataset B with 2 input channels)
    dataset = FlowFieldDataset(
        low_h5_path=config.low_res_folder,
        high_h5_path=config.high_res_folder,
        # channels=2,  # Uncomment to explicitly set channel count
        num_samples=config.use_samples,
        return_coords=False,
        normalize=True,
        mean_std_file=config.mean_std_file,
        x_range=(-9999999, 9999999999999999999),
        y_range=(-9999999, 9999999999999999999),
        return_meta=False,
        return_index=False
    )

    # Train/validation/test split
    train_size = int(config.use_samples * config.train_ratio)
    test_size = int(config.use_samples * config.test_ratio)
    val_size = config.use_samples - train_size - test_size
    train_data, test_data, val_data = random_split(dataset, [train_size, test_size, val_size])

    # Create DataLoaders
    # Adjust num_workers based on your CPU core count (typically 4-8)
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )

    # Print dataset metadata
    print("-" * 30)
    print("Dataset B Configuration:")
    print(f"Input Points (N_low): {dataset.N_low}")
    print(f"Output Points (N_high): {dataset.N_high}")
    print(f"Input Channels: {dataset.input_dim} (Expected: 2)")
    print(f"Output Channels: {dataset.output_dim}")
    print("-" * 30)

    # Set computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model (architecture adapted for Dataset B dimensions)
    model = PointNetRegression(
        input_dim=dataset.input_dim,  # Should be 2 for Dataset B
        global_feat_dim=config.global_feat_dim,
        output_dim=dataset.output_dim,
        N_high=dataset.N_high  # Output point count for Dataset B
    )

    # Move model to device BEFORE initializing optimizer (best practice)
    model.to(device)

    # =========================================================
    # Transfer Learning & Checkpoint Resume Logic
    # =========================================================
    pretrained_path = config.pretrained_files
    best_model_path = os.path.join(config.best_save_dir)

    # Priority: 1. Resume from existing best checkpoint > 2. Load pretrained weights > 3. Train from scratch
    if os.path.exists(best_model_path):
        print(f"Found existing checkpoint at {best_model_path}. Resuming training...")
        model.load_state_dict(torch.load(best_model_path))
    elif os.path.exists(pretrained_path):
        print("Initiating transfer learning from pretrained weights...")
        load_transfer_weights(model, pretrained_path, device)
    else:
        print(f"No pretrained weights found at {pretrained_path}. Training from random initialization.")

    # Define loss function
    criterion = nn.MSELoss()

    # Transfer learning best practice: use differential learning rates
    # Lower LR for pretrained layers, higher LR for reinitialized layers
    # Uncomment below to enable differential learning rates:
    """
    ignored_params = (
        list(map(id, model.decoder.fc.parameters())) +
        list(map(id, model.transform.input_transform.parameters())) +
        list(map(id, model.transform.conv1.parameters()))
    )
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer = optim.Adam([
        {'params': base_params, 'lr': config.learning_rate * 0.1},  # Pretrained backbone
        {'params': model.decoder.fc.parameters(), 'lr': config.learning_rate},
        {'params': model.transform.input_transform.parameters(), 'lr': config.learning_rate},
        {'params': model.transform.conv1.parameters(), 'lr': config.learning_rate}
    ], lr=config.learning_rate)
    """
    # Default: use uniform global learning rate
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)

    # Initialize loss logging
    loss_log_path = config.final_save_loss_dir
    loss_file = open(loss_log_path, "w", encoding="utf-8")
    loss_file.write("epoch,train_loss,test_loss\n")

    # Training state initialization
    best_loss = float('inf')
    start_epoch = 0

    print("Starting training process...")
    start_time = time.time()

    # Main training loop
    for epoch in range(start_epoch, config.num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)  # Shape: (B, N_low, 2)
            targets = targets.to(device)  # Shape: (B, N_high, output_dim)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{config.num_epochs}] Training Loss: {avg_train_loss:.6f}")

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)
                val_outputs = model(val_inputs)
                val_running_loss += criterion(val_outputs, val_targets).item()

        avg_val_loss = val_running_loss / len(val_loader)

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{config.num_epochs}] Validation Loss: {avg_val_loss:.6f}")

        # Write loss log
        loss_file.write(f"{epoch + 1},{avg_train_loss:.6f},{avg_val_loss:.6f}\n")
        loss_file.flush()

        # Save best model checkpoint
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

    # Save final model weights
    final_model_path = os.path.join(config.final_save_dir)
    torch.save(model.state_dict(), final_model_path)
    loss_file.close()

    # Print completion summary
    end_time = time.time()
    print("Training completed successfully.")
    print(f"Total training duration: {end_time - start_time:.2f} seconds")