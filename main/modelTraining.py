import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from config import get_config
from model.FFSRP import PointNetRegression
from utils.datasetGen import FlowFieldDataset
from utils.datasetGen_OneChannel import FlowFieldDataset_OneChannel
import numpy as np

def train_model():
    # Retrieve configuration
    config = get_config()

    # Set random seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)

    # # Ensure data directories exist
    # os.makedirs(config.low_res_folder, exist_ok=True)
    # os.makedirs(con fig.high_res_folder, exist_ok=True)

    # Build the dataset
    dataset = FlowFieldDataset(
        low_h5_path=config.low_res_folder,
        high_h5_path=config.high_res_folder,
        # channels=2, # if you want to predict the depth field, please enter 1, or you want to train the velocity field, please enter 2
        num_samples=config.use_samples,
        return_coords=False,
        normalize=True,
        mean_std_file=config.mean_std_file,
        # For the SWSB case, please comment out the two lines of x _ range and y _ range.
        x_range=(-9999999, 9999999999999999999),
        y_range=(-9999999, 9999999999999999999),
        return_meta = False,   # 不直接返回 dict meta（更稳健）
        return_index = False
    )

    # Split the dataset into training, testing, and validation sets
    train_size = int(config.use_samples * config.train_ratio)
    test_size = int(config.use_samples * config.test_ratio)
    val_size = config.use_samples - train_size - test_size

    train_data, test_data, val_data = random_split(dataset, [train_size, test_size, val_size])

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    print("Number of points in each low-resolution sample:", dataset.N_low)
    print("Number of points in each high-resolution sample:", dataset.N_high)
    print("Training feature dimension:", dataset.input_dim)
    print("Target feature dimension:", dataset.output_dim)
    print("Total number of samples in dataset:", len(dataset))

    # Build the model using input/output dimensions from the dataset
    model = PointNetRegression(
        input_dim=dataset.input_dim,
        global_feat_dim=config.global_feat_dim,
        output_dim=dataset.output_dim,
        N_high=dataset.N_high
    )

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    # Create directory for saving model weights
    os.makedirs(config.save_dir, exist_ok=True)
    best_model_path = os.path.join(config.best_save_dir)
    best_loss = float('inf')
    start_epoch = 0

    # Load existing weights if available
    if os.path.exists(best_model_path):
        print(f"Detected existing weights at {best_model_path}, resuming training...")
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("No existing weights detected, starting training from scratch...")

    # Open the loss log file
    loss_log_path = os.path.join(config.save_dir, "loss_log.txt")
    loss_file = open(loss_log_path, "w", encoding="utf-8")
    loss_file.write("epoch,train_loss,test_loss\n")

    # Set device and move the model to the corresponding device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)    # (B, N_low, input_dim)
            targets = targets.to(device)  # (B, N_high, output_dim)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config.num_epochs}], Training Loss: {avg_loss:.4f}")

        # Validation phase after each epoch
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for test_inputs, test_targets in val_loader:
                test_inputs = test_inputs.to(device)
                test_targets = test_targets.to(device)
                test_output = model(test_inputs)
                test_loss += criterion(test_output, test_targets).item()
        avg_test_loss = test_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{config.num_epochs}], Test Loss: {avg_test_loss:.4f}")

        # Write loss logs
        loss_file.write(f"{epoch + 1},{avg_loss:.6f},{avg_test_loss:.6f}\n")
        loss_file.flush()

        # Save the best model weights
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), best_model_path)

    # Save model weights at the final epoch
    epoch_save_path = os.path.join(config.final_save_dir)
    torch.save(model.state_dict(), epoch_save_path)
    loss_file.close()
    print("Training finished.")