import os
import time
import torch
from torch.utils.data import DataLoader

# Import the required local configuration and model classes
from config import get_config
from utils.datasetGen import FlowFieldDataset
from model.FFSRP_v2 import FFSRPointNetV2

def measure_inference_performance():
    # 1. Load the base configuration
    config = get_config()
    use_samples = config.use_samples_val
    batch_size = 1  # inference batch size
    global_feat_dim = config.global_feat_dim
    mean_std_files = config.mean_std_file
    if not os.path.exists(config.mean_std_file):
        raise FileNotFoundError(
            "Event-safe normalization file not found. Run modelTraining.py first."
        )

    # 2. Device selection (prefer GPU, fall back to CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running benchmark on device: {device}")

    # 3. Initialize the dataset and data loader
    dataset = FlowFieldDataset(
        low_h5_path=config.val_low_res_folder,
        high_h5_path=config.val_high_res_folder,
        num_samples=use_samples,
        return_coords=True,
        normalize=True,
        mean_std_file=mean_std_files,
        x_range=(-9999999, 9999999999999999999),
        y_range=(-9999999, 9999999999999999999),
        return_meta=False,
        return_index=False,
        return_hr_priors=True
    )

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True if device.type == 'cuda' else False
    )

    # 4. Initialize the model and load pretrained weights
    model = FFSRPointNetV2(
        input_dim=config.model_input_dim,
        global_feat_dim=global_feat_dim,
        output_dim=config.output_dim,
        N_high=dataset.N_high,
        prior_dim=config.prior_dim,
    )

    model_path = config.best_save_dir
    assert os.path.exists(model_path), f"Model weight file not found: {model_path}"
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # 5. Warm up the model to remove the first-call GPU latency
    dummy_input = torch.randn(1, dataset.N_low, dataset.input_dim).to(device)
    dummy_priors = torch.rand(1, dataset.N_high, config.prior_dim).to(device)
    with torch.no_grad():
        _ = model(dummy_input, dummy_priors)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 6. Start measuring pure model inference time
    total_samples = 0
    total_inference_time = 0.0

    print("Starting timing inference loop...")
    with torch.no_grad():
        for inputs, _, hr_priors, _, _ in test_loader:
            inputs = inputs.to(device)
            hr_priors = hr_priors.to(device)
            batch_len = inputs.size(0)

            # Time only the model forward pass precisely
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            outputs = model(inputs, hr_priors)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_end = time.perf_counter()

            # Accumulate elapsed time and sample count
            total_inference_time += (t_end - t_start)
            total_samples += batch_len

    # 7. Print the statistics
    print("\n" + "="*40)
    print("INFERENCE BENCHMARK RESULTS")
    print("="*40)
    print(f"Total processed samples : {total_samples}")
    print(f"Total inference time    : {total_inference_time:.6f} seconds")
    print(f"Average time per sample : {(total_inference_time / total_samples)*1000:.3f} ms")
    print("="*40)

if __name__ == "__main__":
    measure_inference_performance()
