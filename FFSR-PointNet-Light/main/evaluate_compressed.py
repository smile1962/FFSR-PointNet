import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config_light import get_config
from model.FFSRP_subsampled_compressed import SubsampledCompressed
from utils.datasetGen import FlowFieldDataset


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    config = get_config()
    dataset = FlowFieldDataset(
        low_h5_path=config.val_low_res_folder,
        high_h5_path=config.val_high_res_folder,
        normalize=True,
        mean_std_file=config.mean_std_file,
        return_hr_priors=True,
    )
    index_file = os.path.join(config.save_dir, "sanjiang_test_indices.json")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            indices = json.load(f)
        eval_ds = Subset(dataset, indices)
        print(f"Using {len(indices)} Sanjiang test samples from split JSON.")
    else:
        eval_ds = dataset
    loader = DataLoader(eval_ds, batch_size=2, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    trained_state = torch.load(
        os.path.join(config.save_dir, "compressed_best.pth"),
        map_location="cpu",
    )
    model.load_state_dict(trained_state)
    model.to(device).eval()

    stats = {
        "depth_rmse": 0.0,
        "depth_r": 0.0,
        "vel_rmse": 0.0,
        "vel_r": 0.0,
    }
    n = 0
    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            pred = model(inputs).float().cpu().numpy()
            true = targets.cpu().numpy() * dataset.std_out + dataset.mean_out
            pred = pred * dataset.std_out + dataset.mean_out
            for b in range(pred.shape[0]):
                stats["depth_rmse"] += float(np.sqrt(np.mean((pred[b, :, 0] - true[b, :, 0]) ** 2)))
                stats["depth_r"] += corr(pred[b, :, 0], true[b, :, 0])
                stats["vel_rmse"] += float(np.sqrt(np.mean((pred[b, :, 1] - true[b, :, 1]) ** 2)))
                stats["vel_r"] += corr(pred[b, :, 1], true[b, :, 1])
                n += 1
    for key in stats:
        stats[key] /= n
    print(stats)


if __name__ == "__main__":
    main()
