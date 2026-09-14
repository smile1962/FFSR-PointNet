"""Focused numerical validation for the new implicit FFSR-PointNet model."""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import get_config
from model.FFSRP_v2 import FFSRPointNetV2
from utils.datasetGen import FlowFieldDataset


def pearson(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def binary_metrics(pred, true, threshold=0.0):
    pred_binary = pred > threshold
    true_binary = true > threshold
    tp = np.logical_and(pred_binary, true_binary).sum()
    fp = np.logical_and(pred_binary, ~true_binary).sum()
    fn = np.logical_and(~pred_binary, true_binary).sum()
    tn = np.logical_and(~pred_binary, ~true_binary).sum()
    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    return recall, precision, accuracy


def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(config.mean_std_file):
        raise FileNotFoundError(
            "Event-safe normalization file not found. Run modelTraining.py first."
        )

    dataset = FlowFieldDataset(
        low_h5_path=config.val_low_res_folder,
        high_h5_path=config.val_high_res_folder,
        num_samples=config.use_samples_val,
        return_coords=False,
        normalize=True,
        mean_std_file=config.mean_std_file,
        return_meta=False,
        return_index=False,
        return_hr_priors=True,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = FFSRPointNetV2(
        input_dim=config.model_input_dim,
        global_feat_dim=config.global_feat_dim,
        output_dim=config.output_dim,
        N_high=dataset.N_high,
        prior_dim=config.prior_dim,
    )
    checkpoint = torch.load(config.best_save_dir, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    records = []
    amp_dtype = (
        torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
    )
    use_amp = config.use_amp and device.type == "cuda"
    autocast_context = torch.autocast(
        device_type="cuda", dtype=amp_dtype, enabled=use_amp
    )

    with torch.no_grad():
        for inputs, targets, hr_priors in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            hr_priors = hr_priors.to(device)

            lr_physical = np.stack(
                [
                    inputs[:, :, 0].cpu().numpy() * dataset.std_depth
                    + dataset.mean_depth,
                    inputs[:, :, 1].cpu().numpy() * dataset.std_velocity
                    + dataset.mean_velocity,
                ],
                axis=-1,
            )
            with autocast_context:
                outputs = model(inputs, hr_priors)

            pred = outputs.float().cpu().numpy() * dataset.std_out + dataset.mean_out
            true = targets.cpu().numpy() * dataset.std_out + dataset.mean_out

            depth_pred = pred[:, :, 0]
            depth_true = true[:, :, 0]
            vel_pred = pred[:, :, 1]
            vel_true = true[:, :, 1]

            for batch_idx in range(pred.shape[0]):
                dp = depth_pred[batch_idx]
                dt = depth_true[batch_idx]
                vp = vel_pred[batch_idx]
                vt = vel_true[batch_idx]
                d_lr = lr_physical[batch_idx, :, 0]
                v_lr = lr_physical[batch_idx, :, 1]
                d_recall, d_precision, d_accuracy = binary_metrics(
                    dp, dt, threshold=0.05
                )
                v_recall, v_precision, v_accuracy = binary_metrics(
                    vp, vt, threshold=0.05
                )
                d_lr_recall, d_lr_precision, d_lr_accuracy = binary_metrics(
                    d_lr, dt, threshold=0.05
                )
                v_lr_recall, v_lr_precision, v_lr_accuracy = binary_metrics(
                    v_lr, vt, threshold=0.05
                )
                records.append(
                    {
                        "depth_rmse": float(np.sqrt(np.mean((dp - dt) ** 2))),
                        "depth_r": pearson(dp, dt),
                        "depth_recall": d_recall,
                        "depth_precision": d_precision,
                        "depth_accuracy": d_accuracy,
                        "velocity_rmse": float(np.sqrt(np.mean((vp - vt) ** 2))),
                        "velocity_r": pearson(vp, vt),
                        "velocity_recall": v_recall,
                        "velocity_precision": v_precision,
                        "velocity_accuracy": v_accuracy,
                        "baseline_depth_rmse": float(
                            np.sqrt(np.mean((d_lr - dt) ** 2))
                        ),
                        "baseline_depth_r": pearson(d_lr, dt),
                        "baseline_depth_recall": d_lr_recall,
                        "baseline_depth_precision": d_lr_precision,
                        "baseline_depth_accuracy": d_lr_accuracy,
                        "baseline_velocity_rmse": float(
                            np.sqrt(np.mean((v_lr - vt) ** 2))
                        ),
                        "baseline_velocity_r": pearson(v_lr, vt),
                        "baseline_velocity_recall": v_lr_recall,
                        "baseline_velocity_precision": v_lr_precision,
                        "baseline_velocity_accuracy": v_lr_accuracy,
                    }
                )

    frame = pd.DataFrame(records)
    summary = frame.mean(numeric_only=True)
    os.makedirs(config.save_root, exist_ok=True)
    metrics_path = os.path.join(config.save_root, "metrics_v2.csv")
    frame.to_csv(metrics_path, index=False)

    print(f"Test samples: {len(frame)}")
    print(summary.round(4).to_string())
    print(f"Saved per-sample metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
