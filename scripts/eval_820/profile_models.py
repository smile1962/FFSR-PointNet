"""Count parameters and GFLOPs for the four compared architectures.

GFLOPs here counts multiply-add operations as two FLOPs (MAC x 2) for all
Conv1d/Conv2d/Linear layers and the two PointNet batched matrix transforms.
Point models use one 126460-point sample with 4 input channels. CNN models
use one padded 4x1200x1800 input (the full watershed grid used by inference).
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAIN = ROOT / "FFSR-PointNet" / "main"
LIGHT_MAIN = ROOT / "FFSR-PointNet-Light" / "main"
UNET = ROOT / "Unet"
N_POINTS = 126460
IN_CHANNELS = 4
# Mean number of ROI-wet 120x120 patches retained per Shouxi training/validation
# sample (248 train + 24 validation samples) after dry-patch filtering.
WET_PATCH_MEAN = 87.74632352941177


def load_file(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_flops(model, input_provider, device="cpu"):
    macs = 0
    counters = {"macs": 0}
    original_bmm = torch.bmm

    def counting_bmm(a, b):
        # a: (B, k, N), b: (B, k, k) for PointNet transforms.
        if a.dim() == 3 and b.dim() == 3:
            batch = a.size(0)
            k = a.size(1)
            n = a.size(2)
            counters["macs"] += batch * k * k * n
        return original_bmm(a, b)

    def hook(module, inp, out):
        if isinstance(module, nn.Conv1d):
            x = inp[0]
            n = x.shape[-1]
            counters["macs"] += (
                x.shape[0]
                * module.out_channels
                * module.in_channels
                * module.kernel_size[0]
                * n
            )
        elif isinstance(module, nn.Conv2d):
            x = inp[0]
            out_h = out.shape[-2]
            out_w = out.shape[-1]
            counters["macs"] += (
                x.shape[0]
                * module.out_channels
                * module.in_channels
                * module.kernel_size[0]
                * module.kernel_size[1]
                * out_h
                * out_w
            )
        elif isinstance(module, nn.Linear):
            x = inp[0]
            counters["macs"] += x.numel() // x.shape[-1] * module.out_features * module.in_features

    handles = []
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    torch.bmm = counting_bmm
    try:
        with torch.no_grad():
            inputs = input_provider(model)
            if isinstance(inputs, (tuple, list)):
                model(*inputs)
            else:
                model(inputs)
    finally:
        torch.bmm = original_bmm
        for handle in handles:
            handle.remove()
    return counters["macs"] * 2.0 / 1e9


def load_models():
    original_mod = load_file(MAIN / "model" / "FFSRP.py", "ffsrp_orig_profile")
    PointNetRegression = original_mod.PointNetRegression

    sys.path.insert(0, str(LIGHT_MAIN))
    light_mod = load_file(
        LIGHT_MAIN / "model" / "FFSRP_subsampled_compressed.py",
        "ffsrp_light_profile",
    )
    SubsampledCompressed = light_mod.SubsampledCompressed

    srunet_mod = load_file(UNET / "model" / "SRUnet.py", "srunet_profile")
    SuperResolutionModel = srunet_mod.SuperResolutionModel

    flo_mod = load_file(UNET / "model" / "FLO_SR.py", "flo_profile")
    FLOSRModel = flo_mod.FLOSRModel

    models = {
        "SR-PointNet": PointNetRegression(
            input_dim=4, global_feat_dim=512, output_dim=2, N_high=N_POINTS
        ),
        "FFSR-PointNet-Light": SubsampledCompressed(
            input_dim=4,
            global_feat_dim=512,
            output_dim=2,
            N_high=N_POINTS,
            rank=128,
            context_points=8192,
        ),
        "SR-Unet": SuperResolutionModel(
            input_channels=4, output_channels=2
        ),
        "FLO-SR": FLOSRModel(input_channels=4, output_channels=2),
    }
    return models


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models()
    rows = []
    for name, model in models.items():
        model.to(device).eval()

        if name in ("SR-PointNet", "FFSR-PointNet-Light"):

            def provider(model=model):
                return torch.zeros(
                    1, N_POINTS, IN_CHANNELS, dtype=torch.float32, device=device
                )

        else:

            def provider(model=model):
                return torch.zeros(
                    1, 4, 120, 120, dtype=torch.float32, device=device
                )

        per_patch_or_full = count_flops(model, provider, device)
        input_label = (
            "1x126460x4"
            if name in ("SR-PointNet", "FFSR-PointNet-Light")
            else "1200x1800 via 150 non-overlap 120x120 patches"
        )
        if name not in ("SR-PointNet", "FFSR-PointNet-Light"):
            gflops = per_patch_or_full * WET_PATCH_MEAN
            full_domain_gflops = per_patch_or_full * 150.0
        else:
            gflops = per_patch_or_full
            full_domain_gflops = per_patch_or_full
        rows.append(
            {
                "Model": name,
                "Params (M)": count_params(model) / 1e6,
                "GFLOPs (MACx2)": gflops,
                "Full-domain GFLOPs (MACx2)": full_domain_gflops,
                "Wet patch mean": (
                    WET_PATCH_MEAN
                    if name not in ("SR-PointNet", "FFSR-PointNet-Light")
                    else None
                ),
                "Input": input_label,
            }
        )
        print(rows[-1])

    import pandas as pd

    frame = pd.DataFrame(rows)
    output = ROOT / "Results_profile_820.csv"
    frame.to_csv(output, index=False)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
