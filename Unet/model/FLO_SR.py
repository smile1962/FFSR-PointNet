import torch
import torch.nn as nn


class EDSRResidualBlock(nn.Module):
    """FLO-SR residual block without batch normalization (EDSR style)."""

    def __init__(self, num_feats=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out * self.res_scale


class FLOSRModel(nn.Module):
    """Same-size FLO-SR adaptation for 4-channel to 2-channel mapping.

    The original FLO-SR is an EDSR-style network with no BatchNorm, 64 initial
    filters and 16 residual blocks. In this codebase LR and HR grids already
    share one spatial size, so the PixelShuffle branch is replaced by a final
    3x3 convolution that keeps the spatial size unchanged while matching the
    SR-Unet input/output contract.
    """

    def __init__(
        self,
        input_channels=4,
        output_channels=2,
        num_resblocks=16,
        num_feats=64,
        res_scale=1.0,
        use_output_relu=True,
    ):
        super().__init__()
        self.head = nn.Conv2d(
            input_channels, num_feats, kernel_size=3, padding=1
        )
        self.body = nn.Sequential(
            *[
                EDSRResidualBlock(num_feats=num_feats, res_scale=res_scale)
                for _ in range(num_resblocks)
            ]
        )
        self.tail = nn.Conv2d(
            num_feats, output_channels, kernel_size=3, padding=1
        )
        self.use_output_relu = use_output_relu
        self.relu = nn.ReLU(inplace=True)

    def forward(self, coarse_map):
        x = self.head(coarse_map)
        residual = self.body(x)
        x = x + residual
        out = self.tail(x)
        if self.use_output_relu:
            out = self.relu(out)
        return out

