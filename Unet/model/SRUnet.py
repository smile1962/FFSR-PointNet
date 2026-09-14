import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual unit ResUnit (core module)
class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUnit, self).__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv_channel = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x  # residual connection
        out = self.conv_block(x)
        """
            Crop the residual: because conv_block has no padding, the size drops by 4
            Crop the residual: two 3x3 convolutions reduce the size by 4 (2 on each side)
            Input x has shape [B, C, H, W]; after cropping it is [B, C, H-4, W-4]
        """
        residual = self.conv_channel(residual)
        # residual = residual[:, :, 2:-2, 2:-2]
        out += residual
        out = self.relu(out)
        return out

# Encoder block (ResUnit + MaxPooling + Crop)
class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodingBlock, self).__init__()
        self.res_unit = ResUnit(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x_res = self.res_unit(x)
        x_pooled = self.max_pool(x_res)
        return x_res, x_pooled  # return the ResUnit output (for skip connections) and the pooled output

# Decoder block (transposed convolution + concatenation + ResUnit)
class DecodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(DecodingBlock, self).__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.res_unit = ResUnit(out_channels + skip_channels, out_channels)  # concatenate the skip-connection features

    def forward(self, x, skip_connection):
        x_up = self.trans_conv(x)
        x_concat = torch.cat([x_up, skip_connection], dim=1)
        x_out = self.res_unit(x_concat)
        return x_out

# Main super-resolution model
class SuperResolutionModel(nn.Module):
    def __init__(self, input_channels=4, fine_topographic_channels=0, output_channels=2):
        super(SuperResolutionModel, self).__init__()
        # Encoder-stage module
        self.encoding1 = EncodingBlock(input_channels + fine_topographic_channels, 64)
        self.encoding2 = EncodingBlock(64, 128)
        self.encoding3 = EncodingBlock(128, 256)
        self.res_unit_bottom = ResUnit(256, 512)  # ResUnit at the bottom of the encoder

        # Decoder-stage module
        self.decoding3 = DecodingBlock(512, 256, 256)
        self.decoding2 = DecodingBlock(256, 128, 128)
        self.decoding1 = DecodingBlock(128, 64, 64)

        # Final output module
        self.final_resunit = ResUnit(64, output_channels)
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, coarse_map):
        """
        Args:
            coarse_map: coarse-grid input, shape [B, 1, H, H]
            fine_topographic: fine topographic features, shape [B, 4, H+88, H+88]
        Returns:
            output: super-resolution result, shape [B, output_channels, H, H]
        """
        # # 1. Bicubic interpolation
        # H = coarse_map.size(2)
        # interpolated = F.interpolate(
        #     coarse_map,
        #     size=(H + 88, H + 88),
        #     mode='bicubic',
        #     align_corners=False
        # )

        # 2. Concatenate fine topographic features
        x = coarse_map  # [B, 5, H, H]

        # 3. Encoding stage
        x_enc1, x_pool1 = self.encoding1(x)
        # # Crop x_enc1 to (H+4)x(H+4) (example: H=64, crop 40 pixels)
        # crop_margin1 = (x_enc1.size(2) - (H + 4)) // 2
        # x_enc1_cropped = x_enc1[:, :, crop_margin1:-crop_margin1, crop_margin1:-crop_margin1]

        x_enc2, x_pool2 = self.encoding2(x_pool1)
        # crop_margin2 = (x_enc2.size(2) - (H // 2 + 6)) // 2
        # x_enc2_cropped = x_enc2[:, :, crop_margin2:-crop_margin2, crop_margin2:-crop_margin2]

        x_enc3, x_pool3 = self.encoding3(x_pool2)
        # crop_margin3 = (x_enc3.size(2) - (H // 4 + 7)) // 2
        # x_enc3_cropped = x_enc3[:, :, crop_margin3:-crop_margin3, crop_margin3:-crop_margin3]

        x_bottom = self.res_unit_bottom(x_pool3)

        # 4. Decoding stage
        x_dec3 = self.decoding3(x_bottom, x_enc3)
        x_dec2 = self.decoding2(x_dec3, x_enc2)
        x_dec1 = self.decoding1(x_dec2, x_enc1)

        # 5. 1x1 convolution to adjust the channel count
        x_final = self.final_conv(x_dec1)
        x_final = nn.ReLU()(x_final)

        return x_final

