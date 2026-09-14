"""FFSR-PointNet v2: original SR-PointNet backbone plus topographic prior.

The encoder and the fixed-size reconstruction decoder keep the structure of
the earlier SR-PointNet model so that the established training behaviour is
retained. The structural difference is an added dual-branch fusion: HR x, y,
DEM, and slope are encoded by a compact pointwise topographic encoder, and
the resulting global terrain descriptor is fused with the hydraulic global
feature before the decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """Transformation network used by the original PointNet encoder."""

    def __init__(self, k=3):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = nn.MaxPool1d(x.size(-1))(x).view(batch_size, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))

        identity = torch.eye(self.k).repeat(batch_size, 1, 1)
        identity = identity.to(x.device)
        matrix = self.fc3(x).view(-1, self.k, self.k) + identity
        return matrix


class Transform(nn.Module):
    """PointNet encoder used in the original SR-PointNet model."""

    def __init__(self, input_dim, global_feat_dim):
        super().__init__()
        self.input_dim = input_dim
        self.global_feat_dim = global_feat_dim

        self.input_transform = TNet(k=input_dim)

        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)

        self.feature_transform = TNet(k=128)
        self.conv4 = nn.Conv1d(128, global_feat_dim, 1)
        self.bn4 = nn.BatchNorm1d(global_feat_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        matrix_input = self.input_transform(x)
        x = torch.bmm(matrix_input, x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        matrix_feat = self.feature_transform(x)
        x = torch.bmm(matrix_feat, x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = nn.MaxPool1d(x.size(-1))(x)
        return x.view(x.size(0), self.global_feat_dim), matrix_input, matrix_feat


class PointNetDecoder(nn.Module):
    """Original fixed-size decoder retained for behaviour compatibility."""

    def __init__(self, global_feat_dim, output_dim, N_high):
        super().__init__()
        self.N_high = N_high
        self.output_dim = output_dim
        self.fc = nn.Sequential(
            nn.Linear(global_feat_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, N_high * output_dim),
        )

    def forward(self, global_feat):
        out = self.fc(global_feat)
        return out.view(-1, self.N_high, self.output_dim)


class TopographyPriorEncoder(nn.Module):
    """Compact encoder producing a global terrain descriptor."""

    def __init__(self, prior_dim, topo_feat_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(prior_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, topo_feat_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(topo_feat_dim)

    def forward(self, hr_priors):
        x = hr_priors.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = nn.MaxPool1d(x.size(-1))(x)
        return x.view(x.size(0), -1)


class FFSRPointNetV2(nn.Module):
    """SR-PointNet backbone with topography-guided feature fusion."""

    def __init__(
        self,
        input_dim,
        global_feat_dim,
        output_dim,
        N_high,
        prior_dim=4,
        topo_feat_dim=128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.transform = Transform(input_dim, global_feat_dim)
        self.topo_encoder = TopographyPriorEncoder(prior_dim, topo_feat_dim)
        self.fusion = nn.Sequential(
            nn.Linear(global_feat_dim + topo_feat_dim, global_feat_dim),
            nn.ReLU(),
        )
        self.decoder = PointNetDecoder(global_feat_dim, output_dim, N_high)

    def forward(self, lr_inputs, hr_priors):
        hydraulic_feat, _, _ = self.transform(lr_inputs)
        topo_feat = self.topo_encoder(hr_priors)
        fused = self.fusion(torch.cat([hydraulic_feat, topo_feat], dim=1))
        return self.decoder(fused)
