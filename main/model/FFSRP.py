import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Model structure design：TNet
# ------------------------------
class TNet(nn.Module):
    """
    A T-Net that predicts a k x k transformation matrix, used for both:
      1) Input Transform (k = input_dim)
      2) Feature Transform (k = 128, etc.)
    """

    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k

        # Convolution layers
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        """
        x shape: (B, k, N)
          B: batch size
          k: number of channels (could be input_dim or 128, etc.)
          N: number of points
        Returns: transform_matrix shape: (B, k, k)
        """
        batch_size = x.size(0)

        # 1D Convs + BN + ReLU
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128, N)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 1024, N)

        # Global max pooling
        x = nn.MaxPool1d(x.size(-1))(x)  # (B, 1024, 1)
        x = x.view(batch_size, 1024)  # (B, 1024)

        # FC layers + BN + ReLU
        x = F.relu(self.bn4(self.fc1(x)))  # (B, 512)
        x = F.relu(self.bn5(self.fc2(x)))  # (B, 256)

        # Regress k*k matrix
        # Initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(batch_size, 1, 1)
        if x.is_cuda:
            init = init.cuda()

        matrix = self.fc3(x)  # (B, k*k)
        matrix = matrix.view(-1, self.k, self.k) + init
        return matrix


# --------------------------------------------------------
# Model structure design：Transform
# --------------------------------------------------------
class Transform(nn.Module):
    """
    This module replicates the reference code's structure:
      - TNet for input transform
      - Some conv layers
      - TNet for feature transform
      - More conv layers
      - Global pooling
    Finally returns the global feature (for regression tasks)
    and optionally the transform matrices.
    """

    def __init__(self, input_dim, global_feat_dim):
        super(Transform, self).__init__()
        self.input_dim = input_dim
        self.global_feat_dim = global_feat_dim

        # Input transform
        self.input_transform = TNet(k=input_dim)

        # conv1, conv2, conv3 -> get 64, 128, 128 features
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)

        # Feature transform (for 128-dim features)
        self.feature_transform = TNet(k=128)

        # conv4 -> global_feat_dim
        self.conv4 = nn.Conv1d(128, self.global_feat_dim, 1)
        self.bn4 = nn.BatchNorm1d(self.global_feat_dim)

    def forward(self, x):
        """
        x shape: (B, N, input_dim)
        returns:
          - global_feat: (B, global_feat_dim)
          - matrix_input: (B, input_dim, input_dim)  # optional
          - matrix_feat: (B, 128, 128)              # optional
        """
        B, N, D = x.shape

        # 1) transpose to (B, D, N) for 1D conv
        x = x.transpose(1, 2)  # (B, input_dim, N)

        # 2) input transform
        matrix_input = self.input_transform(x)  # (B, input_dim, input_dim)
        x = torch.bmm(matrix_input, x)  # (B, input_dim, N)

        # 3) conv1 -> conv2 -> conv3
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64,   N)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128,  N)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 128,  N)

        # 4) feature transform on 128-dim features
        matrix_feat = self.feature_transform(x)  # (B, 128, 128)
        x = torch.bmm(matrix_feat, x)  # (B, 128, N)

        # 5) conv4 -> global_feat_dim
        x = F.relu(self.bn4(self.conv4(x)))  # (B, global_feat_dim, N)

        # 6) global max pooling
        x = nn.MaxPool1d(x.size(-1))(x)  # (B, global_feat_dim, 1)
        global_feat = x.view(B, self.global_feat_dim)

        return global_feat, matrix_input, matrix_feat


class PointNetDecoder(nn.Module):
    def __init__(self, global_feat_dim, output_dim, N_high):
        super(PointNetDecoder, self).__init__()
        self.N_high = N_high
        self.output_dim = output_dim  # output feature dimension (original minus coordinates)
        self.fc = nn.Sequential(
            nn.Linear(global_feat_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, N_high * output_dim)
        )

    def forward(self, global_feat):
        # global_feat: (B, global_feat_dim)
        out = self.fc(global_feat)  # (B, N_high * output_dim)
        out = out.view(-1, self.N_high, self.output_dim)  # (B, N_high, output_dim)
        return out

class PointNetRegression(nn.Module):
    """
    A PointNet-based regression model for SR flow field.
    Incorporates:
      - class Transform(nn.Module) for input & feature transforms
      - class PointNetDecoder for final upsampling/regression
    """
    def __init__(self, input_dim, global_feat_dim, output_dim, N_high):
        super(PointNetRegression, self).__init__()
        self.transform = Transform(input_dim, global_feat_dim)
        self.decoder = PointNetDecoder(global_feat_dim, output_dim, N_high)

    def forward(self, x):
        """
        x: (B, N_low, input_dim)
        returns: (B, N_high, output_dim)
        """
        global_feat, matrix_input, matrix_feat = self.transform(x)
        out = self.decoder(global_feat)
        return out
