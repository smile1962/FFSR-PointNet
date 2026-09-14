"""Original SR-PointNet structure with T-Net computed on a point subset.

The T-Net matrices are global, so they are estimated from a deterministic
subset of context points. The resulting matrices are still applied to all HR
points, preserving the original transform semantics while reducing the
T-Net's per-point cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.FFSRP_original import TNet


class SubsampledTransform(nn.Module):
    def __init__(self, input_dim, global_feat_dim, context_points=8192):
        super().__init__()
        self.input_dim = input_dim
        self.global_feat_dim = global_feat_dim
        self.context_points = context_points

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

    def _context_indices(self, num_points, device):
        if num_points <= self.context_points:
            return torch.arange(num_points, device=device)
        stride = num_points // self.context_points
        idx = torch.arange(0, num_points, stride, device=device)
        return idx[:self.context_points]

    def forward(self, x):
        num_points = x.size(1)
        indices = self._context_indices(num_points, x.device)
        context = x[:, indices, :].transpose(1, 2)

        matrix_input = self.input_transform(context)
        x = x.transpose(1, 2)
        x = torch.bmm(matrix_input, x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        context_feat = x[:, :, indices]
        matrix_feat = self.feature_transform(context_feat)
        x = torch.bmm(matrix_feat, x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = nn.MaxPool1d(x.size(-1))(x)
        return x.view(x.size(0), self.global_feat_dim), matrix_input, matrix_feat


class SubsampledCompressed(nn.Module):
    def __init__(
        self,
        input_dim,
        global_feat_dim,
        output_dim,
        N_high,
        rank,
        context_points=8192,
    ):
        super().__init__()
        self.N_high = N_high
        self.output_dim = output_dim
        self.context_points = context_points
        self.transform = SubsampledTransform(
            input_dim, global_feat_dim, context_points
        )
        self.decoder = nn.Sequential()
        self.decoder.append(nn.Linear(global_feat_dim, 1024))
        self.decoder.append(nn.ReLU())
        self.decoder.append(
            nn.Sequential(
                nn.Linear(1024, rank, bias=False),
                nn.Linear(rank, N_high * output_dim, bias=True),
            )
        )

    @classmethod
    def from_teacher_state(
        cls,
        teacher_state,
        input_dim,
        global_feat_dim,
        output_dim,
        N_high,
        rank,
        context_points=8192,
    ):
        model = cls(
            input_dim,
            global_feat_dim,
            output_dim,
            N_high,
            rank,
            context_points,
        )
        prefix = "transform."
        transform_state = {}
        for key, value in teacher_state.items():
            if key.startswith(prefix):
                transform_state[key[len(prefix):]] = value
        model.transform.load_state_dict(transform_state)
        model.decoder[0].load_state_dict(
            {
                "weight": teacher_state["decoder.fc.0.weight"],
                "bias": teacher_state["decoder.fc.0.bias"],
            }
        )
        weight = teacher_state["decoder.fc.2.weight"]
        bias = teacher_state["decoder.fc.2.bias"]
        u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
        proj = model.decoder[2][0]
        expand = model.decoder[2][1]
        with torch.no_grad():
            proj.weight.copy_(vh[:rank])
            expand.weight.copy_(u[:, :rank] * s[:rank].reshape(1, -1))
            expand.bias.copy_(bias)
        return model

    def forward(self, x):
        global_feat, _, _ = self.transform(x)
        h = self.decoder[0](global_feat)
        h = F.relu(h)
        h = self.decoder[2][0](h)
        out = self.decoder[2][1](h)
        return out.view(-1, self.N_high, self.output_dim)
