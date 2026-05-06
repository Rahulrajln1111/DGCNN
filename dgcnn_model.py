"""
DGCNN – Dynamic Graph CNN for Point Cloud Classification.

Reference: Wang et al., "Dynamic Graph CNN for Learning on Point Clouds", 2019.

Architecture:
  4 EdgeConv layers with dynamic KNN graph → concat → global pool → MLP classifier.

Compatible with PyTorch 1.10+ and PyG 2.0.3+ (works on both A100 and Jetson Nano).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn
from torch_scatter import scatter
from torch_geometric.nn import global_max_pool, global_mean_pool


class EdgeConv(nn.Module):
    """
    EdgeConv layer from DGCNN.

    For each point i, finds k nearest neighbors in feature space,
    computes edge features [x_i, x_j - x_i], applies a shared MLP,
    and aggregates with max pooling.
    """

    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, batch):
        # KNN in feature space: returns (2, N*k) tensor
        # row = query (center), col = neighbor
        edge_index = knn(x, x, self.k, batch, batch)
        row, col = edge_index[0], edge_index[1]

        # Edge features: [center_features, neighbor - center]
        x_i = x[row]   # center point features
        x_j = x[col]   # neighbor features
        edge_feat = torch.cat([x_i, x_j - x_i], dim=-1)

        # Shared MLP on each edge
        edge_feat = self.mlp(edge_feat)

        # Max aggregate to center nodes
        out = scatter(edge_feat, row, dim=0, dim_size=x.size(0), reduce='max')
        return out


class DGCNN(nn.Module):
    """
    DGCNN for 3D point cloud classification.

    Input:  PyG Data with pos (N, 3) and batch (N,)
    Output: (batch_size, num_classes) logits
    """

    def __init__(self, num_classes=10, k=20, dropout=0.5):
        super().__init__()
        self.k = k

        # 4 EdgeConv layers with increasing feature dimensions
        self.conv1 = EdgeConv(3, 64, k=k)
        self.conv2 = EdgeConv(64, 64, k=k)
        self.conv3 = EdgeConv(64, 128, k=k)
        self.conv4 = EdgeConv(128, 256, k=k)

        # Aggregation: concat all layer outputs → project
        feat_dim = 64 + 64 + 128 + 256  # = 512
        self.lin0 = nn.Linear(feat_dim, 1024)
        self.bn0 = nn.BatchNorm1d(1024)

        # Classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 2, 512),   # *2 because max+mean pool
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, data):
        x, batch = data.pos, data.batch

        # EdgeConv layers with dynamic graph recomputation
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        x4 = self.conv4(x3, batch)

        # Concatenate multi-scale features
        x = torch.cat([x1, x2, x3, x4], dim=-1)  # (N, 512)
        x = F.leaky_relu(self.bn0(self.lin0(x)), 0.2)  # (N, 1024)

        # Global pooling: max + mean concatenation
        x_max = global_max_pool(x, batch)    # (B, 1024)
        x_mean = global_mean_pool(x, batch)  # (B, 1024)
        x = torch.cat([x_max, x_mean], dim=-1)  # (B, 2048)

        return self.classifier(x)  # (B, num_classes)
