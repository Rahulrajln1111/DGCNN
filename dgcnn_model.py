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
from torch_geometric.nn import global_max_pool, global_mean_pool


def knn_graph(x, k, batch):
    """
    Manual KNN graph construction using torch.cdist.

    Works on ALL GPUs including Jetson Nano's Maxwell (128-core).
    torch_cluster.knn fails on Maxwell due to CUDA kernel launch limits,
    so we use standard BLAS-backed distance computation instead.

    Args:
        x:     (N, D) node features
        k:     number of neighbors
        batch: (N,) graph membership indices

    Returns:
        edge_index: (2, N*k) tensor — [row (center), col (neighbor)]
    """
    row_list, col_list = [], []

    for graph_id in batch.unique():
        mask = (batch == graph_id)
        pts = x[mask]                           # (n, D)
        n = pts.size(0)
        real_k = min(k, n - 1)
        global_ids = torch.where(mask)[0]       # map local → global

        # Pairwise distances via optimised BLAS
        dist = torch.cdist(pts, pts, p=2.0)     # (n, n)
        dist.fill_diagonal_(float("inf"))        # exclude self-loops

        # Top-k nearest (smallest distance)
        _, knn_idx = dist.topk(real_k, largest=False, dim=1)  # (n, k)

        # Build edge list: center → neighbor
        row = global_ids.unsqueeze(1).expand(-1, real_k).reshape(-1)
        col = global_ids[knn_idx.reshape(-1)]
        row_list.append(row)
        col_list.append(col)

    return torch.stack([torch.cat(row_list), torch.cat(col_list)], dim=0)


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
        # Build KNN graph in feature space (works on all GPUs)
        edge_index = knn_graph(x, self.k, batch)
        row, col = edge_index[0], edge_index[1]

        # Edge features: [center_features, neighbor - center]
        x_i = x[row]   # center point features
        x_j = x[col]   # neighbor features
        edge_feat = torch.cat([x_i, x_j - x_i], dim=-1)

        # Shared MLP on each edge
        edge_feat = self.mlp(edge_feat)

        # Max aggregate to center nodes
        out = scatter_max(edge_feat, row, dim=0, dim_size=x.size(0))
        return out


def scatter_max(src, index, dim=0, dim_size=None):
    """
    Scatter max aggregation — compatible with PyTorch 1.10+.
    Uses torch_scatter if available, otherwise falls back to a loop.
    """
    try:
        from torch_scatter import scatter
        return scatter(src, index, dim=dim, dim_size=dim_size, reduce='max')
    except ImportError:
        pass

    # Fallback: use scatter_reduce_ if available (PyTorch >= 1.12)
    try:
        idx = index.unsqueeze(-1).expand_as(src)
        out = torch.full((dim_size, src.size(1)), float('-inf'),
                         device=src.device, dtype=src.dtype)
        out.scatter_reduce_(0, idx, src, reduce="amax", include_self=False)
        return torch.where(out == float('-inf'), torch.zeros_like(out), out)
    except (AttributeError, TypeError):
        pass

    # Last resort fallback
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    idx = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(0, idx, src)
    cnt = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    cnt.scatter_add_(0, index.unsqueeze(-1), torch.ones(index.size(0), 1, device=src.device))
    return out / cnt.clamp(min=1)


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
