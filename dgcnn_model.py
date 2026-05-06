"""
DGCNN – Dynamic Graph CNN for Point Cloud Classification.

Reference: Wang et al., "Dynamic Graph CNN for Learning on Point Clouds", 2019.

This is a CONFIGURABLE implementation supporting multiple optimizations
for edge-device deployment:
  - Model compression (Full / Lite / Tiny channel widths)
  - Static graph reuse (compute KNN once, reuse across layers)
  - Progressive K-reduction (fewer neighbors in deeper layers)
  - Selectable aggregation (max, mean, sum)
  - Edge attention gating (learn which neighbors matter)
  - FP16 inference support

Compatible with PyTorch 1.10+ and PyG 2.0.3+ (A100 + Jetson Nano).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, global_mean_pool


# ── Model presets ─────────────────────────────────────────────────────────────

MODEL_PRESETS = {
    "full":  {"channels": [64, 64, 128, 256], "emb_dim": 1024},
    "lite":  {"channels": [32, 32, 64, 128],  "emb_dim": 512},
    "tiny":  {"channels": [16, 16, 32, 64],   "emb_dim": 256},
}


# ── KNN graph construction ───────────────────────────────────────────────────

def knn_graph(x, k, batch):
    """
    Manual KNN graph using torch.cdist (works on Jetson Nano Maxwell GPU).
    torch_cluster.knn fails on Maxwell due to CUDA kernel launch limits.
    """
    row_list, col_list = [], []
    for graph_id in batch.unique():
        mask = (batch == graph_id)
        pts = x[mask]
        n = pts.size(0)
        real_k = min(k, n - 1)
        global_ids = torch.where(mask)[0]

        dist = torch.cdist(pts, pts, p=2.0)
        dist.fill_diagonal_(float("inf"))
        _, knn_idx = dist.topk(real_k, largest=False, dim=1)

        row = global_ids.unsqueeze(1).expand(-1, real_k).reshape(-1)
        col = global_ids[knn_idx.reshape(-1)]
        row_list.append(row)
        col_list.append(col)

    return torch.stack([torch.cat(row_list), torch.cat(col_list)], dim=0)


# ── Scatter aggregation (PyTorch 1.10 compatible) ────────────────────────────

def scatter_agg(src, index, dim_size, aggr="max"):
    """Scatter aggregation compatible with PyTorch 1.10+."""
    try:
        from torch_scatter import scatter
        return scatter(src, index, dim=0, dim_size=dim_size, reduce=aggr)
    except ImportError:
        pass

    idx = index.unsqueeze(-1).expand_as(src)
    if aggr == "max":
        try:
            out = torch.full((dim_size, src.size(1)), float('-inf'),
                             device=src.device, dtype=src.dtype)
            out.scatter_reduce_(0, idx, src, reduce="amax", include_self=False)
            return torch.where(out == float('-inf'), torch.zeros_like(out), out)
        except (AttributeError, TypeError):
            pass

    # Fallback: mean aggregation
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    out.scatter_add_(0, idx, src)
    cnt = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    cnt.scatter_add_(0, index.unsqueeze(-1),
                     torch.ones(index.size(0), 1, device=src.device))
    return out / cnt.clamp(min=1)


# ── EdgeConv layer ────────────────────────────────────────────────────────────

class EdgeConv(nn.Module):
    """
    Configurable EdgeConv layer from DGCNN.

    Supports:
    - Selectable aggregation (max/mean/sum)
    - Optional edge attention gating
    """

    def __init__(self, in_channels, out_channels, k=20, aggr="max",
                 use_attention=False):
        super().__init__()
        self.k = k
        self.aggr = aggr
        self.use_attention = use_attention

        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if use_attention:
            self.attn_gate = nn.Sequential(
                nn.Linear(2 * in_channels, 1),
                nn.Sigmoid(),
            )

    def forward(self, x, batch, edge_index=None):
        """
        Args:
            x: (N, D) node features
            batch: (N,) graph membership
            edge_index: optional pre-computed KNN graph (for static mode)
        """
        if edge_index is None:
            edge_index = knn_graph(x, self.k, batch)

        row, col = edge_index[0], edge_index[1]
        x_i = x[row]
        x_j = x[col]
        edge_input = torch.cat([x_i, x_j - x_i], dim=-1)

        edge_feat = self.mlp(edge_input)

        if self.use_attention:
            attn_weight = self.attn_gate(edge_input)
            edge_feat = edge_feat * attn_weight

        out = scatter_agg(edge_feat, row, dim_size=x.size(0), aggr=self.aggr)
        return out


# ── DGCNN model ──────────────────────────────────────────────────────────────

class DGCNN(nn.Module):
    """
    Configurable DGCNN for 3D point cloud classification.

    Args:
        num_classes:    number of output classes
        k:              KNN neighbors (or list for progressive-K)
        dropout:        classifier dropout rate
        channels:       list of output dims per EdgeConv layer
        emb_dim:        embedding dim after concat (0 = auto)
        aggr:           aggregation function ('max', 'mean', 'sum')
        static_graph:   if True, compute KNN once on raw xyz, reuse all layers
        use_attention:  if True, add edge attention gates
        preset:         'full', 'lite', 'tiny' (overrides channels/emb_dim)
    """

    def __init__(self, num_classes=10, k=20, dropout=0.5,
                 channels=None, emb_dim=None, aggr="max",
                 static_graph=False, use_attention=False, preset=None):
        super().__init__()

        # Apply preset if given
        if preset and preset in MODEL_PRESETS:
            p = MODEL_PRESETS[preset]
            channels = channels or p["channels"]
            emb_dim = emb_dim or p["emb_dim"]

        if channels is None:
            channels = [64, 64, 128, 256]
        if emb_dim is None:
            emb_dim = 1024

        self.static_graph = static_graph
        self.aggr = aggr
        self.num_layers = len(channels)

        # Handle progressive K: k can be int or list
        if isinstance(k, int):
            self.k_per_layer = [k] * len(channels)
        else:
            self.k_per_layer = list(k) + [k[-1]] * (len(channels) - len(k))

        # EdgeConv layers
        self.convs = nn.ModuleList()
        in_dim = 3
        for i, out_dim in enumerate(channels):
            self.convs.append(EdgeConv(
                in_dim, out_dim, k=self.k_per_layer[i],
                aggr=aggr, use_attention=use_attention,
            ))
            in_dim = out_dim

        # Aggregation projection
        feat_dim = sum(channels)
        self.lin0 = nn.Linear(feat_dim, emb_dim)
        self.bn0 = nn.BatchNorm1d(emb_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim // 2),
            nn.BatchNorm1d(emb_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, emb_dim // 4),
            nn.BatchNorm1d(emb_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 4, num_classes),
        )

    def forward(self, data):
        x, batch = data.pos, data.batch

        # Static graph: compute KNN once on raw xyz
        static_ei = None
        if self.static_graph:
            static_ei = knn_graph(x, self.k_per_layer[0], batch)

        # EdgeConv layers with multi-scale feature collection
        layer_outputs = []
        for i, conv in enumerate(self.convs):
            ei = static_ei if self.static_graph else None
            x = conv(x, batch, edge_index=ei)
            layer_outputs.append(x)

        # Concat multi-scale features
        x = torch.cat(layer_outputs, dim=-1)
        x = F.leaky_relu(self.bn0(self.lin0(x)), 0.2)

        # Global pooling: max + mean
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x = torch.cat([x_max, x_mean], dim=-1)

        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self):
        """Return config dict for saving/loading."""
        return {
            "num_layers": self.num_layers,
            "k_per_layer": self.k_per_layer,
            "aggr": self.aggr,
            "static_graph": self.static_graph,
        }
