"""
GNN SuperNet for HGNAS (Section 3.3 & 3.4).

Implements the one-shot supernet where each position selects:
  - Sample  : KNN or Random graph construction
  - Aggregate: MessagePassing with configurable aggregator & message type
  - Combine  : MLP that updates node features
  - Connect  : Skip-connection or identity

The supernet is used for weight sharing across all sub-architectures,
following the single-path one-shot NAS methodology [SPOS].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, global_mean_pool

import config as C
from design_space import Architecture, PositionEncoding


def knn_graph_manual(x: torch.Tensor, k: int, batch=None) -> torch.Tensor:
    """
    Fast vectorised KNN graph construction (no torch-cluster dependency).
    Works on Jetson Nano and any platform without torch-cluster.

    Uses the identity  ||a-b||² = ||a||² + ||b||² - 2<a,b>
    for O(N²·d) matmul instead of an O(N²·d) explicit diff loop.

    Returns edge_index of shape (2, N*k).
    """
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    src_list, dst_list = [], []
    for graph_id in batch.unique():
        mask       = batch == graph_id
        pts        = x[mask]                       # (n, d)
        n          = pts.size(0)
        real_k     = min(k, n - 1)
        global_ids = torch.where(mask)[0]

        # Vectorised squared distances: (n, n)
        sq = (pts * pts).sum(dim=1, keepdim=True)  # (n, 1)
        dist2 = sq + sq.t() - 2.0 * (pts @ pts.t())
        dist2.fill_diagonal_(float("inf"))

        knn_idx = dist2.topk(real_k, largest=False).indices   # (n, k)

        src = global_ids.unsqueeze(1).expand(-1, real_k).reshape(-1)
        dst = global_ids[knn_idx.reshape(-1)]
        src_list.append(src)
        dst_list.append(dst)

    return torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)


# ── Operation modules ─────────────────────────────────────────────────────────

class SampleOp(nn.Module):
    """Graph construction: KNN or Random sampling."""

    def __init__(self, k: int = C.KNN_K):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor, sample_type: str, batch=None):
        if sample_type == "knn":
            edge_index = knn_graph_manual(x, k=self.k, batch=batch)
        else:  # random – lightweight fallback for search speed
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            src_list, dst_list = [], []
            for graph_id in batch.unique():
                mask       = batch == graph_id
                global_ids = torch.where(mask)[0]
                n          = global_ids.size(0)
                real_k     = min(self.k, n - 1)
                src = global_ids.unsqueeze(1).expand(-1, real_k).reshape(-1)
                dst_idx = torch.randint(0, n, (n * real_k,), device=x.device) % n
                dst = global_ids[dst_idx]
                src_list.append(src); dst_list.append(dst)
            edge_index = torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)
        return edge_index


class AggregateOp(nn.Module):
    """
    Manual message-passing aggregate with fully dynamic aggregator and message type.

    Avoids PyG MessagePassing so that both the aggregation function and
    message construction can be chosen at runtime without re-instantiation.

    Message types (Table 2):
      source    : h_j
      target    : h_i
      relative  : h_j - h_i
      src_rel   : cat(h_j, h_j - h_i)
      tgt_rel   : cat(h_i, h_j - h_i)
      euclidean : ||h_j - h_i||_2 broadcast to in_dim
      full      : cat(h_i, h_j, h_j - h_i)

    Aggregators: max | sum | mean | min
    """

    _MSG_MULT = {
        "source": 1, "target": 1, "relative": 1,
        "src_rel": 2, "tgt_rel": 2, "euclidean": 1, "full": 3,
    }

    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        # Pre-build one projection per message-size multiplier
        self.projs = nn.ModuleDict({
            "1": nn.Linear(in_dim,     in_dim),
            "2": nn.Linear(in_dim * 2, in_dim),
            "3": nn.Linear(in_dim * 3, in_dim),
        })

    def _build_message(self, x_i: torch.Tensor, x_j: torch.Tensor,
                       msg_type: str) -> torch.Tensor:
        if msg_type == "source":
            m = x_j
        elif msg_type == "target":
            m = x_i
        elif msg_type == "relative":
            m = x_j - x_i
        elif msg_type == "src_rel":
            m = torch.cat([x_j, x_j - x_i], dim=-1)
        elif msg_type == "tgt_rel":
            m = torch.cat([x_i, x_j - x_i], dim=-1)
        elif msg_type == "euclidean":
            m = (x_j - x_i).norm(dim=-1, keepdim=True).expand(-1, self.in_dim)
        else:  # full
            m = torch.cat([x_i, x_j, x_j - x_i], dim=-1)
        return self.projs[str(self._MSG_MULT[msg_type])](m)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                aggr_type: str = "max", msg_type: str = "relative") -> torch.Tensor:
        """
        Args:
            x          : node features (N, in_dim)
            edge_index : (2, E)  src → dst
            aggr_type  : 'max' | 'sum' | 'mean' | 'min'
            msg_type   : one of the 7 message types above

        Returns:
            Aggregated node features (N, in_dim)
        """
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        x_i = x[dst]   # target features
        x_j = x[src]   # source features
        msg = self._build_message(x_i, x_j, msg_type)   # (E, in_dim)

        # Aggregate messages at destination nodes
        if aggr_type == "sum":
            out = torch.zeros(N, self.in_dim, device=x.device, dtype=x.dtype)
            out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        elif aggr_type == "mean":
            out  = torch.zeros(N, self.in_dim, device=x.device, dtype=x.dtype)
            cnt  = torch.zeros(N, 1,           device=x.device, dtype=x.dtype)
            out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
            cnt.scatter_add_(0, dst.unsqueeze(-1),
                             torch.ones(dst.size(0), 1, device=x.device))
            out = out / cnt.clamp(min=1)
        elif aggr_type == "max":
            out = torch.full((N, self.in_dim), -1e9, device=x.device, dtype=x.dtype)
            out.scatter_reduce_(0, dst.unsqueeze(-1).expand_as(msg), msg,
                                reduce="amax", include_self=True)
            out = out.clamp(min=-1e8)   # replace unfilled -inf
        elif aggr_type == "min":
            out = torch.full((N, self.in_dim), 1e9, device=x.device, dtype=x.dtype)
            out.scatter_reduce_(0, dst.unsqueeze(-1).expand_as(msg), msg,
                                reduce="amin", include_self=True)
            out = out.clamp(max=1e8)
        else:
            raise ValueError(f"Unknown aggr_type: {aggr_type}")

        return out


class CombineOp(nn.Module):
    """MLP-based feature update (combine operation)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ── Full SuperNet ─────────────────────────────────────────────────────────────

class GNNSuperNet(nn.Module):
    """
    One-shot GNN SuperNet.

    Covers all architectures in the fine-grained design space by
    sharing weights across sub-architectures (single-path one-shot).
    During a forward pass an Architecture encoding selects which
    operations/functions to activate at each position.
    """

    def __init__(
        self,
        num_positions: int = C.NUM_POSITIONS,
        in_channels:   int = C.IN_CHANNELS,
        num_classes:   int = C.NUM_CLASSES,
        hidden_dim:    int = C.HIDDEN_DIM,
    ):
        super().__init__()
        self.N          = num_positions
        self.hidden_dim = hidden_dim

        # Input projection: xyz → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # One set of shared operators per position
        self.sample_ops  = nn.ModuleList([SampleOp() for _ in range(num_positions)])
        self.agg_ops = nn.ModuleList([
            AggregateOp(hidden_dim) for _ in range(num_positions)
        ])
        self.combine_ops = nn.ModuleList([
            CombineOp(hidden_dim, hidden_dim) for _ in range(num_positions)
        ])

        # Skip-connection projections (for identity the proj is unused)
        self.skip_projs  = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_positions)
        ])

        # Global pooling + classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, data, architecture: Architecture):
        x     = data.pos          # (total_nodes, 3) – xyz point coordinates
        batch = data.batch        # (total_nodes,)

        x = self.input_proj(x)   # → (total_nodes, hidden_dim)

        # Static graph mode: compute one KNN graph from initial features and
        # reuse it across all positions (inspired by Sec 2.2 sample-reuse).
        # This is ~12× faster on CPU (Jetson Nano) with minimal accuracy loss.
        static_edge_index = None
        if C.STATIC_GRAPH:
            static_edge_index = knn_graph_manual(data.pos, k=C.KNN_K, batch=batch)

        for pos_idx, pos_enc in enumerate(architecture.positions):
            x_prev = x

            # 1. Sample – build graph (or reuse cached graph)
            if C.STATIC_GRAPH:
                edge_index = static_edge_index
            else:
                edge_index = self.sample_ops[pos_idx](
                    x, sample_type=pos_enc.sample_op, batch=batch
                )

            # 2. Aggregate – message passing
            aggr_type, msg_type = pos_enc.agg_op
            x_agg = self.agg_ops[pos_idx](x, edge_index,
                                          aggr_type=aggr_type,
                                          msg_type=msg_type)

            # 3. Combine – MLP update (add aggregated signal)
            x = self.combine_ops[pos_idx](x + x_agg)

            # 4. Connect – residual or identity
            if pos_enc.connect_op == "skip":
                x = x + self.skip_projs[pos_idx](x_prev)

        # Global pooling over all nodes in each graph
        x = global_mean_pool(x, batch)       # → (batch_size, hidden_dim)
        return self.classifier(x)            # → (batch_size, num_classes)
