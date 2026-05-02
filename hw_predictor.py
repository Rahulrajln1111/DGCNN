"""
GNN Hardware Performance Predictor (Section 3.5).

Following the paper:
  "By abstracting the GNN architecture into graphs, the hardware-awareness
   problem can be reformulated as a graph representation learning problem."

Architecture → directed graph → 3-layer GCN + MLP → predicted latency (ms)

Node features (one-hot, 21-dim per position + 16-dim global node):
  [sample(2)] + [agg_type(4)] + [msg_type(7)] + [combine(6)] + [connect(2)]

We train a separate predictor for Jetson Nano using simulated profiling
data (since we can't run code on the device from Replit).  On the actual
Jetson you would replace `simulate_latency()` with real `torch.profiler`
measurements.
"""

import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import Data, Batch

import config as C
from design_space import Architecture, PositionEncoding
from memory_estimator import estimate_peak_memory


# ── Latency simulation for Jetson Nano ───────────────────────────────────────
# These constants approximate real Jetson Nano profiling results.
# On the actual device replace with measured values.

_SAMPLE_LAT  = {"knn": 5.0, "random": 1.5}            # ms per position
_AGG_LAT     = {"max": 2.0, "sum": 1.8, "mean": 1.8, "min": 1.9}
_MSG_LAT_MULT= {
    "source": 1.0, "target": 1.0, "relative": 1.1,
    "src_rel": 1.5, "tgt_rel": 1.5, "euclidean": 1.2, "full": 1.8,
}
_COMBINE_LAT = {8: 0.3, 16: 0.5, 32: 0.8, 64: 1.2, 128: 1.8, 256: 3.0}
_CONNECT_LAT = {"identity": 0.0, "skip": 0.3}


def simulate_latency(architecture: Architecture, num_nodes: int = 1024) -> float:
    """
    Simulated per-position latency in ms (Jetson Nano approximation).
    Replace with real on-device profiling for production use.
    """
    lat = 2.0   # base overhead
    for pos in architecture.positions:
        lat += _SAMPLE_LAT[pos.sample_op]
        agg_t, msg_t = pos.agg_op
        lat += _AGG_LAT[agg_t] * _MSG_LAT_MULT[msg_t]
        lat += _COMBINE_LAT[pos.combine_dim]
        lat += _CONNECT_LAT[pos.connect_op]
        lat += random.gauss(0, 0.5)   # device noise
    return max(lat, 1.0)


# ── Architecture → Graph conversion ──────────────────────────────────────────

GLOBAL_NODE_DIM = 16   # graph-level properties encoded into global node


def _graph_properties_to_vec(num_nodes: int = 1024, k: int = C.KNN_K) -> torch.Tensor:
    """Encode dataset graph properties into a 16-dim vector (global node)."""
    density = (num_nodes * k) / (num_nodes ** 2 + 1e-6)
    v = torch.zeros(GLOBAL_NODE_DIM)
    v[0]  = math.log(num_nodes + 1) / 10.0
    v[1]  = density * 100.0
    v[2]  = math.log(k + 1) / 5.0
    # remaining dims: zeros (represent unused properties)
    return v


def arch_to_pyg_graph(architecture: Architecture, num_nodes: int = 1024) -> Data:
    """
    Convert an Architecture to a PyG Data graph for the predictor.

    Node layout:
      0           : input node  (zero feature)
      1 … N       : operation nodes (one-hot feature)
      N+1         : output node (zero feature)
      N+2         : global node (graph-property feature)
    """
    N = len(architecture.positions)
    op_dim  = PositionEncoding.ONEHOT_DIM
    feat_dim = max(op_dim, GLOBAL_NODE_DIM)

    num_graph_nodes = N + 3   # input + ops + output + global
    x = torch.zeros(num_graph_nodes, feat_dim)

    # Operation nodes
    for i, pos in enumerate(architecture.positions):
        oh = pos.to_onehot()
        x[i + 1, :len(oh)] = torch.tensor(oh, dtype=torch.float)

    # Global node
    x[N + 2, :GLOBAL_NODE_DIM] = _graph_properties_to_vec(num_nodes)

    # Edges: sequential dataflow input → op_0 → ... → op_N → output
    src, dst = [], []
    for i in range(N + 1):
        src.append(i); dst.append(i + 1)
    # Global node connects to all operation nodes (both directions)
    for i in range(1, N + 1):
        src.append(N + 2); dst.append(i)
        src.append(i);     dst.append(N + 2)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, num_nodes=num_graph_nodes)


# ── Predictor model (3 GCN layers + MLP) ─────────────────────────────────────

class LatencyPredictor(nn.Module):
    """
    GNN-based latency predictor (Section 3.5).
    Architecture: 3 GCN layers (256→512→512) + MLP (256→128→1).
    """

    def __init__(self, node_feat_dim: int, hidden: list = C.PREDICTOR_HIDDEN,
                 mlp_hidden: list = C.PREDICTOR_MLP_HIDDEN):
        super().__init__()
        dims = [node_feat_dim] + hidden
        self.gcn_layers = nn.ModuleList([
            GCNConv(dims[i], dims[i + 1]) for i in range(len(hidden))
        ])

        # MLP head
        mlp_dims = [hidden[-1]] + mlp_hidden
        mlp_layers = []
        for i in range(len(mlp_dims) - 1):
            mlp_layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:
                mlp_layers.append(nn.LeakyReLU(0.1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for gcn in self.gcn_layers:
            x = F.relu(gcn(x, edge_index))
        x = global_add_pool(x, batch)
        return self.mlp(x).squeeze(-1)


# ── HWPredictor: unified interface ────────────────────────────────────────────

class HWPredictor:
    """
    Combines the GNN-based latency predictor with the
    analytical peak memory estimator.

    Usage:
        predictor = HWPredictor(device="jetson_nano")
        predictor.train_on_samples(n_samples=500)
        lat = predictor.predict_latency(arch)
        mem = predictor.predict_peak_memory(arch)
    """

    def __init__(self, device_name: str = C.DEVICE_NAME,
                 torch_device: str = C.TORCH_DEVICE):
        self.device_name  = device_name
        self.torch_device = torch_device

        # Determine node feature dimension from a dummy architecture
        dummy = Architecture.random(1)
        dummy_graph = arch_to_pyg_graph(dummy)
        node_feat_dim = dummy_graph.x.size(1)

        self.model = LatencyPredictor(node_feat_dim).to(torch_device)
        self._trained = False

    # ── Training ─────────────────────────────────────────────────────────────

    def train_on_samples(self, n_samples: int = C.PREDICTOR_SAMPLES,
                         epochs: int = C.PREDICTOR_EPOCHS,
                         lr: float = C.PREDICTOR_LR,
                         batch_size: int = C.PREDICTOR_BATCH,
                         verbose: bool = True):
        """
        Train the latency predictor on randomly sampled architectures
        with simulated Jetson Nano profiling data.
        """
        print(f"[HWPredictor] Generating {n_samples} architecture samples...")
        from design_space import DesignSpace
        ds   = DesignSpace()
        archs = [ds.random_architecture() for _ in range(n_samples)]
        lats  = [simulate_latency(a)       for a in archs]

        graphs = [arch_to_pyg_graph(a) for a in archs]
        labels = torch.tensor(lats, dtype=torch.float)

        split = int(0.85 * n_samples)
        train_g, val_g = graphs[:split], graphs[split:]
        train_y, val_y = labels[:split], labels[split:]

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                      weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)

        print(f"[HWPredictor] Training latency predictor for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            self.model.train()
            perm = torch.randperm(len(train_g))
            epoch_loss = 0.0
            for start in range(0, len(train_g), batch_size):
                idx   = perm[start:start + batch_size]
                batch = Batch.from_data_list([train_g[i] for i in idx]).to(self.torch_device)
                y     = train_y[idx].to(self.torch_device)

                optimizer.zero_grad()
                pred = self.model(batch)
                # MAPE loss (paper uses MAPE)
                loss = ((pred - y).abs() / (y.abs() + 1e-6)).mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)

            epoch_loss /= len(train_g)
            scheduler.step(epoch_loss)

            if verbose and epoch % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    vb = Batch.from_data_list(val_g).to(self.torch_device)
                    vy = val_y.to(self.torch_device)
                    val_mape = ((self.model(vb) - vy).abs() / (vy.abs() + 1e-6)).mean()
                print(f"  Epoch {epoch:3d}/{epochs} | train MAPE: {epoch_loss:.4f} "
                      f"| val MAPE: {val_mape:.4f}")

        self._trained = True
        print("[HWPredictor] Training complete.")

    def save(self, path: str = C.PREDICTOR_CKPT):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str = C.PREDICTOR_CKPT):
        self.model.load_state_dict(torch.load(path, map_location=self.torch_device))
        self._trained = True

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_latency(self, architecture: Architecture) -> float:
        """Return predicted inference latency in ms."""
        if not self._trained:
            # Fallback: use simulation directly
            return simulate_latency(architecture)
        self.model.eval()
        with torch.no_grad():
            g = arch_to_pyg_graph(architecture)
            b = Batch.from_data_list([g]).to(self.torch_device)
            return self.model(b).item()

    def predict_peak_memory(self, architecture: Architecture) -> float:
        """Return estimated peak memory in MB (analytical model from paper)."""
        return estimate_peak_memory(architecture)
