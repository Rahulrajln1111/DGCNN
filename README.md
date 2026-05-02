# HGNAS – Hardware-Aware Graph Neural Architecture Search

Implementation of the paper:  
**"HGNAS: Hardware-Aware Graph Neural Architecture Search for Edge Devices"**  
Zhou et al., arXiv:2408.12840, 2024.

Targeting the **NVIDIA Jetson Nano** edge device for real-time 3D point cloud classification on **ModelNet10**.

---

## What HGNAS Does

HGNAS automates the design of Graph Neural Networks (GNNs) that are both **accurate** and **efficient** on resource-constrained edge devices. Instead of manually tuning a GNN, it searches a large fine-grained design space and finds architectures that satisfy hardware constraints (inference latency and peak memory).

### Key Contributions

| Component | Description |
|---|---|
| **Fine-grained Design Space** | Each GNN layer is broken into independent Sample / Aggregate / Combine / Connect operations, each with multiple implementation choices |
| **GNN Hardware Predictor** | A small GCN-based model that predicts the latency of a candidate architecture in milliseconds, without running it on the device |
| **Peak Memory Estimator** | Analytical model (from GNN inference profiling) that estimates peak memory from architecture structure |
| **Multi-stage Hierarchical Search** | Two-stage evolutionary search: Stage 1 finds optimal per-position *functions*, Stage 2 finds optimal *operations* under hardware constraints |

---

## Project Structure

```
hgnas/
├── main.py              # Entry point – runs the full pipeline
├── config.py            # All hyperparameters and Jetson Nano constraints
├── design_space.py      # Fine-grained hierarchical design space (Table 2 from paper)
├── supernet.py          # One-shot GNN SuperNet (all operations in one model)
├── hw_predictor.py      # GNN-based latency predictor + Jetson Nano profiling sim
├── memory_estimator.py  # Analytical peak memory estimator (Equations 6–9)
├── trainer.py           # One-shot supernet training with random path sampling
├── search.py            # Multi-stage evolutionary search (Algorithm 1)
├── data_loader.py       # ModelNet10 loader using PyG
├── plot_res.py          # Pareto-front plots
├── get_data.py          # Downloads ModelNet10 via PyTorch Geometric
├── run.sh               # Shell script to run the full pipeline
└── data/                # Downloaded ModelNet10 dataset
```

---

## Design Space (Table 2 from Paper)

Each of the 12 supernet positions can independently choose:

| Operation | Choices |
|---|---|
| **Sample** | KNN, Random |
| **Aggregate** | Aggregator: sum, max, mean, min × Message type: source, target, relative, src‖rel, tgt‖rel, euclidean, full |
| **Combine** | Output hidden dim: 8, 16, 32, 64, 128, 256 |
| **Connect** | Identity, Skip-connection |

Total design space: ~4.2 × 10¹² configurations. HGNAS reduces effective search to ~1.7 × 10⁷ via the two-stage hierarchical decomposition.

---

## Hardware Performance Prediction

### Latency Predictor (Section 3.5)
A GNN architecture is **abstracted as a directed graph** where:
- Nodes = input, operations, output, and a global node (encoding dataset properties)
- Edges = dataflow between operations
- Node features = 21-dim one-hot encoding of operation + function choices

The predictor is a 3-layer GCN (256 → 512 → 512) followed by an MLP (256 → 128 → 1), trained with MAPE loss on sampled architectures profiled on the Jetson Nano.

### Peak Memory Estimator (Equations 6–9)
Analytically derived from GNN inference profiling:

| Operation | Memory Formula |
|---|---|
| Sample | M = N_edges × 2 × U_index |
| Aggregate (msg) | M = N_edges × 2 × L × U_k |
| Aggregate (broadcast) | M = N × L × U_k |
| Combine | M = N × L_out × U_k |

Peak memory is the maximum accumulated memory at any point in the forward pass.

---

## Multi-stage Search Algorithm (Algorithm 1)

```
Stage 1 – Function Search:
  for T iterations:
    Sample sub-functions via evolutionary algorithm
    Evaluate on validation set (few forward passes)
    Keep top-P candidates, mutate/crossover rest
  → Fix optimal function set for upper and lower supernet halves

Re-train supernet with fixed function set (one-shot, shared weights)

Stage 2 – Operation Search:
  for T iterations:
    Sample candidate operations via evolutionary algorithm
    Predict latency and memory via HW predictor
    IF constraints satisfied: evaluate accuracy on validation set
    Score = α × acc − β × efficiency_penalty
    Keep top-P, mutate/crossover rest
  → Return best architecture A*
```

**Objective function (Eq. 4):**
```
F_obj = 0,                        if latency > C_lat OR memory > C_mem
      = α × acc_val − β × E,      otherwise
where E = (latency / C_lat) + (memory / C_mem)
```

---

## Jetson Nano Hardware Constraints

| Constraint | Value |
|---|---|
| Inference latency limit | 150 ms |
| Peak memory limit | 800 MB |
| α (accuracy weight) | 1.0 |
| β (efficiency penalty weight) | 0.5 |

These can be adjusted in `config.py` or via CLI flags.

---

## Setup and Running

### 1. Install Dependencies

```bash
pip install torch torch-geometric pandas matplotlib numpy scikit-learn
```

### 2. Download the Dataset

```bash
python get_data.py
```

This downloads **ModelNet10** (~40 MB) via PyTorch Geometric and processes it into 1024-point clouds.

### 3. Run the Full Pipeline

```bash
bash run.sh
```

Or directly with custom settings:

```bash
# Quick demo (a few minutes)
python main.py \
    --supernet-epochs 5 \
    --predictor-samples 300 \
    --ea-iter-s1 5 \
    --ea-iter-s2 15

# Full search (recommended for final results)
python main.py \
    --supernet-epochs 30 \
    --predictor-samples 2000 \
    --ea-iter-s1 50 \
    --ea-iter-s2 100 \
    --lat-constraint 150 \
    --mem-constraint 800
```

### 4. Skip Pre-training (use saved checkpoints)

```bash
python main.py --skip-train --skip-predictor
```

---

## Outputs

| File | Description |
|---|---|
| `search_results.csv` | All valid architectures found during Stage 2 search |
| `hgnas_pareto_front.png` | Pareto-front plots: Accuracy vs Latency, Accuracy vs Memory, Convergence |
| `supernet_weights.pt` | Saved supernet checkpoint |
| `predictor_weights.pt` | Saved hardware predictor checkpoint |

---

## Deploying on Jetson Nano

When running on the actual Jetson Nano:

1. **Replace simulated latency with real profiling** in `hw_predictor.py`:
   ```python
   # Replace simulate_latency() with:
   import torch.utils.benchmark as benchmark
   timer = benchmark.Timer(stmt="model(data, arch)", ...)
   lat_ms = timer.timeit(10).mean * 1000
   ```

2. **Real peak memory** measurement (GPU):
   ```python
   torch.cuda.reset_peak_memory_stats()
   _ = model(data, arch)
   peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
   ```

3. **Use the CPU backend** – Jetson Nano's Maxwell GPU has limited CUDA support. The code automatically falls back to CPU when CUDA is unavailable.

4. **Reduce batch size** – Jetson Nano has 4 GB shared RAM. Set `SUPERNET_BATCH_SIZE = 4` in `config.py` if you run out of memory.

---

## References

- Zhou et al., "HGNAS: Hardware-Aware Graph Neural Architecture Search for Edge Devices", arXiv:2408.12840, 2024.
- Gao et al., "Single Path One-Shot NAS" (SPOS), ECCV 2020.
- Fey & Lenssen, "Fast Graph Representation Learning with PyTorch Geometric", ICLR-W 2019.
- Wang et al., "Dynamic Graph CNN for Learning on Point Clouds" (DGCNN), TOG 2019.
