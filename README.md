# DGCNN on Edge Devices – GNN for Jetson Nano

[PyTorch](https://pytorch.org/) | [NVIDIA Jetson Nano](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/) | [License: MIT](https://opensource.org/licenses/MIT)

This project implements and optimizes **DGCNN** (Dynamic Graph CNN) for 3D point cloud classification on **ModelNet10**, specifically tailored for deployment on the **NVIDIA Jetson Nano** edge device.

---

## Key Features

- **Dynamic Graph Construction**: Recomputes KNN graphs in feature space to capture local geometric evolution.
- **EdgeConv Optimization**: Custom message-passing layers optimized for Maxwell-generation GPUs (Jetson Nano).
- **Model Compression**: Supports multiple presets (**Full**, **Lite**, **Tiny**) to balance accuracy and latency.
- **Edge-Ready Optimizations**:
  - **Static Graph Reuse**: Compute KNN once and reuse across layers to save compute.
  - **Progressive K-Reduction**: Dynamically reduce neighborhood size in deeper layers.
  - **Edge Attention**: Gated attention mechanism to focus on relevant local features.
  - **FP16 Inference**: Full support for half-precision floating point on Jetson.

## Project Structure

```bash
├── dgcnn_model.py       # Core model definition with optimization toggles
├── dataset.py           # Efficient data loading for .off (ModelNet10) files
├── train.py             # Training script with logging and checkpointing
├── inference.py         # Jetson Nano inference & per-class evaluation
├── benchmark.py         # Latency & throughput benchmarking suite
├── download_data.py     # Script to fetch and prepare ModelNet10
├── profile_model.py     # Layer-wise profiling for A100 vs Jetson
├── experiments.py       # Automated ablation study runner
├── requirements.txt     # Python dependencies
└── checkpoints/         # Pre-trained weights and configurations
```

---

## Usage Guide

### 1. Training (GCP A100 / Local GPU)

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset
python download_data.py

# Standard training (200 epochs)
python train.py --epochs 200 --k 20

# Quick test run
python train.py --quick
```

### 2. Edge Deployment (Jetson Nano)

Transfer the `checkpoints/` folder to your Jetson device, then run:

```bash
# Run full inference evaluation
python inference.py --model checkpoints/dgcnn_best.pt

# Run performance benchmark
python benchmark.py --model checkpoints/dgcnn_best.pt --batch-size 1
```

---

## Results

| Variant | Params | ModelNet10 Acc | Jetson Latency (BS=1) |
| :--- | :--- | :--- | :--- |
| **DGCNN Full** | 1.8M | 93.4% | ~85ms |
| **DGCNN Lite** | 0.5M | 91.8% | ~42ms |
| **DGCNN Tiny** | 0.1M | 88.5% | ~18ms |

*Note: Benchmarks performed on Jetson Nano 4GB (Max-N mode).*

---

## References

1.  **Wang et al.**, "[Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829)", ACM Transactions on Graphics (TOG), 2019.
2.  **Zhou et al.**, "[HGNAS: Hardware-Aware GNN Architecture Search](https://ieeexplore.ieee.org/document/10123010)", IEEE Transactions on Computers, 2024.
