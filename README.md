# DGCNN on Edge Devices – Implementing GNN on Jetson Nano

This project trains a **DGCNN** (Dynamic Graph CNN) for 3D point cloud classification on **ModelNet10**, then deploys inference to an **NVIDIA Jetson Nano** edge device.

## Architecture

**DGCNN** (Wang et al., 2019) is a Graph Neural Network that:
- Dynamically constructs KNN graphs in feature space at each layer
- Uses **EdgeConv** (message passing) to learn local geometric patterns
- Achieves **92-94% accuracy** on ModelNet10

## Project Structure

```
├── dgcnn_model.py       # DGCNN model definition (shared between A100 & Jetson)
├── dataset.py           # Data loading from .off files
├── train.py             # Training script (run on A100/GPU)
├── inference.py         # Inference + timing (run on Jetson Nano)
├── benchmark.py         # Performance benchmarking suite
├── download_data.py     # Download ModelNet10 dataset
├── steps.md             # Step-by-step execution guide
├── requirements.txt     # Python dependencies (for A100)
├── checkpoints/         # Saved model weights (created during training)
├── data/ModelNet10/     # Dataset (already on Jetson, download on A100)
└── nouse/               # Archived old HGNAS implementation
```

## Quick Start

### Train on A100 (GCP)
```bash
pip install -r requirements.txt
python download_data.py
python train.py --epochs 200
```

### Deploy to Jetson Nano
```bash
# Transfer checkpoints/ folder to Jetson
scp -r checkpoints/ jetson@<IP>:~/GNN/DGCNN/
# On Jetson:
python inference.py --model checkpoints/dgcnn_best.pt
python benchmark.py --model checkpoints/dgcnn_best.pt
```

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 92-94% |
| Model Parameters | ~1.8M |
| A100 Training Time | ~20 min |
| Jetson Inference Latency | TBD |

## References

- Wang et al., "Dynamic Graph CNN for Learning on Point Clouds", ACM TOG 2019
- Zhou et al., "HGNAS: Hardware-Aware GNN Architecture Search", IEEE TC 2024
