# Step-by-Step Guide: DGCNN on Jetson Nano

Complete walkthrough to train DGCNN on an A100 (GCP) and deploy to Jetson Nano.

**Total estimated time: ~2 hours**

---

## PHASE 1: Train on A100 (GCP) — ~30 minutes

### Step 1: Create GCP VM with A100 GPU

1. Go to [GCP Console](https://console.cloud.google.com/compute)
2. Create a new VM instance:
   - **Machine type**: `a2-highgpu-1g` (1x A100 40GB)
   - **OS**: Ubuntu 20.04 or 22.04
   - **Boot disk**: 50 GB (SSD)
   - **GPU**: NVIDIA A100
3. SSH into the VM

> **Cost**: ~$3-4/hr. Total job takes ~30 min = **~$2 total cost**.
> **Cheaper option**: Use a T4 GPU (`n1-standard-4` + T4) at ~$0.35/hr — training takes ~45 min.

### Step 2: Set up environment on A100

```bash
# Clone your repo (or upload files)
git clone <your-repo-url>
cd GNN/DGCNN

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.6.0+cu121.html
pip install numpy matplotlib tqdm scikit-learn

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

> **Note**: Adjust the PyTorch/CUDA versions in the URLs above to match what's available.
> Check https://pytorch.org/get-started/locally/ for the latest install command.
> Check https://data.pyg.org/whl/ for compatible PyG extension versions.

### Step 3: Download ModelNet10

```bash
python download_data.py
```

This downloads and extracts ModelNet10 (~500MB) to `./data/ModelNet10/raw/`.

### Step 4: Train DGCNN

```bash
# Full training (200 epochs, ~20 min on A100)
python train.py --epochs 200 --batch-size 32 --k 20 --num-points 1024

# OR quick test first (10 epochs, ~2 min) to verify everything works
python train.py --quick
```

**Expected output:**
```
  DGCNN Training – ModelNet10
  Device     : cuda
  Epochs     : 200
  ...
  Epoch 200/200 | Loss: 0.0523 | Train: 98.2% | Test: 93.1% | Best: 93.5%

  Training Complete!
  Best Test Accuracy : 93.50%
  Model saved to     : checkpoints/dgcnn_best.pt
```

### Step 5: Verify output files

```bash
ls -lh checkpoints/
# Should see:
#   dgcnn_best.pt     (~7 MB)  ← best model weights
#   dgcnn_final.pt    (~7 MB)  ← final epoch weights
#   dgcnn_config.pt   (~1 KB)  ← model config
```

---

## PHASE 2: Transfer to Jetson Nano — ~5 minutes

### Step 6: Copy model files to Jetson

**Option A: Direct SCP (if Jetson is on same network)**
```bash
# From A100 VM:
scp -r checkpoints/ <jetson-user>@<jetson-ip>:~/Desktop/GNN/DGCNN/
```

**Option B: Via your laptop**
```bash
# Download from A100 to your laptop
scp -r <a100-user>@<a100-ip>:~/GNN/DGCNN/checkpoints/ ./checkpoints/

# Upload to Jetson
scp -r ./checkpoints/ <jetson-user>@<jetson-ip>:~/Desktop/GNN/DGCNN/
```

**Option C: Via Google Drive / USB stick**
- Download `checkpoints/` folder from A100
- Transfer to Jetson via USB or cloud storage

> **Important**: You ONLY need to transfer the `checkpoints/` folder (~15 MB).
> The code files should already be on the Jetson (push via git or copy).

### Step 7: Make sure code files are on Jetson

The Jetson needs these files in `~/Desktop/GNN/DGCNN/`:
```
dgcnn_model.py
dataset.py
inference.py
benchmark.py
checkpoints/dgcnn_best.pt
checkpoints/dgcnn_config.pt
data/ModelNet10/raw/   (already there from before)
```

```bash
# Push code via git from A100 or your laptop
git add -A
git commit -m "DGCNN implementation"
git push

# On Jetson:
cd ~/Desktop/GNN/DGCNN
git pull
```

---

## PHASE 3: Run on Jetson Nano — ~15 minutes

### Step 8: Set Jetson to max performance mode

```bash
sudo nvpmodel -m 0        # MAXN power mode
sudo jetson_clocks         # Lock GPU/CPU at max frequency
```

### Step 9: Run inference

```bash
cd ~/Desktop/GNN/DGCNN

# Run inference on full test set (908 samples)
python inference.py --model checkpoints/dgcnn_best.pt --batch-size 8
```

**Expected output:**
```
  DGCNN Inference Results – Jetson Nano
  Device           : cuda
  Test Accuracy    : 93.28%
  Throughput       : XX.X samples/sec
  Latency (batch)  : XX.XX ± X.XX ms
  
  Per-Class Accuracy:
    bathtub         : 92.0%
    bed             : 95.0%
    chair           : 96.0%
    ...
```

### Step 10: Run benchmarks

```bash
python benchmark.py --model checkpoints/dgcnn_best.pt
```

This generates:
- `benchmark_results/batch_benchmark.png` — throughput vs batch size
- `benchmark_results/points_benchmark.png` — accuracy vs point count
- `benchmark_results/benchmark_report.txt` — full text report

### Step 11: Check results

```bash
cat benchmark_results/benchmark_report.txt
# View plots (if GUI available):
# eog benchmark_results/batch_benchmark.png
```

---

## Troubleshooting

### "No .off files found"
Make sure ModelNet10 data is at `./data/ModelNet10/raw/bathtub/train/*.off`.
The directory structure should be:
```
data/ModelNet10/raw/
├── bathtub/
│   ├── train/
│   │   ├── bathtub_0001.off
│   │   └── ...
│   └── test/
│       └── ...
├── bed/
│   ├── train/
│   └── test/
...
```

### CUDA out of memory on Jetson
Reduce batch size:
```bash
python inference.py --batch-size 2
# or even batch-size 1
```

Also free memory:
```bash
# Disable GUI to free ~800MB RAM
sudo systemctl set-default multi-user.target
sudo reboot
```

### Model loading error (PyTorch version mismatch)
If you get errors loading the checkpoint, the model was saved with a different
PyTorch version. Try re-saving on the A100 with explicit compatibility:
```python
import torch
state = torch.load("checkpoints/dgcnn_best.pt")
torch.save(state, "checkpoints/dgcnn_best_compat.pt",
           _use_new_zipfile_serialization=True)
```

### Slow inference on Jetson
- Make sure you ran `sudo nvpmodel -m 0` and `sudo jetson_clocks`
- Use batch-size 4 or 8 (not 1)
- Reduce num_points to 512: `python inference.py --num-points 512`

---

## Summary of Commands

```bash
# === A100 (GCP) ===
pip install -r requirements.txt
python download_data.py
python train.py --epochs 200

# === Transfer ===
scp -r checkpoints/ jetson@<IP>:~/Desktop/GNN/DGCNN/

# === Jetson Nano ===
sudo nvpmodel -m 0 && sudo jetson_clocks
python inference.py --model checkpoints/dgcnn_best.pt
python benchmark.py --model checkpoints/dgcnn_best.pt
```

**Done! You now have a GNN running on an edge device with full benchmarks.** 🎉
