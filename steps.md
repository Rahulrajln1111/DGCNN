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

## Summary of Phase 1-3 Commands

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

---

## PHASE 4: Ablation Studies on A100 — ~2 hours

This trains **14 DGCNN variants** to study the impact of each design choice.

### Step 12: Run all ablation experiments

```bash
# On A100 — runs all 14 experiments (~2 hours total)
python train_ablation.py

# OR run a quick subset first (~30 min)
python train_ablation.py --quick

# OR run specific experiment groups:
python train_ablation.py --experiments baseline lite tiny          # compression
python train_ablation.py --experiments baseline k5 k10 k15        # K sweep
python train_ablation.py --experiments baseline aggr_mean aggr_sum # aggregation
python train_ablation.py --experiments baseline static             # static graph
python train_ablation.py --experiments baseline attention          # attention
python train_ablation.py --experiments baseline depth2 depth3 depth5  # depth

# Override epochs for faster runs:
python train_ablation.py --epochs 100
```

**What each experiment tests:**

| Experiment | What Changes | Why |
|-----------|-------------|-----|
| `baseline` | Standard DGCNN (k=20, max, dynamic) | Reference point |
| `k5`, `k10`, `k15` | Number of KNN neighbors | Fewer K = faster on Jetson |
| `prog_k` | K decreases per layer [20,15,10,5] | Adaptive computation |
| `lite` | Channels [32,32,64,128] | 4× fewer params |
| `tiny` | Channels [16,16,32,64] | 16× fewer params |
| `aggr_mean` | Mean instead of max aggregation | Simpler computation |
| `aggr_sum` | Sum instead of max aggregation | Different inductive bias |
| `static` | KNN computed once, reused all layers | ~3-4× faster inference |
| `attention` | Edge attention gates | Learn which neighbors matter |
| `depth2/3/5` | Fewer/more EdgeConv layers | Depth vs efficiency |

### Step 13: Check ablation results

```bash
cat results/ablation_results.csv
# Shows: name, accuracy, params, training time for each variant
```

### Step 14: Generate training visualizations (on A100)

```bash
python visualize.py --skip-model-plots
# Generates plots in plots/ directory from CSV results
```

---

## PHASE 5: Benchmark ALL variants on Jetson — ~30 minutes

### Step 15: Transfer all checkpoints to Jetson

```bash
# From A100:
scp -r checkpoints/ jetson@<IP>:~/Desktop/GNN/DGCNN/
scp -r results/ jetson@<IP>:~/Desktop/GNN/DGCNN/

# Also push code updates:
git add -A && git commit -m "ablation studies" && git push
# On Jetson: git pull
```

### Step 16: Benchmark all variants on Jetson

```bash
# On Jetson — benchmarks every model in checkpoints/
python benchmark.py

# Also benchmark with FP16 (half precision):
python benchmark.py --fp16
```

**Expected output:**
```
  Name                 Acc%   Lat(ms)     Tput     Params
  -------------------------------------------------------
  baseline             93.2%    45.3      176    1,803,914
  lite                 90.1%    28.7      279      451,082
  tiny                 85.3%    18.2      440      113,546
  static               91.5%    12.1      661    1,803,914
  ...
```

### Step 17: Profile layer-by-layer timing

```bash
# Profile the baseline model
python profile_model.py --model checkpoints/dgcnn_baseline.pt

# Profile with FP16
python profile_model.py --model checkpoints/dgcnn_baseline.pt --fp16
```

This shows exactly where time is spent (KNN vs message passing vs aggregation).

---

## PHASE 6: Generate All Visualizations — ~10 minutes

### Step 18: Generate all plots

```bash
# On Jetson (or wherever you have all results):
python visualize.py
```

**Plots generated (in `plots/` directory):**

| Plot | What It Shows |
|------|---------------|
| `accuracy_comparison.png` | Bar chart of all variants ranked by accuracy |
| `params_comparison.png` | Model size comparison |
| `pareto_front.png` | Accuracy vs model size tradeoff |
| `k_sweep.png` | Impact of K on accuracy |
| `aggregation_comparison.png` | max vs mean vs sum |
| `static_vs_dynamic.png` | Speed vs accuracy tradeoff |
| `compression_summary.png` | Full vs Lite vs Tiny |
| `training_curves.png` | Loss/accuracy over epochs for all variants |
| `confusion_matrix.png` | 10×10 classification heatmap |
| `tsne_features.png` | t-SNE of learned features |
| `profile_breakdown_fp32.png` | Time per component (FP32) |
| `profile_breakdown_fp16.png` | Time per component (FP16) |
| `fp16_comparison.png` | FP32 vs FP16 per component |
| `jetson_benchmarks.png` | Latency + accuracy on Jetson |

### Step 19: Collect results for presentation

```bash
ls plots/
# 14+ PNG files ready for your presentation/report

cat results/ablation_results.csv       # Training results
cat results/jetson_benchmark.csv       # Jetson performance
cat benchmark_results/benchmark_report.txt  # Detailed report
```

---

## Complete Command Summary

```bash
# ═══════════════════════════════════════════════════
# PHASE 1-3: Basic pipeline (already done)
# ═══════════════════════════════════════════════════
pip install -r requirements.txt
python download_data.py
python train.py --epochs 200
scp -r checkpoints/ jetson@<IP>:~/Desktop/GNN/DGCNN/
# On Jetson:
python inference.py --model checkpoints/dgcnn_best.pt

# ═══════════════════════════════════════════════════
# PHASE 4: Ablation studies (on A100)
# ═══════════════════════════════════════════════════
python train_ablation.py                    # all experiments
python visualize.py --skip-model-plots      # generate plots

# ═══════════════════════════════════════════════════
# PHASE 5: Benchmark all variants (on Jetson)
# ═══════════════════════════════════════════════════
python benchmark.py --fp16                  # benchmark all models
python profile_model.py --model checkpoints/dgcnn_baseline.pt

# ═══════════════════════════════════════════════════
# PHASE 6: Visualizations (on Jetson or A100)
# ═══════════════════════════════════════════════════
python visualize.py                         # generate all 15+ plots
```

**Your project now demonstrates:**
- ✅ Custom GNN implementation (not just a library call)
- ✅ 5 edge-device optimizations with measured impact
- ✅ 14 ablation experiments with training curves
- ✅ 15+ publication-quality plots
- ✅ FP32 vs FP16 comparison
- ✅ Layer-wise profiling on real hardware
- ✅ Pareto-optimal model selection for edge deployment

