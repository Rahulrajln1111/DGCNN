#!/usr/bin/env python3
"""
Layer-wise profiling for DGCNN on Jetson Nano.

Measures time spent in each component:
  - KNN graph construction
  - Message computation (edge features)
  - Aggregation
  - Classifier

Usage:
    python profile_model.py --model checkpoints/dgcnn_best.pt
    python profile_model.py --model checkpoints/dgcnn_baseline.pt --fp16
"""

import argparse
import os
import time

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dgcnn_model import DGCNN, knn_graph, scatter_agg
from dataset import get_test_loader


def parse_args():
    p = argparse.ArgumentParser(description="Profile DGCNN layer-by-layer")
    p.add_argument("--model", type=str, default="checkpoints/dgcnn_best.pt")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--data-root", type=str, default="./data/ModelNet10")
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-runs", type=int, default=20)
    p.add_argument("--fp16", action="store_true", help="Profile in FP16 mode")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=str, default="plots")
    return p.parse_args()


def sync():
    """CUDA synchronize if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def profile_forward_pass(model, data, device, fp16=False):
    """
    Profile a single forward pass, returning timing per component.
    """
    if fp16:
        model = model.half()

    model.eval()
    x = data.pos.to(device)
    batch = data.batch.to(device)

    if fp16:
        x = x.half()

    timings = {"knn": [], "message": [], "aggregate": [], "mlp": [],
               "projection": [], "pooling": [], "classifier": []}

    # Input projection is inside the first conv (3 -> 64)
    static_ei = None
    if model.static_graph:
        sync()
        t0 = time.perf_counter()
        static_ei = knn_graph(x, model.k_per_layer[0], batch)
        sync()
        timings["knn"].append(time.perf_counter() - t0)

    layer_outputs = []
    cur_x = x

    for i, conv in enumerate(model.convs):
        # KNN
        sync()
        t0 = time.perf_counter()
        if static_ei is not None:
            edge_index = static_ei
        else:
            edge_index = knn_graph(cur_x, conv.k, batch)
        sync()
        timings["knn"].append(time.perf_counter() - t0)

        row, col = edge_index[0], edge_index[1]

        # Message computation
        sync()
        t0 = time.perf_counter()
        x_i = cur_x[row]
        x_j = cur_x[col]
        edge_input = torch.cat([x_i, x_j - x_i], dim=-1)
        sync()
        timings["message"].append(time.perf_counter() - t0)

        # MLP on edges
        sync()
        t0 = time.perf_counter()
        edge_feat = conv.mlp(edge_input)
        if conv.use_attention:
            attn = conv.attn_gate(edge_input)
            edge_feat = edge_feat * attn
        sync()
        timings["mlp"].append(time.perf_counter() - t0)

        # Aggregation
        sync()
        t0 = time.perf_counter()
        cur_x = scatter_agg(edge_feat, row, dim_size=x.size(0), aggr=conv.aggr)
        sync()
        timings["aggregate"].append(time.perf_counter() - t0)

        layer_outputs.append(cur_x)

    # Projection
    sync()
    t0 = time.perf_counter()
    cat_x = torch.cat(layer_outputs, dim=-1)
    proj_x = torch.nn.functional.leaky_relu(model.bn0(model.lin0(cat_x)), 0.2)
    sync()
    timings["projection"].append(time.perf_counter() - t0)

    # Pooling
    sync()
    t0 = time.perf_counter()
    from torch_geometric.nn import global_max_pool, global_mean_pool
    x_max = global_max_pool(proj_x, batch)
    x_mean = global_mean_pool(proj_x, batch)
    pooled = torch.cat([x_max, x_mean], dim=-1)
    sync()
    timings["pooling"].append(time.perf_counter() - t0)

    # Classifier
    sync()
    t0 = time.perf_counter()
    _ = model.classifier(pooled)
    sync()
    timings["classifier"].append(time.perf_counter() - t0)

    # Sum each component across layers
    return {k: sum(v) * 1000 for k, v in timings.items()}  # ms


def plot_profile(profile_results, output_dir, suffix=""):
    """Stacked bar chart of time per component."""
    components = ["knn", "message", "mlp", "aggregate", "projection", "pooling", "classifier"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#00BCD4", "#795548"]

    means = {k: np.mean(profile_results[k]) for k in components}
    total = sum(means.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Stacked bar
    bottom = 0
    for comp, color in zip(components, colors):
        ax1.bar("DGCNN", means[comp], bottom=bottom, color=color, label=comp,
                edgecolor="black", linewidth=0.3)
        if means[comp] / total > 0.05:
            ax1.text(0, bottom + means[comp] / 2, f"{means[comp]:.1f}ms",
                     ha="center", va="center", fontsize=9, fontweight="bold")
        bottom += means[comp]

    ax1.set_ylabel("Time (ms)")
    ax1.set_title(f"Forward Pass Breakdown ({total:.1f}ms total)")
    ax1.legend(loc="upper right", fontsize=9)

    # Pie chart
    sizes = [means[k] for k in components]
    labels = [f"{k}\n{v:.1f}ms ({v/total*100:.0f}%)" for k, v in zip(components, sizes)]
    ax2.pie(sizes, labels=labels, colors=colors, startangle=90,
            textprops={"fontsize": 9})
    ax2.set_title("Time Distribution")

    plt.tight_layout()
    fname = f"profile_breakdown{suffix}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()
    print(f"  ✓ {fname}")


def plot_fp16_comparison(fp32_results, fp16_results, output_dir):
    """Compare FP32 vs FP16 timing."""
    components = ["knn", "message", "mlp", "aggregate", "projection", "pooling", "classifier"]
    fp32_vals = [np.mean(fp32_results[c]) for c in components]
    fp16_vals = [np.mean(fp16_results[c]) for c in components]

    x = np.arange(len(components))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, fp32_vals, width, label="FP32", color="#2196F3",
                   edgecolor="black", linewidth=0.3)
    bars2 = ax.bar(x + width / 2, fp16_vals, width, label="FP16", color="#4CAF50",
                   edgecolor="black", linewidth=0.3)

    ax.set_xlabel("Component")
    ax.set_ylabel("Time (ms)")
    ax.set_title("FP32 vs FP16 – Per-Component Latency")
    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=30, ha="right")
    ax.legend()

    total32 = sum(fp32_vals)
    total16 = sum(fp16_vals)
    speedup = total32 / total16 if total16 > 0 else 0
    ax.annotate(f"Total: {total32:.1f}ms → {total16:.1f}ms ({speedup:.1f}× speedup)",
                xy=(0.5, 0.95), xycoords="axes fraction", fontsize=12,
                ha="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fp16_comparison.png"), dpi=150)
    plt.close()
    print("  ✓ fp16_comparison.png")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  DGCNN Layer-wise Profiling")
    print(f"  Device: {args.device}")
    print(f"  FP16: {args.fp16}")
    print("=" * 60)

    # Load model
    config_path = args.config or args.model.replace("dgcnn_", "config_")
    model_kwargs = {"num_classes": 10, "k": 20, "dropout": 0.0}
    if os.path.exists(config_path):
        config = torch.load(config_path, map_location="cpu")
        for key in ["k", "channels", "emb_dim", "aggr", "static_graph", "use_attention", "preset"]:
            if key in config:
                model_kwargs[key] = config[key]

    model = DGCNN(**model_kwargs).to(args.device)
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    model.eval()
    print(f"[Model] Loaded: {model.count_parameters():,} params")

    # Load test data
    test_loader = get_test_loader(args.data_root, args.num_points,
                                   args.batch_size, num_workers=0)

    # Get one batch for profiling
    batch = next(iter(test_loader)).to(args.device)

    # Warmup
    print("[Profile] Warming up...")
    for _ in range(3):
        with torch.no_grad():
            _ = model(batch)
        sync()

    # Profile FP32
    print(f"[Profile] Running {args.num_runs} profiling passes (FP32)...")
    fp32_results = {k: [] for k in ["knn", "message", "mlp", "aggregate",
                                     "projection", "pooling", "classifier"]}
    for _ in range(args.num_runs):
        timings = profile_forward_pass(model, batch, args.device, fp16=False)
        for k, v in timings.items():
            fp32_results[k].append(v)

    plot_profile(fp32_results, args.output_dir, "_fp32")

    # Profile FP16
    print(f"[Profile] Running {args.num_runs} profiling passes (FP16)...")
    fp16_results = {k: [] for k in fp32_results.keys()}
    try:
        for _ in range(args.num_runs):
            timings = profile_forward_pass(model, batch, args.device, fp16=True)
            for k, v in timings.items():
                fp16_results[k].append(v)

        plot_profile(fp16_results, args.output_dir, "_fp16")
        plot_fp16_comparison(fp32_results, fp16_results, args.output_dir)
    except Exception as e:
        print(f"  ⚠ FP16 profiling failed: {e}")

    # Print summary
    print("\n  Component Breakdown (FP32):")
    total = 0
    for k in ["knn", "message", "mlp", "aggregate", "projection", "pooling", "classifier"]:
        mean = np.mean(fp32_results[k])
        total += mean
        print(f"    {k:15s}: {mean:7.2f} ms")
    print(f"    {'TOTAL':15s}: {total:7.2f} ms")

    print(f"\n  Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
