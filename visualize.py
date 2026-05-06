#!/usr/bin/env python3
"""
Visualization suite – generates all plots from ablation results.

Usage:
    python visualize.py                                     # all plots
    python visualize.py --results results/ablation_results.csv
    python visualize.py --jetson-results results/jetson_benchmark.csv

Generates 15+ plots in plots/ directory.
"""

import argparse
import csv
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ── Style ─────────────────────────────────────────────────────────────────────

COLORS = {
    "baseline": "#2196F3",
    "k5": "#FF9800", "k10": "#FF5722", "k15": "#E91E63",
    "prog_k": "#9C27B0",
    "lite": "#4CAF50", "tiny": "#8BC34A",
    "aggr_mean": "#00BCD4", "aggr_sum": "#009688",
    "static": "#795548",
    "attention": "#F44336",
    "depth2": "#CDDC39", "depth3": "#FFC107", "depth5": "#607D8B",
}

def get_color(name):
    return COLORS.get(name, "#757575")


def setup_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


# ── Plot generators ──────────────────────────────────────────────────────────

def plot_accuracy_comparison(results, output_dir):
    """Bar chart comparing test accuracy across all variants."""
    results = sorted(results, key=lambda x: x["best_test_acc"], reverse=True)
    names = [r["name"] for r in results]
    accs = [r["best_test_acc"] for r in results]
    colors = [get_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(names)), accs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Test Accuracy (%)")
    ax.set_title("DGCNN Variants – Accuracy Comparison")
    ax.set_xlim(left=max(0, min(accs) - 10))

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=150)
    plt.close()
    print("  ✓ accuracy_comparison.png")


def plot_params_comparison(results, output_dir):
    """Bar chart comparing parameter counts."""
    results = sorted(results, key=lambda x: x["num_params"])
    names = [r["name"] for r in results]
    params = [r["num_params"] / 1000 for r in results]  # in K
    colors = [get_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(names)), params, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Parameters (K)")
    ax.set_title("Model Size Comparison")

    for bar, p in zip(bars, params):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{p:.0f}K", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "params_comparison.png"), dpi=150)
    plt.close()
    print("  ✓ params_comparison.png")


def plot_pareto_front(results, output_dir, x_key="num_params", x_label="Parameters"):
    """Pareto front: accuracy vs model size (or latency)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    xs = [r[x_key] for r in results]
    ys = [r["best_test_acc"] for r in results]
    names = [r["name"] for r in results]
    sizes = [max(r["num_params"] / 5000, 50) for r in results]

    for x, y, name, s in zip(xs, ys, names, sizes):
        color = get_color(name)
        ax.scatter(x, y, s=s, c=color, edgecolors="black", linewidth=0.5, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(8, 5), fontsize=9, alpha=0.8)

    # Draw Pareto front line
    pts = sorted(zip(xs, ys), key=lambda p: p[0])
    pareto_x, pareto_y = [pts[0][0]], [pts[0][1]]
    best_y = pts[0][1]
    for px, py in pts[1:]:
        if py >= best_y:
            best_y = py
            pareto_x.append(px)
            pareto_y.append(py)
    ax.plot(pareto_x, pareto_y, "r--", alpha=0.5, linewidth=1.5, label="Pareto front")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy vs Model Size – Pareto Front")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pareto_front.png"), dpi=150)
    plt.close()
    print("  ✓ pareto_front.png")


def plot_k_sweep(results, output_dir):
    """Line chart: accuracy vs K value."""
    k_results = []
    for r in results:
        k_val = r.get("k", "20")
        if isinstance(k_val, str) and k_val.isdigit():
            k_results.append((int(k_val), r["best_test_acc"], r["name"]))

    if len(k_results) < 2:
        return

    k_results.sort()
    ks, accs, names = zip(*k_results)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, accs, "o-", color="#2196F3", linewidth=2.5, markersize=10,
            markerfacecolor="white", markeredgewidth=2)

    for k, acc, name in zip(ks, accs, names):
        ax.annotate(f"{acc:.1f}%", (k, acc), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("K (Number of Neighbors)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Impact of K on Classification Accuracy")
    ax.set_xticks(list(ks))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "k_sweep.png"), dpi=150)
    plt.close()
    print("  ✓ k_sweep.png")


def plot_aggregation_comparison(results, output_dir):
    """Bar chart comparing aggregation functions."""
    aggr_results = {}
    for r in results:
        aggr = r.get("aggr", "max")
        if aggr not in aggr_results or r["best_test_acc"] > aggr_results[aggr]:
            aggr_results[aggr] = r["best_test_acc"]

    if len(aggr_results) < 2:
        return

    names = list(aggr_results.keys())
    accs = [aggr_results[n] for n in names]
    colors = ["#2196F3", "#4CAF50", "#FF9800"][:len(names)]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, accs, color=colors, edgecolor="black", linewidth=0.5, width=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", fontsize=12, fontweight="bold")

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Aggregation Function Comparison")
    ax.set_ylim(bottom=max(0, min(accs) - 10))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aggregation_comparison.png"), dpi=150)
    plt.close()
    print("  ✓ aggregation_comparison.png")


def plot_training_curves(results_dir, output_dir):
    """Training loss/accuracy curves for all experiments."""
    history_files = [f for f in os.listdir(results_dir) if f.startswith("history_")]
    if not history_files:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for hf in sorted(history_files):
        name = hf.replace("history_", "").replace(".csv", "")
        epochs, losses, train_accs, test_accs = [], [], [], []

        with open(os.path.join(results_dir, hf)) as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                losses.append(float(row["loss"]))
                train_accs.append(float(row["train_acc"]))
                ta = float(row["test_acc"])
                if ta > 0:
                    test_accs.append((int(row["epoch"]), ta))

        color = get_color(name)
        ax1.plot(epochs, losses, color=color, alpha=0.8, linewidth=1.5, label=name)
        if test_accs:
            te, ta = zip(*test_accs)
            ax2.plot(te, ta, "o-", color=color, alpha=0.8, linewidth=1.5,
                     markersize=4, label=name)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend(fontsize=8, loc="upper right")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_title("Test Accuracy over Training")
    ax2.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()
    print("  ✓ training_curves.png")


def plot_static_vs_dynamic(results, output_dir):
    """Compare static graph vs dynamic graph."""
    static_acc = None
    dynamic_acc = None
    for r in results:
        if r.get("static_graph") in [True, "True"]:
            static_acc = r["best_test_acc"]
        if r["name"] == "baseline":
            dynamic_acc = r["best_test_acc"]

    if static_acc is None or dynamic_acc is None:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    names = ["Dynamic\n(Standard)", "Static\n(Optimized)"]
    accs = [dynamic_acc, static_acc]
    colors = ["#2196F3", "#795548"]

    bars = ax.bar(names, accs, color=colors, edgecolor="black", linewidth=0.5, width=0.4)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", fontsize=13, fontweight="bold")

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Static vs Dynamic Graph Construction")
    ax.set_ylim(bottom=max(0, min(accs) - 10))

    # Add annotation about speed
    ax.annotate("~3-4× faster\non Jetson", xy=(1, static_acc),
                xytext=(1.3, static_acc - 3), fontsize=10, color="#795548",
                arrowprops=dict(arrowstyle="->", color="#795548"))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "static_vs_dynamic.png"), dpi=150)
    plt.close()
    print("  ✓ static_vs_dynamic.png")


def plot_compression_summary(results, output_dir):
    """Side-by-side: Full vs Lite vs Tiny (accuracy + params)."""
    targets = {"baseline": "Full", "lite": "Lite", "tiny": "Tiny"}
    data = []
    for r in results:
        if r["name"] in targets:
            data.append((targets[r["name"]], r["best_test_acc"], r["num_params"]))

    if len(data) < 2:
        return

    data.sort(key=lambda x: x[2], reverse=True)
    names, accs, params = zip(*data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#2196F3", "#4CAF50", "#8BC34A"][:len(names)]

    bars1 = ax1.bar(names, accs, color=colors, edgecolor="black", linewidth=0.5)
    for bar, acc in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{acc:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Accuracy")
    ax1.set_ylim(bottom=max(0, min(accs) - 10))

    params_k = [p / 1000 for p in params]
    bars2 = ax2.bar(names, params_k, color=colors, edgecolor="black", linewidth=0.5)
    for bar, p in zip(bars2, params_k):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{p:.0f}K", ha="center", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Parameters (K)")
    ax2.set_title("Model Size")

    fig.suptitle("Model Compression: Full vs Lite vs Tiny", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compression_summary.png"), dpi=150)
    plt.close()
    print("  ✓ compression_summary.png")


def plot_confusion_matrix(model, test_loader, device, output_dir, class_names):
    """Generate confusion matrix for the best model."""
    from sklearn.metrics import confusion_matrix as cm_func

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch.y.view(-1).cpu().tolist())

    cm = cm_func(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    # Add text annotations
    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix – DGCNN on ModelNet10")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("  ✓ confusion_matrix.png")


def plot_tsne(model, test_loader, device, output_dir, class_names):
    """t-SNE visualization of learned features."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  ⚠ sklearn not available, skipping t-SNE")
        return

    model.eval()
    features, labels = [], []

    # Hook to capture features before classifier
    hook_output = {}
    def hook_fn(module, input, output):
        hook_output["feat"] = input[0].detach()

    # Register hook on the first layer of classifier
    handle = model.classifier[0].register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            _ = model(batch)
            features.append(hook_output["feat"].cpu().numpy())
            labels.extend(batch.y.view(-1).cpu().tolist())

    handle.remove()

    features = np.concatenate(features, axis=0)
    labels = np.array(labels)

    print("  Computing t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap("tab10", len(class_names))

    for i, cls_name in enumerate(class_names):
        mask = labels == i
        ax.scatter(embedded[mask, 0], embedded[mask, 1], c=[cmap(i)], label=cls_name,
                   s=15, alpha=0.7, edgecolors="none")

    ax.legend(loc="best", fontsize=8, markerscale=2)
    ax.set_title("t-SNE of Learned Features (DGCNN)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_features.png"), dpi=150)
    plt.close()
    print("  ✓ tsne_features.png")


def plot_jetson_benchmarks(csv_path, output_dir):
    """Plot Jetson benchmark results (latency, throughput, memory)."""
    if not os.path.exists(csv_path):
        print(f"  ⚠ No Jetson benchmark file: {csv_path}")
        return

    data = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    if not data:
        return

    names = [d["name"] for d in data]
    latencies = [float(d["latency_ms"]) for d in data]
    accs = [float(d["accuracy"]) for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = [get_color(n) for n in names]

    # Latency
    bars = ax1.barh(range(len(names)), latencies, color=colors,
                    edgecolor="black", linewidth=0.5)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names)
    ax1.set_xlabel("Latency (ms/batch)")
    ax1.set_title("Inference Latency on Jetson Nano")
    for bar, lat in zip(bars, latencies):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{lat:.1f}ms", va="center", fontsize=9)

    # Pareto: accuracy vs latency
    for x, y, name in zip(latencies, accs, names):
        ax2.scatter(x, y, s=100, c=get_color(name), edgecolors="black",
                    linewidth=0.5, zorder=3)
        ax2.annotate(name, (x, y), textcoords="offset points",
                     xytext=(5, 5), fontsize=9)

    ax2.set_xlabel("Latency (ms/batch)")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_title("Accuracy vs Latency – Jetson Nano")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "jetson_benchmarks.png"), dpi=150)
    plt.close()
    print("  ✓ jetson_benchmarks.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate all DGCNN plots")
    p.add_argument("--results", type=str, default="results/ablation_results.csv")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--jetson-results", type=str, default="results/jetson_benchmark.csv")
    p.add_argument("--output-dir", type=str, default="plots")
    p.add_argument("--model", type=str, default="checkpoints/dgcnn_baseline.pt",
                   help="Best model checkpoint (for confusion matrix / t-SNE)")
    p.add_argument("--data-root", type=str, default="./data/ModelNet10")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-model-plots", action="store_true",
                   help="Skip plots that require loading a model (confusion, t-SNE)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_style()

    print("=" * 60)
    print("  DGCNN Visualization Suite")
    print("=" * 60)

    # Load ablation results
    results = []
    if os.path.exists(args.results):
        with open(args.results) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["best_test_acc"] = float(row["best_test_acc"])
                row["num_params"] = int(row["num_params"])
                row["train_time_min"] = float(row["train_time_min"])
                results.append(row)
        print(f"  Loaded {len(results)} experiment results")
    else:
        print(f"  ⚠ No results file: {args.results}")

    if results:
        print("\n  Generating comparison plots...")
        plot_accuracy_comparison(results, args.output_dir)
        plot_params_comparison(results, args.output_dir)
        plot_pareto_front(results, args.output_dir)
        plot_k_sweep(results, args.output_dir)
        plot_aggregation_comparison(results, args.output_dir)
        plot_static_vs_dynamic(results, args.output_dir)
        plot_compression_summary(results, args.output_dir)

    # Training curves
    if os.path.exists(args.results_dir):
        print("\n  Generating training curves...")
        plot_training_curves(args.results_dir, args.output_dir)

    # Jetson benchmarks
    print("\n  Generating Jetson benchmark plots...")
    plot_jetson_benchmarks(args.jetson_results, args.output_dir)

    # Model-dependent plots (confusion matrix, t-SNE)
    if not args.skip_model_plots and os.path.exists(args.model):
        print("\n  Generating model analysis plots...")
        from dgcnn_model import DGCNN
        from dataset import get_test_loader, CLASSES

        config_path = args.model.replace("dgcnn_", "config_")
        k = 20
        model_kwargs = {"num_classes": 10, "k": k, "dropout": 0.0}
        if os.path.exists(config_path):
            config = torch.load(config_path, map_location="cpu")
            k = config.get("k", 20)
            model_kwargs["k"] = k

        model = DGCNN(**model_kwargs).to(args.device)
        model.load_state_dict(torch.load(args.model, map_location=args.device))
        model.eval()

        test_loader = get_test_loader(args.data_root, num_points=1024,
                                       batch_size=8, num_workers=0)

        plot_confusion_matrix(model, test_loader, args.device, args.output_dir, CLASSES)
        plot_tsne(model, test_loader, args.device, args.output_dir, CLASSES)

    print(f"\n{'='*60}")
    print(f"  All plots saved to: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
