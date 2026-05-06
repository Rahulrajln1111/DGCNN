#!/usr/bin/env python3
"""
Benchmark DGCNN on Jetson Nano – measures latency, throughput, memory.

Generates performance plots and a summary report.

Usage:
    python benchmark.py --model checkpoints/dgcnn_best.pt
"""

import argparse
import time
import os

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from dgcnn_model import DGCNN
from dataset import get_test_loader, CLASSES, ModelNet10Dataset
from torch_geometric.data import Data

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark DGCNN")
    p.add_argument("--model", type=str, default="checkpoints/dgcnn_best.pt")
    p.add_argument("--config", type=str, default="checkpoints/dgcnn_config.pt")
    p.add_argument("--data-root", type=str, default="./data/ModelNet10")
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=str, default="benchmark_results")
    return p.parse_args()


def load_model(model_path, config_path, device):
    k = 20
    if os.path.exists(config_path):
        config = torch.load(config_path, map_location="cpu")
        k = config.get("k", 20)

    model = DGCNN(num_classes=10, k=k, dropout=0.0)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def benchmark_batch_sizes(model, data_root, num_points, device, batch_sizes=[1, 2, 4, 8, 16]):
    """Benchmark latency/throughput across different batch sizes."""
    results = []

    for bs in batch_sizes:
        print(f"\n[Benchmark] Batch size = {bs}")
        try:
            loader = get_test_loader(data_root, num_points, batch_size=bs, num_workers=0)

            # Warmup
            for i, batch in enumerate(loader):
                if i >= 3:
                    break
                batch = batch.to(device)
                _ = model(batch)
                if device == "cuda":
                    torch.cuda.synchronize()

            # Timed run
            latencies = []
            total_samples = 0
            for batch in loader:
                batch = batch.to(device)
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(batch)
                if device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)
                total_samples += batch.y.size(0)

            lat = np.array(latencies)
            total_time = lat.sum() / 1000  # seconds
            throughput = total_samples / total_time

            results.append({
                "batch_size": bs,
                "mean_latency_ms": lat.mean(),
                "std_latency_ms": lat.std(),
                "throughput_sps": throughput,
                "per_sample_ms": lat.mean() / bs if bs > 0 else 0,
            })

            print(f"  Latency: {lat.mean():.2f} ± {lat.std():.2f} ms/batch | "
                  f"Throughput: {throughput:.1f} samples/sec")

        except RuntimeError as e:
            print(f"  FAILED (likely OOM): {e}")
            results.append({"batch_size": bs, "error": str(e)})

    return results


@torch.no_grad()
def benchmark_num_points(model_path, config_path, data_root, device,
                         point_counts=[256, 512, 1024]):
    """Benchmark accuracy vs num_points."""
    results = []

    for np_ in point_counts:
        print(f"\n[Benchmark] Num points = {np_}")
        model = load_model(model_path, config_path, device)
        loader = get_test_loader(data_root, np_, batch_size=8, num_workers=0)

        correct, total = 0, 0
        latencies = []
        for batch in loader:
            batch = batch.to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(batch)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

            pred = logits.argmax(dim=1)
            correct += (pred == batch.y.view(-1)).sum().item()
            total += batch.y.size(0)

        acc = 100.0 * correct / total
        lat = np.array(latencies)
        results.append({
            "num_points": np_,
            "accuracy": acc,
            "mean_latency_ms": lat.mean(),
        })
        print(f"  Accuracy: {acc:.2f}% | Latency: {lat.mean():.2f} ms/batch")

    return results


def plot_results(batch_results, point_results, output_dir):
    """Generate benchmark plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Batch size vs Throughput
    valid = [r for r in batch_results if "error" not in r]
    if valid:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        bs = [r["batch_size"] for r in valid]
        tp = [r["throughput_sps"] for r in valid]
        lat = [r["mean_latency_ms"] for r in valid]

        ax1.bar(range(len(bs)), tp, color="#4CAF50", edgecolor="black")
        ax1.set_xticks(range(len(bs)))
        ax1.set_xticklabels(bs)
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Throughput (samples/sec)")
        ax1.set_title("Throughput vs Batch Size")

        ax2.bar(range(len(bs)), lat, color="#2196F3", edgecolor="black")
        ax2.set_xticks(range(len(bs)))
        ax2.set_xticklabels(bs)
        ax2.set_xlabel("Batch Size")
        ax2.set_ylabel("Latency (ms/batch)")
        ax2.set_title("Latency vs Batch Size")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "batch_benchmark.png"), dpi=150)
        plt.close()
        print(f"[Plot] Saved batch_benchmark.png")

    # Plot 2: Num points vs Accuracy/Latency
    if point_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        nps = [r["num_points"] for r in point_results]
        accs = [r["accuracy"] for r in point_results]
        lats = [r["mean_latency_ms"] for r in point_results]

        ax1.plot(nps, accs, "o-", color="#FF5722", linewidth=2, markersize=8)
        ax1.set_xlabel("Number of Points")
        ax1.set_ylabel("Test Accuracy (%)")
        ax1.set_title("Accuracy vs Point Count")
        ax1.grid(True, alpha=0.3)

        ax2.plot(nps, lats, "s-", color="#9C27B0", linewidth=2, markersize=8)
        ax2.set_xlabel("Number of Points")
        ax2.set_ylabel("Latency (ms/batch)")
        ax2.set_title("Latency vs Point Count")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "points_benchmark.png"), dpi=150)
        plt.close()
        print(f"[Plot] Saved points_benchmark.png")


def write_report(batch_results, point_results, output_dir, device):
    """Write a text summary report."""
    path = os.path.join(output_dir, "benchmark_report.txt")
    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  DGCNN Benchmark Report – Jetson Nano\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Device: {device}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"PyTorch: {torch.__version__}\n\n")

        f.write("--- Batch Size Benchmark ---\n")
        for r in batch_results:
            if "error" in r:
                f.write(f"  BS={r['batch_size']}: FAILED ({r['error'][:50]})\n")
            else:
                f.write(f"  BS={r['batch_size']}: "
                        f"{r['mean_latency_ms']:.2f}ms/batch | "
                        f"{r['throughput_sps']:.1f} samples/sec | "
                        f"{r['per_sample_ms']:.2f}ms/sample\n")

        f.write("\n--- Point Count Benchmark ---\n")
        for r in point_results:
            f.write(f"  {r['num_points']} points: "
                    f"{r['accuracy']:.2f}% accuracy | "
                    f"{r['mean_latency_ms']:.2f}ms/batch\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"[Report] Saved to {path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  DGCNN Benchmark Suite")
    print(f"  Device : {args.device}")
    print("=" * 60)

    model = load_model(args.model, args.config, args.device)

    # Benchmark 1: Batch sizes
    print("\n" + "=" * 40)
    print("  Benchmark 1: Batch Size Sweep")
    print("=" * 40)
    batch_results = benchmark_batch_sizes(
        model, args.data_root, args.num_points, args.device,
        batch_sizes=[1, 2, 4, 8]
    )

    # Benchmark 2: Point counts
    print("\n" + "=" * 40)
    print("  Benchmark 2: Point Count Sweep")
    print("=" * 40)
    point_results = benchmark_num_points(
        args.model, args.config, args.data_root, args.device,
        point_counts=[256, 512, 1024]
    )

    # Generate plots and report
    plot_results(batch_results, point_results, args.output_dir)
    write_report(batch_results, point_results, args.output_dir, args.device)

    print("\n" + "=" * 60)
    print("  Benchmark Complete!")
    print(f"  Results in: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
