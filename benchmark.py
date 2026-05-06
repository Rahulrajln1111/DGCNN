#!/usr/bin/env python3
"""
Benchmark all trained DGCNN variants on Jetson Nano.

Loads every checkpoint in checkpoints/, runs inference on test set,
measures latency/accuracy/throughput, and saves CSV for visualization.

Usage:
    python benchmark.py                           # benchmark all checkpoints
    python benchmark.py --models baseline lite    # specific models only
    python benchmark.py --fp16                    # also benchmark FP16
"""

import argparse
import csv
import os
import time
import glob

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dgcnn_model import DGCNN
from dataset import get_test_loader, CLASSES


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark DGCNN variants")
    p.add_argument("--models", nargs="+", default=None,
                   help="Model names to benchmark (default: all in checkpoints/)")
    p.add_argument("--data-root", type=str, default="./data/ModelNet10")
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--fp16", action="store_true", help="Also benchmark FP16")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=str, default="results")
    return p.parse_args()


def discover_models(models_filter=None):
    """Find all model checkpoints and their configs."""
    found = []
    for ckpt in sorted(glob.glob("checkpoints/dgcnn_*.pt")):
        name = os.path.basename(ckpt).replace("dgcnn_", "").replace(".pt", "")
        if name in ["config", "final"]:
            continue
        if models_filter and name not in models_filter:
            continue

        config_path = f"checkpoints/config_{name}.pt"
        found.append({"name": name, "ckpt": ckpt, "config": config_path})

    return found


def load_model(info, device):
    """Load a model from checkpoint + config."""
    model_kwargs = {"num_classes": 10, "dropout": 0.0}

    if os.path.exists(info["config"]):
        config = torch.load(info["config"], map_location="cpu")
        for key in ["k", "channels", "emb_dim", "aggr", "static_graph",
                     "use_attention", "preset"]:
            if key in config:
                model_kwargs[key] = config[key]

    model = DGCNN(**model_kwargs).to(device)
    model.load_state_dict(torch.load(info["ckpt"], map_location=device))
    model.eval()
    return model


@torch.no_grad()
def benchmark_model(model, test_loader, device, name, fp16=False):
    """Benchmark a single model. Returns results dict."""
    if fp16:
        model = model.half()

    model.eval()

    # Warmup
    for i, batch in enumerate(test_loader):
        if i >= 3:
            break
        batch = batch.to(device)
        x = batch
        if fp16:
            batch.pos = batch.pos.half()
        _ = model(batch)
        if device == "cuda":
            torch.cuda.synchronize()

    # Timed inference
    correct, total = 0, 0
    latencies = []
    class_correct = [0] * 10
    class_total = [0] * 10

    for batch in test_loader:
        batch = batch.to(device)
        if fp16:
            batch.pos = batch.pos.half()

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)
        pred = logits.argmax(dim=1)
        labels = batch.y.view(-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        for p, l in zip(pred.cpu().tolist(), labels.cpu().tolist()):
            class_total[l] += 1
            if p == l:
                class_correct[l] += 1

    lat = np.array(latencies)
    accuracy = 100.0 * correct / total
    throughput = total / (lat.sum() / 1000)
    num_params = sum(p.numel() for p in model.parameters())

    return {
        "name": f"{name}_fp16" if fp16 else name,
        "accuracy": accuracy,
        "latency_ms": lat.mean(),
        "latency_std": lat.std(),
        "throughput_sps": throughput,
        "num_params": num_params,
        "fp16": fp16,
        "class_correct": class_correct,
        "class_total": class_total,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  DGCNN Benchmark Suite")
    print(f"  Device     : {args.device}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  FP16       : {args.fp16}")
    print("=" * 60)

    models = discover_models(args.models)
    if not models:
        print("[ERROR] No model checkpoints found in checkpoints/")
        return

    print(f"\n  Found {len(models)} models: {[m['name'] for m in models]}")

    test_loader = get_test_loader(args.data_root, args.num_points,
                                   args.batch_size, num_workers=0)

    all_results = []

    for info in models:
        print(f"\n  Benchmarking: {info['name']}")
        model = load_model(info, args.device)
        print(f"    Params: {model.count_parameters():,}")

        # FP32
        result = benchmark_model(model, test_loader, args.device, info["name"])
        all_results.append(result)
        print(f"    FP32: {result['accuracy']:.1f}% | {result['latency_ms']:.1f}ms | "
              f"{result['throughput_sps']:.0f} samp/s")

        # FP16
        if args.fp16:
            try:
                result16 = benchmark_model(model, test_loader, args.device,
                                            info["name"], fp16=True)
                all_results.append(result16)
                print(f"    FP16: {result16['accuracy']:.1f}% | {result16['latency_ms']:.1f}ms | "
                      f"{result16['throughput_sps']:.0f} samp/s")
            except Exception as e:
                print(f"    FP16 FAILED: {e}")

    # Save CSV (without per-class data for cleanliness)
    csv_rows = []
    for r in all_results:
        csv_rows.append({
            "name": r["name"],
            "accuracy": f"{r['accuracy']:.2f}",
            "latency_ms": f"{r['latency_ms']:.2f}",
            "latency_std": f"{r['latency_std']:.2f}",
            "throughput_sps": f"{r['throughput_sps']:.1f}",
            "num_params": r["num_params"],
            "fp16": r["fp16"],
        })

    csv_path = os.path.join(args.output_dir, "jetson_benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  {'Name':<20} {'Acc%':>6} {'Lat(ms)':>8} {'Tput':>8} {'Params':>10}")
    print(f"  {'-'*55}")
    for r in all_results:
        print(f"  {r['name']:<20} {r['accuracy']:>5.1f}% {r['latency_ms']:>7.1f} "
              f"{r['throughput_sps']:>7.0f} {r['num_params']:>10,}")
    print(f"{'='*60}")
    print(f"  Results saved: {csv_path}")


if __name__ == "__main__":
    main()
