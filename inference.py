#!/usr/bin/env python3
"""
Jetson Nano Inference – run trained DGCNN model on ModelNet10 test set.

Usage:
    python inference.py --model checkpoints/dgcnn_best.pt
    python inference.py --model checkpoints/dgcnn_best.pt --data-root ./data/ModelNet10

Requirements: Only packages already installed on Jetson Nano.
No new pip installs needed.
"""

import argparse
import time
import os

import torch
import numpy as np
from tqdm import tqdm

from dgcnn_model import DGCNN
from dataset import get_test_loader, CLASSES


def parse_args():
    p = argparse.ArgumentParser(description="DGCNN Inference on Jetson Nano")
    p.add_argument("--model", type=str, default="checkpoints/dgcnn_best.pt",
                   help="Path to trained model weights")
    p.add_argument("--config", type=str, default="checkpoints/dgcnn_config.pt",
                   help="Path to model config")
    p.add_argument("--data-root", type=str, default="./data/ModelNet10")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--warmup-runs", type=int, default=5,
                   help="Warmup inference runs before timing")
    return p.parse_args()


def load_model(args):
    """Load trained DGCNN model."""
    # Load config if available
    k = 20
    if os.path.exists(args.config):
        config = torch.load(args.config, map_location="cpu")
        k = config.get("k", 20)
        print(f"[Model] Config: k={k}, num_classes={config.get('num_classes', 10)}")

    model = DGCNN(num_classes=10, k=k, dropout=0.0)  # no dropout at inference
    state_dict = torch.load(args.model, map_location=args.device)
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Loaded DGCNN ({num_params:,} params) on {args.device}")
    return model


@torch.no_grad()
def run_inference(model, test_loader, device, warmup_runs=5):
    """
    Run full test-set inference. Returns:
      - accuracy (%)
      - per-class accuracy
      - latency stats (mean, std, min, max per batch)
      - throughput (samples/sec)
    """
    model.eval()

    # Warmup (important for GPU timing)
    print("[Inference] Warming up ...")
    for i, batch in enumerate(test_loader):
        if i >= warmup_runs:
            break
        batch = batch.to(device)
        _ = model(batch)
        if device == "cuda":
            torch.cuda.synchronize()

    # Timed inference
    print("[Inference] Running timed inference on test set ...")
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    latencies = []
    all_preds = []
    all_labels = []

    for batch in tqdm(test_loader, desc="Inference"):
        batch = batch.to(device)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits = model(batch)

        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)  # ms

        pred = logits.argmax(dim=1)
        labels = batch.y.view(-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        for p, l in zip(pred.cpu().tolist(), labels.cpu().tolist()):
            class_total[l] += 1
            if p == l:
                class_correct[l] += 1

    accuracy = 100.0 * correct / total
    latencies = np.array(latencies)
    throughput = total / (latencies.sum() / 1000)

    return {
        "accuracy": accuracy,
        "class_correct": class_correct,
        "class_total": class_total,
        "latencies_ms": latencies,
        "throughput_sps": throughput,
        "total_samples": total,
        "predictions": all_preds,
        "labels": all_labels,
    }


def print_results(results, device):
    """Print formatted inference results."""
    lat = results["latencies_ms"]

    print("\n" + "=" * 60)
    print("  DGCNN Inference Results – Jetson Nano")
    print("=" * 60)
    print(f"  Device           : {device}")
    print(f"  Test Accuracy    : {results['accuracy']:.2f}%")
    print(f"  Total Samples    : {results['total_samples']}")
    print(f"  Throughput       : {results['throughput_sps']:.1f} samples/sec")
    print(f"  Latency (batch)  : {lat.mean():.2f} ± {lat.std():.2f} ms")
    print(f"  Latency (min)    : {lat.min():.2f} ms")
    print(f"  Latency (max)    : {lat.max():.2f} ms")
    print(f"  Latency (median) : {np.median(lat):.2f} ms")

    print("\n  Per-Class Accuracy:")
    print("  " + "-" * 40)
    for i, cls_name in enumerate(CLASSES):
        total_i = results["class_total"][i]
        correct_i = results["class_correct"][i]
        acc_i = 100.0 * correct_i / total_i if total_i > 0 else 0
        print(f"    {cls_name:15s} : {acc_i:5.1f}% ({correct_i}/{total_i})")
    print("=" * 60)


def save_results(results, path="inference_results.txt"):
    """Save results to a text file."""
    with open(path, "w") as f:
        lat = results["latencies_ms"]
        f.write("DGCNN Inference Results\n")
        f.write(f"Accuracy: {results['accuracy']:.2f}%\n")
        f.write(f"Throughput: {results['throughput_sps']:.1f} samples/sec\n")
        f.write(f"Latency mean: {lat.mean():.2f} ms\n")
        f.write(f"Latency std: {lat.std():.2f} ms\n")
        f.write(f"Latency min: {lat.min():.2f} ms\n")
        f.write(f"Latency max: {lat.max():.2f} ms\n\n")
        f.write("Per-Class:\n")
        for i, cls_name in enumerate(CLASSES):
            total_i = results["class_total"][i]
            correct_i = results["class_correct"][i]
            acc_i = 100.0 * correct_i / total_i if total_i > 0 else 0
            f.write(f"  {cls_name}: {acc_i:.1f}% ({correct_i}/{total_i})\n")
    print(f"[Results] Saved to {path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("  DGCNN – Jetson Nano Inference")
    print(f"  Model      : {args.model}")
    print(f"  Device     : {args.device}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Num points : {args.num_points}")
    print("=" * 60)

    model = load_model(args)

    test_loader = get_test_loader(
        root=args.data_root,
        num_points=args.num_points,
        batch_size=args.batch_size,
        num_workers=0,  # 0 workers for Jetson (less memory)
    )

    results = run_inference(model, test_loader, args.device,
                            warmup_runs=args.warmup_runs)

    print_results(results, args.device)
    save_results(results)


if __name__ == "__main__":
    main()
