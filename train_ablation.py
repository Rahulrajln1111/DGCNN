#!/usr/bin/env python3
"""
Ablation Study Runner – trains all DGCNN variants and logs results.

Usage:
    python train_ablation.py                           # run ALL experiments
    python train_ablation.py --experiments baseline lite tiny static
    python train_ablation.py --quick                   # run quick subset
    python train_ablation.py --epochs 50               # override epochs

Each experiment trains a model variant, evaluates on test set,
and saves results to results/ablation_results.csv.
"""

import argparse
import csv
import os
import time

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dgcnn_model import DGCNN
from dataset import get_dataloaders
from experiments import EXPERIMENTS, QUICK_EXPERIMENTS


def parse_args():
    p = argparse.ArgumentParser(description="DGCNN Ablation Studies")
    p.add_argument("--experiments", nargs="+", default=None,
                   help="Specific experiments to run (default: all)")
    p.add_argument("--quick", action="store_true",
                   help="Run quick subset only")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epochs for all experiments")
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--data-root", type=str, default="./data/ModelNet10")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        pred = logits.argmax(dim=1)
        correct += (pred == batch.y.view(-1)).sum().item()
        total += batch.y.size(0)
    return 100.0 * correct / total


def train_one_experiment(name, config, train_loader, test_loader, args):
    """Train a single experiment and return results dict."""
    desc = config["desc"]
    model_args = config["model_args"].copy()
    train_args = config["train_args"].copy()

    if args.epochs is not None:
        train_args["epochs"] = args.epochs

    model_args["num_classes"] = 10
    model_args["dropout"] = 0.5

    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"  {desc}")
    print(f"  Model args: {model_args}")
    print(f"  Epochs: {train_args['epochs']}")
    print(f"{'='*60}")

    model = DGCNN(**model_args).to(args.device)
    num_params = model.count_parameters()
    print(f"  Parameters: {num_params:,}")

    optimizer = Adam(model.parameters(), lr=train_args["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=train_args["epochs"], eta_min=1e-5)

    best_acc = 0.0
    train_history = []
    start_time = time.time()

    for epoch in range(1, train_args["epochs"] + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            batch = batch.to(args.device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * batch.y.size(0)
            correct += (logits.argmax(1) == batch.y.view(-1)).sum().item()
            total += batch.y.size(0)

        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_loss = total_loss / total

        test_acc = 0.0
        if epoch % 10 == 0 or epoch == train_args["epochs"]:
            test_acc = evaluate(model, test_loader, args.device)
            if test_acc > best_acc:
                best_acc = test_acc
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), f"checkpoints/dgcnn_{name}.pt")

            print(f"  Epoch {epoch:3d}/{train_args['epochs']} | Loss: {avg_loss:.4f} | "
                  f"Train: {train_acc:.1f}% | Test: {test_acc:.1f}% | Best: {best_acc:.1f}%")

        train_history.append({
            "epoch": epoch, "loss": avg_loss,
            "train_acc": train_acc, "test_acc": test_acc,
        })

    elapsed = time.time() - start_time

    # Save training history
    os.makedirs("results", exist_ok=True)
    with open(f"results/history_{name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "train_acc", "test_acc"])
        writer.writeheader()
        writer.writerows(train_history)

    # Save model config
    config_save = model_args.copy()
    config_save["name"] = name
    config_save["best_acc"] = best_acc
    config_save["num_params"] = num_params
    torch.save(config_save, f"checkpoints/config_{name}.pt")

    print(f"\n  ✓ {name}: Best={best_acc:.2f}%, Params={num_params:,}, Time={elapsed/60:.1f}min")

    return {
        "name": name,
        "desc": desc,
        "best_test_acc": best_acc,
        "num_params": num_params,
        "train_time_min": elapsed / 60,
        "k": str(model_args.get("k", 20)),
        "channels": str(model_args.get("channels", "")),
        "aggr": model_args.get("aggr", "max"),
        "static_graph": model_args.get("static_graph", False),
        "use_attention": model_args.get("use_attention", False),
    }


def main():
    args = parse_args()

    # Determine which experiments to run
    if args.experiments:
        exp_names = args.experiments
    elif args.quick:
        exp_names = QUICK_EXPERIMENTS
    else:
        exp_names = list(EXPERIMENTS.keys())

    print(f"Running {len(exp_names)} experiments: {exp_names}")

    # Load data once
    batch_size = max(config["train_args"]["batch_size"]
                     for name, config in EXPERIMENTS.items() if name in exp_names)
    train_loader, test_loader = get_dataloaders(
        root=args.data_root, num_points=args.num_points,
        batch_size=batch_size, augment_train=True, num_workers=args.num_workers,
    )

    # Run experiments
    all_results = []
    for name in exp_names:
        if name not in EXPERIMENTS:
            print(f"[WARN] Unknown experiment: {name}, skipping")
            continue
        result = train_one_experiment(name, EXPERIMENTS[name],
                                      train_loader, test_loader, args)
        all_results.append(result)

    # Save summary CSV
    os.makedirs("results", exist_ok=True)
    summary_path = "results/ablation_results.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'='*60}")
    print(f"  All experiments complete!")
    print(f"  Summary: {summary_path}")
    print(f"{'='*60}")

    # Print summary table
    print(f"\n  {'Name':<15} {'Acc%':>6} {'Params':>10} {'Time':>8}")
    print(f"  {'-'*42}")
    for r in sorted(all_results, key=lambda x: x["best_test_acc"], reverse=True):
        print(f"  {r['name']:<15} {r['best_test_acc']:>5.1f}% "
              f"{r['num_params']:>10,} {r['train_time_min']:>6.1f}min")


if __name__ == "__main__":
    main()
