#!/usr/bin/env python3
"""
Train DGCNN on ModelNet10 – run this on A100 (GCP).

Usage:
    python train.py                          # defaults: 200 epochs, k=20, 1024 pts
    python train.py --epochs 100 --k 20      # custom
    python train.py --quick                  # fast test run (10 epochs)

Output:
    checkpoints/dgcnn_best.pt   – best model weights (for Jetson deployment)
    checkpoints/dgcnn_final.pt  – final epoch weights
    training_log.csv            – epoch-by-epoch metrics
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


def parse_args():
    p = argparse.ArgumentParser(description="Train DGCNN on ModelNet10")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--k", type=int, default=20, help="KNN neighbors")
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--data-root", type=str, default="./data/ModelNet10")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quick", action="store_true", help="Quick test (10 epochs)")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model accuracy on a data loader."""
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        pred = logits.argmax(dim=1)
        correct += (pred == batch.y.view(-1)).sum().item()
        total += batch.y.size(0)
    return 100.0 * correct / total


def train(args):
    device = args.device
    if args.quick:
        args.epochs = 10
        args.batch_size = 16

    print("=" * 60)
    print("  DGCNN Training – ModelNet10")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  K (KNN)    : {args.k}")
    print(f"  Num points : {args.num_points}")
    print(f"  LR         : {args.lr}")
    print("=" * 60)

    # Data
    train_loader, test_loader = get_dataloaders(
        root=args.data_root,
        num_points=args.num_points,
        batch_size=args.batch_size,
        augment_train=True,
        num_workers=args.num_workers,
    )

    # Model
    model = DGCNN(num_classes=10, k=args.k, dropout=0.5).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] DGCNN with {num_params:,} parameters")

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)

    best_test_acc = 0.0
    log_rows = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        epoch_start = time.time()

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * batch.y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == batch.y.view(-1)).sum().item()
            total += batch.y.size(0)

        scheduler.step()

        train_acc = 100.0 * correct / total
        avg_loss = total_loss / total
        epoch_time = time.time() - epoch_start

        # Evaluate every 5 epochs (or last epoch)
        test_acc = 0.0
        if epoch % 5 == 0 or epoch == args.epochs or epoch <= 3:
            test_acc = evaluate(model, test_loader, device)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), "checkpoints/dgcnn_best.pt")

        log_rows.append({
            "epoch": epoch, "loss": f"{avg_loss:.4f}",
            "train_acc": f"{train_acc:.2f}", "test_acc": f"{test_acc:.2f}",
            "lr": f"{scheduler.get_last_lr()[0]:.6f}", "time_s": f"{epoch_time:.1f}",
        })

        print(f"  Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f} | "
              f"Train: {train_acc:.1f}% | Test: {test_acc:.1f}% | "
              f"Best: {best_test_acc:.1f}% | {epoch_time:.1f}s")

    # Save final model
    torch.save(model.state_dict(), "checkpoints/dgcnn_final.pt")

    # Save training log
    with open("training_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Best Test Accuracy : {best_test_acc:.2f}%")
    print(f"  Total Time         : {elapsed / 60:.1f} minutes")
    print(f"  Model saved to     : checkpoints/dgcnn_best.pt")
    print(f"  Training log       : training_log.csv")
    print("=" * 60)

    # Save model config alongside weights (for loading on Jetson)
    config = {"num_classes": 10, "k": args.k, "dropout": 0.5, "num_points": args.num_points}
    torch.save(config, "checkpoints/dgcnn_config.pt")
    print(f"  Model config saved : checkpoints/dgcnn_config.pt")


if __name__ == "__main__":
    args = parse_args()
    train(args)
