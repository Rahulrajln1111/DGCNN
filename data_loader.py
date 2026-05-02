"""
ModelNet10 data loading for HGNAS.

Uses the dataset already downloaded by get_data.py.
Returns PyG DataLoader objects for training and validation.
"""

import torch
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

import config as C


def get_loaders(
    root        : str   = C.DATA_ROOT,
    num_points  : int   = C.NUM_POINTS,
    batch_size  : int   = C.SUPERNET_BATCH_SIZE,
    val_split   : float = 0.1,
    max_samples : int   = 0,   # 0 = use full dataset; >0 = subsample for speed
):
    """
    Returns (train_loader, val_loader, test_loader).

    The ModelNet10 dataset is expected to be pre-downloaded at `root`.
    run get_data.py first if the data folder is missing.

    Args:
        max_samples : Cap total training+val samples (useful for CPU demos).
                      Set to 0 for full dataset (recommended for Jetson Nano).
    """
    pre_transform = T.NormalizeScale()
    transform     = T.SamplePoints(num_points)

    print(f"[Data] Loading ModelNet10 from {root} ...")
    train_full = ModelNet(
        root=root, name="10", train=True,
        transform=transform, pre_transform=pre_transform,
    )
    test_set = ModelNet(
        root=root, name="10", train=False,
        transform=transform, pre_transform=pre_transform,
    )

    # Optional subsample for quick CPU demo
    total = len(train_full)
    if max_samples > 0 and max_samples < total:
        indices = torch.randperm(total, generator=torch.Generator().manual_seed(42))
        train_full = torch.utils.data.Subset(train_full, indices[:max_samples].tolist())
        total = max_samples

    # Split training into train / val
    n_val   = max(1, int(total * val_split))
    n_train = total - n_val
    train_set, val_set = torch.utils.data.random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[Data] Train: {n_train} | Val: {n_val} | Test: {len(test_set)}")
    return train_loader, val_loader, test_loader
