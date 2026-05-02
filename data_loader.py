"""
ModelNet10 data loading for HGNAS.

Uses the dataset already downloaded by get_data.py.
Returns PyG DataLoader objects for training and validation.
"""

import os
import shutil

import torch
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

import config as C


def _clear_processed(root: str):
    """Remove pre-processed cache so torch_geometric re-builds it."""
    processed_dir = os.path.join(root, "processed")
    if os.path.isdir(processed_dir):
        print(f"[Data] Removing stale processed cache at {processed_dir} ...")
        shutil.rmtree(processed_dir)


def _load_modelnet(root, train, transform, pre_transform, retry=True):
    """
    Load ModelNet, automatically clearing the processed cache if the saved
    format is incompatible with the installed torch_geometric version
    (ValueError: too many values to unpack).
    """
    try:
        return ModelNet(
            root=root, name="10", train=train,
            transform=transform, pre_transform=pre_transform,
        )
    except ValueError as e:
        if "too many values to unpack" in str(e) and retry:
            print(f"[Data] Cached files incompatible with this torch_geometric "
                  f"version ({e}). Clearing cache and re-processing ...")
            _clear_processed(root)
            return _load_modelnet(root, train, transform, pre_transform, retry=False)
        raise


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
    train_full = _load_modelnet(root, train=True,
                                transform=transform, pre_transform=pre_transform)
    test_set   = _load_modelnet(root, train=False,
                                transform=transform, pre_transform=pre_transform)

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
