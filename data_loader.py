"""
ModelNet10 data loading for HGNAS.

Uses the dataset already downloaded by get_data.py.
Returns PyG DataLoader objects for training and validation.

NOTE: We intentionally do NOT use pre_transform (which triggers the slow
"Processing..." step that can hang on Jetson Nano). NormalizeScale is applied
at runtime instead — slightly slower per batch but avoids the multi-minute
disk processing stage entirely.
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
    """Remove pre-processed cache so torch_geometric does not try to load it."""
    processed_dir = os.path.join(root, "processed")
    if os.path.isdir(processed_dir):
        print(f"[Data] Removing cached processed dir (avoids Processing... hang): {processed_dir}")
        shutil.rmtree(processed_dir)


def _load_modelnet(root, train, transform, retry=True):
    """
    Load ModelNet with NO pre_transform so no "Processing..." step occurs.
    NormalizeScale + SamplePoints are both applied at runtime per batch.
    Auto-clears stale processed cache on version mismatch.
    """
    try:
        return ModelNet(
            root=root, name="10", train=train,
            transform=transform,
            pre_transform=None,   # ← no disk processing step
        )
    except ValueError as e:
        if "too many values to unpack" in str(e) and retry:
            print(f"[Data] Cached files incompatible ({e}). Clearing and retrying ...")
            _clear_processed(root)
            return _load_modelnet(root, train, transform, retry=False)
        raise


def get_loaders(
    root        : str   = C.DATA_ROOT,
    num_points  : int   = C.NUM_POINTS,
    batch_size  : int   = C.SUPERNET_BATCH_SIZE,
    val_split   : float = 0.1,
    max_samples : int   = 0,
):
    """
    Returns (train_loader, val_loader, test_loader).

    NormalizeScale and SamplePoints are applied as runtime transforms
    (no slow pre-processing to disk).
    """
    # Both transforms applied at runtime — no pre_transform / Processing step
    transform = T.Compose([
        T.NormalizeScale(),
        T.SamplePoints(num_points),
    ])

    # Clear any existing processed cache that was built with a pre_transform —
    # it would be incompatible now that we pass pre_transform=None.
    _clear_processed(root)

    print(f"[Data] Loading ModelNet10 from {root} ...")
    train_full = _load_modelnet(root, train=True,  transform=transform)
    test_set   = _load_modelnet(root, train=False, transform=transform)

    # Optional subsample for quick runs
    total = len(train_full)
    if max_samples > 0 and max_samples < total:
        indices = torch.randperm(total, generator=torch.Generator().manual_seed(42))
        train_full = torch.utils.data.Subset(train_full, indices[:max_samples].tolist())
        total = max_samples

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
