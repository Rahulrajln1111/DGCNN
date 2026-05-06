"""
Dataset loading for DGCNN – reads ModelNet10 .off files directly.

Works on both A100 (training) and Jetson Nano (inference) without
requiring PyG's InMemoryDataset processing step.
"""

import os
import glob
import random

import torch
import torch.utils.data
from torch_geometric.data import Data

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader


# ModelNet10 classes (alphabetical order)
CLASSES = [
    "bathtub", "bed", "chair", "desk", "dresser",
    "monitor", "night_stand", "sofa", "table", "toilet",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = 10


# ── OFF file parser ──────────────────────────────────────────────────────────

def parse_off(path):
    """Parse a .off mesh file → vertex positions as (N, 3) float tensor."""
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    header = lines[0]
    if header.startswith("OFF") and len(header) > 3:
        counts = header[3:].split()
        start = 1
    else:
        counts = lines[1].split()
        start = 2

    num_verts = int(counts[0])
    verts = []
    for i in range(start, start + num_verts):
        x, y, z = lines[i].split()[:3]
        verts.append([float(x), float(y), float(z)])

    return torch.tensor(verts, dtype=torch.float)


# ── Point cloud preprocessing ────────────────────────────────────────────────

def sample_points(pos, num_points, seed=None):
    """Sample num_points from vertices. Deterministic if seed is provided."""
    N = pos.size(0)
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    else:
        g = None

    if N >= num_points:
        idx = torch.randperm(N, generator=g)[:num_points]
    else:
        idx = torch.randint(0, N, (num_points,), generator=g)

    return pos[idx]


def normalize_points(pos):
    """Center and normalize point cloud to unit sphere."""
    pos = pos - pos.mean(dim=0, keepdim=True)
    scale = pos.abs().max()
    if scale > 0:
        pos = pos / scale
    return pos


# ── Data augmentation (training only) ────────────────────────────────────────

def augment_point_cloud(pos):
    """Apply random augmentation: rotation, jitter, scale."""
    # Random rotation around Y axis
    theta = random.uniform(0, 2 * 3.14159265)
    cos_t, sin_t = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    R = torch.tensor([
        [cos_t, 0, sin_t],
        [0, 1, 0],
        [-sin_t, 0, cos_t],
    ], dtype=pos.dtype)
    pos = pos @ R.T

    # Random jitter
    pos = pos + torch.randn_like(pos) * 0.01

    # Random scale
    scale = random.uniform(0.8, 1.2)
    pos = pos * scale

    return pos


# ── Dataset ──────────────────────────────────────────────────────────────────

class ModelNet10Dataset(torch.utils.data.Dataset):
    """
    ModelNet10 dataset reading .off files directly.

    Args:
        root: Path to ModelNet10 directory containing 'raw/' subdirectory
        split: 'train' or 'test'
        num_points: Points to sample per object
        augment: Apply data augmentation (for training)
    """

    def __init__(self, root, split="train", num_points=1024, augment=False):
        super().__init__()
        self.num_points = num_points
        self.augment = augment

        self.samples = []
        for cls_name in CLASSES:
            label = CLASS_TO_IDX[cls_name]
            pattern = os.path.join(root, "raw", cls_name, split, "*.off")
            files = sorted(glob.glob(pattern))
            for f in files:
                self.samples.append((f, label))

        if not self.samples:
            raise RuntimeError(
                f"No .off files found under {root}/raw/*/{{train,test}}/. "
                f"Run: python download_data.py"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pos = parse_off(path)
        pos = sample_points(pos, self.num_points, seed=idx)
        pos = normalize_points(pos)

        if self.augment:
            pos = augment_point_cloud(pos)

        return Data(
            pos=pos,
            y=torch.tensor([label], dtype=torch.long),
        )


# ── DataLoader factory ───────────────────────────────────────────────────────

def get_dataloaders(root="./data/ModelNet10", num_points=1024, batch_size=32,
                    augment_train=True, num_workers=4):
    """
    Returns (train_loader, test_loader).

    For training on A100:  augment_train=True,  num_workers=4, batch_size=32
    For inference on Jetson: only test_loader needed
    """
    train_set = ModelNet10Dataset(root, split="train", num_points=num_points,
                                  augment=augment_train)
    test_set = ModelNet10Dataset(root, split="test", num_points=num_points,
                                 augment=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    print(f"[Data] Train: {len(train_set)} | Test: {len(test_set)}")
    return train_loader, test_loader


def get_test_loader(root="./data/ModelNet10", num_points=1024, batch_size=8,
                    num_workers=0):
    """Get only test loader (for Jetson inference)."""
    test_set = ModelNet10Dataset(root, split="test", num_points=num_points,
                                 augment=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    print(f"[Data] Test samples: {len(test_set)}")
    return test_loader
