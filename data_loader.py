"""
ModelNet10 data loading for HGNAS.

Loads .off files DIRECTLY — completely bypasses torch_geometric's
InMemoryDataset / Processing... pipeline which hangs on Jetson Nano.
No disk caching, no "Processing..." message, starts instantly.
"""

import os
import glob

import torch
import torch.utils.data
from torch_geometric.data import Data

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

import config as C

# ModelNet10 class name → integer label (alphabetical order)
CLASSES = [
    "bathtub", "bed", "chair", "desk", "dresser",
    "monitor", "night_stand", "sofa", "table", "toilet",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# ── OFF file parser ────────────────────────────────────────────────────────────

def _parse_off(path: str) -> torch.Tensor:
    """
    Parse a .off mesh file and return vertex positions as (N, 3) float tensor.
    Handles both 'OFF' on its own line and 'OFF<num> <num> <num>' on one line.
    """
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    # First line is always 'OFF' or 'OFF<n> <n> <n>'
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

    return torch.tensor(verts, dtype=torch.float)   # (N, 3)


# ── Point sampling & normalisation ────────────────────────────────────────────

def _sample_and_normalize(pos: torch.Tensor, num_points: int, seed: int = 0) -> torch.Tensor:
    """
    Randomly sample `num_points` from vertex positions and normalise to unit sphere.
    Uses a deterministic seed based on the input for reproducibility.
    """
    N = pos.size(0)

    # Deterministic sampling: seed based on vertex count and a sample-specific seed
    # so the same object always produces the same point cloud.
    g = torch.Generator()
    g.manual_seed(seed)

    if N >= num_points:
        idx = torch.randperm(N, generator=g)[:num_points]
    else:
        idx = torch.randint(0, N, (num_points,), generator=g)
    pos = pos[idx]

    # NormalizeScale: centre then scale to [-1, 1]
    pos = pos - pos.mean(dim=0, keepdim=True)
    scale = pos.abs().max()
    if scale > 0:
        pos = pos / scale

    return pos   # (num_points, 3)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ModelNet10Raw(torch.utils.data.Dataset):
    """
    Reads ModelNet10 .off files directly.
    No torch_geometric InMemoryDataset, no Processing... step.
    """

    def __init__(self, root: str, train: bool = True, num_points: int = 32):
        super().__init__()
        self.num_points = num_points
        split = "train" if train else "test"

        self.samples = []   # list of (path, label)
        for cls_name in CLASSES:
            label = CLASS_TO_IDX[cls_name]
            pattern = os.path.join(root, "raw", cls_name, split, "*.off")
            files = sorted(glob.glob(pattern))
            for f in files:
                self.samples.append((f, label))

        if not self.samples:
            raise RuntimeError(
                f"No .off files found under {root}/raw/. "
                "Make sure get_data.py ran successfully."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pos = _parse_off(path)
        # Use idx as seed for deterministic but unique sampling per object
        pos = _sample_and_normalize(pos, self.num_points, seed=idx * 31 + self.num_points)
        return Data(
            pos=pos,
            y=torch.tensor([label], dtype=torch.long),
        )


# ── Collate helper (needed by older PyG DataLoader) ──────────────────────────

def _collate(batch):
    try:
        from torch_geometric.data import Batch
        return Batch.from_data_list(batch)
    except Exception:
        return batch


# ── Public API ────────────────────────────────────────────────────────────────

def get_loaders(
    root        : str   = C.DATA_ROOT,
    num_points  : int   = C.NUM_POINTS,
    batch_size  : int   = C.SUPERNET_BATCH_SIZE,
    val_split   : float = 0.15,
    max_samples : int   = 0,
):
    """
    Returns (train_loader, val_loader, test_loader).
    Loads .off files directly — no Processing... hang.
    """
    print(f"[Data] Loading ModelNet10 from {root} (direct .off reader) ...")

    train_full = ModelNet10Raw(root=root, train=True,  num_points=num_points)
    test_set   = ModelNet10Raw(root=root, train=False, num_points=num_points)

    # Optional subsample
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

    # Use torch_geometric DataLoader so it returns Batch objects
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[Data] Train: {n_train} | Val: {n_val} | Test: {len(test_set)}")
    return train_loader, val_loader, test_loader
