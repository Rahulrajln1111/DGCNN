"""
One-shot SuperNet training (Section 3.4, Stage 2 pre-training).

In each mini-batch a random architecture is sampled and only its
sub-path through the supernet is trained.  This decouples supernet
training from architecture search (single-path one-shot NAS).
"""

import torch
import torch.nn.functional as F

import config as C
from supernet import GNNSuperNet
from design_space import DesignSpace


def train_supernet(
    supernet     : GNNSuperNet,
    train_loader,
    val_loader,
    design_space : DesignSpace,
    epochs       : int   = C.SUPERNET_EPOCHS,
    lr           : float = C.SUPERNET_LR,
    device       : str   = C.TORCH_DEVICE,
    verbose      : bool  = True,
) -> GNNSuperNet:
    """
    Train the supernet with one-shot path sampling.

    Args:
        supernet     : the GNNSuperNet to train
        train_loader : PyG DataLoader for training data
        val_loader   : PyG DataLoader for validation
        design_space : used to sample random architectures each step
        epochs       : number of training epochs
        lr           : learning rate
        device       : torch device string
        verbose      : print progress

    Returns:
        Trained supernet (in-place modification + return for convenience).
    """
    supernet = supernet.to(device)
    optimizer = torch.optim.Adam(
        supernet.parameters(), lr=lr, weight_decay=C.SUPERNET_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"[Trainer] Pre-training supernet for {epochs} epochs on {device} ...")
    for epoch in range(1, epochs + 1):
        supernet.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            batch = batch.to(device)
            # Sample a random sub-architecture for this step
            arch  = design_space.random_architecture()

            optimizer.zero_grad()
            logits = supernet(batch, arch)
            loss   = F.cross_entropy(logits, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(supernet.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            pred        = logits.argmax(dim=1)
            correct    += (pred == batch.y.view(-1)).sum().item()
            total      += batch.num_graphs

        scheduler.step()
        train_acc = 100.0 * correct / total

        if verbose:
            val_acc = evaluate_supernet(supernet, val_loader, design_space, device)
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {total_loss / total:.4f} | "
                  f"Train Acc: {train_acc:.1f}% | "
                  f"Val Acc: {val_acc:.1f}%")

    return supernet


@torch.no_grad()
def evaluate_supernet(
    supernet     : GNNSuperNet,
    loader,
    design_space : DesignSpace,
    device       : str = C.TORCH_DEVICE,
    n_archs      : int = 1,
) -> float:
    """
    Evaluate a supernet using a single consistent random architecture.

    Using ONE architecture for the entire evaluation pass gives a clean
    accuracy signal.  Averaging over multiple random architectures (as
    was done before) destroys the signal because different sub-networks
    produce conflicting predictions.
    """
    supernet.eval()
    total_correct, total_samples = 0, 0

    # Use ONE architecture consistently across all batches
    arch = design_space.random_architecture()

    for batch in loader:
        batch  = batch.to(device)
        logits = supernet(batch, arch)
        pred   = logits.argmax(dim=1)
        total_correct  += (pred == batch.y.view(-1)).sum().item()
        total_samples  += batch.num_graphs

    return 100.0 * total_correct / total_samples


@torch.no_grad()
def evaluate_architecture(
    supernet : GNNSuperNet,
    arch,
    loader,
    device   : str = C.TORCH_DEVICE,
) -> float:
    """Evaluate a specific architecture on a data loader. Returns accuracy %."""
    supernet.eval()
    correct, total = 0, 0
    for batch in loader:
        batch  = batch.to(device)
        logits = supernet(batch, arch)
        pred   = logits.argmax(dim=1)
        correct += (pred == batch.y.view(-1)).sum().item()
        total   += batch.num_graphs
    return 100.0 * correct / total
