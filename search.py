"""
Multi-stage Hierarchical Search Strategy (Section 3.4 & Algorithm 1).

Stage 1 – Function Search:
  Evolutionary search over the Function Space (aggregator type,
  message type, combine dim, sample type) to maximise supernet
  validation accuracy.  Upper and lower halves of the supernet
  share their own function settings.

Stage 2 – Operation Search:
  With the optimal function set fixed, the supernet is pre-trained
  once.  An EA then searches the Operation Space for architectures
  that jointly maximise accuracy and meet Jetson Nano hardware
  constraints (latency < C_lat, peak memory < C_mem).

Objective (Eq. 4 in paper):
  F_obj = 0,                       if E >= C
        = α * acc_val - β * E,     if E < C
  where E = normalised efficiency penalty.
"""

import csv
import random
from typing import Tuple, List, Optional

import config as C
from design_space import Architecture, DesignSpace, PositionEncoding
from hw_predictor  import HWPredictor
from trainer       import evaluate_architecture


# ── Objective function ────────────────────────────────────────────────────────

def compute_objective(
    acc     : float,
    latency : float,
    memory  : float,
    lat_c   : float = C.LATENCY_CONSTRAINT_MS,
    mem_c   : float = C.MEMORY_CONSTRAINT_MB,
    alpha   : float = C.ALPHA,
    beta    : float = C.BETA,
) -> float:
    """Eq. 4: return 0 if constraints violated, else α*acc - β*efficiency_penalty."""
    if latency > lat_c or memory > mem_c:
        return 0.0
    eff = (latency / lat_c) + (memory / mem_c)
    return alpha * acc - beta * eff


# ── Stage 1: Function Search ──────────────────────────────────────────────────

def function_search(
    supernet,
    val_loader,
    design_space : DesignSpace,
    pop_size     : int = C.EA_POP_SIZE,
    max_iter     : int = C.EA_MAX_ITER_STAGE1,
    device       : str = C.TORCH_DEVICE,
    verbose      : bool = True,
) -> Tuple[Architecture, Architecture]:
    """
    Stage 1: Find optimal function settings for upper and lower halves.

    Returns (upper_arch, lower_arch) – architectures whose function
    choices will be fixed for Stage 2 supernet initialisation.
    """
    print("\n[Stage 1] Function Search ...")
    half = design_space.half

    population = [design_space.random_architecture() for _ in range(pop_size)]

    best_score = -float("inf")
    best_arch  = population[0]

    for it in range(1, max_iter + 1):
        scored = []
        for arch in population:
            acc = evaluate_architecture(supernet, arch, val_loader, device)
            scored.append((acc, arch))

        scored.sort(key=lambda x: x[0], reverse=True)
        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_arch  = scored[0][1]

        if verbose and it % 10 == 0:
            print(f"  Iter {it:3d}/{max_iter} | Best Acc: {best_score:.2f}%")

        # EA: keep top half, mutate/crossover for the rest
        survivors = [a for _, a in scored[:pop_size // 2]]
        new_pop   = survivors.copy()
        while len(new_pop) < pop_size:
            if random.random() < 0.5:
                parent   = random.choice(survivors)
                new_pop.append(design_space.mutate(parent))
            else:
                p1, p2 = random.sample(survivors, 2)
                new_pop.append(design_space.crossover(p1, p2))
        population = new_pop

    # Upper half: positions 0 … half-1
    upper_arch = Architecture(best_arch.positions[:half])
    # Lower half: positions half … N-1
    lower_arch = Architecture(best_arch.positions[half:])

    print(f"[Stage 1] Done. Best Val Acc: {best_score:.2f}%")
    return upper_arch, lower_arch


# ── Stage 2: Operation Search ─────────────────────────────────────────────────

def _apply_function_constraint(arch: Architecture,
                               upper_func: Optional[Architecture],
                               lower_func: Optional[Architecture],
                               half: int) -> Architecture:
    """
    Fix the sample_op and agg_op of each position to match the function
    set found in Stage 1 (paper Section 3.4: Stage 2 fixes functions).
    Only combine_dim and connect_op remain free for Stage 2 to search.
    """
    if upper_func is None and lower_func is None:
        return arch

    new_positions = []
    for i, pos in enumerate(arch.positions):
        ref = (upper_func if i < half else lower_func)
        if ref is None:
            new_positions.append(pos)
            continue
        ref_pos = ref.positions[i % len(ref.positions)]
        new_positions.append(PositionEncoding(
            sample_idx  = ref_pos.sample_idx,   # fixed from Stage 1
            agg_idx     = ref_pos.agg_idx,       # fixed from Stage 1
            combine_idx = pos.combine_idx,        # free to search
            connect_idx = pos.connect_idx,        # free to search
        ))
    return Architecture(new_positions)


def operation_search(
    supernet,
    val_loader,
    design_space : DesignSpace,
    predictor    : HWPredictor,
    upper_func   : Optional[Architecture] = None,
    lower_func   : Optional[Architecture] = None,
    pop_size     : int   = C.EA_POP_SIZE,
    max_iter     : int   = C.EA_MAX_ITER_STAGE2,
    lat_c        : float = C.LATENCY_CONSTRAINT_MS,
    mem_c        : float = C.MEMORY_CONSTRAINT_MB,
    device       : str   = C.TORCH_DEVICE,
    csv_path     : str   = C.SEARCH_RESULTS_CSV,
    verbose      : bool  = True,
) -> Architecture:
    """
    Stage 2: Multi-objective operation search with hardware constraints.

    upper_func / lower_func fix the sample and aggregate functions from
    Stage 1. Only combine_dim and connect_op are searched here (paper §3.4).
    """
    print("\n[Stage 2] Operation Search ...")
    half = design_space.half

    # Seed population respecting fixed function constraints from Stage 1
    population = [
        _apply_function_constraint(
            design_space.random_architecture(), upper_func, lower_func, half
        )
        for _ in range(pop_size)
    ]

    best_score = -float("inf")
    best_arch  = population[0]
    results    : List[dict] = []

    for it in range(1, max_iter + 1):
        scored = []
        for arch in population:
            lat = predictor.predict_latency(arch)
            mem = predictor.predict_peak_memory(arch)

            # Only evaluate accuracy if constraints are satisfied
            if lat <= lat_c and mem <= mem_c:
                acc = evaluate_architecture(supernet, arch, val_loader, device)
            else:
                acc = 0.0

            score = compute_objective(acc, lat, mem, lat_c, mem_c)
            scored.append((score, arch, acc, lat, mem))

            if score > 0:
                results.append({
                    "Iteration": it,
                    "Accuracy":  acc,
                    "Latency_ms": lat,
                    "Memory_MB": mem,
                    "Score":     score,
                })

        scored.sort(key=lambda x: x[0], reverse=True)
        top_score, top_arch, top_acc, top_lat, top_mem = scored[0]

        if top_score > best_score:
            best_score = top_score
            best_arch  = top_arch

        if verbose and it % 10 == 0:
            print(f"  Iter {it:3d}/{max_iter} | Best Score: {best_score:.4f} | "
                  f"Acc: {top_acc:.1f}% | Lat: {top_lat:.1f}ms | Mem: {top_mem:.1f}MB")

        # EA selection + mutation (re-apply function constraints after each op)
        survivors = [a for _, a, *_ in scored[:pop_size // 2]]
        new_pop   = survivors.copy()
        while len(new_pop) < pop_size:
            if random.random() < 0.5:
                child = design_space.mutate(random.choice(survivors))
            else:
                p1, p2 = random.sample(survivors, 2)
                child = design_space.crossover(p1, p2)
            # Re-pin functions so mutation never breaks Stage 1 constraints
            child = _apply_function_constraint(child, upper_func, lower_func, half)
            new_pop.append(child)
        population = new_pop

    # Write CSV log
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"[Stage 2] Results logged to '{csv_path}'")

    print(f"[Stage 2] Done. Best Score: {best_score:.4f}")
    return best_arch


# ── Unified search entry point ────────────────────────────────────────────────

def run_hgnas_search(
    supernet,
    train_supernet_fn,
    val_loader,
    design_space : DesignSpace,
    predictor    : HWPredictor,
    device       : str = C.TORCH_DEVICE,
    verbose      : bool = True,
) -> Architecture:
    """
    Full HGNAS two-stage search (Algorithm 1).

    1. Stage 1 – function search on initial supernet
    2. Fix function set and re-initialise / pre-train supernet
    3. Stage 2 – operation search with hardware constraints

    Returns the best found Architecture.
    """
    # ── Stage 1 ──────────────────────────────────────────────────────────────
    upper_func, lower_func = function_search(
        supernet, val_loader, design_space, verbose=verbose, device=device
    )

    # ── Re-initialise and pre-train with fixed functions ──────────────────────
    print("\n[HGNAS] Re-initialising supernet with optimal function set ...")
    from supernet import GNNSuperNet
    supernet = GNNSuperNet().to(device)
    supernet = train_supernet_fn(supernet)

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    best_arch = operation_search(
        supernet, val_loader, design_space, predictor,
        upper_func=upper_func, lower_func=lower_func,
        verbose=verbose, device=device,
    )

    return best_arch
