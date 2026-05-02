"""
HGNAS – Hardware-Aware GNN Architecture Search
Main entry point.

Pipeline:
  1. Load ModelNet10 (must be pre-downloaded with get_data.py)
  2. Train HW Predictor on simulated Jetson Nano profiling data
  3. (Optional) Pre-train supernet with random path sampling
  4. Stage 1: Function search → find best aggregator/message/dim/sample functions
  5. Stage 2: Operation search → find best per-position operations under HW constraints
  6. Evaluate the best found architecture on the test set
  7. Plot Pareto-front results
"""

import argparse
import torch

import config as C
from data_loader  import get_loaders
from supernet     import GNNSuperNet
from design_space import DesignSpace
from hw_predictor import HWPredictor
from trainer      import train_supernet, evaluate_architecture
from search       import function_search, operation_search
from plot_res     import plot_pareto_fronts


def parse_args():
    p = argparse.ArgumentParser(description="HGNAS for Jetson Nano")
    p.add_argument("--skip-train",      action="store_true",
                   help="Skip supernet pre-training (load from checkpoint).")
    p.add_argument("--skip-predictor",  action="store_true",
                   help="Skip predictor training (load from checkpoint).")
    p.add_argument("--supernet-epochs", type=int, default=C.SUPERNET_EPOCHS)
    p.add_argument("--predictor-samples", type=int, default=C.PREDICTOR_SAMPLES)
    p.add_argument("--predictor-epochs",  type=int, default=C.PREDICTOR_EPOCHS)
    p.add_argument("--ea-iter-s1",      type=int, default=C.EA_MAX_ITER_STAGE1)
    p.add_argument("--ea-iter-s2",      type=int, default=C.EA_MAX_ITER_STAGE2)
    p.add_argument("--lat-constraint",  type=float, default=C.LATENCY_CONSTRAINT_MS)
    p.add_argument("--mem-constraint",  type=float, default=C.MEMORY_CONSTRAINT_MB)
    p.add_argument("--device",          type=str, default=C.TORCH_DEVICE)
    p.add_argument("--max-samples",     type=int, default=0,
                   help="Max training+val samples (0=full dataset). Use ~400 for CPU demos.")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    print("=" * 60)
    print("  HGNAS – Hardware-Aware GNN Architecture Search")
    print(f"  Target  : {C.DEVICE_NAME}")
    print(f"  Dataset : ModelNet10")
    print(f"  Compute : {device}")
    print(f"  Lat ≤ {args.lat_constraint}ms  |  Mem ≤ {args.mem_constraint}MB")
    print("=" * 60)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_loaders(max_samples=args.max_samples)

    design_space = DesignSpace(num_positions=C.NUM_POSITIONS)

    # ── 2. Hardware Performance Predictor ─────────────────────────────────────
    predictor = HWPredictor(device_name=C.DEVICE_NAME, torch_device=device)

    if args.skip_predictor:
        try:
            predictor.load(C.PREDICTOR_CKPT)
            print(f"[Main] Loaded predictor from '{C.PREDICTOR_CKPT}'")
        except FileNotFoundError:
            print("[Main] Checkpoint not found, training predictor ...")
            predictor.train_on_samples(n_samples=args.predictor_samples,
                                       epochs=args.predictor_epochs)
            predictor.save(C.PREDICTOR_CKPT)
    else:
        predictor.train_on_samples(n_samples=args.predictor_samples,
                                   epochs=args.predictor_epochs)
        predictor.save(C.PREDICTOR_CKPT)

    # ── 3. SuperNet Initialisation & Pre-training ─────────────────────────────
    supernet = GNNSuperNet(
        num_positions=C.NUM_POSITIONS,
        in_channels  =C.IN_CHANNELS,
        num_classes  =C.NUM_CLASSES,
        hidden_dim   =C.HIDDEN_DIM,
    ).to(device)

    if args.skip_train:
        try:
            supernet.load_state_dict(torch.load(C.SUPERNET_CKPT, map_location=device))
            print(f"[Main] Loaded supernet from '{C.SUPERNET_CKPT}'")
        except FileNotFoundError:
            print("[Main] Checkpoint not found, training supernet ...")
            supernet = train_supernet(
                supernet, train_loader, val_loader, design_space,
                epochs=args.supernet_epochs, device=device,
            )
            torch.save(supernet.state_dict(), C.SUPERNET_CKPT)
    else:
        supernet = train_supernet(
            supernet, train_loader, val_loader, design_space,
            epochs=args.supernet_epochs, device=device,
        )
        torch.save(supernet.state_dict(), C.SUPERNET_CKPT)

    print(f"\n[Main] Supernet saved → '{C.SUPERNET_CKPT}'")

    # ── 4. Stage 1 – Function Search ─────────────────────────────────────────
    upper_func, lower_func = function_search(
        supernet, val_loader, design_space,
        pop_size =C.EA_POP_SIZE,
        max_iter =args.ea_iter_s1,
        device   =device,
    )
    print("\n[Main] Optimal function set determined.")
    print(f"  Upper half sample: {upper_func.positions[0].sample_op}")
    print(f"  Upper half agg   : {upper_func.positions[0].agg_op}")

    # ── 5. Stage 2 – Operation Search ────────────────────────────────────────
    # Re-initialise and re-train supernet with fixed function set
    print("\n[Main] Re-initialising supernet with fixed function set ...")
    supernet2 = GNNSuperNet(
        num_positions=C.NUM_POSITIONS,
        in_channels  =C.IN_CHANNELS,
        num_classes  =C.NUM_CLASSES,
        hidden_dim   =C.HIDDEN_DIM,
    ).to(device)
    supernet2 = train_supernet(
        supernet2, train_loader, val_loader, design_space,
        epochs=args.supernet_epochs, device=device,
    )

    best_arch = operation_search(
        supernet2, val_loader, design_space, predictor,
        upper_func=upper_func, lower_func=lower_func,
        pop_size  =C.EA_POP_SIZE,
        max_iter  =args.ea_iter_s2,
        lat_c     =args.lat_constraint,
        mem_c     =args.mem_constraint,
        device    =device,
        csv_path  =C.SEARCH_RESULTS_CSV,
    )

    # ── 6. Final Evaluation ───────────────────────────────────────────────────
    print("\n[Main] Evaluating best architecture on test set ...")
    test_acc = evaluate_architecture(supernet2, best_arch, test_loader, device)
    lat      = predictor.predict_latency(best_arch)
    mem      = predictor.predict_peak_memory(best_arch)

    print("\n" + "=" * 60)
    print("  HGNAS Search Complete!")
    print(f"  Test Accuracy   : {test_acc:.2f}%")
    print(f"  Predicted Lat.  : {lat:.2f} ms")
    print(f"  Predicted Mem.  : {mem:.2f} MB")
    print("  Best Architecture Encoding:")
    for i, pos in enumerate(best_arch.positions):
        print(f"    Pos {i:2d}: sample={pos.sample_op:6s}  "
              f"agg=({pos.agg_op[0]:4s},{pos.agg_op[1]:8s})  "
              f"dim={pos.combine_dim:3d}  connect={pos.connect_op}")
    print("=" * 60)

    # ── 7. Plot ───────────────────────────────────────────────────────────────
    print("\n[Main] Generating Pareto-front plots ...")
    plot_pareto_fronts(C.SEARCH_RESULTS_CSV, C.PARETO_PLOT_FILE)
    print(f"[Main] Plot saved → '{C.PARETO_PLOT_FILE}'")


if __name__ == "__main__":
    main()
