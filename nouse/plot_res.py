"""
Visualise HGNAS search results.

Generates:
  1. Accuracy vs Inference Latency  (Pareto-front style)
  2. Accuracy vs Peak Memory        (Pareto-front style)
  3. Score distribution over search iterations
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless backend for server / Jetson
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import config as C


def pareto_front(points):
    """Return indices of Pareto-optimal points (maximise both axes)."""
    pts   = np.array(points)
    n     = len(pts)
    front = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if j == i: continue
            if pts[j, 0] >= pts[i, 0] and pts[j, 1] >= pts[i, 1] and \
               (pts[j, 0] > pts[i, 0] or pts[j, 1] > pts[i, 1]):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def plot_pareto_fronts(csv_file: str = C.SEARCH_RESULTS_CSV,
                       out_file: str = C.PARETO_PLOT_FILE):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"[Plot] Error: '{csv_file}' not found. Run main.py first.")
        return

    df = df[df["Score"] > 0].reset_index(drop=True)
    if df.empty:
        print("[Plot] No valid architectures found to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"HGNAS Architecture Search – Jetson Nano  "
        f"(Lat≤{C.LATENCY_CONSTRAINT_MS}ms, Mem≤{C.MEMORY_CONSTRAINT_MB}MB)",
        fontsize=14, fontweight="bold",
    )

    # ── Plot 1: Accuracy vs Latency ───────────────────────────────────────────
    ax = axes[0]
    sc = ax.scatter(df["Latency_ms"], df["Accuracy"],
                    c=df["Score"], cmap="viridis", alpha=0.6,
                    edgecolors="w", s=60)
    # Pareto front
    pts   = list(zip(-df["Latency_ms"], df["Accuracy"]))   # min latency, max acc
    front = pareto_front(pts)
    pdf   = df.iloc[front].sort_values("Latency_ms")
    ax.plot(pdf["Latency_ms"], pdf["Accuracy"], "r--", lw=1.5, label="Pareto front")
    ax.axvline(C.LATENCY_CONSTRAINT_MS, color="tomato", ls=":", lw=1, label=f"Constraint ({C.LATENCY_CONSTRAINT_MS}ms)")
    fig.colorbar(sc, ax=ax, label="Score")
    ax.set_xlabel("Inference Latency (ms) – lower is better")
    ax.set_ylabel("Accuracy (%)  – higher is better")
    ax.set_title("Accuracy vs Latency")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.5)

    # ── Plot 2: Accuracy vs Memory ────────────────────────────────────────────
    ax = axes[1]
    sc = ax.scatter(df["Memory_MB"], df["Accuracy"],
                    c=df["Score"], cmap="plasma", alpha=0.6,
                    edgecolors="w", s=60)
    pts   = list(zip(-df["Memory_MB"], df["Accuracy"]))
    front = pareto_front(pts)
    pdf   = df.iloc[front].sort_values("Memory_MB")
    ax.plot(pdf["Memory_MB"], pdf["Accuracy"], "r--", lw=1.5, label="Pareto front")
    ax.axvline(C.MEMORY_CONSTRAINT_MB, color="tomato", ls=":", lw=1, label=f"Constraint ({C.MEMORY_CONSTRAINT_MB}MB)")
    fig.colorbar(sc, ax=ax, label="Score")
    ax.set_xlabel("Peak Memory (MB) – lower is better")
    ax.set_ylabel("Accuracy (%)  – higher is better")
    ax.set_title("Accuracy vs Peak Memory")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.5)

    # ── Plot 3: Score over iterations ─────────────────────────────────────────
    ax = axes[2]
    iter_best = df.groupby("Iteration")["Score"].max().reset_index()
    ax.plot(iter_best["Iteration"], iter_best["Score"], color="steelblue", lw=2)
    ax.fill_between(iter_best["Iteration"], iter_best["Score"],
                    alpha=0.15, color="steelblue")
    ax.set_xlabel("Search Iteration")
    ax.set_ylabel("Best Score")
    ax.set_title("Search Convergence")
    ax.grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → '{out_file}'")


if __name__ == "__main__":
    plot_pareto_fronts()
