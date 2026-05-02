import pandas as pd
import matplotlib.pyplot as plt

def plot_pareto_fronts(csv_file):
    # Load the logged data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}. Run the search script first.")
        return

    # Filter out entries with 0 score (violating constraints)
    df = df[df["Score"] > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Accuracy vs Latency ---
    ax1.scatter(df["Latency_ms"], df["Accuracy"], alpha=0.6, edgecolors='w', s=50, c='blue')
    ax1.set_title("Search Space: Accuracy vs. Inference Latency", fontsize=14)
    ax1.set_xlabel("Latency (ms) - Lower is better", fontsize=12)
    ax1.set_ylabel("Accuracy (%) - Higher is better", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # --- Plot 2: Accuracy vs Memory ---
    ax2.scatter(df["Memory_MB"], df["Accuracy"], alpha=0.6, edgecolors='w', s=50, c='green')
    ax2.set_title("Search Space: Accuracy vs. Peak Memory", fontsize=14)
    ax2.set_xlabel("Peak Memory (MB) - Lower is better", fontsize=12)
    ax2.set_ylabel("Accuracy (%) - Higher is better", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("hgnas_pareto_front.png", dpi=300)
    print("[INFO] Graph successfully saved as 'hgnas_pareto_front.png'")

if __name__ == "__main__":
    plot_pareto_fronts("search_results.csv")
