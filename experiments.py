"""
Experiment configurations for DGCNN ablation studies.

Each config trains a model variant and benchmarks it.
"""

# ── Ablation Experiments ──────────────────────────────────────────────────────

EXPERIMENTS = {
    # ── Baseline ──────────────────────────────────────────────────────────────
    "baseline": {
        "desc": "Standard DGCNN (Full model, k=20, dynamic graph, max aggr)",
        "model_args": {"k": 20, "channels": [64, 64, 128, 256], "emb_dim": 1024,
                       "aggr": "max", "static_graph": False, "use_attention": False},
        "train_args": {"epochs": 200, "batch_size": 32, "lr": 1e-3},
    },

    # ── K sweep ───────────────────────────────────────────────────────────────
    "k5":  {"desc": "k=5 neighbors",
            "model_args": {"k": 5, "channels": [64, 64, 128, 256], "emb_dim": 1024, "aggr": "max"},
            "train_args": {"epochs": 200, "batch_size": 32, "lr": 1e-3}},
    "k10": {"desc": "k=10 neighbors",
            "model_args": {"k": 10, "channels": [64, 64, 128, 256], "emb_dim": 1024, "aggr": "max"},
            "train_args": {"epochs": 200, "batch_size": 32, "lr": 1e-3}},
    "k15": {"desc": "k=15 neighbors",
            "model_args": {"k": 15, "channels": [64, 64, 128, 256], "emb_dim": 1024, "aggr": "max"},
            "train_args": {"epochs": 200, "batch_size": 32, "lr": 1e-3}},

    # ── Progressive K ─────────────────────────────────────────────────────────
    "prog_k": {"desc": "Progressive K: [20, 15, 10, 5]",
               "model_args": {"k": [20, 15, 10, 5], "channels": [64, 64, 128, 256],
                               "emb_dim": 1024, "aggr": "max"},
               "train_args": {"epochs": 200, "batch_size": 32, "lr": 1e-3}},

    # ── Model compression ────────────────────────────────────────────────────
    "lite": {"desc": "DGCNN-Lite: [32,32,64,128], emb=512",
             "model_args": {"k": 20, "preset": "lite", "aggr": "max"},
             "train_args": {"epochs": 200, "batch_size": 32, "lr": 1e-3}},
    "tiny": {"desc": "DGCNN-Tiny: [16,16,32,64], emb=256",
             "model_args": {"k": 20, "preset": "tiny", "aggr": "max"},
             "train_args": {"epochs": 200, "batch_size": 64, "lr": 1e-3}},

    # ── Aggregation comparison ────────────────────────────────────────────────
    "aggr_mean": {"desc": "Mean aggregation",
                  "model_args": {"k": 20, "channels": [64, 64, 128, 256], "emb_dim": 1024,
                                  "aggr": "mean"},
                  "train_args": {"epochs": 200, "batch_size": 32, "lr": 1e-3}},
    "aggr_sum":  {"desc": "Sum aggregation",
                  "model_args": {"k": 20, "channels": [64, 64, 128, 256], "emb_dim": 1024,
                                  "aggr": "sum"},
                  "train_args": {"epochs": 200, "batch_size": 32, "lr": 1e-3}},

    # ── Static vs Dynamic graph ───────────────────────────────────────────────
    "static": {"desc": "Static graph (KNN computed once on raw xyz)",
               "model_args": {"k": 20, "channels": [64, 64, 128, 256], "emb_dim": 1024,
                                "aggr": "max", "static_graph": True},
               "train_args": {"epochs": 200, "batch_size": 32, "lr": 1e-3}},

    # ── Attention ─────────────────────────────────────────────────────────────
    "attention": {"desc": "Edge attention gating",
                  "model_args": {"k": 20, "channels": [64, 64, 128, 256], "emb_dim": 1024,
                                  "aggr": "max", "use_attention": True},
                  "train_args": {"epochs": 200, "batch_size": 32, "lr": 1e-3}},

    # ── Layer depth ───────────────────────────────────────────────────────────
    "depth2": {"desc": "2 EdgeConv layers",
               "model_args": {"k": 20, "channels": [64, 128], "emb_dim": 512, "aggr": "max"},
               "train_args": {"epochs": 200, "batch_size": 32, "lr": 1e-3}},
    "depth3": {"desc": "3 EdgeConv layers",
               "model_args": {"k": 20, "channels": [64, 64, 128], "emb_dim": 768, "aggr": "max"},
               "train_args": {"epochs": 200, "batch_size": 32, "lr": 1e-3}},
    "depth5": {"desc": "5 EdgeConv layers",
               "model_args": {"k": 20, "channels": [64, 64, 64, 128, 256], "emb_dim": 1024,
                                "aggr": "max"},
               "train_args": {"epochs": 200, "batch_size": 32, "lr": 1e-3}},
}

# Subsets for quick runs
QUICK_EXPERIMENTS = ["baseline", "lite", "tiny", "static", "aggr_mean"]
K_SWEEP = ["k5", "k10", "k15", "baseline"]
COMPRESSION = ["baseline", "lite", "tiny"]
AGGREGATION = ["baseline", "aggr_mean", "aggr_sum"]
DEPTH_SWEEP = ["depth2", "depth3", "baseline", "depth5"]
