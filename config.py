import torch

# ─────────────────────────────────────────────
#  Target platform: Jetson Nano
# ─────────────────────────────────────────────
DEVICE_NAME = "jetson_nano"

# Hardware constraints for Jetson Nano
# 4 GB LPDDR4 RAM, 128-core Maxwell GPU
LATENCY_CONSTRAINT_MS  = 120.0   # ms  – real-time threshold for point cloud
MEMORY_CONSTRAINT_MB   = 600.0   # MB  – conservative limit for Jetson Nano

# ─────────────────────────────────────────────
#  Design space
# ─────────────────────────────────────────────
NUM_POSITIONS = 12          # 4 GNN layers × 3 ops (sample/agg/combine)

# Sample operations
SAMPLE_OPS = ["knn", "random"]

# Aggregate: (aggregator_type, message_type) pairs
AGGREGATE_TYPES = ["sum", "max", "mean", "min"]
MESSAGE_TYPES   = [
    "source",        # source node position
    "target",        # target node position
    "relative",      # u - v (relative position)
    "src_rel",       # source || relative
    "tgt_rel",       # target || relative
    "euclidean",     # Euclidean distance scalar
    "full",          # source || target || relative (full message)
]

# Combine: MLP output hidden dimension choices
COMBINE_DIMS = [8, 16, 32, 64, 128, 256]

# Connect: residual/skip options
CONNECT_OPS = ["identity", "skip"]

# ─────────────────────────────────────────────
#  Supernet / Training
# ─────────────────────────────────────────────
IN_CHANNELS  = 3    # xyz coordinates
NUM_CLASSES  = 10   # ModelNet10
HIDDEN_DIM   = 64   # supernet hidden dim (kept uniform for one-shot)
KNN_K        = 10   # neighbours for KNN sampling

SUPERNET_EPOCHS      = 30    # one-shot pre-training epochs
SUPERNET_LR          = 1e-3
SUPERNET_BATCH_SIZE  = 8
SUPERNET_WEIGHT_DECAY= 1e-4
NUM_POINTS           = 128   # points sampled per object (lower = faster on CPU)

# Whether to use static (pre-computed once) KNN graphs during training.
# Faster on CPU (Jetson Nano), trades some accuracy for speed.
# The paper mentions reusing sample results is nearly iso-accurate (Sec 2.2, Fig 3).
STATIC_GRAPH = True

# ─────────────────────────────────────────────
#  Evolutionary algorithm (both stages)
# ─────────────────────────────────────────────
EA_POP_SIZE          = 20
EA_MAX_ITER_STAGE1   = 50    # function search iterations
EA_MAX_ITER_STAGE2   = 100   # operation search iterations
EA_MUTATION_PROB     = 0.2

# ─────────────────────────────────────────────
#  Hardware predictor
# ─────────────────────────────────────────────
PREDICTOR_HIDDEN      = [256, 512, 512]
PREDICTOR_MLP_HIDDEN  = [256, 128, 1]
PREDICTOR_EPOCHS      = 50
PREDICTOR_LR          = 8e-4
PREDICTOR_BATCH       = 32
PREDICTOR_SAMPLES     = 2000   # architectures to profile for training

# ─────────────────────────────────────────────
#  Objective scaling (Eq. 2 in paper)
# ─────────────────────────────────────────────
ALPHA = 1.0   # accuracy weight
BETA  = 0.5   # efficiency penalty weight

# ─────────────────────────────────────────────
#  Compute device
# ─────────────────────────────────────────────
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
DATA_ROOT          = "./data/ModelNet10"
SEARCH_RESULTS_CSV = "search_results.csv"
PARETO_PLOT_FILE   = "hgnas_pareto_front.png"
PREDICTOR_CKPT     = "predictor_weights.pt"
SUPERNET_CKPT      = "supernet_weights.pt"
