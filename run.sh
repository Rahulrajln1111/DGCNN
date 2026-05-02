#!/bin/bash
set -e

echo "============================================================"
echo "  HGNAS – Hardware-Aware GNN Architecture Search"
echo "  Target Device : Jetson Nano"
echo "  Dataset       : ModelNet10"
echo "============================================================"

# ── ARM64 / aarch64 fix ───────────────────────────────────────────────────────
# On Jetson Nano (Python 3.6 + aarch64), scikit-learn's libgomp fails to load
# due to a static TLS block limitation. Preloading it fixes the ImportError.
GOMP_LIB=$(find "${VIRTUAL_ENV:-/usr}" -name "libgomp*.so*" 2>/dev/null | head -1)
if [ -z "$GOMP_LIB" ]; then
    GOMP_LIB=$(ldconfig -p 2>/dev/null | grep libgomp | awk '{print $NF}' | head -1)
fi
if [ -n "$GOMP_LIB" ]; then
    echo "[INFO] Preloading $GOMP_LIB (ARM64 TLS fix)"
    export LD_PRELOAD="$GOMP_LIB"
fi
# ─────────────────────────────────────────────────────────────────────────────

# Verify RAW data is present (not processed — that gets auto-generated).
# Delete data/ModelNet10/processed/ if you get a "too many values to unpack"
# error — it means the cached files were built by a different PyG version.
if [ ! -d "data/ModelNet10/raw" ] && [ ! -d "data/ModelNet10/ModelNet10" ]; then
    echo "[INFO] ModelNet10 raw data not found – running get_data.py ..."
    python get_data.py
fi

# ─────────────────────────────────────────────────────────────────────
#  Three run modes — uncomment the one you want:
#
#  TINY  (default) – fits in 4 GB RAM, ~10-15 min, good for first run
#  DEMO            – original quick demo (~5 min on GPU, needs ~6 GB)
#  FULL            – full paper settings, needs ~8+ GB, best results
# ─────────────────────────────────────────────────────────────────────

# --- MICRO RUN (default – fastest possible, ~3-5 min on 4 GB Jetson Nano) ---
python main.py \
    --supernet-epochs    2   \
    --predictor-samples  50  \
    --predictor-epochs   10  \
    --ea-iter-s1         3   \
    --ea-iter-s2         5   \
    --max-samples        100 \
    --lat-constraint     120 \
    --mem-constraint     600

# --- TINY RUN (uncomment – ~10-15 min, slightly better results) ---
# python main.py \
#     --supernet-epochs    5   \
#     --predictor-samples  100 \
#     --predictor-epochs   15  \
#     --ea-iter-s1         5   \
#     --ea-iter-s2         10  \
#     --max-samples        200 \
#     --lat-constraint     120 \
#     --mem-constraint     600

# --- DEMO RUN (uncomment – ~5 min, needs ~6 GB RAM) ---
# python main.py \
#     --supernet-epochs    5   \
#     --predictor-samples  200 \
#     --predictor-epochs   30  \
#     --ea-iter-s1         10  \
#     --ea-iter-s2         20  \
#     --max-samples        400 \
#     --lat-constraint     120 \
#     --mem-constraint     600

# --- FULL RUN (uncomment – best results, needs ~8 GB RAM) ---
# python main.py \
#     --supernet-epochs    30  \
#     --predictor-samples  2000\
#     --predictor-epochs   50  \
#     --ea-iter-s1         50  \
#     --ea-iter-s2         100 \
#     --lat-constraint     120 \
#     --mem-constraint     600

echo "============================================================"
echo "  Done!  Outputs:"
echo "    search_results.csv     – all valid architectures found"
echo "    hgnas_pareto_front.png – Pareto-front plots"
echo "    supernet_weights.pt    – trained supernet checkpoint"
echo "    predictor_weights.pt   – trained HW predictor checkpoint"
echo "============================================================"
