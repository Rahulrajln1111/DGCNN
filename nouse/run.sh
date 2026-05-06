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
#GOMP_LIB=$(find "${VIRTUAL_ENV:-/usr}" -name "libgomp*.so*" 2>/dev/null | head -1)
#if [ -z "$GOMP_LIB" ]; then
 #   GOMP_LIB=$(ldconfig -p 2>/dev/null | grep libgomp | awk '{print $NF}' | head -1)
#fi
#if [ -n "$GOMP_LIB" ]; then
 #   echo "[INFO] Preloading $GOMP_LIB (ARM64 TLS fix)"
 #   export LD_PRELOAD="$GOMP_LIB"
#fi
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
#  BALANCED (default) – fits in 4 GB RAM, ~20-30 min, good accuracy
#  STRONG             – ~60-90 min on Jetson Nano, best results
#  QUICK              – ~5-10 min, for testing the pipeline
# ─────────────────────────────────────────────────────────────────────

# --- QUICK TEST (~5-10 min, for verifying pipeline works) ---
# python main.py \
#     --supernet-epochs    5   \
#     --predictor-samples  100 \
#     --predictor-epochs   15  \
#     --ea-iter-s1         5   \
#     --ea-iter-s2         10  \
#     --max-samples        600 \
#     --lat-constraint     120 \
#     --mem-constraint     600

# --- BALANCED RUN (default – ~20-30 min on Jetson Nano, ~60-75% accuracy) ---
# Uses FULL dataset (max-samples=0), more training epochs, better search.
# Key fixes applied: deterministic sampling, proper gradients, full data.
python main.py \
    --supernet-epochs    20  \
    --predictor-samples  200 \
    --predictor-epochs   30  \
    --ea-iter-s1         15  \
    --ea-iter-s2         30  \
    --max-samples        0   \
    --lat-constraint     120 \
    --mem-constraint     600

# --- STRONG RUN (uncomment – ~60-90 min on Jetson Nano, ~70-80% accuracy) ---
# python main.py \
#     --supernet-epochs    30  \
#     --predictor-samples  500 \
#     --predictor-epochs   40  \
#     --ea-iter-s1         25  \
#     --ea-iter-s2         50  \
#     --max-samples        0   \
#     --lat-constraint     120 \
#     --mem-constraint     600

# --- FULL RUN (uncomment – best results, ~80%+, several hours on Jetson Nano) ---
# python main.py \
#     --supernet-epochs    50  \
#     --predictor-samples  2000\
#     --predictor-epochs   50  \
#     --ea-iter-s1         50  \
#     --ea-iter-s2         100 \
#     --max-samples        0   \
#     --lat-constraint     120 \
#     --mem-constraint     600

echo "============================================================"
echo "  Done!  Outputs:"
echo "    search_results.csv     – all valid architectures found"
echo "    hgnas_pareto_front.png – Pareto-front plots"
echo "    supernet_weights.pt    – trained supernet checkpoint"
echo "    predictor_weights.pt   – trained HW predictor checkpoint"
echo "============================================================"
