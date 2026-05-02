#!/bin/bash
set -e

echo "============================================================"
echo "  HGNAS – Hardware-Aware GNN Architecture Search"
echo "  Target Device : Jetson Nano"
echo "  Dataset       : ModelNet10"
echo "============================================================"

# Verify RAW data is present (not processed — that gets auto-generated).
# Delete data/ModelNet10/processed/ if you get a "too many values to unpack"
# error — it means the cached files were built by a different PyG version.
if [ ! -d "data/ModelNet10/raw" ] && [ ! -d "data/ModelNet10/ModelNet10" ]; then
    echo "[INFO] ModelNet10 raw data not found – running get_data.py ..."
    python get_data.py
fi

# ─────────────────────────────────────────────────────────────────────
#  Run the full HGNAS pipeline.
#
#  QUICK DEMO (uncomment for a fast ~3-5 min CPU test):
#    Uses a 400-sample subset, few epochs, small EA iterations.
#
#  FULL RUN (default – recommended for Jetson Nano):
#    Uses the complete dataset and full search iterations for real results.
# ─────────────────────────────────────────────────────────────────────

# --- FULL RUN on Jetson Nano (default) ---
python main.py \
    --supernet-epochs    30  \
    --predictor-samples  2000\
    --predictor-epochs   50  \
    --ea-iter-s1         50  \
    --ea-iter-s2         100 \
    --lat-constraint     120 \
    --mem-constraint     600

# --- QUICK DEMO (uncomment to use instead) ---
# python main.py \
#     --supernet-epochs    5   \
#     --predictor-samples  200 \
#     --predictor-epochs   30  \
#     --ea-iter-s1         10  \
#     --ea-iter-s2         20  \
#     --max-samples        400 \
#     --lat-constraint     120 \
#     --mem-constraint     600

echo "============================================================"
echo "  Done!  Outputs:"
echo "    search_results.csv     – all valid architectures found"
echo "    hgnas_pareto_front.png – Pareto-front plots"
echo "    supernet_weights.pt    – trained supernet checkpoint"
echo "    predictor_weights.pt   – trained HW predictor checkpoint"
echo "============================================================"
