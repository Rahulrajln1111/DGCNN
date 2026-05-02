#!/bin/bash
set -e

echo "============================================================"
echo "  HGNAS – Hardware-Aware GNN Architecture Search"
echo "  Target Device : Jetson Nano"
echo "  Dataset       : ModelNet10"
echo "============================================================"

# Verify data is present; if not, download it automatically
if [ ! -f "data/ModelNet10/processed/training.pt" ]; then
    echo "[INFO] ModelNet10 data not found – running get_data.py ..."
    python get_data.py
fi

# ─────────────────────────────────────────────────────────────────────
#  Run the full HGNAS pipeline.
#
#  QUICK DEMO (default, runs in ~3-5 min on CPU / Replit):
#    Uses a 400-sample subset, few epochs, small EA iterations.
#    Demonstrates the full pipeline end-to-end.
#
#  FULL RUN (recommended for Jetson Nano – uncomment below):
#    Uses the complete dataset and full search iterations for real results.
# ─────────────────────────────────────────────────────────────────────

# --- QUICK DEMO (default) ---
python main.py \
    --supernet-epochs    5   \
    --predictor-samples  200 \
    --predictor-epochs   30  \
    --ea-iter-s1         10  \
    --ea-iter-s2         20  \
    --max-samples        400 \
    --lat-constraint     120 \
    --mem-constraint     600

# --- FULL RUN on Jetson Nano (uncomment to use) ---
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
