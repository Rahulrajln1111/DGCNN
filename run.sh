#!/bin/bash
set -e

echo "==================================================="
echo "   HGNAS Simulation on Jetson Edge Device          "
echo "==================================================="

# Execute the search
echo "[INFO] Running Architecture Search..."
python hgnas_search.py

# Execute the plotting script
echo "[INFO] Plotting trade-off graphs..."
python plot_res.py

echo "==================================================="
echo "   Done! Check 'hgnas_pareto_front.png'            "
echo "==================================================="
