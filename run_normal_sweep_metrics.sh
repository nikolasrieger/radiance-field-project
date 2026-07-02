#!/usr/bin/env bash
# Run eval_benchmark.py on all normal-sweep runs:
#   minipointnet, minipointnetpp, pointnet2 × normal weights 0.2, 0.1, 0.05, 0.0
# Usage: bash run_normal_sweep_metrics.sh [2>&1 | tee normal_sweep_metrics.log]
set -euo pipefail

cd "$(dirname "$0")"

RUNS=(
    runs/minipointnet_l1_l1normal
    runs/minipointnet_l2_cos
    runs/pointnet2_l1_l1normal
    runs/pointnet2_l2_cos
)

for run in "${RUNS[@]}"; do
    echo "=========================================="
    echo "Evaluating: $(basename "$run")"
    echo "=========================================="
    uv run python eval_benchmark.py \
        --run "$run" \
        --train 6 \
        --out-csv "${run}/benchmark_results.csv"
done

echo "All evaluations complete."
