#!/usr/bin/env bash
# Sweep normal loss weights (0.2, 0.1, 0.05, 0.0) for minipointnet and minipointnetpp.
# Each run uses 10 training scenes.
# Usage: bash run_normal_sweep.sh [2>&1 | tee normal_sweep.log]
set -euo pipefail

cd "$(dirname "$0")"

CONFIGS=(
    configs/minipointnet_normal_0.2.yaml
    configs/minipointnet_normal_0.1.yaml
    configs/minipointnet_normal_0.05.yaml
    configs/minipointnet_normal_0.0.yaml
    configs/minipointnetpp_normal_0.2.yaml
    configs/minipointnetpp_normal_0.1.yaml
    configs/minipointnetpp_normal_0.05.yaml
    configs/minipointnetpp_normal_0.0.yaml
)

for cfg in "${CONFIGS[@]}"; do
    echo "=========================================="
    echo "Running: $cfg"
    echo "=========================================="
    uv run training --config "$cfg"
done

echo "All runs complete."
