#!/usr/bin/env bash
# Loss/normal-loss-type ablation: l1+l1normal vs l2+cosine for minipointnet and pointnet2.
# Usage: bash run_loss_ablation.sh [2>&1 | tee loss_ablation.log]
set -euo pipefail

cd "$(dirname "$0")"

CONFIGS=(
    configs/minipointnet_l1_l1normal.yaml
    configs/minipointnet_l2_cos.yaml
    configs/pointnet2_l1_l1normal.yaml
    configs/pointnet2_l2_cos.yaml
)

for cfg in "${CONFIGS[@]}"; do
    echo "=========================================="
    echo "Running: $cfg"
    echo "=========================================="
    uv run training --config "$cfg"
done

echo "All runs complete."
