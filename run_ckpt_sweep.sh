#!/usr/bin/env bash
# Sweep all checkpoints for the 5 specified runs.
# Usage: bash run_ckpt_sweep.sh [2>&1 | tee sweep.log]
set -euo pipefail

cd "$(dirname "$0")"

uv run python eval_ckpt_sweep.py
