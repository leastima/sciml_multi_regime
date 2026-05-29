#!/usr/bin/env bash
# =============================================================================
# CNN (ResNet-18) training — single experiment (CIFAR-10)
#
# Trains ResNet-18 on CIFAR-10 with SGD; used as a loss-landscape
# comparison baseline (Figure 2 / Appendix).
#
# Edit the variables below to change the experimental setting.
# =============================================================================
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CNN_DIR="${REPO_ROOT}/CNN"
OUTDIR="${OUTDIR:-/pscratch/sd/w/wyx345/sciml_multi_regime/experiments/cnn_single}"

GPU="${GPU:-0}"

# Single experimental setting
DATASET="${DATASET:-CIFAR-10}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-128}"
# ─────────────────────────────────────────────────────────────────────────────

echo "=== CNN (ResNet-18) training ==="
echo "Setting: dataset=${DATASET}  epochs=${EPOCHS}  batch=${BATCH_SIZE}"
echo "Output:  ${OUTDIR}"
mkdir -p "${OUTDIR}"

python "${CNN_DIR}/run_exp.py" \
    --dataset "${DATASET}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --save_model \
    --model ResNet18

echo ""
echo "=== Done. ==="
