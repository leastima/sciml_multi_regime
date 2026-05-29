#!/usr/bin/env bash
# =============================================================================
# FNO training — single experiment (2D Poisson)
#
# Trains FNO on the 2D Poisson equation using a config yaml.
# Vary YAML_CONFIG / CONFIG to switch between Poisson / Helmholtz / AD.
#
# Edit the variables below to change the experimental setting.
# =============================================================================
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FNO_DIR="${REPO_ROOT}/FNO"
OUTDIR="${OUTDIR:-/pscratch/sd/w/wyx345/sciml_multi_regime/experiments/fno_single}"

GPU="${GPU:-0}"

# Config (select PDE / data size via yaml + config key)
YAML_CONFIG="${YAML_CONFIG:-${FNO_DIR}/config/operators_poisson_64K.yaml}"
CONFIG="${CONFIG:-poisson_scale_k1.0_2.5_val1024_64K}"

# Training hyper-parameters
SEED="${SEED:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-500}"
LR="${LR:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SUBSAMPLE="${SUBSAMPLE:-32}"
# ─────────────────────────────────────────────────────────────────────────────

echo "=== FNO training ==="
echo "Config: ${CONFIG}  epochs=${MAX_EPOCHS}  lr=${LR}  seed=${SEED}"
echo "Output: ${OUTDIR}"
mkdir -p "${OUTDIR}"

CUDA_VISIBLE_DEVICES="${GPU}" python "${FNO_DIR}/train.py" \
    --yaml_config "${YAML_CONFIG}" \
    --config "${CONFIG}" \
    --root_dir "${OUTDIR}" \
    --run_num "0" \
    --seed "${SEED}" \
    --max_epochs "${MAX_EPOCHS}" \
    --lr "${LR}" \
    --batch_size "${BATCH_SIZE}" \
    --subsample "${SUBSAMPLE}"

echo ""
echo "=== Done. Results in ${OUTDIR} ==="
