#!/usr/bin/env bash
# =============================================================================
# PINODE optimizer comparison — single experiment (nonlinear pendulum)
#
# Runs one (inv_b, horizon, seed) cell through all 5 optimizers:
#   Adam → L-BFGS → ALM → NNCG → CL
#
# Edit the variables below to change the experimental setting.
# =============================================================================
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE_DIR="${REPO_ROOT}/NeuralODE"
OUTDIR="${OUTDIR:-/pscratch/sd/w/wyx345/sciml_multi_regime/experiments/node_single}"

# Single experimental setting
INV_B="${INV_B:-8}"        # 1/b (physical difficulty)
HORIZON="${HORIZON:-20}"   # training horizon T
SEED="${SEED:-0}"

# Training epochs
ADAM_EPOCHS="${ADAM_EPOCHS:-600}"
ALM_OUTER="${ALM_OUTER:-20}"
# ─────────────────────────────────────────────────────────────────────────────

echo "=== PINODE optimizer comparison ==="
echo "Setting: 1/b=${INV_B}  horizon=${HORIZON}  seed=${SEED}"
echo "Output:  ${OUTDIR}"
mkdir -p "${OUTDIR}"

COMMON="--inv-b-values ${INV_B} --horizon-values ${HORIZON} --seeds ${SEED} --cuda"

# ── Adam ──────────────────────────────────────────────────────────────────────
echo ""
echo "--- Adam (${ADAM_EPOCHS} epochs) ---"
python "${NODE_DIR}/run_sweep.py" \
    --optimizer Adam \
    --physics-mode pinn \
    --epochs "${ADAM_EPOCHS}" \
    --out-dir "${OUTDIR}/adam" \
    ${COMMON}

# ── L-BFGS ────────────────────────────────────────────────────────────────────
echo ""
echo "--- L-BFGS ---"
python "${NODE_DIR}/run_sweep.py" \
    --optimizer LBFGS \
    --physics-mode pinn_alm \
    --epochs 1 \
    --alm-outer-iters 1 \
    --out-dir "${OUTDIR}/lbfgs" \
    ${COMMON}

# ── ALM ───────────────────────────────────────────────────────────────────────
echo ""
echo "--- ALM (outer=${ALM_OUTER}) ---"
python "${NODE_DIR}/run_sweep.py" \
    --optimizer LBFGS \
    --physics-mode pinn_alm \
    --epochs 1 \
    --alm-outer-iters "${ALM_OUTER}" \
    --out-dir "${OUTDIR}/alm" \
    ${COMMON}

# ── NNCG ──────────────────────────────────────────────────────────────────────
echo ""
echo "--- NNCG ---"
python "${NODE_DIR}/run_sweep.py" \
    --optimizer Adam_NNCG \
    --physics-mode pinn \
    --epochs "${ADAM_EPOCHS}" \
    --out-dir "${OUTDIR}/nncg" \
    ${COMMON}

# ── CL ────────────────────────────────────────────────────────────────────────
echo ""
echo "--- CL ---"
python "${NODE_DIR}/run_sweep.py" \
    --optimizer Adam \
    --physics-mode pinn \
    --cl-warmup \
    --epochs "${ADAM_EPOCHS}" \
    --out-dir "${OUTDIR}/cl" \
    ${COMMON}

echo ""
echo "=== Done. Results in ${OUTDIR} ==="
