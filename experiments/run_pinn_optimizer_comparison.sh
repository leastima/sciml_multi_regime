#!/usr/bin/env bash
# =============================================================================
# PINN optimizer comparison — single experiment (1D Convection)
#
# Runs one (beta, n_res, seed) setting through all 5 optimizers:
#   RoPINN, L-BFGS, ALM, NNCG, CL
#
# Edit the variables below to change the experimental setting.
# =============================================================================
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PINN_DIR="${REPO_ROOT}/PINN"
OUTDIR="${OUTDIR:-/pscratch/sd/w/wyx345/sciml_multi_regime/experiments/pinn_single}"

GPU="${GPU:-0}"

# Single experimental setting
BETA="${BETA:-10}"          # convection coefficient (physical difficulty)
N_RES="${N_RES:-5000}"      # number of collocation points
SEED="${SEED:-0}"
# ─────────────────────────────────────────────────────────────────────────────

echo "=== PINN optimizer comparison ==="
echo "Setting: beta=${BETA}  n_res=${N_RES}  seed=${SEED}"
echo "Output:  ${OUTDIR}"
mkdir -p "${OUTDIR}"

COMMON="--pde convection --pde_params '{\"beta\":${BETA}}' --num_res ${N_RES} --initial_seed ${SEED} --device ${GPU}"

for opt in ropinn lbfgs alm nncg cl; do
    echo ""
    echo "--- optimizer: ${opt} ---"
    python "${PINN_DIR}/run_experiment.py" \
        --opt "${opt}" \
        --save_path "${OUTDIR}/${opt}" \
        ${COMMON}
done

echo ""
echo "=== Done. Results in ${OUTDIR} ==="
