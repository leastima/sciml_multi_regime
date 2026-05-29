#!/usr/bin/env bash
# =============================================================================
# Reproduce Figure 3(k–o): PINODE optimizer comparison on nonlinear pendulum
#
# Sweeps 1/b (inverse damping / physical difficulty) vs horizon T_train
# for 5 optimizer settings: Adam, L-BFGS, ALM, NNCG, CL.
#
# Paper setup (Table 1): NODE/PINODE, nonlinear pendulum, 3 seeds,
# 1/b in {1,2,4,6,8,10,16,32}, T in {2,4,8,10,16,20,30,40}.
# =============================================================================
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE_DIR="${REPO_ROOT}/NeuralODE"
OUTDIR="${OUTDIR:-/pscratch/sd/w/wyx345/sciml_multi_regime/experiments/node_fig3}"

SEEDS="${SEEDS:-0,1,2}"

# Sweep axes (match paper Table 1) — comma-separated for --inv-b-values / --horizon-values
INV_B_VALUES="1,2,4,6,8,10,16,32"
HORIZON_VALUES="2,4,8,10,16,20,30,40"

# Training epochs for each optimizer
ADAM_EPOCHS=600
LBFGS_EPOCHS=1       # 1 outer call × --lbfgs-max-iter inner iters
ALM_OUTER=20
# ─────────────────────────────────────────────────────────────────────────────

echo "=== PINODE optimizer comparison ==="
echo "Output: ${OUTDIR}"
mkdir -p "${OUTDIR}"

# ── Adam ──────────────────────────────────────────────────────────────────────
echo "--- optimizer: Adam ---"
python "${NODE_DIR}/run_sweep.py" \
    --optimizer Adam \
    --physics-mode pinn \
    --inv-b-values "${INV_B_VALUES}" \
    --horizon-values "${HORIZON_VALUES}" \
    --seeds "${SEEDS}" \
    --epochs "${ADAM_EPOCHS}" \
    --out-dir "${OUTDIR}/adam" \
    --cuda

# ── L-BFGS ────────────────────────────────────────────────────────────────────
echo "--- optimizer: L-BFGS ---"
python "${NODE_DIR}/run_sweep.py" \
    --optimizer LBFGS \
    --physics-mode pinn_alm \
    --inv-b-values "${INV_B_VALUES}" \
    --horizon-values "${HORIZON_VALUES}" \
    --seeds "${SEEDS}" \
    --epochs "${LBFGS_EPOCHS}" \
    --alm-outer-iters 1 \
    --out-dir "${OUTDIR}/lbfgs" \
    --cuda

# ── ALM ───────────────────────────────────────────────────────────────────────
echo "--- optimizer: ALM ---"
python "${NODE_DIR}/run_sweep.py" \
    --optimizer LBFGS \
    --physics-mode pinn_alm \
    --inv-b-values "${INV_B_VALUES}" \
    --horizon-values "${HORIZON_VALUES}" \
    --seeds "${SEEDS}" \
    --epochs 1 \
    --alm-outer-iters "${ALM_OUTER}" \
    --out-dir "${OUTDIR}/alm" \
    --cuda

# ── NNCG (post-Adam fine-tuning) ──────────────────────────────────────────────
echo "--- optimizer: NNCG ---"
python "${NODE_DIR}/run_sweep.py" \
    --optimizer Adam_NNCG \
    --physics-mode pinn \
    --inv-b-values "${INV_B_VALUES}" \
    --horizon-values "${HORIZON_VALUES}" \
    --seeds "${SEEDS}" \
    --epochs "${ADAM_EPOCHS}" \
    --out-dir "${OUTDIR}/nncg" \
    --cuda

# ── CL (curriculum on damping b) ─────────────────────────────────────────────
echo "--- optimizer: CL ---"
python "${NODE_DIR}/run_sweep.py" \
    --optimizer Adam \
    --physics-mode pinn \
    --cl-warmup \
    --inv-b-values "${INV_B_VALUES}" \
    --horizon-values "${HORIZON_VALUES}" \
    --seeds "${SEEDS}" \
    --epochs "${ADAM_EPOCHS}" \
    --out-dir "${OUTDIR}/cl" \
    --cuda

echo ""
echo "=== Done. Results in ${OUTDIR} ==="
