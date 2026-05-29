#!/usr/bin/env bash
# =============================================================================
# PINO optimizer comparison — single experiment (2D Darcy flow)
#
# Runs one (r, n_samples, seed) cell through all 5 optimizers:
#   Adam → L-BFGS → ALM → NNCG → CL
#
# All post-training optimizers warm-start from the same Adam checkpoint.
# Edit the variables below to change the experimental setting.
# =============================================================================
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PINO_DIR="${REPO_ROOT}/PINO"
OUTDIR="${OUTDIR:-/pscratch/sd/w/wyx345/sciml_multi_regime/experiments/pino_single}"

GPU="${GPU:-0}"

# Single experimental setting
R="${R:-10}"
N_SAMPLES="${N_SAMPLES:-1000}"
SEED="${SEED:-0}"

# Adam warm-start steps
ADAM_STEPS="${ADAM_STEPS:-15000}"
CKPT_DIR="${CKPT_DIR:-${OUTDIR}/adam_ckpts}"
# ─────────────────────────────────────────────────────────────────────────────

echo "=== PINO optimizer comparison ==="
echo "Setting: r=${R}  n_samples=${N_SAMPLES}  seed=${SEED}"
echo "Output:  ${OUTDIR}"
mkdir -p "${OUTDIR}" "${CKPT_DIR}"

COMMON="--r ${R} --n_samples ${N_SAMPLES} --seed ${SEED} --gpu ${GPU}"

# ── Adam warm-start ───────────────────────────────────────────────────────────
echo ""
echo "--- Adam (${ADAM_STEPS} steps) ---"
python "${PINO_DIR}/scripts/darcy_sweep.py" \
    --optimizer adam \
    --steps "${ADAM_STEPS}" \
    --ckpt_dir "${CKPT_DIR}" \
    --outdir "${OUTDIR}/adam" \
    ${COMMON}

# ── L-BFGS ────────────────────────────────────────────────────────────────────
echo ""
echo "--- L-BFGS ---"
python "${PINO_DIR}/scripts/darcy_sweep.py" \
    --optimizer lbfgs \
    --steps 1 \
    --ckpt_dir "${CKPT_DIR}" \
    --outdir "${OUTDIR}/lbfgs" \
    ${COMMON}

# ── ALM ───────────────────────────────────────────────────────────────────────
echo ""
echo "--- ALM (mu=${ALM_MU:-2} rho=${ALM_RHO:-1.05} outer=${ALM_OUTER:-50} inner=${ALM_INNER:-500}) ---"
python "${PINO_DIR}/scripts/darcy_sweep.py" \
    --optimizer alm \
    --steps 1 \
    --ckpt_dir "${CKPT_DIR}" \
    --outdir "${OUTDIR}/alm" \
    ${COMMON}

# ── NNCG ──────────────────────────────────────────────────────────────────────
echo ""
echo "--- NNCG ---"
python "${PINO_DIR}/scripts/darcy_sweep.py" \
    --optimizer nncg \
    --steps 1 \
    --ckpt_dir "${CKPT_DIR}" \
    --outdir "${OUTDIR}/nncg" \
    ${COMMON}

# ── CL ────────────────────────────────────────────────────────────────────────
echo ""
echo "--- CL ---"
python "${PINO_DIR}/scripts/darcy_sweep.py" \
    --optimizer cl \
    --steps 1 \
    --ckpt_dir "${CKPT_DIR}" \
    --outdir "${OUTDIR}/cl" \
    ${COMMON}

echo ""
echo "=== Done. Results in ${OUTDIR} ==="
