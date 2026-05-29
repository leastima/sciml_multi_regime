#!/usr/bin/env bash
# =============================================================================
# Reproduce Figure 3(f–j): PINO optimizer comparison on 2D Darcy flow
#
# Sweeps r (contrast ratio / physical difficulty) vs n_samples (training data)
# for 5 optimizer settings: Adam, L-BFGS, ALM, NNCG, CL.
#
# Paper setup (Table 1): PINO, 2D Darcy flow, 3 seeds,
# r in {4,6,8,10}, n_samples in {250,500,750,1000}.
# =============================================================================
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PINO_DIR="${REPO_ROOT}/PINO"
OUTDIR="${OUTDIR:-/pscratch/sd/w/wyx345/sciml_multi_regime/experiments/pino_fig3}"

GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"

# Sweep axes
R_VALUES="4 6 8 10"
N_SAMPLES_VALUES="250 500 750 1000"

# Adam warm-start steps and per-optimizer post-training steps
ADAM_STEPS=10000

# Adam ckpt dir (set via PINO_CKPT_DIR or override here)
CKPT_DIR="${PINO_CKPT_DIR:-${OUTDIR}/adam_ckpts}"
# ─────────────────────────────────────────────────────────────────────────────

echo "=== PINO optimizer comparison ==="
echo "Output: ${OUTDIR}"
mkdir -p "${OUTDIR}" "${CKPT_DIR}"

# ── Step 1: Adam warm-start (shared across all post-training optimizers) ──────
echo ""
echo "--- Step 1: Adam warm-start (${ADAM_STEPS} steps) ---"
for r in ${R_VALUES}; do
    for n in ${N_SAMPLES_VALUES}; do
        for seed in ${SEEDS}; do
            python "${PINO_DIR}/scripts/darcy_sweep.py" \
                --optimizer adam \
                --r "${r}" --n_samples "${n}" --seed "${seed}" \
                --steps "${ADAM_STEPS}" \
                --ckpt_dir "${CKPT_DIR}" \
                --outdir "${OUTDIR}/adam" \
                --gpu "${GPU}" \
                || echo "WARN: adam failed r=${r} n=${n} seed=${seed}"
        done
    done
done

# ── Step 2: Post-training optimizers (warm-start from Adam ckpts) ─────────────
for opt in lbfgs alm nncg cl; do
    echo ""
    echo "--- optimizer: ${opt} ---"
    for r in ${R_VALUES}; do
        for n in ${N_SAMPLES_VALUES}; do
            for seed in ${SEEDS}; do
                python "${PINO_DIR}/scripts/darcy_sweep.py" \
                    --optimizer "${opt}" \
                    --r "${r}" --n_samples "${n}" --seed "${seed}" \
                    --steps 1 \
                    --ckpt_dir "${CKPT_DIR}" \
                    --outdir "${OUTDIR}/${opt}" \
                    --gpu "${GPU}" \
                    || echo "WARN: ${opt} failed r=${r} n=${n} seed=${seed}"
            done
        done
    done
done

echo ""
echo "=== Done. Results in ${OUTDIR} ==="
