#!/usr/bin/env bash
# =============================================================================
# Reproduce Figure 3(a–e): PINN optimizer comparison on 1D Convection
#
# Sweeps beta (physical difficulty) vs n_res (collocation points) for 5 optimizer
# settings: RoPINN, L-BFGS, ALM, NNCG, CL.
#
# Paper setup (Table 1): PINN, 1D convection, 5 seeds, beta in {1,5,10,20,40},
# n_res in {1000,2000,5000,10000}.
# =============================================================================
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PINN_DIR="${REPO_ROOT}/PINN"
OUTDIR="${OUTDIR:-/pscratch/sd/w/wyx345/sciml_multi_regime/experiments/pinn_fig3}"

GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2 3 4}"

# Sweep axes (match paper)
BETA_VALUES="1 5 10 20 40"
N_RES_VALUES="1000 2000 5000 10000"

# Optimizers to compare (Figure 3 columns a–e)
OPTIMIZERS="ropinn lbfgs alm nncg cl"
# ─────────────────────────────────────────────────────────────────────────────

echo "=== PINN optimizer comparison ==="
echo "Output: ${OUTDIR}"
mkdir -p "${OUTDIR}"

for opt in ${OPTIMIZERS}; do
    echo "--- optimizer: ${opt} ---"
    for beta in ${BETA_VALUES}; do
        for n_res in ${N_RES_VALUES}; do
            for seed in ${SEEDS}; do
                python "${PINN_DIR}/run_experiment.py" \
                    --pde convection \
                    --pde_params "{\"beta\":${beta}}" \
                    --opt "${opt}" \
                    --num_res "${n_res}" \
                    --initial_seed "${seed}" \
                    --outdir "${OUTDIR}/${opt}" \
                    --gpu "${GPU}" \
                    || echo "WARN: failed opt=${opt} beta=${beta} n_res=${n_res} seed=${seed}"
            done
        done
    done
done

echo "=== Done. Results in ${OUTDIR} ==="
