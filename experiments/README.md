# Experiments — Cross-module Optimizer Comparisons

Reproduces the optimizer-comparison regime maps from **Figure 3** of the paper:

> **Unveiling Multi-regime Patterns in SciML: Distinct Failure Modes and Regime-specific Optimization** (arXiv:2605.29153)

---

## Scripts

| Script | Reproduces | Model | PDE / system |
|--------|-----------|-------|--------------|
| `run_pinn_optimizer_comparison.sh` | Figure 3(a–e) | PINN | 1D Convection |
| `run_pino_optimizer_comparison.sh` | Figure 3(f–j) | PINO (FNO2d) | 2D Darcy flow |
| `run_node_optimizer_comparison.sh` | Figure 3(k–o) | PINODE | Nonlinear pendulum |

Each script sweeps one optimizer across the full 2D configuration space
(physical difficulty × data availability), using 3 random seeds.

---

## Usage

Edit the `OUTDIR` and resource variables at the top of each script,
then submit via SLURM or run interactively:

```bash
cd experiments

# PINN (1D convection, Figure 3a–e)
bash run_pinn_optimizer_comparison.sh

# PINO (2D Darcy flow, Figure 3f–j)
bash run_pino_optimizer_comparison.sh

# PINODE (nonlinear pendulum, Figure 3k–o)
bash run_node_optimizer_comparison.sh
```

---

## Resource requirements

| Model | GPU memory | Approx. wall time (8×8 sweep, 3 seeds) |
|-------|-----------|----------------------------------------|
| PINN | ≤ 4 GB | ~2 h (A100) |
| PINO | 16–32 GB | ~8–24 h (A100, depends on optimizer) |
| PINODE | ≤ 8 GB | ~4 h (A100) |
