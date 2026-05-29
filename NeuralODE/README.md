# NeuralODE — Nonlinear Pendulum Experiments

Multi-regime analysis of **Neural ODEs (NODEs)** and **Physics-Informed NODEs (PINODEs)**
on the nonlinear pendulum benchmark, covering all optimizer and training strategies
from the paper:

> **Unveiling Multi-regime Patterns in SciML** (arXiv:2605.29153)

---

## Module Structure

```
NeuralODE/
  run_sweep.py               ← CLI entry point (calls main() below)
  run_sweep_horizon_physics_cl.py  ← full training + sweep implementation
  data.py                    ← pendulum simulation and data-loading utilities
  model.py                   ← ShallowODE model and helpers
  requirements.txt           ← extra dependencies (torchdiffeq)
  README.md
```

### File responsibilities

| File | Contents |
|------|----------|
| `data.py` | `pendulum()`, `build_data()`, `build_loader()`, `numpy_to_torch()` |
| `model.py` | `ShallowODE`, `shallow()`, `set_seed()` |
| `run_sweep_horizon_physics_cl.py` | Full training loop (`train_odenet`), ALM/CL/NNCG chains, sweep cell executor, plotting |
| `run_sweep.py` | Thin entry point; parses args, dispatches to main sweep |

---

## Physical Setup

- **System**: Nonlinear pendulum  `θ̈ = −sin θ − b θ̇`
- **Input representation**: Spherical embedding `(sin θ cos φ, sin θ sin φ, −cos θ)` (embedded)  
  or raw state `(θ, ω)` (state; required for PINN physics loss)
- **Regime axes**:
  - x-axis: inverse damping `1/b` (physical difficulty; larger = harder)
  - y-axis: training horizon `T_train` (data availability)

---

## Optimizers

| Optimizer flag | Description |
|---------------|-------------|
| `Adam` | Mini-batch Adam |
| `LBFGS` | Full-batch L-BFGS (default for PINN/PINODE) |
| `NNCG` | Nyström–Newton-CG post-training fine-tuning |
| `Adam_LBFGS_NNCG` | Chained: Adam → L-BFGS → NNCG |
| `Adam_NNCG` | Chained: Adam → NNCG |

Physics constraint modes (``--physics-mode``):

| Mode | Description |
|------|-------------|
| `none` | Pure data loss (NODE) |
| `sphere` | Spherical constraint on embedded representation |
| `pinn` | Physics residual on state representation (pendulum ODE) |
| `pinn_alm` | ALM hard constraint on PINN residual (PINODE) |

---

## Quickstart

### Install extra dependency

```bash
pip install torchdiffeq
```

### Single cell (PINODE, L-BFGS, 1/b=8, T=20)

```bash
cd NeuralODE
python run_sweep.py \
    --optimizer LBFGS --physics-mode pinn_alm \
    --inv-b 8 --horizon 20 --seed 0 \
    --epochs 1 --alm-outer-iters 20 \
    --outdir /path/to/output
```

### 8×8 regime sweep (Adam)

```bash
python run_sweep.py \
    --optimizer Adam --physics-mode pinn \
    --inv-b 1 2 4 6 8 10 16 32 \
    --horizon 2 4 8 10 16 20 30 40 \
    --seeds 0 1 2 --epochs 600 \
    --outdir /path/to/output
```

### Adam → L-BFGS → NNCG chained

```bash
python run_sweep.py \
    --optimizer Adam_LBFGS_NNCG \
    --physics-mode pinn_alm \
    --inv-b 8 --horizon 20 --seed 0 \
    --outdir /path/to/output
```

---

## Output

Each sweep writes to ``--outdir``:

```
<outdir>/
  results.json            ← per-cell metrics (test rel-L2, train loss, etc.)
  plots/
    regime_map_*.png      ← 2D regime heatmaps
    loss_curves_*.png     ← training loss vs iteration
```

---

## Key Implementation Notes

- **Coupled loaders**: NODE uses a single `(inputs, targets, dt, idx)` loader;
  PINN/PINODE compute the physics residual on the same mini-batch.
- **ALM multipliers**: Per-sample Lagrange multipliers `λᵢ` stored as a tensor
  of shape `(N_train,)`, updated after each outer iteration.
- **Curriculum (CL)**: Gradually increases the damping coefficient `b` from
  `cl_init_b` to the target value, training `cl_inner_step` Adam steps at each stage.
