# Unveiling Multi-regime Patterns in SciML

**Distinct Failure Modes and Regime-specific Optimization**

[![arXiv](https://img.shields.io/badge/arXiv-2605.29153-b31b1b.svg)](https://arxiv.org/abs/2605.29153)

This repository contains the code for the paper:

> **Unveiling Multi-regime Patterns in SciML: Distinct Failure Modes and Regime-specific Optimization**  
> Yuxin Wang, Yuanzhe Hu, Xiaokun Zhong, Xiaopeng Wang, Haiquan Lu, Tianyu Pang, Michael W. Mahoney, Yujun Yan, Pu Ren, Yaoqing Yang  
> arXiv 2025. [[PDF]](https://arxiv.org/pdf/2605.29153)

---

## Overview

Neural networks trained under different hyperparameter settings can fall into distinct training **regimes**, with consistent behavior within regimes and qualitative differences across regimes. This paper studies such multi-regime behavior in scientific machine learning (SciML) through a **regime-aware diagnostic framework** that jointly analyzes performance, training dynamics, and loss-landscape geometry.

### Three-Regime Structure

Across all SciML models studied, a consistent three-regime pattern emerges on the (physical difficulty, data availability) configuration space:

| Regime | Name | Training Loss | Test Error | Description |
|--------|------|--------------|------------|-------------|
| **I** | Well-Trained | Low | Low | Optimization and generalization both succeed |
| **II** | Under-Trained | High | High | Optimization difficulty dominates; model fails to fit training objective |
| **III** | Over-Trained | Low | High | Data-limited failure; model fits supervision but does not generalize |

### Key Findings

1. The three-regime structure appears consistently across PINN, FNO, PINO, NODE, and PINODE.
2. Optimization effectiveness is **regime-specific**: no single optimizer performs well across all regimes.
3. SciML models exhibit novel pathological phenomena (e.g., *deceptive sharpness*, *deceptive flatness*) that challenge standard loss-landscape interpretations from computer vision.

---

## Repository Structure

```
sciml_multi_regime/
  PINN/           Physics-Informed Neural Networks
  PINO/           Physics-Informed Neural Operators (Darcy flow)
  FNO/            Fourier Neural Operators
  NeuralODE/      Neural ODEs / Physics-Informed NODEs
  CNN/            CNN baseline (ResNet-18, for landscape comparison)
  experiments/    Cross-module comparison scripts (Figure 3 reproducibility)
  sh/             Shared SLURM/batch submission helpers
```

### Module Map

| Module | Model | Benchmark PDE | Optimizers |
|--------|-------|---------------|------------|
| `PINN/multiadam` | PINN | 1D Convection, Reaction, Wave, Reaction-Diffusion | Adam, L-BFGS, NNCG, ALM, CL, RoPINN |
| `PINN/RoPINN` | PINN | 1D Wave, Reaction, Convection | RoPINN |
| `PINO/` | PINO (FNO2d backbone) | 2D Darcy Flow | Adam, L-BFGS, NNCG, ALM, CL |
| `FNO/` | FNO | 2D Poisson, Advection-Diffusion, Helmholtz | Adam |
| `NeuralODE/` | NODE / PINODE | Nonlinear Pendulum | Adam, L-BFGS, ALM, NNCG, CL |
| `CNN/` | ResNet-18 | Image classification (CIFAR) | SGD (comparison baseline) |

---

## Environment Setup

This repository uses separate environments for different modules due to distinct dependency requirements.

### Common dependencies (all modules)

```bash
pip install torch>=2.0 numpy scipy matplotlib tqdm wandb
```

### PINN

```bash
cd PINN
pip install -r requirements.txt
# For RoPINN sub-module:
pip install -r RoPINN/requirements.txt
```

### PINO

```bash
cd PINO
# Core dependencies are shared with the common stack.
# No extra requirements.txt needed beyond torch, numpy, scipy, matplotlib.
```

### FNO

```bash
cd FNO
pip install -r requirements.txt   # torch, numpy, scipy, h5py, wandb, ruamel.yaml
```

### NeuralODE

```bash
pip install torchdiffeq
```

> **Note**: All experiments were run on NERSC Perlmutter (A100 GPUs). Module-specific SLURM scripts are in each module's `sh/` or `scripts/` subdirectory.

---

## Quickstart

### PINN — 1D Convection sweep (Adam → L-BFGS)

```bash
cd PINN
python run_experiment.py --pde convection --optimizer lbfgs --beta 1 5 10 20 40
```

### PINO — 2D Darcy Flow, single run (r=10, N=1000, Adam → L-BFGS → NNCG)

```bash
cd PINO
# Generate data (if not using piececonst benchmark mats):
python scripts/generate_darcy.py --r 10 --n_samples 1200 --seed 0

# Adam warm-start + L-BFGS + NNCG chained:
bash scripts/run_r10_n1000.sh
```

Or to run a single optimizer interactively:

```bash
cd PINO
python scripts/darcy_sweep.py \
    --r 10 --n_samples 1000 --seeds 0 --gpu 0 \
    --optimizer adam --steps 10000 \
    --outdir /path/to/output
```

### NeuralODE — Nonlinear Pendulum sweep

```bash
cd NeuralODE
python run_sweep_horizon_physics_cl.py \
    --optimizer lbfgs --horizon 2 4 8 16 --n_train 100 500 1000
```

---

## Reproducing Paper Figures

The `experiments/` directory contains shell scripts that reproduce the main regime-map comparisons from the paper (Figure 3):

```bash
cd experiments
bash run_pinn_optimizer_comparison.sh   # Figure 3(a–e): PINN, 1D convection
bash run_pino_optimizer_comparison.sh   # Figure 3(f–j): PINO, 2D Darcy flow
bash run_node_optimizer_comparison.sh   # Figure 3(k–o): PINODE, nonlinear pendulum
```

---

## Citation

```bibtex
@article{wang2025sciml,
  title   = {Unveiling Multi-regime Patterns in SciML: Distinct Failure Modes and Regime-specific Optimization},
  author  = {Wang, Yuxin and Hu, Yuanzhe and Zhong, Xiaokun and Wang, Xiaopeng and Lu, Haiquan
             and Pang, Tianyu and Mahoney, Michael W. and Yan, Yujun and Ren, Pu and Yang, Yaoqing},
  journal = {arXiv preprint arXiv:2605.29153},
  year    = {2025}
}
```
