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
  PINN/           Physics-Informed Neural Networks (1D convection / reaction / wave)
  PINO/           Physics-Informed Neural Operators (2D Darcy flow)
  FNO/            Fourier Neural Operators (2D Poisson / Helmholtz / AD)
  NeuralODE/      Neural ODEs / Physics-Informed NODEs (nonlinear pendulum)
  CNN/            CNN baseline (ResNet-18, CIFAR-10)
  experiments/    Single-run experiment scripts for each module
```

### Module Map

| Module | Model | Benchmark | Entry Point | Optimizers |
|--------|-------|-----------|-------------|------------|
| `PINN/` | PINN | 1D Convection, Reaction, Wave | `PINN/run_experiment.py` | Adam, L-BFGS, ALM, NNCG, CL, RoPINN |
| `PINO/` | PINO (FNO2d backbone) | 2D Darcy Flow | `PINO/scripts/darcy_sweep.py` | Adam, L-BFGS, ALM, NNCG, CL |
| `FNO/` | FNO | 2D Poisson, Helmholtz, AD | `FNO/train.py` | Adam |
| `NeuralODE/` | NODE / PINODE | Nonlinear Pendulum | `NeuralODE/run_sweep.py` | Adam, L-BFGS, ALM, NNCG, CL |
| `CNN/` | ResNet-18 | CIFAR-10 | `CNN/run_exp.py` | SGD |

---

## Environment Setup

All experiments were run on NERSC Perlmutter (A100 GPUs) using a shared conda environment.

```bash
# Create environment (example using conda)
conda create -n sciml python=3.11
conda activate sciml

# Core dependencies
pip install torch torchvision numpy scipy matplotlib tqdm wandb h5py
pip install torchdiffeq          # NeuralODE
pip install ruamel.yaml          # FNO config parsing

# Module-specific extras
pip install -r PINN/requirements.txt
pip install -r NeuralODE/requirements.txt
pip install -r FNO/requirements.txt
```

---

## Quickstart

### PINN — 1D Convection, single run

```bash
cd PINN
python run_experiment.py \
    --pde convection \
    --pde_params '{"beta":10}' \
    --opt lbfgs \
    --num_res 5000 \
    --initial_seed 0 \
    --save_path /path/to/output \
    --new_data
```

Supported `--opt` values: `adam`, `lbfgs`, `alm`, `nncg`, `cl`, `adam_lbfgs`.

### PINO — 2D Darcy flow, single run

```bash
cd PINO

# (Optional) generate data:
python scripts/generate_darcy.py --r 10 --n_samples 1200 --seed 0

# Adam warm-start:
python scripts/darcy_sweep.py \
    --optimizer adam --steps 15000 \
    --r 10 --n_samples 1000 --seed 0 --gpu 0 \
    --ckpt_dir /path/to/ckpts \
    --outdir /path/to/output/adam

# ALM fine-tuning from the Adam checkpoint:
python scripts/darcy_sweep.py \
    --optimizer alm --steps 1 \
    --r 10 --n_samples 1000 --seed 0 --gpu 0 \
    --ckpt_dir /path/to/ckpts \
    --outdir /path/to/output/alm
```

Supported `--optimizer` values: `adam`, `lbfgs`, `alm`, `nncg`, `cl`.

### NeuralODE — Nonlinear Pendulum, single run

```bash
cd NeuralODE

# Adam baseline:
python run_sweep.py \
    --optimizer Adam --physics-mode pinn \
    --inv-b-values 8 --horizon-values 20 --seeds 0 \
    --epochs 600 --cuda \
    --out-dir /path/to/output/adam

# ALM (LBFGS inner, 500-step warmup):
python run_sweep.py \
    --optimizer LBFGS --physics-mode pinn_alm \
    --inv-b-values 8 --horizon-values 20 --seeds 0 \
    --alm-outer-iters 50 --alm-warmup-epochs 500 --cuda \
    --out-dir /path/to/output/alm
```

Supported `--optimizer` values: `Adam`, `LBFGS`, `Adam_NNCG`. CL is enabled via `--cl-warmup`.

---

## Running All Optimizers at Once

The `experiments/` directory contains ready-to-run single-experiment scripts for each module. Each script accepts environment variables to override the default setting.

```bash
# PINN (1D Convection, default: beta=10, n_res=5000, seed=0)
bash experiments/run_pinn_optimizer_comparison.sh

# PINO (2D Darcy, default: r=10, n_samples=1000, seed=0)
bash experiments/run_pino_optimizer_comparison.sh

# PINODE (Pendulum, default: 1/b=8, horizon=20, seed=0)
bash experiments/run_node_optimizer_comparison.sh

# FNO (2D Poisson, default config)
bash experiments/run_fno_training.sh

# CNN (ResNet-18 on CIFAR-10)
bash experiments/run_cnn_training.sh
```

Override any setting via environment variables, e.g.:

```bash
R=5 N_SAMPLES=500 SEED=1 GPU=2 bash experiments/run_pino_optimizer_comparison.sh
BETA=30 N_RES=10000 bash experiments/run_pinn_optimizer_comparison.sh
INV_B=4 HORIZON=40 bash experiments/run_node_optimizer_comparison.sh
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
