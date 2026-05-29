#!/usr/bin/env python3
"""
Darcy PINO experiment — CLI entry point.

Runs a single (r, f, n_samples, seed, optimizer) cell and appends
results to a CSV file.  All heavy logic lives in sibling modules:

  darcy_data.py            — data generation / loading
  darcy_model.py           — FNO model, metrics
  darcy_optimizers/alm.py  — Augmented Lagrangian Method
  darcy_optimizers/nncg.py — Nyström–Newton-CG
  darcy_optimizers/cl.py   — Curriculum Learning
  darcy_train.py           — train_one() orchestrator

Usage examples
--------------
Adam pretrain (10 k steps):
    python scripts/darcy_sweep.py --optimizer adam --steps 10000 \\
        --r 10 --n_samples 1000 --seeds 0 --gpu 0 \\
        --outdir /pscratch/sd/w/wyx345/pino/my_adam_run

ALM (warm-start from existing Adam ckpt):
    python scripts/darcy_sweep.py --optimizer alm --steps 1 \\
        --r 10 --n_samples 1000 --seeds 0 --gpu 0 \\
        --outdir /pscratch/sd/w/wyx345/pino/my_alm_run \\
        --ckpt_dir /path/to/adam_ckpts

Phase sweep (vary r and n_samples):
    python scripts/darcy_sweep.py --optimizer lbfgs \\
        --r 4 6 10 --n_samples 500 1000 --seeds 0 1 2 --gpu 0
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from darcy_data import (  # noqa: E402
    DEFAULT_A_LOW, DEFAULT_GRF_ALPHA, DEFAULT_GRF_TAU,
    PDE_SUB, _use_legacy_mat_name,
    coeff_tag_suffix, ensure_data, r_sweep_token, f_sweep_token,
    resolve_piececonst_paths,
)
from darcy_train import train_one  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-cell Darcy PINO sweep (writes results.csv).",
    )

    # ── Physical / data parameters ──────────────────────────────────────────
    parser.add_argument("--r",   type=float, default=6.0,
        help="Contrast ratio r (piececonst: CSV tag only).")
    parser.add_argument("--f",   type=float, default=1.0,
        help="Forcing f in -∇·(a∇u)=f (must match generated .mat).")
    parser.add_argument("--tau",   type=float, default=DEFAULT_GRF_TAU,
        help="GRF parameter τ in (-Δ+τ²)^(-α) coefficient sampling.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_GRF_ALPHA,
        help="GRF spectral exponent α.")
    parser.add_argument("--a_low", type=float, default=DEFAULT_A_LOW,
        help="Low permeability before thresholding; a_high = r * a_low.")
    parser.add_argument("--n_samples", type=int, default=1000,
        help="Training trajectories.")
    parser.add_argument("--pde_sub",  type=int, default=PDE_SUB,
        help="DarcyIC subsample for PDE loss (data uses sub=7).")
    parser.add_argument("--seed",  type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
        help="Run multiple seeds sequentially (overrides --seed).")

    # ── Training / optimizer parameters ─────────────────────────────────────
    parser.add_argument("--steps", type=int, default=1,
        help="Adam steps before ALM/CL/NNCG (default 1: load *_adam.pt and skip Adam).")
    parser.add_argument("--optimizer", type=str, default="alm",
        choices=("adam", "alm", "cl", "nncg", "lbfgs"),
        help="Optimizer / outer trainer.")
    parser.add_argument("--lbfgs_max_iter", type=int, default=10000,
        help="For optimizer=lbfgs: max_iter in a single LBFGS step().")
    parser.add_argument("--xy_loss", type=float, default=5.0,
        help="Data-loss weight in `xy_loss*data + f_loss*pde`.")
    parser.add_argument("--f_loss",  type=float, default=1.0,
        help="PDE-loss weight in `xy_loss*data + f_loss*pde`.")

    # ── ALM parameters ───────────────────────────────────────────────────────
    parser.add_argument("--alm_outer_iters",    type=int,   default=100)
    parser.add_argument("--alm_inner",          type=str,   default="adam",
        choices=("lbfgs", "adam"))
    parser.add_argument("--alm_lbfgs_max_iter", type=int,   default=200)
    parser.add_argument("--alm_lbfgs_chunks",   type=int,   default=0,
        help="0=auto: ceil(n/500) chunks.")
    parser.add_argument("--alm_inner_step",     type=int,   default=4000,
        help="Adam inner steps per outer when --alm_inner adam.")
    parser.add_argument("--alm_pde_eps_slack",  type=float, default=0.0)
    parser.add_argument("--alm_data_eps_slack", type=float, default=0.0)
    parser.add_argument("--alm_cons_item",      type=str,   default="pde",
        choices=("data", "pde"))
    parser.add_argument("--alm_uncon_weight",   type=float, default=200.0)
    parser.add_argument("--alm_mu", "--alm_mu0", type=float, default=2.0,
        dest="alm_mu", help="Initial ALM penalty μ₀.")
    parser.add_argument("--alm_rho",            type=float, default=1.2,
        help="ALM μ multiplier each outer.")
    parser.add_argument("--alm_save_best",      action="store_true")
    parser.add_argument("--alm_adam_lr",        type=float, default=None)
    parser.add_argument("--alm_adam_warmup_frac", type=float, default=0.2)
    parser.add_argument("--alm_adam_no_cosine", action="store_true")

    # ── CL parameters ────────────────────────────────────────────────────────
    parser.add_argument("--cl_init_f",   type=float, default=1.0)
    parser.add_argument("--cl_delta_f",  type=float, default=0.1)
    parser.add_argument("--cl_target_f", type=float, default=None)
    parser.add_argument("--cl_inner_step", type=int, default=500)

    # ── NNCG parameters ──────────────────────────────────────────────────────
    parser.add_argument("--nncg_steps", type=int, default=5)

    # ── I/O paths ────────────────────────────────────────────────────────────
    parser.add_argument("--gpu",  type=int, default=0)
    parser.add_argument("--outdir", type=str,
        default="/pscratch/sd/w/wyx345/pino/sweep/darcy")
    parser.add_argument("--progress_dir", type=str, default=None,
        help="Per-cell progress ckpt dir (resume support).")
    parser.add_argument("--ckpt_every", type=int, default=2000)
    parser.add_argument("--ckpt_dir", type=str,
        default=os.environ.get(
            "PINO_CKPT_DIR",
            "/global/homes/w/wyx345/pscratch/pino/adam_15k_8x8_seed01_20260523_123416/ckpts",
        ),
        help="Adam/ALM ckpt dir (override with PINO_CKPT_DIR env var).")
    parser.add_argument("--piececonst_dir", type=str,
        default=os.environ.get("PINO_PIECECONST_DIR"),
        help="Directory with piececonst_r421_N1024_smooth1/2.mat.")
    parser.add_argument("--test_num", type=int, default=None,
        help="Test trajectories (default 500 if piececonst, else 200).")
    parser.add_argument("--force", action="store_true",
        help="Rerun cells even if results.csv already has a matching row.")

    args = parser.parse_args()

    if args.test_num is None:
        args.test_num = 500 if args.piececonst_dir else 200
    if args.piececonst_dir:
        if not _use_legacy_mat_name(args.tau, args.alpha, args.a_low):
            raise SystemExit(
                "With --piececonst_dir, τ=3, α=2, a_low=3 only (piececonst dataset)."
            )
    if args.piececonst_dir and args.n_samples > 1024:
        raise SystemExit("piececonst MAT files contain 1024 trajectories; use --n_samples ≤ 1024.")

    seeds = args.seeds if args.seeds is not None else [args.seed]
    outdir = os.path.join(args.outdir, "pino")
    os.makedirs(outdir, exist_ok=True)
    if args.progress_dir:
        os.makedirs(args.progress_dir, exist_ok=True)
    if args.ckpt_dir:
        os.makedirs(args.ckpt_dir, exist_ok=True)

    csv_path = os.path.join(outdir, "results.csv")
    device   = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    fieldnames = [
        "optimizer", "r", "f", "tau", "alpha", "a_low",
        "n_samples", "seed",
        "train_loss", "train_loss_last50_mean", "train_loss_last50_min",
        "data_l2", "pde_res", "test_error", "test_error_abs", "elapsed_s",
    ]

    # Migrate legacy CSV that lacks the optimizer column.
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            old_fields = csv.reader(f).__next__()
        if "optimizer" not in old_fields:
            bak = csv_path + f".bak_{time.strftime('%Y%m%d_%H%M%S')}"
            os.replace(csv_path, bak)
            print(f"  [csv] Renamed legacy results (no optimizer column) → {bak}", flush=True)

    done: set = set()
    if os.path.exists(csv_path) and not args.force:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if not row.get("optimizer"):
                    continue
                try:
                    fv    = float(row.get("f", "1.0"))
                    tau_k = float(row["tau"])   if row.get("tau")   else DEFAULT_GRF_TAU
                    alp_k = float(row["alpha"]) if row.get("alpha") else DEFAULT_GRF_ALPHA
                    al_k  = float(row["a_low"]) if row.get("a_low") else DEFAULT_A_LOW
                    done.add((
                        str(row["optimizer"]), float(row["r"]),
                        fv, tau_k, alp_k, al_k,
                        int(row["n_samples"]), int(row["seed"]),
                    ))
                except (ValueError, TypeError):
                    pass

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    csv_lock_path = csv_path + ".lock"
    r         = float(args.r)
    f_rhs     = float(args.f)
    tau       = float(args.tau)
    alpha_v   = float(args.alpha)
    a_low     = float(args.a_low)
    n_samples = int(args.n_samples)
    n_pool_train = n_samples
    n_pool_test  = max(200, args.test_num)

    print(f"Darcy PINO run (61×61, FNO modes 20)  →  {csv_path}", flush=True)
    if args.optimizer == "alm":
        print(
            f"  [config] optimizer=alm  cons={args.alm_cons_item}  inner={args.alm_inner}  "
            f"outer={args.alm_outer_iters}  μ0={args.alm_mu}  ρ={args.alm_rho}  "
            f"steps={args.steps}  ckpt_dir={args.ckpt_dir}",
            flush=True,
        )
    if args.piececonst_dir:
        print(
            f"  piececonst_dir={os.path.abspath(args.piececonst_dir)}  "
            f"test_num={args.test_num}",
            flush=True,
        )

    for seed in seeds:
        cell_key = (args.optimizer, r, f_rhs, tau, alpha_v, a_low, n_samples, seed)
        if cell_key in done and not args.force:
            print(
                f"SKIP optimizer={args.optimizer} r={r} f={f_rhs} tau={tau} "
                f"alpha={alpha_v} a_low={a_low} n={n_samples} seed={seed} "
                f"(in {csv_path}; use --force to rerun)",
                flush=True,
            )
            continue


        print(
            f"\nRun: r={r}  f={f_rhs}  tau={tau} alpha={alpha_v} a_low={a_low}  "
            f"n_samples={n_samples}  seed={seed}  steps={args.steps}  "
            f"optimizer={args.optimizer}",
            flush=True,
        )
        t0 = time.time()

        if args.piececonst_dir:
            train_path, test_path = resolve_piececonst_paths(args.piececonst_dir)
        else:
            train_path = ensure_data(r, seed=0, n_samples=n_pool_train,
                f=f_rhs, tau=tau, alpha=alpha_v, a_low=a_low)
            test_path  = ensure_data(r, seed=1, n_samples=n_pool_test,
                f=f_rhs, tau=tau, alpha=alpha_v, a_low=a_low)

        tag = (
            f"r{r_sweep_token(r)}_f{f_sweep_token(f_rhs)}"
            f"_n{n_samples}_s{seed}"
            f"{coeff_tag_suffix(tau, alpha_v, a_low)}"
        )
        progress_ckpt = (
            os.path.join(args.progress_dir, f"{tag}_progress.pt")
            if args.progress_dir else None
        )
        adam_ckpt = (
            os.path.join(args.ckpt_dir, f"{tag}_adam.pt")
            if args.ckpt_dir else None
        )
        alm_ckpt = (
            os.path.join(args.ckpt_dir, f"{tag}_alm.pt")
            if args.ckpt_dir and args.optimizer == "alm" else None
        )
        cl_progress_ckpt = (
            os.path.join(args.progress_dir, f"{tag}_cl_progress.pt")
            if args.progress_dir and args.optimizer == "cl" else None
        )

        result = train_one(
            train_path, test_path, n_samples, args.steps, device, seed,
            xy_loss=args.xy_loss, f_loss=args.f_loss,
            progress_ckpt_path=progress_ckpt, ckpt_every=args.ckpt_every,
            adam_ckpt_path=adam_ckpt, adam_save_path=adam_ckpt,
            alm_ckpt_path=alm_ckpt,
            alm_outer_iters=args.alm_outer_iters, alm_inner_step=args.alm_inner_step,
            alm_cons_item=args.alm_cons_item,  alm_uncon_weight=args.alm_uncon_weight,
            alm_mu=args.alm_mu,    alm_rho=args.alm_rho,
            alm_inner=args.alm_inner,
            alm_lbfgs_max_iter=args.alm_lbfgs_max_iter,
            alm_lbfgs_chunks=args.alm_lbfgs_chunks,
            alm_pde_eps_slack=args.alm_pde_eps_slack,
            alm_data_eps_slack=args.alm_data_eps_slack,
            alm_save_best=args.alm_save_best,
            alm_adam_lr=args.alm_adam_lr,
            alm_adam_warmup_frac=args.alm_adam_warmup_frac,
            alm_adam_cosine=(not args.alm_adam_no_cosine),
            cl_init_f=args.cl_init_f,   cl_delta_f=args.cl_delta_f,
            cl_target_f=args.cl_target_f, cl_inner_step=args.cl_inner_step,
            cl_progress_ckpt_path=cl_progress_ckpt,
            nncg_steps=args.nncg_steps, lbfgs_max_iter=args.lbfgs_max_iter,
            f_rhs=f_rhs, test_num=args.test_num, opt=args.optimizer,
            pde_sub=args.pde_sub, darcy_r=r,
            tau=tau, alpha=alpha_v, a_low=a_low,
        )
        elapsed = time.time() - t0

        if args.optimizer == "alm":
            pt_te = result.get("pretrain_test_error")
            if pt_te is not None and not math.isnan(float(pt_te)):
                print(
                    f"  → pretrain ckpt  "
                    f"test_rel={result['pretrain_test_error']:.4f}  "
                    f"test_abs={result['pretrain_test_error_abs']:.4g}  "
                    f"data_loss={result['pretrain_data_l2']:.4f}  "
                    f"physics_loss={result['pretrain_pde_res']:.4f}",
                    flush=True,
                )
            print(
                f"  → ALM final  test_rel={result['test_error']:.4f}  "
                f"test_abs={result['test_error_abs']:.4g}  "
                f"data_loss={result['data_l2']:.4f}  "
                f"physics_loss={result['pde_res']:.4f}  ({elapsed:.0f}s)",
                flush=True,
            )
        else:
            print(
                f"  → test_rel={result['test_error']:.4f}  "
                f"test_abs={result['test_error_abs']:.4g}  "
                f"data_loss={result['data_l2']:.4f}  "
                f"physics_loss={result['pde_res']:.4f}  ({elapsed:.0f}s)",
                flush=True,
            )

        with open(csv_path, "a", newline="") as csvfile, open(csv_lock_path, "a") as _lk:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            fcntl.flock(_lk.fileno(), fcntl.LOCK_EX)
            try:
                writer.writerow({
                    "optimizer": args.optimizer,
                    "r": r, "f": f_rhs, "tau": tau, "alpha": alpha_v, "a_low": a_low,
                    "n_samples": n_samples, "seed": seed,
                    "train_loss":             round(result["train_loss"], 6),
                    "train_loss_last50_mean": round(result["train_loss_last50_mean"], 8),
                    "train_loss_last50_min":  round(result["train_loss_last50_min"],  8),
                    "data_l2":    round(result["data_l2"],    6),
                    "pde_res":    round(result["pde_res"],    6),
                    "test_error": round(result["test_error"], 6),
                    "test_error_abs": round(result["test_error_abs"], 8),
                    "elapsed_s":  round(elapsed, 1),
                })
                csvfile.flush()
            finally:
                fcntl.flock(_lk.fileno(), fcntl.LOCK_UN)

    print(f"\nDone. Results: {csv_path}")


if __name__ == "__main__":
    main()
