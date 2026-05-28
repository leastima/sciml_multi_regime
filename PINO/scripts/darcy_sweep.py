#!/usr/bin/env python3
"""
Darcy PINO experiment.

Data loss on ``DarcyFlow`` grid (default ``sub=7`` → 61×61); PDE loss on ``DarcyIC`` grid
(``pde_sub``, default 7 — decoupled loaders like ``train_darcy.py``),
FNO **modes 20** (``build_model``, same spirit as ``configs/pretrain/Darcy-pretrain.yaml``).

* Synthetic data: ``ensure_data`` writes under ``DATADIR`` unless files exist.
* Benchmark: pass ``--piececonst_dir`` with author ``piececonst_r421_N1024_smooth{1,2}.mat``;
  ``--test_num`` defaults to 500 when piececonst is set.

Phase plots: typically vary ``f`` and ``n_samples`` at fixed ``r``; coefficient-field
hardness can be swept via ``--tau``, ``--alpha``, ``--a_low`` (defaults match piececonst).

Example (ALM, physics constraint, warm-start from Adam ckpt):
    python scripts/darcy_sweep.py --r 4 --n_samples 1000 --seeds 0 --gpu 0 \\
        --outdir /pscratch/sd/w/wyx345/pino/my_alm_run

Example (Adam pretrain):
    python scripts/darcy_sweep.py --optimizer adam --steps 10000 \\
        --piececonst_dir /path/to/mats --f 0.5 1 2 --n_samples 1000 --gpu 0
"""

from __future__ import annotations

import argparse
import collections
import csv
import fcntl
import inspect
import math
import os
import re
import sys
import time
from functools import partial
from typing import Any

import numpy as np
import scipy.io
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fourier2d import FNO2d
from train_utils.datasets import DarcyFlow, DarcyIC, sample_data
from train_utils.losses import FDM_Darcy, LpLoss, darcy_loss

from scripts._cell_shard import cell_shard
from scripts.generate_darcy import gen_coeff, solve_darcy_batch

# ──────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────

DATADIR = "/pscratch/sd/w/wyx345/pino/darcy_gen"
N_FINE = 421
SUB = 7  # data loss grid: 421//7+1 → 61×61
PDE_SUB = 7  # PDE / IC grid (train_darcy ``pde_sub``; override via --pde_sub)

def _darcy_mollifier(mesh: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    return (
        0.001
        * torch.sin(math.pi * mesh[..., 0])
        * torch.sin(math.pi * mesh[..., 1])
    ).unsqueeze(0).to(device)


def _load_torch_ckpt(path: str, map_location):
    """Load training checkpoint (PyTorch 2.6+ defaults weights_only=True)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


PIECECONST_TRAIN_NAME = "piececonst_r421_N1024_smooth1.mat"
PIECECONST_TEST_NAME = "piececonst_r421_N1024_smooth2.mat"


def _validate_piececonst_mat(path: str) -> None:
    keys = {x[0] for x in scipy.io.whosmat(path)}
    if not {"coeff", "sol"}.issubset(keys):
        raise ValueError(
            f"{path}: expected MATLAB variables coeff, sol (got {sorted(keys)})."
        )


def resolve_piececonst_paths(directory: str) -> tuple[str, str]:
    d = os.path.abspath(directory)
    t1 = os.path.join(d, PIECECONST_TRAIN_NAME)
    t2 = os.path.join(d, PIECECONST_TEST_NAME)
    if not os.path.isfile(t1) or not os.path.isfile(t2):
        raise SystemExit(
            f"Missing piececonst MAT files.\n  Expected: {t1}\n  Expected: {t2}"
        )
    _validate_piececonst_mat(t1)
    _validate_piececonst_mat(t2)
    return t1, t2


def r_sweep_token(r: float) -> str:
    """Stable path token for contrast ratio r (e.g. 1, 1.5, 10)."""
    return f"{float(r):g}"


def f_sweep_token(fv: float) -> str:
    """Stable path token for forcing f."""
    return f"{float(fv):g}"


# Defaults match ``generate_darcy.gen_coeff`` / piececonst paper setup.
DEFAULT_GRF_TAU = 3.0
DEFAULT_GRF_ALPHA = 2.0
DEFAULT_A_LOW = 3.0
DEFAULT_SIGMA_SCALE = 1.0


def _use_legacy_mat_name(
    tau: float, alpha: float, a_low: float, sigma_scale: float = DEFAULT_SIGMA_SCALE
) -> bool:
    """If True, use the historical filename without τ/α/a_low tokens (cache-compatible)."""
    return (
        abs(float(tau) - DEFAULT_GRF_TAU) < 1e-9
        and abs(float(alpha) - DEFAULT_GRF_ALPHA) < 1e-9
        and abs(float(a_low) - DEFAULT_A_LOW) < 1e-9
        and abs(float(sigma_scale) - DEFAULT_SIGMA_SCALE) < 1e-9
    )


def coeff_tag_suffix(
    tau: float,
    alpha: float,
    a_low: float,
    sigma_scale: float = DEFAULT_SIGMA_SCALE,
) -> str:
    """Append to checkpoint tags when GRF / ``a_low`` differ from piececonst defaults."""
    if _use_legacy_mat_name(tau, alpha, a_low, sigma_scale):
        return ""
    tt = f"{float(tau):g}".replace("-", "m")
    at = f"{float(alpha):g}".replace("-", "m")
    alt = f"{float(a_low):g}".replace("-", "m")
    sgt = f"{float(sigma_scale):g}".replace("-", "m")
    return f"_tau{tt}_a{at}_al{alt}_sg{sgt}"


def _mat_readable(path: str) -> bool:
    try:
        scipy.io.whosmat(path)
        return True
    except Exception:
        return False


def ensure_data(
    r: float,
    seed: int,
    n_samples: int = 1024,
    f: float = 1.0,
    *,
    tau: float = DEFAULT_GRF_TAU,
    alpha: float = DEFAULT_GRF_ALPHA,
    a_low: float = DEFAULT_A_LOW,
    sigma_scale: float = DEFAULT_SIGMA_SCALE,
) -> str:
    """Return path to (or generate) Darcy ``.mat`` under ``DATADIR``.

    Filenames use the legacy pattern (no τ/α/a_low/sigma in the name) when
    ``tau=3, alpha=2, a_low=3, sigma_scale=1`` so existing caches keep working;
    otherwise ``…_tau{t}_a{a}_al{a_low}_sg{sigma}_seed{seed}.mat`` disambiguates
    coefficient stats.
    """
    os.makedirs(DATADIR, exist_ok=True)
    rt = r_sweep_token(r)
    ft = f_sweep_token(f)
    if _use_legacy_mat_name(tau, alpha, a_low, sigma_scale):
        fname = f"darcy_N{N_FINE}_n{n_samples}_r{rt}_f{ft}_seed{seed}.mat"
    else:
        tt = f"{float(tau):g}".replace("-", "m")
        at = f"{float(alpha):g}".replace("-", "m")
        alt = f"{float(a_low):g}".replace("-", "m")
        sgt = f"{float(sigma_scale):g}".replace("-", "m")
        fname = (
            f"darcy_N{N_FINE}_n{n_samples}_r{rt}_f{ft}_"
            f"tau{tt}_a{at}_al{alt}_sg{sgt}_seed{seed}.mat"
        )
    fpath = os.path.join(DATADIR, fname)
    lock_path = fpath + ".lock"
    with open(lock_path, "w") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        try:
            if os.path.exists(fpath) and os.path.getsize(fpath) > 1_000_000:
                if _mat_readable(fpath):
                    print(f"  [data] Using existing {fname}", flush=True)
                    return fpath
                print(
                    f"  [data] Removing unreadable/truncated {fname}; regenerating …",
                    flush=True,
                )
                try:
                    os.remove(fpath)
                except OSError as e:
                    print(f"  [data] WARN could not remove bad file: {e}", flush=True)

            print(f"  [data] Generating {fname} …", flush=True)
            t0 = time.time()
            a = gen_coeff(
                N=N_FINE,
                n_samples=n_samples,
                r=r,
                seed=seed,
                tau=float(tau),
                alpha=float(alpha),
                a_low=float(a_low),
                sigma_scale=float(sigma_scale),
            )
            u = solve_darcy_batch(a, N_FINE, f=f)
            tmp = fpath + f".{os.getpid()}.tmp"
            scipy.io.savemat(
                tmp,
                {
                    "coeff": a.astype(np.float64),
                    "sol": u.astype(np.float64),
                    "f": float(f),
                    "tau": float(tau),
                    "alpha": float(alpha),
                    "a_low": float(a_low),
                    "sigma_scale": float(sigma_scale),
                },
                do_compression=True,
            )
            os.replace(tmp, fpath)
            print(f"  [data] Saved {fname}  ({time.time() - t0:.0f}s)", flush=True)
            return fpath
        finally:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


# ──────────────────────────────────────────────────────────────────
# Model (Table 2 / pretrain yaml: modes 20 on 61×61)
# ──────────────────────────────────────────────────────────────────


def _darcy_du_f_tensors(out: torch.Tensor, a: torch.Tensor, f_rhs: float):
    """Same Du, f as ``darcy_loss`` (strong residual vs forcing)."""
    batchsize = out.size(0)
    size = out.size(1)
    u = out.reshape(batchsize, size, size)
    a2 = a.reshape(batchsize, size, size)
    Du = FDM_Darcy(u, a2)
    f_tensor = torch.full(Du.shape, float(f_rhs), device=out.device, dtype=Du.dtype)
    return Du, f_tensor


def build_model(device):
    return FNO2d(
        modes1=[20, 20, 20, 20],
        modes2=[20, 20, 20, 20],
        layers=[64, 64, 64, 64, 64],
        fc_dim=128,
        act="gelu",
        pad_ratio=0.0,
    ).to(device)


def mean_test_rel_abs_errors(
    model,
    test_loader,
    device,
    lploss: LpLoss,
    mollifier: torch.Tensor,
) -> tuple[float, float]:
    """Batch-mean relative and absolute (mesh-weighted) L2 — matches ``LpLoss``."""
    model.eval()
    rel_errs: list[float] = []
    abs_errs: list[float] = []
    with torch.no_grad():
        for ic, u_true in test_loader:
            ic, u_true = ic.to(device), u_true.to(device)
            out = model(ic).squeeze(-1) * mollifier
            rel_errs.append(lploss(out, u_true).item())
            abs_errs.append(lploss.abs(out, u_true).item())
    return float(np.mean(rel_errs)), float(np.mean(abs_errs))


def mean_train_data_pde(
    model,
    train_u_loader,
    ic_loader,
    device,
    lploss: LpLoss,
    mollifier_u: torch.Tensor,
    mollifier_ic: torch.Tensor,
    f_rhs: float = 1.0,
) -> tuple[float, float]:
    """Full-pass means: data L2 on ``DarcyFlow`` grid; PDE residual on ``DarcyIC`` grid."""
    was_training = model.training
    model.eval()
    data_vals: list[float] = []
    pde_vals: list[float] = []
    with torch.no_grad():
        for ic, u_true in train_u_loader:
            ic, u_true = ic.to(device), u_true.to(device)
            out = model(ic).squeeze(-1) * mollifier_u
            data_vals.append(lploss(out, u_true).item())
        for ic in ic_loader:
            ic = ic.to(device)
            out = model(ic).squeeze(-1) * mollifier_ic
            a = ic[..., 0]
            pde_vals.append(darcy_loss(out, a, f_rhs).item())
    if was_training:
        model.train()
    return float(np.mean(data_vals)), float(np.mean(pde_vals))


def mean_train_viol_mean(
    model,
    train_u_loader,
    ic_loader,
    device,
    mollifier_u: torch.Tensor,
    mollifier_ic: torch.Tensor,
    f_rhs: float = 1.0,
    cons_item: str = "pde",
) -> float:
    """Mean per-sample constraint residual (``LpLoss.rel``), same as ALM ``h`` before slack."""
    lploss_vec = LpLoss(size_average=False, reduction=False)
    was_training = model.training
    model.eval()
    vals: list[float] = []
    with torch.no_grad():
        if str(cons_item).lower() == "pde":
            for ic in ic_loader:
                ic = ic.to(device)
                out = model(ic).squeeze(-1) * mollifier_ic
                a = ic[..., 0]
                Du, f_tensor = _darcy_du_f_tensors(out, a, f_rhs)
                vals.append(lploss_vec.rel(Du, f_tensor).mean().item())
        else:
            for ic, u_true in train_u_loader:
                ic, u_true = ic.to(device), u_true.to(device)
                out = model(ic).squeeze(-1) * mollifier_u
                vals.append(lploss_vec.rel(out, u_true).mean().item())
    if was_training:
        model.train()
    return float(np.mean(vals))


def print_run_metrics(
    label: str,
    model,
    train_u_loader,
    ic_loader,
    test_loader,
    device,
    lploss: LpLoss,
    mollifier_u: torch.Tensor,
    mollifier_ic: torch.Tensor,
    f_rhs: float = 1.0,
    t0: float | None = None,
    xy_loss: float = 5.0,
    f_loss: float = 1.0,
    cons_item: str | None = None,
) -> tuple[float, float, float, float]:
    """Print test + full-train ``data_loss`` / ``physics_loss`` (and PINO weighted sum)."""
    data_loss, physics_loss = mean_train_data_pde(
        model,
        train_u_loader,
        ic_loader,
        device,
        lploss,
        mollifier_u,
        mollifier_ic,
        f_rhs,
    )
    pino_loss = float(xy_loss) * data_loss + float(f_loss) * physics_loss
    viol_extra = ""
    if cons_item is not None:
        viol_m = mean_train_viol_mean(
            model,
            train_u_loader,
            ic_loader,
            device,
            mollifier_u,
            mollifier_ic,
            f_rhs,
            cons_item=cons_item,
        )
        viol_extra = f"  viol_mean={viol_m:.4f}"
    elapsed = f"  ({time.time() - t0:.0f}s)" if t0 is not None else ""
    if test_loader is not None:
        test_rel, test_abs = mean_test_rel_abs_errors(
            model, test_loader, device, lploss, mollifier_u
        )
        print(
            f"  → {label}  test_rel={test_rel:.4f}  test_abs={test_abs:.4g}  "
            f"data_loss={data_loss:.4f}  physics_loss={physics_loss:.4f}  "
            f"pino_loss={pino_loss:.4f}{viol_extra}{elapsed}",
            flush=True,
        )
        return test_rel, test_abs, data_loss, physics_loss
    print(
        f"  → {label}  data_loss={data_loss:.4f}  physics_loss={physics_loss:.4f}  "
        f"pino_loss={pino_loss:.4f}{viol_extra}{elapsed}",
        flush=True,
    )
    return float("nan"), float("nan"), data_loss, physics_loss


def mean_test_l2_l1_errors(
    model,
    test_loader,
    device,
    mollifier: torch.Tensor,
) -> tuple[float, float, float, float]:
    """One test pass: batch-mean rel/abs for L2 and L1 (same ``LpLoss`` weighting)."""
    l2 = LpLoss(size_average=True, p=2)
    l1 = LpLoss(size_average=True, p=1)
    model.eval()
    r2: list[float] = []
    a2: list[float] = []
    r1: list[float] = []
    a1: list[float] = []
    with torch.no_grad():
        for ic, u_true in test_loader:
            ic, u_true = ic.to(device), u_true.to(device)
            out = model(ic).squeeze(-1) * mollifier
            r2.append(l2(out, u_true).item())
            a2.append(l2.abs(out, u_true).item())
            r1.append(l1(out, u_true).item())
            a1.append(l1.abs(out, u_true).item())
    return (
        float(np.mean(r2)),
        float(np.mean(a2)),
        float(np.mean(r1)),
        float(np.mean(a1)),
    )


# ──────────────────────────────────────────────────────────────────
# Training (PINO: data + PDE on same 61×61 grid)
# ──────────────────────────────────────────────────────────────────


class _DatasetWithIndex(torch.utils.data.Dataset):
    """Return ``(ic, u, global_index)`` for per-sample ALM multipliers (data grid)."""

    def __init__(self, base: torch.utils.data.Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        ic, u = self.base[idx]
        return ic, u, idx


class _ICDatasetWithIndex(torch.utils.data.Dataset):
    """Return ``(ic, global_index)`` on the PDE / IC grid (``DarcyIC``)."""

    def __init__(self, base: torch.utils.data.Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        return self.base[idx], idx


def _lbfgs_chunk_slices(n_samples: int, n_chunks: int) -> list[slice]:
    """Split sample index [0, n) into ``n_chunks`` contiguous slices (for LBFGS grad accumulation)."""
    n_chunks = max(1, int(n_chunks))
    if n_chunks <= 1 or n_samples <= 0:
        return [slice(0, n_samples)]
    base, rem = divmod(n_samples, n_chunks)
    out: list[slice] = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        out.append(slice(start, start + size))
        start += size
    return out


def alm(
    cons_item: str = "pde",
    uncon_weight: float = 1.0,
    mu: float = 2.0,
    rho: float = 1.1,
    outer_iters: int = 20,
    inner_step: int = 200,
) -> dict[str, Any] | None:
    """ALM on full training set (default ``alm_inner=adam``: indexed mini-batch inner).

    ``cons_item=pde``: unconstrained = ``uncon_weight×xy_loss×data`` (weight tunable);
    constrain per-sample ``LpLoss.rel(Du,f) ≤ ε`` (``ε = slack×h_pretrain``, default slack=0).
    """
    fr = inspect.currentframe()
    try:
        loc = fr.f_back.f_locals
    finally:
        del fr

    model = loc.get("model")
    device = loc.get("device")
    train_u_loader = loc.get("train_u_loader") or loc.get("train_loader")
    ic_loader = loc.get("ic_loader")
    ds_train_u = loc.get("ds_train_u") or loc.get("ds_train")
    ds_train_ic = loc.get("ds_train_ic")
    n_samples = int(loc.get("n_samples") or 0)
    f_rhs = loc.get("f_rhs", 1.0)
    base_lr = loc.get("base_lr", 1e-3)
    mollifier_u = loc.get("mollifier_u")
    if mollifier_u is None:
        mollifier_u = loc.get("mollifier")
    mollifier_ic = loc.get("mollifier_ic")
    if mollifier_ic is None:
        mollifier_ic = loc.get("mollifier")
    lploss = loc.get("lploss")
    xy_w = float(loc.get("xy_loss", 5.0))
    f_w = float(loc.get("f_loss", 1.0))
    test_loader = loc.get("test_loader")
    alm_ckpt_path = loc.get("alm_ckpt_path")
    inner_solver = str(loc.get("alm_inner", "adam")).lower().strip()
    lbfgs_max_iter = int(loc.get("alm_lbfgs_max_iter", 200))
    _lbfgs_chunks_arg = int(loc.get("alm_lbfgs_chunks", 0))
    pde_eps_slack = float(loc.get("alm_pde_eps_slack", 0.0))
    data_eps_slack = float(loc.get("alm_data_eps_slack", 0.0))
    _adam_lr_raw = loc.get("alm_adam_lr", None)
    adam_inner_lr = float(_adam_lr_raw) if _adam_lr_raw is not None else float(base_lr) * 0.1
    adam_inner_warmup = float(loc.get("alm_adam_warmup_frac", 0.2))
    adam_inner_cosine = bool(loc.get("alm_adam_cosine", True))

    if (
        model is None
        or device is None
        or train_u_loader is None
        or ic_loader is None
        or mollifier_u is None
        or mollifier_ic is None
        or lploss is None
    ):
        return None
    if ds_train_u is not None:
        n_samples = len(ds_train_u)
    elif n_samples <= 0:
        n_samples = len(train_u_loader.dataset)

    # Resolve LBFGS chunk count.
    # 0/negative -> auto: 1 chunk per ≤500 samples (n=500→1, n=1000→2, n=2000→4)
    if _lbfgs_chunks_arg <= 0:
        lbfgs_chunks = max(1, (n_samples + 499) // 500)
    else:
        lbfgs_chunks = max(1, _lbfgs_chunks_arg)

    ci = str(cons_item).lower().strip()
    if ci not in ("data", "pde"):
        raise ValueError("cons_item must be 'data' or 'pde'")

    lploss_vec = LpLoss(
        d=float(getattr(lploss, "d", 2)),
        p=float(getattr(lploss, "p", 2)),
        size_average=False,
        reduction=False,
    )

    def pde_per_sample(out: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        Du, f_tensor = _darcy_du_f_tensors(out, a, f_rhs)
        return lploss_vec.rel(Du, f_tensor)

    def data_per_sample(out: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
        return lploss_vec.rel(out, u_true)

    def uncon_scalar(l_data: torch.Tensor, l_pde: torch.Tensor) -> torch.Tensor:
        if ci == "pde":
            return float(uncon_weight) * xy_w * l_data
        return float(uncon_weight) * f_w * l_pde

    mu_cur = float(mu)
    cell_t0 = loc.get("cell_t0")
    inner_loss_trace: list[float] = []
    outer_test_l2: list[float] = []
    lam = torch.zeros(1)
    opt_alm = None
    alm_save_best = bool(loc.get("alm_save_best", False))
    best_test = float("inf")
    best_outer = 0
    best_state: dict[str, torch.Tensor] | None = None

    if ci == "pde":
        eff_uncon = f"data×{float(uncon_weight) * xy_w:g}"
    else:
        eff_uncon = f"pde×{float(uncon_weight) * f_w:g}"

    inner_desc = (
        f"inner_steps={inner_step}/outer"
        if inner_solver == "adam"
        else (
            f"lbfgs_max_iter={lbfgs_max_iter}  chunks={lbfgs_chunks}  "
            f"(grad-accum ~{math.ceil(n_samples / lbfgs_chunks)}/chunk)"
        )
    )
    print(
        f"  [ALM] cons={ci}  inner={inner_solver}  {inner_desc}  "
        f"unconstrained={eff_uncon}  n_constraints={n_samples}  "
        f"mu0={mu}  rho={rho}  outer={outer_iters}  pde_eps_slack={pde_eps_slack}",
        flush=True,
    )

    if inner_solver == "lbfgs":
        if ds_train_u is None or ds_train_ic is None:
            raise ValueError("alm_inner=lbfgs requires ds_train_u and ds_train_ic in train_one")
        all_ic_u = torch.stack([ds_train_u[i][0] for i in range(n_samples)]).to(device)
        all_u = torch.stack([ds_train_u[i][1] for i in range(n_samples)]).to(device)
        all_ic_pde = torch.stack([ds_train_ic[i] for i in range(n_samples)]).to(device)
        # When sub == pde_sub, DarcyFlow[i][0] and DarcyIC[i] yield identical ic; both
        # mollifiers come from the same grid. Detecting equality lets the closure run
        # a single forward (saves ~2× VRAM and wall on the full-batch LBFGS path).
        _alm_lbfgs_single_fwd = (
            all_ic_u.shape == all_ic_pde.shape
            and torch.equal(all_ic_u, all_ic_pde)
            and mollifier_u.shape == mollifier_ic.shape
            and torch.equal(mollifier_u, mollifier_ic)
        )
        if _alm_lbfgs_single_fwd:
            print(
                "  [ALM] LBFGS path: sub==pde_sub detected → single forward reuse "
                "(out_u≡out_p) for halved VRAM/wall",
                flush=True,
            )
        lam = torch.zeros(n_samples, device=device)

        with torch.no_grad():
            if ci == "pde":
                out0 = model(all_ic_pde).squeeze(-1) * mollifier_ic
                h0 = pde_per_sample(out0, all_ic_pde[..., 0]).detach()
                slack = float(pde_eps_slack)
                eps_vec = slack * h0 if slack > 0 else torch.zeros_like(h0)
            else:
                out0 = model(all_ic_u).squeeze(-1) * mollifier_u
                h0 = data_per_sample(out0, all_u).detach()
                slack = float(data_eps_slack)
                eps_vec = slack * h0 if slack > 0 else torch.zeros_like(h0)
            print(
                f"  [ALM] pretrain viol_mean={h0.mean():.5g}  "
                f"ε slack={slack:g}  ε_mean={eps_vec.mean():.5g}",
                flush=True,
            )

        for i in range(int(outer_iters)):
            mu_k = float(mu_cur)
            lam_k = lam.detach().clone()
            print(
                f"  [ALM] outer {i + 1}/{outer_iters}  start  μ={mu_k:.6g}  "
                f"(L-BFGS max_iter={lbfgs_max_iter}, n={n_samples}) …",
                flush=True,
            )
            t_outer = time.time()
            model.train()
            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=1.0,
                max_iter=lbfgs_max_iter,
                history_size=50,
                line_search_fn="strong_wolfe",
            )
            chunk_slices = _lbfgs_chunk_slices(n_samples, lbfgs_chunks)
            n_chunk = len(chunk_slices)

            def _alm_loss_on_slice(
                sl: slice, _lam: torch.Tensor, _mu: float, _eps: torch.Tensor
            ) -> torch.Tensor:
                ic_u = all_ic_u[sl]
                u = all_u[sl]
                ic_p = all_ic_pde[sl]
                if _alm_lbfgs_single_fwd:
                    out_u = model(ic_u).squeeze(-1) * mollifier_u
                    out_p = out_u  # identical ic & mollifier → reuse
                else:
                    out_u = model(ic_u).squeeze(-1) * mollifier_u
                    out_p = model(ic_p).squeeze(-1) * mollifier_ic
                l_data = lploss(out_u, u)
                a_p = ic_p[..., 0]
                l_pde = darcy_loss(out_p, a_p, f_rhs)
                if ci == "pde":
                    viol = pde_per_sample(out_p, a_p) - _eps[sl]
                else:
                    viol = data_per_sample(out_u, u) - _eps[sl]
                lam_sl = _lam[sl]
                return (
                    uncon_scalar(l_data, l_pde)
                    + torch.mean(lam_sl * viol)
                    + 0.5 * _mu * torch.mean(viol**2)
                )

            def closure(_lam=lam_k, _mu=mu_k, _eps=eps_vec):
                lbfgs.zero_grad()
                last_loss: torch.Tensor | None = None
                for sl in chunk_slices:
                    loss_j = _alm_loss_on_slice(sl, _lam, _mu, _eps) / n_chunk
                    loss_j.backward()
                    last_loss = loss_j
                if last_loss is None:
                    raise RuntimeError("LBFGS closure: empty chunk list")
                return last_loss

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                loss_inner = float(lbfgs.step(closure))
            except Exception as e:
                print(f"  [ALM] WARN L-BFGS failed at outer {i + 1}: {e}", flush=True)
                break
            inner_loss_trace.append(loss_inner)
            print(
                f"  [ALM] outer {i + 1}/{outer_iters}  L-BFGS done  "
                f"loss={loss_inner:.6g}  wall={time.time() - t_outer:.1f}s",
                flush=True,
            )

            with torch.no_grad():
                if ci == "pde":
                    out_p = model(all_ic_pde).squeeze(-1) * mollifier_ic
                    viol = pde_per_sample(out_p, all_ic_pde[..., 0]) - eps_vec
                else:
                    out_u = model(all_ic_u).squeeze(-1) * mollifier_u
                    viol = data_per_sample(out_u, all_u) - eps_vec
                lam = lam + mu_k * viol.detach()

            te, _, data_loss, physics_loss = print_run_metrics(
                f"ALM outer {i + 1}/{outer_iters}",
                model,
                train_u_loader,
                ic_loader,
                test_loader,
                device,
                lploss,
                mollifier_u,
                mollifier_ic,
                float(f_rhs),
                cell_t0,
                xy_w,
                f_w,
            )
            if test_loader is not None and not math.isnan(te):
                outer_test_l2.append(te)
                if te < best_test:
                    best_test = te
                    best_outer = i + 1
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    print(
                        f"  [ALM] ★ best outer {best_outer}/{outer_iters}  test_rel={best_test:.4f}",
                        flush=True,
                    )
            print(
                f"    [ALM] outer {i + 1}/{outer_iters}  "
                f"data_loss={data_loss:.6g}  physics_loss={physics_loss:.6g}  "
                f"viol_mean={float(viol.mean()):.6g}",
                flush=True,
            )

            lam_np = lam.detach().cpu().numpy()
            if lam_np.size <= 16:
                lam_str = str(lam_np)
            else:
                lam_str = (
                    f"mean={float(np.mean(lam_np)):.5g} std={float(np.std(lam_np)):.5g} "
                    f"min={float(np.min(lam_np)):.5g} max={float(np.max(lam_np)):.5g} len={lam_np.size}"
                )
            print(
                f"    [ALM] outer {i + 1}/{outer_iters}  lambda={lam_str}  mu={mu_cur:.6g}",
                flush=True,
            )
            mu_cur *= float(rho)

    else:
        if ds_train_u is None:
            ds_train_u = train_u_loader.dataset
        if ds_train_ic is None:
            ds_train_ic = ic_loader.dataset
        alm_u_loader = torch.utils.data.DataLoader(
            _DatasetWithIndex(ds_train_u),
            batch_size=train_u_loader.batch_size,
            shuffle=True,
            drop_last=False,
        )
        alm_ic_loader = torch.utils.data.DataLoader(
            _ICDatasetWithIndex(ds_train_ic),
            batch_size=ic_loader.batch_size,
            shuffle=True,
            drop_last=False,
        )
        lam = torch.zeros(n_samples, device=device)
        eps_vec = torch.zeros(n_samples, device=device)
        slack = float(pde_eps_slack if ci == "pde" else data_eps_slack)
        h0_sum = 0.0
        h0_cnt = 0
        with torch.no_grad():
            if ci == "pde":
                for ic, idx in alm_ic_loader:
                    ic = ic.to(device)
                    idx = idx.to(device)
                    out = model(ic).squeeze(-1) * mollifier_ic
                    h0 = pde_per_sample(out, ic[..., 0])
                    h0_sum += float(h0.sum().item())
                    h0_cnt += int(h0.numel())
                    if slack > 0:
                        eps_vec[idx] = slack * h0
            else:
                for ic, u_true, idx in alm_u_loader:
                    ic, u_true = ic.to(device), u_true.to(device)
                    idx = idx.to(device)
                    out = model(ic).squeeze(-1) * mollifier_u
                    h0 = data_per_sample(out, u_true)
                    h0_sum += float(h0.sum().item())
                    h0_cnt += int(h0.numel())
                    if slack > 0:
                        eps_vec[idx] = slack * h0
        print(
            f"  [ALM] pretrain viol_mean={h0_sum / max(h0_cnt, 1):.5g}  "
            f"ε slack={slack:g}  ε_mean={eps_vec.mean():.5g}",
            flush=True,
        )

        opt_alm = None  # set per-outer below
        u_iter = iter(alm_u_loader)
        ic_iter = iter(alm_ic_loader)
        model.train()
        print(
            f"  [ALM] Adam inner: lr={adam_inner_lr:g}  "
            f"warmup_frac={adam_inner_warmup:g}  cosine={adam_inner_cosine}  "
            f"per-sample λ (n={n_samples}), decoupled data/PDE loaders  "
            f"(opt+sch reset per outer)",
            flush=True,
        )

        for i in range(int(outer_iters)):
            mu_k = float(mu_cur)
            print(
                f"  [ALM] outer {i + 1}/{outer_iters}  start  μ={mu_k:.6g}  "
                f"(Adam inner_steps={inner_step}, batch={alm_u_loader.batch_size}) …",
                flush=True,
            )
            t_outer = time.time()
            opt_alm = torch.optim.Adam(model.parameters(), lr=adam_inner_lr)
            if adam_inner_cosine and int(inner_step) > 1:
                _warmup_end = max(1, int(int(inner_step) * adam_inner_warmup))
                def _lr_lambda(step, we=_warmup_end, ns=int(inner_step)):
                    if step < we:
                        return 1.0
                    p = (step - we) / max(1, ns - we)
                    return 0.5 * (1.0 + math.cos(math.pi * p))
                sch_alm = torch.optim.lr_scheduler.LambdaLR(opt_alm, _lr_lambda)
            else:
                sch_alm = None
            for _ in range(int(inner_step)):
                try:
                    ic_d, u_true, _idx_d = next(u_iter)
                except StopIteration:
                    u_iter = iter(alm_u_loader)
                    ic_d, u_true, _idx_d = next(u_iter)
                ic_d, u_true = ic_d.to(device), u_true.to(device)
                out_d = model(ic_d).squeeze(-1) * mollifier_u
                l_data = lploss(out_d, u_true)

                try:
                    ic_p, idx_p = next(ic_iter)
                except StopIteration:
                    ic_iter = iter(alm_ic_loader)
                    ic_p, idx_p = next(ic_iter)
                ic_p = ic_p.to(device)
                idx_p = idx_p.to(device)
                out_p = model(ic_p).squeeze(-1) * mollifier_ic
                a_p = ic_p[..., 0]
                l_pde = darcy_loss(out_p, a_p, f_rhs)
                h = (
                    pde_per_sample(out_p, a_p)
                    if ci == "pde"
                    else data_per_sample(out_d, u_true)
                )
                h = h - eps_vec[idx_p]
                lam_b = lam[idx_p].detach()
                loss_alm = (
                    uncon_scalar(l_data, l_pde)
                    + torch.mean(lam_b * h)
                    + 0.5 * mu_k * torch.mean(h**2)
                )
                # !!! test for only penalty on PDE loss
                # loss_alm = (
                #     uncon_scalar(l_data, l_pde)
                #     # + torch.mean(lam_b * h)
                #     + 0.5 * mu_k * l_pde
                # )
                inner_loss_trace.append(float(loss_alm.detach().cpu()))
                opt_alm.zero_grad()
                loss_alm.backward()
                opt_alm.step()
                if sch_alm is not None:
                    sch_alm.step()

            viol_sum = 0.0
            viol_count = 0
            with torch.no_grad():
                if ci == "pde":
                    for ic, idx in alm_ic_loader:
                        ic = ic.to(device)
                        idx = idx.to(device)
                        out = model(ic).squeeze(-1) * mollifier_ic
                        h = pde_per_sample(out, ic[..., 0]) - eps_vec[idx]
                        lam[idx] = lam[idx] + mu_k * h
                        viol_sum += float(h.sum().item())
                        viol_count += int(h.numel())
                else:
                    for ic, u_true, idx in alm_u_loader:
                        ic, u_true = ic.to(device), u_true.to(device)
                        idx = idx.to(device)
                        out = model(ic).squeeze(-1) * mollifier_u
                        h = data_per_sample(out, u_true) - eps_vec[idx]
                        lam[idx] = lam[idx] + mu_k * h
                        viol_sum += float(h.sum().item())
                        viol_count += int(h.numel())
            viol_mean = viol_sum / max(viol_count, 1)

            print(
                f"  [ALM] outer {i + 1}/{outer_iters}  Adam done  "
                f"wall={time.time() - t_outer:.1f}s",
                flush=True,
            )
            te, _, data_loss, physics_loss = print_run_metrics(
                f"ALM outer {i + 1}/{outer_iters}",
                model,
                train_u_loader,
                ic_loader,
                test_loader,
                device,
                lploss,
                mollifier_u,
                mollifier_ic,
                float(f_rhs),
                cell_t0,
                xy_w,
                f_w,
            )
            if test_loader is not None and not math.isnan(te):
                outer_test_l2.append(te)
                if te < best_test:
                    best_test = te
                    best_outer = i + 1
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    print(
                        f"  [ALM] ★ best outer {best_outer}/{outer_iters}  test_rel={best_test:.4f}",
                        flush=True,
                    )
            print(
                f"    [ALM] outer {i + 1}/{outer_iters}  "
                f"data_loss={data_loss:.6g}  physics_loss={physics_loss:.6g}  "
                f"viol_mean={viol_mean:.6g}",
                flush=True,
            )

            lam_np = lam.detach().cpu().numpy()
            if lam_np.size <= 16:
                lam_str = str(lam_np)
            else:
                lam_str = (
                    f"mean={float(np.mean(lam_np)):.5g} std={float(np.std(lam_np)):.5g} "
                    f"min={float(np.min(lam_np)):.5g} max={float(np.max(lam_np)):.5g} len={lam_np.size}"
                )
            print(
                f"    [ALM] outer {i + 1}/{outer_iters}  lambda={lam_str}  mu={mu_cur:.6g}",
                flush=True,
            )
            mu_cur *= float(rho)

    if best_state is not None and alm_save_best:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(
            f"  [ALM] restored best outer {best_outer} weights (test_rel={best_test:.4f}) for save",
            flush=True,
        )
    elif best_outer > 0 and not alm_save_best and not math.isnan(best_test):
        print(
            f"  [ALM] note: best test_rel={best_test:.4f} at outer {best_outer} "
            f"(saving final-outer weights; pass --alm_save_best to restore best)",
            flush=True,
        )

    last50 = inner_loss_trace[-50:] if len(inner_loss_trace) >= 50 else list(inner_loss_trace)
    out: dict[str, Any] = {
        "inner_loss_trace": inner_loss_trace,
        "last50_inner_loss": last50,
        "n_inner_steps": len(inner_loss_trace),
        "outer_test_l2": outer_test_l2,
    }

    if alm_ckpt_path:
        try:
            d = os.path.dirname(os.path.abspath(alm_ckpt_path))
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = alm_ckpt_path + ".tmp"
            ckpt: dict[str, Any] = {
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "best_outer": int(best_outer),
                "best_test_rel": float(best_test) if best_outer > 0 else None,
                "lam": lam.detach().cpu(),
                "mu": float(mu_cur),
                "cons_item": ci,
                "alm_inner": inner_solver,
                "uncon_weight": float(uncon_weight),
                "xy_loss": xy_w,
                "f_loss": f_w,
                "f_rhs": float(f_rhs),
                "last50_inner_loss": last50,
                "inner_loss_trace": inner_loss_trace,
                "outer_iters": int(outer_iters),
                "inner_step": int(inner_step),
            }
            if opt_alm is not None:
                ckpt["opt_alm_state"] = opt_alm.state_dict()
            torch.save(ckpt, tmp)
            os.replace(tmp, alm_ckpt_path)
            print(f"  [ALM] saved checkpoint  {alm_ckpt_path}", flush=True)
        except Exception as e:
            print(f"  [ALM] WARN save ckpt: {e}", flush=True)

    return out



def nncg(
    nncg_steps: int = 5,
    lr: float = 1.0,
    rank: int = 10,
    mu: float = 0.01,
    precond_update_freq: int = 1,
    cg_tol: float = 1e-5,
    cg_max_iters: int = 200,
) -> dict[str, Any] | None:
    """Post-train Nyström–Newton-CG. Set env ``NNCG_FB_CHUNK`` (e.g. 50–150) to use
    ``ChunkedNysNewtonCG`` (chunked full-batch HVP) like ``darcy_sweep_adam_nncg.py``."""
    fr = inspect.currentframe()
    try:
        loc = fr.f_back.f_locals
    finally:
        del fr

    model = loc.get("model")
    device = loc.get("device")
    ds_train_u = loc.get("ds_train_u") or loc.get("ds_train")
    ds_train_ic = loc.get("ds_train_ic")
    n_samples = loc.get("n_samples")
    f_rhs = loc.get("f_rhs", 1.0)
    xy_loss = loc.get("xy_loss", 5.0)
    f_loss = loc.get("f_loss", 1.0)
    mollifier_u = loc.get("mollifier_u")
    if mollifier_u is None:
        mollifier_u = loc.get("mollifier")
    mollifier_ic = loc.get("mollifier_ic")
    if mollifier_ic is None:
        mollifier_ic = loc.get("mollifier")
    lploss = loc.get("lploss")

    nncg_steps = int(loc.get("nncg_steps", nncg_steps))

    if (
        model is None
        or device is None
        or ds_train_u is None
        or ds_train_ic is None
        or mollifier_u is None
        or mollifier_ic is None
        or lploss is None
        or n_samples is None
    ):
        return None

    ic_all_u = torch.stack([ds_train_u[i][0] for i in range(int(n_samples))]).to(device)
    u_all = torch.stack([ds_train_u[i][1] for i in range(int(n_samples))]).to(device)
    ic_all_pde = torch.stack([ds_train_ic[i] for i in range(int(n_samples))]).to(device)

    test_loader = loc.get("test_loader")

    def _nncg_maybe_print_test_err(step_1based: int) -> None:
        """Print test rel-L2 after **each** NNCG outer iteration."""
        tl = test_loader
        if tl is None:
            return
        model.eval()
        errs: list[float] = []
        with torch.no_grad():
            for ic, u_true in tl:
                ic = ic.to(device)
                u_true = u_true.to(device)
                out = model(ic).squeeze(-1) * mollifier_u
                errs.append(lploss(out, u_true).item())
        model.train()
        te = float(np.mean(errs)) if errs else float("nan")
        print(
            f"  [NNCG] iter {step_1based}/{nncg_steps}  test_error={te:.6g}",
            flush=True,
        )

    fb_chunk = int(os.environ.get("NNCG_FB_CHUNK", "0"))
    dl_fn = partial(darcy_loss, f_rhs=float(f_rhs))

    def _nncg_real_loss(loss: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(loss):
            return torch.real(loss)
        return loss

    def _nncg_real_grad_tuple(grad_tuple):
        out = []
        for g in grad_tuple:
            if g is None:
                out.append(None)
            elif torch.is_complex(g):
                out.append(torch.real(g))
            else:
                out.append(g)
        return tuple(out)

    rank = int(os.environ.get("NNCG_RANK", str(rank)))
    chunk_size = max(1, int(os.environ.get("NNCG_CHUNK_SIZE", "1")))

    if fb_chunk > 0:
        from scripts.nys_newton_cg_chunked import (
            ChunkedNysNewtonCG,
            make_chunked_grad_fn,
            make_chunked_hvp_fn,
            make_chunked_loss_fn,
        )

        params_list = list(model.parameters())
        opt_n = ChunkedNysNewtonCG(
            model.parameters(),
            lr=float(lr),
            rank=int(rank),
            mu=float(mu),
            chunk_size=int(chunk_size),
            cg_tol=float(cg_tol),
            cg_max_iters=int(cg_max_iters),
            line_search_fn="armijo",
            verbose=False,
        )
        _gfn = make_chunked_grad_fn(
            model,
            ic_all_u,
            u_all,
            mollifier_u,
            lploss,
            dl_fn,
            float(xy_loss),
            float(f_loss),
            params_list,
            fb_chunk,
            loss_mode="pino",
        )
        _hfn = make_chunked_hvp_fn(
            model,
            ic_all_u,
            u_all,
            mollifier_u,
            lploss,
            dl_fn,
            float(xy_loss),
            float(f_loss),
            params_list,
            fb_chunk,
            loss_mode="pino",
        )
        _lfn = make_chunked_loss_fn(
            model,
            ic_all_u,
            u_all,
            mollifier_u,
            lploss,
            dl_fn,
            float(xy_loss),
            float(f_loss),
            fb_chunk,
            loss_mode="pino",
        )
        opt_n.attach_callbacks(grad_fn=_gfn, hvp_fn=_hfn, loss_fn=_lfn)

        print(
            f"  [NNCG] Chunked Nyström–CG  steps={nncg_steps}  lr={lr}  rank={rank}  mu={mu}  "
            f"fb_chunk={fb_chunk}  functorch_chunk={chunk_size}  precond_every={precond_update_freq}",
            flush=True,
        )

        model.train()
        pf = max(1, int(precond_update_freq))
        loss_history: list[float] = []
        for k in range(int(nncg_steps)):
            if k % pf == 0:
                opt_n.update_preconditioner_chunked()
            opt_n.step_chunked()
            if torch.cuda.is_available() and (k % 5 == 0):
                torch.cuda.empty_cache()
            with torch.no_grad():
                out_u = model(ic_all_u).squeeze(-1) * mollifier_u
                l_data = lploss(out_u, u_all)
                out_p = model(ic_all_pde).squeeze(-1) * mollifier_ic
                l_pde = darcy_loss(out_p, ic_all_pde[..., 0], f_rhs)
                loss_t = _nncg_real_loss(xy_loss * l_data + f_loss * l_pde)
            loss_history.append(float(loss_t.detach().cpu()))
            step = k + 1
            _nncg_maybe_print_test_err(step)

        return {"loss_history": loss_history, "nncg_steps": int(nncg_steps)}

    from scripts.nys_newton_cg import NysNewtonCG

    opt_n = NysNewtonCG(
        model.parameters(),
        lr=float(lr),
        rank=int(rank),
        mu=float(mu),
        chunk_size=int(chunk_size),
        cg_tol=float(cg_tol),
        cg_max_iters=int(cg_max_iters),
        line_search_fn="armijo",
        verbose=False,
    )

    def closure():
        opt_n.zero_grad(set_to_none=True)
        out_u = model(ic_all_u).squeeze(-1) * mollifier_u
        l_data = lploss(out_u, u_all)
        out_p = model(ic_all_pde).squeeze(-1) * mollifier_ic
        l_pde = darcy_loss(out_p, ic_all_pde[..., 0], f_rhs)
        loss = _nncg_real_loss(xy_loss * l_data + f_loss * l_pde)
        grad_tuple = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        return loss, _nncg_real_grad_tuple(grad_tuple)

    print(
        f"  [NNCG] Nyström–CG (dense graph)  steps={nncg_steps}  lr={lr}  rank={rank}  mu={mu}  "
        f"chunk_size={chunk_size}  precond_every={precond_update_freq}  "
        f"(set NNCG_FB_CHUNK>0 for chunked HVP)",
        flush=True,
    )

    model.train()
    pf = max(1, int(precond_update_freq))
    loss_history = []
    for k in range(int(nncg_steps)):
        if k % pf == 0:
            _, grad_tuple = closure()
            opt_n.update_preconditioner(grad_tuple)
        opt_n.step(closure)
        with torch.no_grad():
            out_u = model(ic_all_u).squeeze(-1) * mollifier_u
            l_data = lploss(out_u, u_all)
            out_p = model(ic_all_pde).squeeze(-1) * mollifier_ic
            l_pde = darcy_loss(out_p, ic_all_pde[..., 0], f_rhs)
            loss_t = _nncg_real_loss(xy_loss * l_data + f_loss * l_pde)
        loss_history.append(float(loss_t.detach().cpu()))
        step = k + 1
        _nncg_maybe_print_test_err(step)

    return {"loss_history": loss_history, "nncg_steps": int(nncg_steps)}


def cl(init_f: float = 0.01, delta_f: float = 0.1, target_f: float | None = None, inner_step: int = 500):
    """Pre-Adam curriculum on forcing ``f``: increase from ``init_f`` by ``delta_f`` until reaching ``target_f``.

    Each stage trains at ``f_stage = min(cur_f, target)`` so the **final** stage always reaches ``target_f``.
    Mini-batch sampling uses **decoupled** ``DarcyFlow`` (data) and ``DarcyIC`` (PDE) loaders, matching
    ``train_one`` Adam / ALM (independent random batches per loss term).
    Optional caller locals (from ``train_one``): ``cl_progress_ckpt_path`` — save/resume **within** the CL column
    (between curriculum stages and every ``cl_ckpt_every`` inner steps); cleared by ``train_one`` after success.
    """
    fr = inspect.currentframe()
    try:
        loc = fr.f_back.f_locals
    finally:
        del fr

    model = loc.get("model")
    device = loc.get("device")
    train_path = loc.get("train_path")
    n_samples = loc.get("n_samples")
    optimizer = loc.get("optimizer")
    batchsize = loc.get("batchsize", 32)
    f_loss = loc.get("f_loss", 1.0)
    xy_loss = loc.get("xy_loss", 5.0)
    cl_progress_ckpt_path = loc.get("cl_progress_ckpt_path")
    cl_ckpt_every = max(50, int(loc.get("cl_ckpt_every", 500)))

    init_f = float(loc.get("cl_init_f", init_f))
    delta_f = float(loc.get("cl_delta_f", delta_f))
    inner_step = int(loc.get("cl_inner_step", inner_step))
    tf_ov = loc.get("cl_target_f", target_f)

    if model is None or device is None or not train_path or optimizer is None:
        return

    r_ov = loc.get("darcy_r")
    if r_ov is not None:
        r = float(r_ov)
    else:
        mpath = re.search(r"_r([^_]+)_f([^_]+)_seed", train_path)
        if not mpath:
            print("  [CL] skip: cannot parse r,f from train_path", flush=True)
            return
        r = float(mpath.group(1).replace("p", "."))
    tau_cl = float(loc.get("tau", DEFAULT_GRF_TAU))
    alpha_cl = float(loc.get("alpha", DEFAULT_GRF_ALPHA))
    a_low_cl = float(loc.get("a_low", DEFAULT_A_LOW))
    pool_n = int(n_samples)

    sweep_seed = int(loc.get("seed", 0))
    target = float(tf_ov) if tf_ov is not None else float(loc.get("f_rhs", 1.0))
    cur_f = float(init_f)
    inner_done = 0

    if cl_progress_ckpt_path and os.path.isfile(cl_progress_ckpt_path):
        try:
            pc = _load_torch_ckpt(cl_progress_ckpt_path, device)
            cur_f = float(pc["cur_f"])
            inner_done = int(pc.get("inner_done", 0))
            ms = pc.get("model_state")
            if ms is None:
                raise KeyError("model_state")
            model.load_state_dict(ms)
            optimizer.load_state_dict(pc["opt_state"])
            if "torch_rng" in pc:
                torch.set_rng_state(pc["torch_rng"])
            if pc.get("numpy_rng") is not None:
                np.random.set_state(pc["numpy_rng"])
            if torch.cuda.is_available() and pc.get("cuda_rng") is not None:
                try:
                    torch.cuda.set_rng_state_all(pc["cuda_rng"])
                except Exception:
                    pass
            print(
                f"  [CL resume] cur_f={cur_f:g}  inner_done={inner_done}/{inner_step}",
                flush=True,
            )
        except Exception as e:
            print(f"  [CL resume] WARN load failed: {e}; starting curriculum from scratch", flush=True)
            cur_f = float(init_f)
            inner_done = 0

    def _save_cl_column_progress(inner_done_save: int) -> None:
        if not cl_progress_ckpt_path:
            return
        try:
            d = os.path.dirname(os.path.abspath(cl_progress_ckpt_path))
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = cl_progress_ckpt_path + ".tmp"
            torch.save(
                {
                    "version": 2,
                    "cur_f": float(cur_f),
                    "inner_done": int(inner_done_save),
                    "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "opt_state": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "numpy_rng": np.random.get_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None,
                },
                tmp,
            )
            os.replace(tmp, cl_progress_ckpt_path)
        except Exception as e:
            print(f"  [CL ckpt] WARN save: {e}", flush=True)

    print(
        f"  [CL] f-curriculum  init_f={init_f:g}  delta_f={delta_f}  target_f={target}  "
        f"inner_step={inner_step}  (decoupled DarcyFlow + DarcyIC loaders)",
        flush=True,
    )

    pde_sub_use = int(loc.get("pde_sub_use", PDE_SUB))
    lp = LpLoss(size_average=True)

    while True:
        f_use = min(cur_f, target)
        tp = ensure_data(
            r,
            seed=sweep_seed,
            n_samples=pool_n,
            f=f_use,
            tau=tau_cl,
            alpha=alpha_cl,
            a_low=a_low_cl,
        )
        ds_tr_u = DarcyFlow(tp, nx=N_FINE, sub=SUB, offset=0, num=n_samples)
        ds_tr_ic = DarcyIC(tp, nx=N_FINE, sub=pde_sub_use, offset=0, num=n_samples)
        u_loader = torch.utils.data.DataLoader(
            ds_tr_u, batch_size=batchsize, shuffle=True, drop_last=False
        )
        ic_loader = torch.utils.data.DataLoader(
            ds_tr_ic, batch_size=batchsize, shuffle=True, drop_last=False
        )
        u_iter = sample_data(u_loader)
        ic_iter = sample_data(ic_loader)
        mollifier_u = _darcy_mollifier(ds_tr_u.mesh, device)
        mollifier_ic = _darcy_mollifier(ds_tr_ic.mesh, device)
        model.train()
        loss_trace = loc.get("cl_final_loss_trace")
        final_stage = f_use >= target - 1e-7
        resume_inner = int(inner_done)
        inner_done = 0

        for inner_i in range(resume_inner, int(inner_step)):
            if xy_loss > 0:
                ic_d, u_true = next(u_iter)
                ic_d, u_true = ic_d.to(device), u_true.to(device)
                out_d = model(ic_d).squeeze(-1) * mollifier_u
                l_data = lp(out_d, u_true)
            else:
                l_data = torch.zeros(1, device=device)

            if f_loss > 0:
                ic_p = next(ic_iter)
                ic_p = ic_p.to(device)
                out_p = model(ic_p).squeeze(-1) * mollifier_ic
                l_pde = darcy_loss(out_p, ic_p[..., 0], f_use)
            else:
                l_pde = torch.zeros(1, device=device)

            loss = xy_loss * l_data + f_loss * l_pde
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss_trace is not None and final_stage:
                loss_trace.append(float(loss.detach().cpu()))
            done_cnt = inner_i + 1
            if cl_progress_ckpt_path and done_cnt % cl_ckpt_every == 0:
                _save_cl_column_progress(done_cnt)

        print(f"    [CL] finished stage f={f_use:g}", flush=True)
        if f_use >= target - 1e-7:
            break
        cur_f = cur_f + delta_f
        if cl_progress_ckpt_path:
            _save_cl_column_progress(0)


def train_one(
    train_path,
    test_path,
    n_samples,
    steps,
    device,
    seed,
    f_loss=1.0,
    xy_loss=5.0,
    batchsize=32,
    base_lr=1e-3,
    progress_ckpt_path=None,
    ckpt_every=2000,
    adam_ckpt_path=None,
    adam_save_path=None,
    f_rhs=1.0,
    test_num=200,
    opt="adam",
    alm_ckpt_path=None,
    alm_outer_iters=10,
    alm_inner_step=100,
    cl_init_f=1.0,
    cl_delta_f=0.1,
    cl_target_f=None,
    cl_inner_step=500,
    cl_progress_ckpt_path=None,
    cl_ckpt_every=500,
    alm_cons_item="pde",
    alm_uncon_weight=1.0,
    alm_mu=1.0,
    alm_rho=1.02,
    alm_inner="adam",
    alm_lbfgs_max_iter=200,
    alm_lbfgs_chunks=2,
    alm_pde_eps_slack=0.0,
    alm_data_eps_slack=0.0,
    alm_save_best=False,
    alm_adam_lr=None,
    alm_adam_warmup_frac=0.2,
    alm_adam_cosine=True,
    nncg_steps=5,
    lbfgs_max_iter=10000,
    pde_sub: int | None = None,
    darcy_r: float | None = None,
    tau: float = DEFAULT_GRF_TAU,
    alpha: float = DEFAULT_GRF_ALPHA,
    a_low: float = DEFAULT_A_LOW,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    pde_sub_use = int(PDE_SUB if pde_sub is None else pde_sub)
    ds_train_u = DarcyFlow(train_path, nx=N_FINE, sub=SUB, offset=0, num=n_samples)
    ds_train_ic = DarcyIC(train_path, nx=N_FINE, sub=pde_sub_use, offset=0, num=n_samples)
    ds_test = DarcyFlow(test_path, nx=N_FINE, sub=SUB, offset=0, num=test_num)
    train_u_loader = torch.utils.data.DataLoader(
        ds_train_u, batch_size=batchsize, shuffle=True, drop_last=False
    )
    ic_loader = torch.utils.data.DataLoader(
        ds_train_ic, batch_size=batchsize, shuffle=True, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        ds_test, batch_size=batchsize, shuffle=False
    )
    train_loader = train_u_loader  # alias for legacy locals
    ds_train = ds_train_u

    model = build_model(device)
    if opt == "lbfgs":
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=int(lbfgs_max_iter),
            history_size=50,
            line_search_fn="strong_wolfe",
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    # Optional base weights before curriculum (same file key as ``*_adam.pt`` — CL never skips Adam via this branch).
    if opt == "cl" and adam_ckpt_path and os.path.isfile(adam_ckpt_path):
        try:
            adc = _load_torch_ckpt(adam_ckpt_path, device)
            sd = adc.get("state_dict") or adc.get("model_state")
            if sd is None:
                raise KeyError("no state_dict / model_state")
            model.load_state_dict(sd)
            print(f"  [CL] loaded base init from  {adam_ckpt_path}", flush=True)
        except Exception as e:
            print(f"  [CL] WARN base ckpt load failed: {e}", flush=True)

    cl_final_loss_trace: list[float] = []
    if opt == "cl":
        cl()

    milestones = [int(steps * frac) for frac in (0.2, 0.4, 0.6, 0.8)]
    scheduler = (
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.5
        )
        if opt != "lbfgs"
        else None
    )

    lploss = LpLoss(size_average=True)

    mollifier_u = _darcy_mollifier(ds_train_u.mesh, device)
    mollifier_ic = _darcy_mollifier(ds_train_ic.mesh, device)
    mollifier = mollifier_u
    print(
        f"  [data] DarcyFlow sub={SUB}  grid {ds_train_u.S}×{ds_train_u.S}  "
        f"[pde] DarcyIC pde_sub={pde_sub_use}  grid {ds_train_ic.S}×{ds_train_ic.S}  "
        f"(decoupled loaders, train_darcy style)",
        flush=True,
    )

    u_loader = sample_data(train_u_loader)
    ic_pde_loader = sample_data(ic_loader)
    step = 0
    acc_data = acc_pde = acc_total = acc_n = 0
    recent_train_losses: collections.deque[float] = collections.deque(maxlen=50)
    cell_t0 = time.time()

    # Resume partial run, else warm-start from a finished ``*_adam.pt`` (never when opt==``cl``).
    pretrain_reported = False
    pretrain_metrics: dict[str, float] | None = None

    def _snapshot_pretrain(label: str) -> None:
        nonlocal pretrain_reported, pretrain_metrics
        te, ta, d_l2, pde_r = print_run_metrics(
            label,
            model,
            train_u_loader,
            ic_loader,
            test_loader,
            device,
            lploss,
            mollifier_u,
            mollifier_ic,
            f_rhs,
            cell_t0,
            xy_loss,
            f_loss,
            cons_item=alm_cons_item if opt == "alm" else None,
        )
        pretrain_metrics = {
            "test_error": float(te),
            "test_error_abs": float(ta),
            "data_l2": float(d_l2),
            "pde_res": float(pde_r),
        }
        pretrain_reported = True

    if progress_ckpt_path and os.path.exists(progress_ckpt_path):
        try:
            pc = _load_torch_ckpt(progress_ckpt_path, device)
            model.load_state_dict(pc["model_state"])
            optimizer.load_state_dict(pc["opt_state"])
            scheduler.load_state_dict(pc["sch_state"])
            step = int(pc["step"])
            acc_data = float(pc["acc_data"])
            acc_pde = float(pc["acc_pde"])
            acc_total = float(pc["acc_total"])
            acc_n = int(pc["acc_n"])
            if "torch_rng" in pc:
                torch.set_rng_state(pc["torch_rng"])
            if pc.get("numpy_rng") is not None:
                np.random.set_state(pc["numpy_rng"])
            if torch.cuda.is_available() and pc.get("cuda_rng") is not None:
                try:
                    torch.cuda.set_rng_state_all(pc["cuda_rng"])
                except Exception:
                    pass
            print(f"  [resume] Loaded Adam progress: step={step}/{steps}", flush=True)
            if step >= steps:
                _snapshot_pretrain("loaded pretrain (progress ckpt)")
        except Exception as e:
            print(f"  [resume] WARN load failed: {e}; restarting cell", flush=True)
            step = 0
            acc_data = acc_pde = acc_total = 0.0
            acc_n = 0

    elif opt != "cl" and adam_ckpt_path:
        if not os.path.isfile(adam_ckpt_path):
            print(
                f"  [resume] WARN no Adam ckpt at {adam_ckpt_path}  "
                f"(will train Adam for {steps} step(s) from scratch)",
                flush=True,
            )
    if opt != "cl" and adam_ckpt_path and os.path.isfile(adam_ckpt_path):
        try:
            adc = _load_torch_ckpt(adam_ckpt_path, device)
            sd = adc.get("state_dict")
            if sd is None:
                sd = adc.get("model_state")
            if sd is None:
                raise KeyError("checkpoint has no state_dict / model_state")
            model.load_state_dict(sd)
            step = steps
            avg_d = float(adc.get("adam_avg_data", 0.0))
            avg_p = float(adc.get("adam_avg_pde", 0.0))
            saved_steps = int(adc.get("adam_steps", steps))
            saved_steps = max(1, saved_steps)
            acc_n = saved_steps
            acc_data = avg_d * saved_steps
            acc_pde = avg_p * saved_steps
            acc_total = (xy_loss * avg_d + f_loss * avg_p) * saved_steps
            ckpt_steps = int(adc.get("adam_steps", saved_steps))
            ckpt_test = adc.get("test_after_adam")
            extra = (
                f"  ckpt_adam_steps={ckpt_steps}  ckpt_test_rel={float(ckpt_test):.4g}"
                if ckpt_test is not None
                else f"  ckpt_adam_steps={ckpt_steps}"
            )
            print(
                f"  [resume] Loaded Adam weights from {adam_ckpt_path}  "
                f"(skip Adam phase; target steps={steps}){extra}",
                flush=True,
            )
            _snapshot_pretrain("loaded pretrain ckpt")
        except Exception as e:
            print(
                f"  [resume] WARN Adam ckpt load failed: {e}; training Adam from scratch.",
                flush=True,
            )
            step = 0
            acc_data = acc_pde = acc_total = acc_n = 0

    model.train()

    if opt == "lbfgs":
        # Loop ``--steps`` L-BFGS calls, each with up to ``--lbfgs_max_iter``
        # inner iterations, over full-batch train data. Total inner iters ≈
        # ``steps * lbfgs_max_iter``.
        all_ic_u = torch.stack([ds_train_u[i][0] for i in range(n_samples)]).to(device)
        all_u = torch.stack([ds_train_u[i][1] for i in range(n_samples)]).to(device)
        all_ic_pde = torch.stack([ds_train_ic[i] for i in range(n_samples)]).to(device)

        def closure():
            optimizer.zero_grad()
            out_u = model(all_ic_u).squeeze(-1) * mollifier_u
            l_data = lploss(out_u, all_u)
            out_p = model(all_ic_pde).squeeze(-1) * mollifier_ic
            l_pde = darcy_loss(out_p, all_ic_pde[..., 0], f_rhs)
            loss = xy_loss * l_data + f_loss * l_pde
            loss.backward()
            return loss

        while step < steps:
            optimizer.step(closure)
            with torch.no_grad():
                out_u = model(all_ic_u).squeeze(-1) * mollifier_u
                l_data = lploss(out_u, all_u)
                out_p = model(all_ic_pde).squeeze(-1) * mollifier_ic
                l_pde = darcy_loss(out_p, all_ic_pde[..., 0], f_rhs)
                loss = xy_loss * l_data + f_loss * l_pde
            acc_data += float(l_data.item())
            acc_pde += float(l_pde.item())
            acc_total += float(loss.item())
            acc_n += 1
            recent_train_losses.append(float(loss.item()))
            step += 1
    else:
        while step < steps:
            if xy_loss > 0:
                ic_d, u_true = next(u_loader)
                ic_d, u_true = ic_d.to(device), u_true.to(device)
                out_d = model(ic_d).squeeze(-1) * mollifier_u
                l_data = lploss(out_d, u_true)
            else:
                l_data = torch.zeros(1, device=device)

            if f_loss > 0:
                ic_p = next(ic_pde_loader)
                ic_p = ic_p.to(device)
                out_p = model(ic_p).squeeze(-1) * mollifier_ic
                l_pde = darcy_loss(out_p, ic_p[..., 0], f_rhs)
            else:
                l_pde = 0.0

            loss = xy_loss * l_data + f_loss * l_pde

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            acc_data += float(l_data.item())
            acc_pde += float(l_pde.item() if torch.is_tensor(l_pde) else l_pde)
            acc_total += loss.item()
            acc_n += 1
            recent_train_losses.append(float(loss.item()))
            step += 1

            if progress_ckpt_path and (step % ckpt_every == 0 or step == steps):
                try:
                    os.makedirs(os.path.dirname(progress_ckpt_path), exist_ok=True)
                    tmp = progress_ckpt_path + ".tmp"
                    torch.save(
                        {
                            "step": step,
                            "model_state": {
                                k: v.cpu().clone() for k, v in model.state_dict().items()
                            },
                            "opt_state": optimizer.state_dict(),
                            "sch_state": scheduler.state_dict(),
                            "acc_data": acc_data,
                            "acc_pde": acc_pde,
                            "acc_total": acc_total,
                            "acc_n": acc_n,
                            "torch_rng": torch.get_rng_state(),
                            "numpy_rng": np.random.get_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available()
                            else None,
                        },
                        tmp,
                    )
                    os.replace(tmp, progress_ckpt_path)
                except Exception as e:
                    print(f"  [ckpt] WARN save: {e}", flush=True)

    if not pretrain_reported and step >= steps and acc_n > 0:
        _snapshot_pretrain("after pretrain")

    alm_stats = None
    nncg_stats = None
    if opt == "alm":
        print(
            f"  [ALM] phase start  cons={alm_cons_item}  inner={alm_inner}  "
            f"outer={alm_outer_iters}  μ0={alm_mu}  ρ={alm_rho}  "
            f"uncon_weight={alm_uncon_weight}  pde_eps_slack={alm_pde_eps_slack}",
            flush=True,
        )
        if not pretrain_reported:
            _snapshot_pretrain("after pretrain")
        alm_stats = alm(
            cons_item=alm_cons_item,
            uncon_weight=float(alm_uncon_weight),
            mu=float(alm_mu),
            rho=float(alm_rho),
            outer_iters=alm_outer_iters,
            inner_step=alm_inner_step,
        )
    elif opt == "nncg":
        nncg_stats = nncg(nncg_steps=nncg_steps)

    test_err, test_err_abs = mean_test_rel_abs_errors(
        model, test_loader, device, lploss, mollifier_u
    )
    if opt == "alm":
        report_data_l2, report_pde = mean_train_data_pde(
            model,
            train_u_loader,
            ic_loader,
            device,
            lploss,
            mollifier_u,
            mollifier_ic,
            f_rhs,
        )
    else:
        report_data_l2 = acc_data / max(1, acc_n)
        report_pde = acc_pde / max(1, acc_n)

    if recent_train_losses:
        lt50 = list(recent_train_losses)
        train_loss_last50_mean = float(sum(lt50) / len(lt50))
        train_loss_last50_min = float(min(lt50))
    else:
        train_loss_last50_mean = float(acc_total / max(1, acc_n))
        train_loss_last50_min = train_loss_last50_mean

    if opt == "cl" and cl_final_loss_trace:
        lt = cl_final_loss_trace[-50:]
        train_loss_last50_mean = float(sum(lt) / len(lt))
        train_loss_last50_min = float(min(lt))

    if progress_ckpt_path and os.path.exists(progress_ckpt_path):
        try:
            os.remove(progress_ckpt_path)
        except Exception:
            pass

    # ALM warm-starts from *_adam.pt; do not overwrite that file with post-ALM weights.
    if opt == "alm":
        save_path = adam_save_path
    else:
        save_path = adam_save_path if adam_save_path is not None else adam_ckpt_path
    ckpt_saved = False
    if save_path:
        try:
            d = os.path.dirname(os.path.abspath(save_path))
            if d:
                os.makedirs(d, exist_ok=True)
            torch.save(
                {
                    "state_dict": {
                        k: v.cpu() for k, v in model.state_dict().items()
                    },
                    "test_after_adam": test_err,
                    "test_after_adam_abs": test_err_abs,
                    "adam_avg_data": acc_data / max(1, acc_n),
                    "adam_avg_pde": acc_pde / max(1, acc_n),
                    "adam_steps": step,
                },
                save_path,
            )
            ckpt_saved = True
        except Exception as e:
            print(f"  [ckpt] WARN save adam ckpt: {e}", flush=True)

    if ckpt_saved and opt == "cl" and cl_progress_ckpt_path and os.path.isfile(cl_progress_ckpt_path):
        try:
            os.remove(cl_progress_ckpt_path)
            print("  [CL] removed column progress after successful save", flush=True)
        except Exception as e:
            print(f"  [CL] WARN remove column progress: {e}", flush=True)

    return {
        "train_loss": acc_total / max(1, acc_n),
        "train_loss_last50_mean": train_loss_last50_mean,
        "train_loss_last50_min": train_loss_last50_min,
        "data_l2": report_data_l2,
        "pde_res": report_pde,
        "test_error": test_err,
        "test_error_abs": test_err_abs,
        "alm_last50_inner_loss": (alm_stats or {}).get("last50_inner_loss"),
        "alm_n_inner_steps": (alm_stats or {}).get("n_inner_steps"),
        "alm_outer_test_l2": (alm_stats or {}).get("outer_test_l2"),
        "nncg_loss_history": (nncg_stats or {}).get("loss_history"),
        "pretrain_test_error": (pretrain_metrics or {}).get("test_error"),
        "pretrain_test_error_abs": (pretrain_metrics or {}).get("test_error_abs"),
        "pretrain_data_l2": (pretrain_metrics or {}).get("data_l2"),
        "pretrain_pde_res": (pretrain_metrics or {}).get("pde_res"),
    }


# ──────────────────────────────────────────────────────────────────
# Main (single experiment)
# ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--r",
        type=float,
        default=6.0,
        help="Contrast ratio r (piececonst: CSV tag only).",
    )
    parser.add_argument(
        "--f",
        type=float,
        default=1.0,
        help="Forcing f in -∇·(a∇u)=f (must match generated .mat).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=DEFAULT_GRF_TAU,
        help="GRF parameter τ in (-Δ+τ²)^(-α) coefficient sampling (see generate_darcy).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_GRF_ALPHA,
        help="GRF spectral exponent α.",
    )
    parser.add_argument(
        "--a_low",
        type=float,
        default=DEFAULT_A_LOW,
        help="Low permeability before thresholding; a_high = r * a_low.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Training trajectories.",
    )
    parser.add_argument(
        "--pde_sub",
        type=int,
        default=PDE_SUB,
        help="DarcyIC subsample for PDE loss (train_darcy ``pde_sub``; data uses sub=7).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Adam steps before ALM/CL/NNCG (default 1: load *_adam.pt from --ckpt_dir and skip Adam).",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--outdir",
        type=str,
        default="/pscratch/sd/w/wyx345/pino/sweep/darcy",
    )
    parser.add_argument(
        "--progress_dir",
        type=str,
        default=None,
        help="Per-cell progress ckpt dir (resume)",
    )
    parser.add_argument("--ckpt_every", type=int, default=2000)
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=os.environ.get(
            "PINO_CKPT_DIR",
            "/global/homes/w/wyx345/pscratch/pino/adam_15k_8x8_seed01_20260523_123416/ckpts",
        ),
        help="Adam/ALM ckpt dir (default: 15k Adam 8×8 sweep ckpts; override with PINO_CKPT_DIR).",
    )
    parser.add_argument("--shard_id", type=int, default=-1)
    parser.add_argument("--n_shards", type=int, default=8)
    parser.add_argument(
        "--piececonst_dir",
        type=str,
        default=os.environ.get("PINO_PIECECONST_DIR"),
        help="Directory with piececonst_r421_N1024_smooth1/2.mat (optional).",
    )
    parser.add_argument(
        "--test_num",
        type=int,
        default=None,
        help="Test trajectories (default 500 if piececonst, else 200).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="alm",
        choices=("adam", "alm", "cl", "nncg", "lbfgs"),
        help="Optimizer / outer trainer (default alm: PDE hard constraint after Adam warm-start).",
    )
    parser.add_argument(
        "--lbfgs_max_iter",
        type=int,
        default=10000,
        help="For optimizer=lbfgs: max_iter in a single LBFGS step().",
    )
    parser.add_argument(
        "--alm_outer_iters",
        type=int,
        default=100,
        help="ALM outer iterations (optimizer=alm).",
    )
    parser.add_argument(
        "--alm_inner",
        type=str,
        default="adam",
        choices=("lbfgs", "adam"),
        help="ALM inner solver: adam=mini-batch (default); lbfgs=full-batch (slow).",
    )
    parser.add_argument(
        "--alm_lbfgs_max_iter",
        type=int,
        default=200,
        help="L-BFGS max_iter per outer when --alm_inner lbfgs.",
    )
    parser.add_argument(
        "--alm_lbfgs_chunks",
        type=int,
        default=0,
        help=(
            "When --alm_inner lbfgs: split n_samples into this many contiguous chunks; "
            "each closure accumulates gradients (lower peak VRAM, ~full-batch ALM). "
            "0 (default) = auto: ceil(n/500) chunks (n≤500→1, n=1000→2, n=2000→4)."
        ),
    )
    parser.add_argument(
        "--alm_inner_step",
        type=int,
        default=4000,
        help="Adam inner steps per outer when --alm_inner adam (default).",
    )
    parser.add_argument(
        "--alm_pde_eps_slack",
        type=float,
        default=0.0,
        help="PDE slack: ε = slack × per-sample |Du-f| at pretrain (0 = hard constraint).",
    )
    parser.add_argument(
        "--alm_data_eps_slack",
        type=float,
        default=0.0,
        help="Data constraint slack when --alm_cons_item data.",
    )
    parser.add_argument(
        "--alm_cons_item",
        type=str,
        default="pde",
        choices=("data", "pde"),
        help="ALM constrained branch (vs PINO unconstrained branch).",
    )
    parser.add_argument(
        "--alm_uncon_weight",
        type=float,
        default=0.000001,
        help="Scale on unconstrained branch. cons=pde: (alm_uncon_weight×xy_loss)×data; "
        "cons=data: (alm_uncon_weight×f_loss)×pde.",
    )
    parser.add_argument(
        "--alm_mu",
        "--alm_mu0",
        type=float,
        default=2.0,
        dest="alm_mu",
        help="Initial ALM penalty μ (μ0).",
    )
    parser.add_argument(
        "--alm_rho",
        type=float,
        default=1.2,
        help="ALM μ multiplier each outer iteration (default 1.02, gentle).",
    )
    parser.add_argument(
        "--alm_save_best",
        action="store_true",
        help="Restore/save weights from best test_rel outer (default: save final outer only).",
    )
    parser.add_argument(
        "--alm_adam_lr",
        type=float,
        default=None,
        help="Adam inner LR for ALM phase. Default: base_lr × 0.1 (e.g. 1e-4). "
             "Lower than pretrain LR to avoid kicking model out of basin.",
    )
    parser.add_argument(
        "--alm_adam_warmup_frac",
        type=float,
        default=0.2,
        help="Warmup fraction of inner_step before cosine decay (default 0.2).",
    )
    parser.add_argument(
        "--alm_adam_no_cosine",
        action="store_true",
        help="Disable cosine LR schedule for ALM Adam inner (constant LR).",
    )
    parser.add_argument(
        "--cl_init_f",
        type=float,
        default=1.0,
        help="Curriculum starting forcing f (optimizer=cl).",
    )
    parser.add_argument(
        "--cl_delta_f",
        type=float,
        default=0.1,
        help="Curriculum f increment per stage (optimizer=cl).",
    )
    parser.add_argument(
        "--cl_target_f",
        type=float,
        default=None,
        help="Curriculum target f (default: same as --f for this cell).",
    )
    parser.add_argument(
        "--cl_inner_step",
        type=int,
        default=500,
        help="Adam substeps per curriculum stage (optimizer=cl).",
    )
    parser.add_argument(
        "--nncg_steps",
        type=int,
        default=5,
        help="Nyström–CG iterations after Adam (optimizer=nncg).",
    )
    parser.add_argument(
        "--xy_loss",
        type=float,
        default=5.0,
        help="Data-loss weight in `xy_loss*data + f_loss*pde` (default 5.0).",
    )
    parser.add_argument(
        "--f_loss",
        type=float,
        default=1.0,
        help="PDE-loss weight in `xy_loss*data + f_loss*pde` (default 1.0; set 0 for data-only).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun cells even if results.csv already has a matching row.",
    )
    args = parser.parse_args()

    if args.test_num is None:
        args.test_num = 500 if args.piececonst_dir else 200

    if args.piececonst_dir:
        if not _use_legacy_mat_name(args.tau, args.alpha, args.a_low):
            raise SystemExit(
                "With --piececonst_dir, τ=3, α=2, a_low=3 only (piececonst dataset)."
            )

    if args.piececonst_dir and args.n_samples > 1024:
        raise SystemExit(
            "piececonst MAT files contain 1024 trajectories; use --n_samples ≤ 1024."
        )

    n_pool_train = args.n_samples

    outdir = os.path.join(args.outdir, "pino")
    os.makedirs(outdir, exist_ok=True)
    if args.progress_dir:
        os.makedirs(args.progress_dir, exist_ok=True)
    if args.ckpt_dir:
        os.makedirs(args.ckpt_dir, exist_ok=True)
    csv_path = os.path.join(outdir, "results.csv")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

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

    n_pool_test = max(200, args.test_num)

    fieldnames = [
        "optimizer",
        "r",
        "f",
        "tau",
        "alpha",
        "a_low",
        "n_samples",
        "seed",
        "train_loss",
        "train_loss_last50_mean",
        "train_loss_last50_min",
        "data_l2",
        "pde_res",
        "test_error",
        "test_error_abs",
        "elapsed_s",
    ]

    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            old_fields = csv.reader(f).__next__()
        if "optimizer" not in old_fields:
            bak = csv_path + f".bak_{time.strftime('%Y%m%d_%H%M%S')}"
            os.replace(csv_path, bak)
            print(
                f"  [csv] Renamed legacy results (no optimizer column) → {bak}",
                flush=True,
            )

    done = set()
    if os.path.exists(csv_path) and not args.force:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if not row.get("optimizer"):
                    continue
                try:
                    fv = float(row.get("f", "1.0"))
                    tau_k = (
                        float(row["tau"])
                        if "tau" in row and row["tau"] != ""
                        else DEFAULT_GRF_TAU
                    )
                    alpha_k = (
                        float(row["alpha"])
                        if "alpha" in row and row["alpha"] != ""
                        else DEFAULT_GRF_ALPHA
                    )
                    a_low_k = (
                        float(row["a_low"])
                        if "a_low" in row and row["a_low"] != ""
                        else DEFAULT_A_LOW
                    )
                    done.add(
                        (
                            str(row["optimizer"]),
                            float(row["r"]),
                            fv,
                            tau_k,
                            alpha_k,
                            a_low_k,
                            int(row["n_samples"]),
                            int(row["seed"]),
                        )
                    )
                except (ValueError, TypeError):
                    pass

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    csv_lock_path = csv_path + ".lock"
    r = float(args.r)
    f_rhs = float(args.f)
    tau = float(args.tau)
    alpha = float(args.alpha)
    a_low = float(args.a_low)
    n_samples = int(args.n_samples)
    seed = int(args.seed)

    cell_key = (
        args.optimizer,
        r,
        f_rhs,
        tau,
        alpha,
        a_low,
        n_samples,
        seed,
    )
    if cell_key in done and not args.force:
        print(
            f"SKIP optimizer={args.optimizer} r={r} f={f_rhs} tau={tau} alpha={alpha} "
            f"a_low={a_low} n={n_samples} seed={seed} "
            f"(in {csv_path}; use --force to rerun)",
            flush=True,
        )
        print(f"\nDone. Results: {csv_path}")
        return

    if args.shard_id >= 0 and cell_shard(
        r,
        n_samples,
        seed,
        args.n_shards,
        f=f_rhs,
        tau=tau,
        alpha=alpha,
        a_low=a_low,
    ) != args.shard_id:
        print(
            f"SKIP shard_id={args.shard_id} (cell assigned to another shard)",
            flush=True,
        )
        print(f"\nDone. Results: {csv_path}")
        return

    print(
        f"\nRun: r={r}  f={f_rhs}  tau={tau} alpha={alpha} a_low={a_low}  "
        f"n_samples={n_samples}  seed={seed}  steps={args.steps}  "
        f"optimizer={args.optimizer}",
        flush=True,
    )
    t0 = time.time()

    if args.piececonst_dir:
        train_path, test_path = resolve_piececonst_paths(args.piececonst_dir)
    else:
        train_path = ensure_data(
            r,
            seed=0,
            n_samples=n_pool_train,
            f=f_rhs,
            tau=tau,
            alpha=alpha,
            a_low=a_low,
        )
        test_path = ensure_data(
            r,
            seed=1,
            n_samples=n_pool_test,
            f=f_rhs,
            tau=tau,
            alpha=alpha,
            a_low=a_low,
        )

    tag = (
        f"r{r_sweep_token(r)}_f{f_sweep_token(f_rhs)}"
        f"_n{n_samples}_s{seed}"
        f"{coeff_tag_suffix(tau, alpha, a_low)}"
    )
    progress_ckpt = (
        os.path.join(args.progress_dir, f"{tag}_progress.pt")
        if args.progress_dir
        else None
    )
    adam_ckpt = (
        os.path.join(args.ckpt_dir, f"{tag}_adam.pt")
        if args.ckpt_dir
        else None
    )
    alm_ckpt = (
        os.path.join(args.ckpt_dir, f"{tag}_alm.pt")
        if args.ckpt_dir and args.optimizer == "alm"
        else None
    )
    cl_progress_ckpt = (
        os.path.join(args.progress_dir, f"{tag}_cl_progress.pt")
        if args.progress_dir and args.optimizer == "cl"
        else None
    )
    result = train_one(
        train_path,
        test_path,
        n_samples,
        args.steps,
        device,
        seed,
        xy_loss=args.xy_loss,
        f_loss=args.f_loss,
        progress_ckpt_path=progress_ckpt,
        ckpt_every=args.ckpt_every,
        adam_ckpt_path=adam_ckpt,
        adam_save_path=adam_ckpt,
        alm_ckpt_path=alm_ckpt,
        alm_outer_iters=args.alm_outer_iters,
        alm_inner_step=args.alm_inner_step,
        alm_cons_item=args.alm_cons_item,
        alm_uncon_weight=args.alm_uncon_weight,
        alm_mu=args.alm_mu,
        alm_rho=args.alm_rho,
        alm_inner=args.alm_inner,
        alm_lbfgs_max_iter=args.alm_lbfgs_max_iter,
        alm_lbfgs_chunks=args.alm_lbfgs_chunks,
        alm_pde_eps_slack=args.alm_pde_eps_slack,
        alm_data_eps_slack=args.alm_data_eps_slack,
        alm_save_best=args.alm_save_best,
        alm_adam_lr=args.alm_adam_lr,
        alm_adam_warmup_frac=args.alm_adam_warmup_frac,
        alm_adam_cosine=(not args.alm_adam_no_cosine),
        cl_init_f=args.cl_init_f,
        cl_delta_f=args.cl_delta_f,
        cl_target_f=args.cl_target_f,
        cl_inner_step=args.cl_inner_step,
        cl_progress_ckpt_path=cl_progress_ckpt,
        nncg_steps=args.nncg_steps,
        lbfgs_max_iter=args.lbfgs_max_iter,
        f_rhs=f_rhs,
        test_num=args.test_num,
        opt=args.optimizer,
        pde_sub=args.pde_sub,
        darcy_r=r,
        tau=tau,
        alpha=alpha,
        a_low=a_low,
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
            f"physics_loss={result['pde_res']:.4f}  "
            f"({elapsed:.0f}s)",
            flush=True,
        )
    else:
        print(
            f"  → test_rel={result['test_error']:.4f}  "
            f"test_abs={result['test_error_abs']:.4g}  "
            f"data_loss={result['data_l2']:.4f}  "
            f"physics_loss={result['pde_res']:.4f}  "
            f"({elapsed:.0f}s)",
            flush=True,
        )

    with open(csv_path, "a", newline="") as csvfile, open(csv_lock_path, "a") as _lk:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        fcntl.flock(_lk.fileno(), fcntl.LOCK_EX)
        try:
            writer.writerow(
                {
                    "optimizer": args.optimizer,
                    "r": r,
                    "f": f_rhs,
                    "tau": tau,
                    "alpha": alpha,
                    "a_low": a_low,
                    "n_samples": n_samples,
                    "seed": seed,
                    "train_loss": round(result["train_loss"], 6),
                    "train_loss_last50_mean": round(
                        result["train_loss_last50_mean"], 8
                    ),
                    "train_loss_last50_min": round(
                        result["train_loss_last50_min"], 8
                    ),
                    "data_l2": round(result["data_l2"], 6),
                    "pde_res": round(result["pde_res"], 6),
                    "test_error": round(result["test_error"], 6),
                    "test_error_abs": round(result["test_error_abs"], 8),
                    "elapsed_s": round(elapsed, 1),
                }
            )
            csvfile.flush()
        finally:
            fcntl.flock(_lk.fileno(), fcntl.LOCK_UN)

    print(f"\nDone. Results: {csv_path}")


if __name__ == "__main__":
    main()
