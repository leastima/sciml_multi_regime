"""
Darcy PINO — FNO model, dataset wrappers, metrics, and training utilities.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
import torch.utils.data

# ── Path setup ────────────────────────────────────────────────────────────────
import os, sys as _sys
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PINO_DIR = os.path.dirname(_SCRIPTS_DIR)
for _p in (_PINO_DIR, _SCRIPTS_DIR):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

from models.fourier2d import FNO2d  # noqa: E402
from train_utils.losses import FDM_Darcy, LpLoss, darcy_loss  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# FNO model (Table 2 / Darcy-pretrain.yaml: modes 20 on 61×61)
# ──────────────────────────────────────────────────────────────────────────────

def build_model(device) -> FNO2d:
    return FNO2d(
        modes1=[20, 20, 20, 20],
        modes2=[20, 20, 20, 20],
        layers=[64, 64, 64, 64, 64],
        fc_dim=128,
        act="gelu",
        pad_ratio=0.0,
    ).to(device)


# ──────────────────────────────────────────────────────────────────────────────
# PDE residual helper
# ──────────────────────────────────────────────────────────────────────────────

def _darcy_du_f_tensors(
    out: torch.Tensor, a: torch.Tensor, f_rhs: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (Du, f_tensor) — same strong residual as ``darcy_loss``."""
    batchsize = out.size(0)
    size = out.size(1)
    u = out.reshape(batchsize, size, size)
    a2 = a.reshape(batchsize, size, size)
    Du = FDM_Darcy(u, a2)
    f_tensor = torch.full(Du.shape, float(f_rhs), device=out.device, dtype=Du.dtype)
    return Du, f_tensor


# ──────────────────────────────────────────────────────────────────────────────
# Dataset wrappers (needed by ALM per-sample multipliers)
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# L-BFGS chunk helper
# ──────────────────────────────────────────────────────────────────────────────

def _lbfgs_chunk_slices(n_samples: int, n_chunks: int) -> list[slice]:
    """Split [0, n) into ``n_chunks`` contiguous slices (gradient accumulation)."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ──────────────────────────────────────────────────────────────────────────────

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
    """Print test + full-train data/physics losses and optional constraint violation."""
    data_loss, physics_loss = mean_train_data_pde(
        model, train_u_loader, ic_loader, device, lploss,
        mollifier_u, mollifier_ic, f_rhs,
    )
    pino_loss = float(xy_loss) * data_loss + float(f_loss) * physics_loss
    viol_extra = ""
    if cons_item is not None:
        viol_m = mean_train_viol_mean(
            model, train_u_loader, ic_loader, device,
            mollifier_u, mollifier_ic, f_rhs, cons_item=cons_item,
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
