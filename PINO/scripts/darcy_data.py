"""
Darcy PINO — data helpers.

Handles GRF-based synthetic data generation, piececonst benchmark loading,
filename tokens, and the ``ensure_data`` cache-and-generate routine.
"""

from __future__ import annotations

import fcntl
import math
import os
import time

import numpy as np
import scipy.io
import torch

# ── Path setup (allows importing when this module is run standalone) ──────────
import sys as _sys
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PINO_DIR = os.path.dirname(_SCRIPTS_DIR)
for _p in (_PINO_DIR, _SCRIPTS_DIR):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

from scripts.generate_darcy import gen_coeff, solve_darcy_batch  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Global constants
# ──────────────────────────────────────────────────────────────────────────────

DATADIR = "/pscratch/sd/w/wyx345/pino/darcy_gen"
N_FINE = 421
SUB = 7        # data-loss grid: 421//7+1 → 61×61
PDE_SUB = 7    # PDE/IC grid (train_darcy ``pde_sub``; override via --pde_sub)

PIECECONST_TRAIN_NAME = "piececonst_r421_N1024_smooth1.mat"
PIECECONST_TEST_NAME  = "piececonst_r421_N1024_smooth2.mat"

# GRF / piececonst-matching defaults
DEFAULT_GRF_TAU    = 3.0
DEFAULT_GRF_ALPHA  = 2.0
DEFAULT_A_LOW      = 3.0
DEFAULT_SIGMA_SCALE = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Mollifier (boundary-zeroing factor)
# ──────────────────────────────────────────────────────────────────────────────

def _darcy_mollifier(mesh: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    return (
        0.001
        * torch.sin(math.pi * mesh[..., 0])
        * torch.sin(math.pi * mesh[..., 1])
    ).unsqueeze(0).to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_torch_ckpt(path: str, map_location):
    """Load training checkpoint; handles PyTorch 2.6+ ``weights_only`` default."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ──────────────────────────────────────────────────────────────────────────────
# Piececonst benchmark helpers
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Filename / tag utilities
# ──────────────────────────────────────────────────────────────────────────────

def r_sweep_token(r: float) -> str:
    """Stable path token for contrast ratio r (e.g. 1, 1.5, 10)."""
    return f"{float(r):g}"


def f_sweep_token(fv: float) -> str:
    """Stable path token for forcing f."""
    return f"{float(fv):g}"


def _use_legacy_mat_name(
    tau: float,
    alpha: float,
    a_low: float,
    sigma_scale: float = DEFAULT_SIGMA_SCALE,
) -> bool:
    """Return True if defaults match piececonst setup (enables legacy filename cache)."""
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
    """Return tag suffix when GRF params differ from piececonst defaults."""
    if _use_legacy_mat_name(tau, alpha, a_low, sigma_scale):
        return ""
    tt  = f"{float(tau):g}".replace("-", "m")
    at  = f"{float(alpha):g}".replace("-", "m")
    alt = f"{float(a_low):g}".replace("-", "m")
    sgt = f"{float(sigma_scale):g}".replace("-", "m")
    return f"_tau{tt}_a{at}_al{alt}_sg{sgt}"


# ──────────────────────────────────────────────────────────────────────────────
# Data generation / cache
# ──────────────────────────────────────────────────────────────────────────────

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

    Uses legacy filename pattern when params match piececonst defaults
    (``tau=3, alpha=2, a_low=3, sigma_scale=1``) for cache compatibility.
    Otherwise uses ``…_tau{t}_a{a}_al{a_low}_sg{sigma}_seed{seed}.mat``.
    """
    os.makedirs(DATADIR, exist_ok=True)
    rt = r_sweep_token(r)
    ft = f_sweep_token(f)
    if _use_legacy_mat_name(tau, alpha, a_low, sigma_scale):
        fname = f"darcy_N{N_FINE}_n{n_samples}_r{rt}_f{ft}_seed{seed}.mat"
    else:
        tt  = f"{float(tau):g}".replace("-", "m")
        at  = f"{float(alpha):g}".replace("-", "m")
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
