"""
Curriculum Learning (CL) pre-training for PINO Darcy.

Increases forcing ``f`` from ``init_f`` by ``delta_f`` per stage until
reaching ``target_f``, training ``inner_step`` Adam steps at each stage.
Uses decoupled ``DarcyFlow`` (data) and ``DarcyIC`` (PDE) loaders per stage.
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import torch
import torch.utils.data

# ── Path setup ────────────────────────────────────────────────────────────────
import sys as _sys
import os as _os
_SCRIPTS_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_PINO_DIR = _os.path.dirname(_SCRIPTS_DIR)
for _p in (_PINO_DIR, _SCRIPTS_DIR):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

from train_utils.datasets import DarcyFlow, DarcyIC, sample_data  # noqa: E402
from train_utils.losses import LpLoss, darcy_loss  # noqa: E402
from darcy_data import (  # noqa: E402
    _darcy_mollifier,
    _load_torch_ckpt,
    ensure_data,
    N_FINE,
    SUB,
    PDE_SUB,
    DEFAULT_GRF_TAU,
    DEFAULT_GRF_ALPHA,
    DEFAULT_A_LOW,
)


def run_cl(
    *,
    model,
    device,
    train_path: str,
    n_samples: int,
    optimizer,
    batchsize: int = 32,
    f_loss: float = 1.0,
    xy_loss: float = 5.0,
    pde_sub_use: int = PDE_SUB,
    tau: float = DEFAULT_GRF_TAU,
    alpha: float = DEFAULT_GRF_ALPHA,
    a_low: float = DEFAULT_A_LOW,
    seed: int = 0,
    darcy_r: float | None = None,
    # CL hyper-parameters
    init_f: float = 1.0,
    delta_f: float = 0.1,
    target_f: float | None = None,
    inner_step: int = 500,
    # Checkpoint
    cl_progress_ckpt_path: str | None = None,
    cl_ckpt_every: int = 500,
    # Output: caller should pass a pre-allocated list; CL appends losses at final stage.
    final_loss_trace: list[float] | None = None,
) -> None:
    """Run f-curriculum Adam pre-training, modifying ``model`` in-place.

    Each stage trains on data at ``f_stage = min(cur_f, target_f)`` using
    decoupled DarcyFlow + DarcyIC loaders (independent random batches per loss term).
    """
    if darcy_r is not None:
        r = float(darcy_r)
    else:
        mpath = re.search(r"_r([^_]+)_f([^_]+)_seed", train_path)
        if not mpath:
            print("  [CL] skip: cannot parse r,f from train_path", flush=True)
            return
        r = float(mpath.group(1).replace("p", "."))

    target   = float(target_f) if target_f is not None else float(1.0)
    cur_f    = float(init_f)
    inner_done = 0

    if cl_progress_ckpt_path and os.path.isfile(cl_progress_ckpt_path):
        try:
            pc = _load_torch_ckpt(cl_progress_ckpt_path, device)
            cur_f      = float(pc["cur_f"])
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
            print(f"  [CL resume] WARN load failed: {e}; starting from scratch", flush=True)
            cur_f      = float(init_f)
            inner_done = 0

    def _save_progress(inner_done_save: int) -> None:
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
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
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

    lp = LpLoss(size_average=True)

    while True:
        f_use = min(cur_f, target)
        tp = ensure_data(r, seed=seed, n_samples=n_samples, f=f_use, tau=tau, alpha=alpha, a_low=a_low)
        ds_tr_u  = DarcyFlow(tp, nx=N_FINE, sub=SUB,          offset=0, num=n_samples)
        ds_tr_ic = DarcyIC( tp, nx=N_FINE, sub=pde_sub_use,   offset=0, num=n_samples)
        u_loader  = torch.utils.data.DataLoader(ds_tr_u,  batch_size=batchsize, shuffle=True, drop_last=False)
        ic_loader = torch.utils.data.DataLoader(ds_tr_ic, batch_size=batchsize, shuffle=True, drop_last=False)
        u_iter    = sample_data(u_loader)
        ic_iter   = sample_data(ic_loader)
        mollifier_u  = _darcy_mollifier(ds_tr_u.mesh,  device)
        mollifier_ic = _darcy_mollifier(ds_tr_ic.mesh, device)
        model.train()
        final_stage  = f_use >= target - 1e-7
        resume_inner = int(inner_done)
        inner_done   = 0

        for inner_i in range(resume_inner, int(inner_step)):
            if xy_loss > 0:
                ic_d, u_true = next(u_iter)
                ic_d, u_true = ic_d.to(device), u_true.to(device)
                out_d  = model(ic_d).squeeze(-1) * mollifier_u
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
            if final_loss_trace is not None and final_stage:
                final_loss_trace.append(float(loss.detach().cpu()))
            done_cnt = inner_i + 1
            if cl_progress_ckpt_path and done_cnt % cl_ckpt_every == 0:
                _save_progress(done_cnt)

        print(f"    [CL] finished stage f={f_use:g}", flush=True)
        if f_use >= target - 1e-7:
            break
        cur_f = cur_f + delta_f
        if cl_progress_ckpt_path:
            _save_progress(0)
