"""
Nyström–Newton-CG (NNCG) post-training fine-tuning for PINO Darcy.

Set env ``NNCG_FB_CHUNK`` (e.g. 50–150) to use ``ChunkedNysNewtonCG``
(chunked full-batch HVP), matching the ``darcy_sweep_adam_nncg.py`` approach.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Any

import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
import sys as _sys
import os as _os
_SCRIPTS_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_PINO_DIR = _os.path.dirname(_SCRIPTS_DIR)
for _p in (_PINO_DIR, _SCRIPTS_DIR):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

from train_utils.losses import LpLoss, darcy_loss  # noqa: E402


def run_nncg(
    *,
    model,
    device,
    ds_train_u,
    ds_train_ic,
    n_samples: int,
    f_rhs: float,
    mollifier_u: torch.Tensor,
    mollifier_ic: torch.Tensor,
    lploss: LpLoss,
    xy_loss: float,
    f_loss: float,
    test_loader,
    # NNCG hyper-parameters
    nncg_steps: int = 5,
    lr: float = 1.0,
    rank: int = 10,
    mu: float = 0.01,
    precond_update_freq: int = 1,
    cg_tol: float = 1e-5,
    cg_max_iters: int = 200,
) -> dict[str, Any]:
    """Post-train Nyström–Newton-CG fine-tuning on the full Darcy batch.

    ``NNCG_FB_CHUNK``  env var (int): activates ``ChunkedNysNewtonCG``
    ``NNCG_RANK``      env var (int): override ``rank``
    ``NNCG_CHUNK_SIZE`` env var (int): functorch vmap chunk size
    """
    ic_all_u   = torch.stack([ds_train_u[i][0] for i in range(int(n_samples))]).to(device)
    u_all      = torch.stack([ds_train_u[i][1] for i in range(int(n_samples))]).to(device)
    ic_all_pde = torch.stack([ds_train_ic[i]   for i in range(int(n_samples))]).to(device)

    def _maybe_print_test_err(step_1based: int) -> None:
        if test_loader is None:
            return
        model.eval()
        errs: list[float] = []
        with torch.no_grad():
            for ic, u_true in test_loader:
                ic, u_true = ic.to(device), u_true.to(device)
                out = model(ic).squeeze(-1) * mollifier_u
                errs.append(lploss(out, u_true).item())
        model.train()
        te = float(np.mean(errs)) if errs else float("nan")
        print(f"  [NNCG] iter {step_1based}/{nncg_steps}  test_error={te:.6g}", flush=True)

    fb_chunk   = int(os.environ.get("NNCG_FB_CHUNK", "0"))
    rank_eff   = int(os.environ.get("NNCG_RANK", str(rank)))
    chunk_size = max(1, int(os.environ.get("NNCG_CHUNK_SIZE", "1")))
    dl_fn      = partial(darcy_loss, f_rhs=float(f_rhs))

    def _real(x: torch.Tensor) -> torch.Tensor:
        return torch.real(x) if torch.is_complex(x) else x

    def _real_grads(grads):
        return tuple(_real(g) if g is not None else None for g in grads)

    pf = max(1, int(precond_update_freq))
    loss_history: list[float] = []

    if fb_chunk > 0:
        from scripts.nys_newton_cg_chunked import (
            ChunkedNysNewtonCG,
            make_chunked_grad_fn,
            make_chunked_hvp_fn,
            make_chunked_loss_fn,
        )
        params_list = list(model.parameters())
        opt_n = ChunkedNysNewtonCG(
            model.parameters(), lr=float(lr), rank=int(rank_eff),
            mu=float(mu), chunk_size=int(chunk_size),
            cg_tol=float(cg_tol), cg_max_iters=int(cg_max_iters),
            line_search_fn="armijo", verbose=False,
        )
        kw = dict(
            model=model, ic_all=ic_all_u, u_all=u_all, mollifier=mollifier_u,
            lploss=lploss, darcy_loss_fn=dl_fn,
            xy_loss=float(xy_loss), f_loss=float(f_loss),
            params_list=params_list, chunk_size=fb_chunk, loss_mode="pino",
        )
        opt_n.attach_callbacks(
            grad_fn=make_chunked_grad_fn(**kw),
            hvp_fn=make_chunked_hvp_fn(**kw),
            loss_fn=make_chunked_loss_fn(**{k: v for k, v in kw.items() if k != "params_list"}),
        )
        print(
            f"  [NNCG] Chunked Nyström–CG  steps={nncg_steps}  lr={lr}  rank={rank_eff}  "
            f"mu={mu}  fb_chunk={fb_chunk}  functorch_chunk={chunk_size}  "
            f"precond_every={precond_update_freq}",
            flush=True,
        )
        model.train()
        for k in range(int(nncg_steps)):
            if k % pf == 0:
                opt_n.update_preconditioner_chunked()
            opt_n.step_chunked()
            if torch.cuda.is_available() and (k % 5 == 0):
                torch.cuda.empty_cache()
            with torch.no_grad():
                out_u  = model(ic_all_u).squeeze(-1) * mollifier_u
                l_data = lploss(out_u, u_all)
                out_p  = model(ic_all_pde).squeeze(-1) * mollifier_ic
                l_pde  = darcy_loss(out_p, ic_all_pde[..., 0], f_rhs)
                loss_t = _real(xy_loss * l_data + f_loss * l_pde)
            loss_history.append(float(loss_t.detach().cpu()))
            _maybe_print_test_err(k + 1)
        return {"loss_history": loss_history, "nncg_steps": int(nncg_steps)}

    # ── Standard (dense-graph) NysNewtonCG ────────────────────────────────────
    from scripts.nys_newton_cg import NysNewtonCG  # noqa: E402

    opt_n = NysNewtonCG(
        model.parameters(), lr=float(lr), rank=int(rank_eff),
        mu=float(mu), chunk_size=int(chunk_size),
        cg_tol=float(cg_tol), cg_max_iters=int(cg_max_iters),
        line_search_fn="armijo", verbose=False,
    )

    def closure():
        opt_n.zero_grad(set_to_none=True)
        out_u  = model(ic_all_u).squeeze(-1) * mollifier_u
        l_data = lploss(out_u, u_all)
        out_p  = model(ic_all_pde).squeeze(-1) * mollifier_ic
        l_pde  = darcy_loss(out_p, ic_all_pde[..., 0], f_rhs)
        loss   = _real(xy_loss * l_data + f_loss * l_pde)
        grads  = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        return loss, _real_grads(grads)

    print(
        f"  [NNCG] Nyström–CG (dense graph)  steps={nncg_steps}  lr={lr}  "
        f"rank={rank_eff}  mu={mu}  chunk_size={chunk_size}  "
        f"precond_every={precond_update_freq}  "
        f"(set NNCG_FB_CHUNK>0 for chunked HVP)",
        flush=True,
    )
    model.train()
    for k in range(int(nncg_steps)):
        if k % pf == 0:
            _, grad_tuple = closure()
            opt_n.update_preconditioner(grad_tuple)
        opt_n.step(closure)
        with torch.no_grad():
            out_u  = model(ic_all_u).squeeze(-1) * mollifier_u
            l_data = lploss(out_u, u_all)
            out_p  = model(ic_all_pde).squeeze(-1) * mollifier_ic
            l_pde  = darcy_loss(out_p, ic_all_pde[..., 0], f_rhs)
            loss_t = _real(xy_loss * l_data + f_loss * l_pde)
        loss_history.append(float(loss_t.detach().cpu()))
        _maybe_print_test_err(k + 1)

    return {"loss_history": loss_history, "nncg_steps": int(nncg_steps)}
