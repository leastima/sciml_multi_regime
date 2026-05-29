"""
Augmented Lagrangian Method (ALM) for PINO Darcy training.

Supports two inner solvers:
  - ``inner="adam"`` : mini-batch Adam (default, faster)
  - ``inner="lbfgs"``: full-batch L-BFGS with gradient accumulation
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import numpy as np
import torch
import torch.utils.data

# ── Path setup ────────────────────────────────────────────────────────────────
import sys as _sys
_HERE = __file__
import os as _os
_SCRIPTS_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(_HERE)))
_PINO_DIR = _os.path.dirname(_SCRIPTS_DIR)
for _p in (_PINO_DIR, _SCRIPTS_DIR):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

from train_utils.losses import LpLoss, darcy_loss  # noqa: E402
from darcy_model import (  # noqa: E402
    _darcy_du_f_tensors,
    _DatasetWithIndex,
    _ICDatasetWithIndex,
    _lbfgs_chunk_slices,
    print_run_metrics,
)


def run_alm(
    *,
    model,
    device,
    train_u_loader,
    ic_loader,
    ds_train_u,
    ds_train_ic,
    n_samples: int,
    f_rhs: float,
    base_lr: float,
    mollifier_u: torch.Tensor,
    mollifier_ic: torch.Tensor,
    lploss: LpLoss,
    xy_loss: float,
    f_loss: float,
    test_loader,
    cell_t0: float,
    # ALM hyper-parameters
    cons_item: str = "pde",
    uncon_weight: float = 1.0,
    mu: float = 2.0,
    rho: float = 1.1,
    outer_iters: int = 20,
    inner_step: int = 200,
    # ALM inner-solver options
    alm_inner: str = "adam",
    alm_lbfgs_max_iter: int = 200,
    alm_lbfgs_chunks: int = 0,
    alm_pde_eps_slack: float = 0.0,
    alm_data_eps_slack: float = 0.0,
    alm_adam_lr: float | None = None,
    alm_adam_warmup_frac: float = 0.2,
    alm_adam_cosine: bool = True,
    alm_save_best: bool = False,
    alm_ckpt_path: str | None = None,
) -> dict[str, Any]:
    """Run Augmented Lagrangian Method on the full Darcy training set.

    ``cons_item='pde'``: constrain per-sample PDE residual ≤ ε, minimise data loss.
    ``cons_item='data'``: constrain per-sample data loss ≤ ε, minimise PDE residual.
    """
    xy_w = float(xy_loss)
    f_w  = float(f_loss)
    inner_solver = str(alm_inner).lower().strip()

    if ds_train_u is not None:
        n_samples = len(ds_train_u)
    elif n_samples <= 0:
        n_samples = len(train_u_loader.dataset)

    # Resolve LBFGS chunk count.
    # 0/negative → auto: 1 chunk per ≤500 samples (n=500→1, n=1000→2, n=2000→4)
    if alm_lbfgs_chunks <= 0:
        lbfgs_chunks = max(1, (n_samples + 499) // 500)
    else:
        lbfgs_chunks = max(1, alm_lbfgs_chunks)

    ci = str(cons_item).lower().strip()
    if ci not in ("data", "pde"):
        raise ValueError("cons_item must be 'data' or 'pde'")

    adam_inner_lr = float(alm_adam_lr) if alm_adam_lr is not None else float(base_lr) * 0.1

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
    inner_loss_trace: list[float] = []
    outer_test_l2: list[float] = []
    lam = torch.zeros(1)
    opt_alm = None
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
            f"lbfgs_max_iter={alm_lbfgs_max_iter}  chunks={lbfgs_chunks}  "
            f"(grad-accum ~{math.ceil(n_samples / lbfgs_chunks)}/chunk)"
        )
    )
    print(
        f"  [ALM] cons={ci}  inner={inner_solver}  {inner_desc}  "
        f"unconstrained={eff_uncon}  n_constraints={n_samples}  "
        f"mu0={mu}  rho={rho}  outer={outer_iters}  pde_eps_slack={alm_pde_eps_slack}",
        flush=True,
    )

    if inner_solver == "lbfgs":
        if ds_train_u is None or ds_train_ic is None:
            raise ValueError("alm_inner=lbfgs requires ds_train_u and ds_train_ic")
        all_ic_u  = torch.stack([ds_train_u[i][0] for i in range(n_samples)]).to(device)
        all_u     = torch.stack([ds_train_u[i][1] for i in range(n_samples)]).to(device)
        all_ic_pde = torch.stack([ds_train_ic[i]  for i in range(n_samples)]).to(device)
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
        eps_vec: torch.Tensor

        with torch.no_grad():
            if ci == "pde":
                out0 = model(all_ic_pde).squeeze(-1) * mollifier_ic
                h0 = pde_per_sample(out0, all_ic_pde[..., 0]).detach()
                slack = float(alm_pde_eps_slack)
                eps_vec = slack * h0 if slack > 0 else torch.zeros_like(h0)
            else:
                out0 = model(all_ic_u).squeeze(-1) * mollifier_u
                h0 = data_per_sample(out0, all_u).detach()
                slack = float(alm_data_eps_slack)
                eps_vec = slack * h0 if slack > 0 else torch.zeros_like(h0)
            print(
                f"  [ALM] pretrain viol_mean={h0.mean():.5g}  "
                f"ε slack={slack:g}  ε_mean={eps_vec.mean():.5g}",
                flush=True,
            )

        for i in range(int(outer_iters)):
            mu_k  = float(mu_cur)
            lam_k = lam.detach().clone()
            print(
                f"  [ALM] outer {i + 1}/{outer_iters}  start  μ={mu_k:.6g}  "
                f"(L-BFGS max_iter={alm_lbfgs_max_iter}, n={n_samples}) …",
                flush=True,
            )
            t_outer = time.time()
            model.train()
            lbfgs = torch.optim.LBFGS(
                model.parameters(), lr=1.0,
                max_iter=alm_lbfgs_max_iter, history_size=50,
                line_search_fn="strong_wolfe",
            )
            chunk_slices = _lbfgs_chunk_slices(n_samples, lbfgs_chunks)
            n_chunk = len(chunk_slices)

            def _alm_loss_on_slice(
                sl: slice, _lam: torch.Tensor, _mu: float, _eps: torch.Tensor
            ) -> torch.Tensor:
                ic_u  = all_ic_u[sl]
                u     = all_u[sl]
                ic_p  = all_ic_pde[sl]
                if _alm_lbfgs_single_fwd:
                    out_u = model(ic_u).squeeze(-1) * mollifier_u
                    out_p = out_u
                else:
                    out_u = model(ic_u).squeeze(-1) * mollifier_u
                    out_p = model(ic_p).squeeze(-1) * mollifier_ic
                l_data = lploss(out_u, u)
                a_p    = ic_p[..., 0]
                l_pde  = darcy_loss(out_p, a_p, f_rhs)
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
                    viol  = pde_per_sample(out_p, all_ic_pde[..., 0]) - eps_vec
                else:
                    out_u = model(all_ic_u).squeeze(-1) * mollifier_u
                    viol  = data_per_sample(out_u, all_u) - eps_vec
                lam = lam + mu_k * viol.detach()

            te, _, data_loss, physics_loss = print_run_metrics(
                f"ALM outer {i + 1}/{outer_iters}",
                model, train_u_loader, ic_loader, test_loader,
                device, lploss, mollifier_u, mollifier_ic,
                float(f_rhs), cell_t0, xy_w, f_w,
            )
            if test_loader is not None and not math.isnan(te):
                outer_test_l2.append(te)
                if te < best_test:
                    best_test  = te
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
            _print_lambda_stats(lam, i + 1, outer_iters, mu_cur)
            mu_cur *= float(rho)

    else:
        # ── Adam inner solver ─────────────────────────────────────────────────
        if ds_train_u is None:
            ds_train_u = train_u_loader.dataset
        if ds_train_ic is None:
            ds_train_ic = ic_loader.dataset
        alm_u_loader = torch.utils.data.DataLoader(
            _DatasetWithIndex(ds_train_u),
            batch_size=train_u_loader.batch_size, shuffle=True, drop_last=False,
        )
        alm_ic_loader = torch.utils.data.DataLoader(
            _ICDatasetWithIndex(ds_train_ic),
            batch_size=ic_loader.batch_size, shuffle=True, drop_last=False,
        )
        lam = torch.zeros(n_samples, device=device)
        eps_vec = torch.zeros(n_samples, device=device)
        slack = float(alm_pde_eps_slack if ci == "pde" else alm_data_eps_slack)
        h0_sum = 0.0
        h0_cnt = 0
        with torch.no_grad():
            if ci == "pde":
                for ic, idx in alm_ic_loader:
                    ic  = ic.to(device)
                    idx = idx.to(device)
                    out = model(ic).squeeze(-1) * mollifier_ic
                    h0  = pde_per_sample(out, ic[..., 0])
                    h0_sum += float(h0.sum().item())
                    h0_cnt += int(h0.numel())
                    if slack > 0:
                        eps_vec[idx] = slack * h0
            else:
                for ic, u_true, idx in alm_u_loader:
                    ic, u_true = ic.to(device), u_true.to(device)
                    idx = idx.to(device)
                    out = model(ic).squeeze(-1) * mollifier_u
                    h0  = data_per_sample(out, u_true)
                    h0_sum += float(h0.sum().item())
                    h0_cnt += int(h0.numel())
                    if slack > 0:
                        eps_vec[idx] = slack * h0
        print(
            f"  [ALM] pretrain viol_mean={h0_sum / max(h0_cnt, 1):.5g}  "
            f"ε slack={slack:g}  ε_mean={eps_vec.mean():.5g}",
            flush=True,
        )

        print(
            f"  [ALM] Adam inner: lr={adam_inner_lr:g}  "
            f"warmup_frac={alm_adam_warmup_frac:g}  cosine={alm_adam_cosine}  "
            f"per-sample λ (n={n_samples}), decoupled data/PDE loaders  "
            f"(opt+sch reset per outer)",
            flush=True,
        )
        u_iter  = iter(alm_u_loader)
        ic_iter = iter(alm_ic_loader)
        model.train()

        for i in range(int(outer_iters)):
            mu_k = float(mu_cur)
            print(
                f"  [ALM] outer {i + 1}/{outer_iters}  start  μ={mu_k:.6g}  "
                f"(Adam inner_steps={inner_step}, batch={alm_u_loader.batch_size}) …",
                flush=True,
            )
            t_outer = time.time()
            opt_alm = torch.optim.Adam(model.parameters(), lr=adam_inner_lr)
            if alm_adam_cosine and int(inner_step) > 1:
                _warmup_end = max(1, int(int(inner_step) * alm_adam_warmup_frac))
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
                out_d  = model(ic_d).squeeze(-1) * mollifier_u
                l_data = lploss(out_d, u_true)

                try:
                    ic_p, idx_p = next(ic_iter)
                except StopIteration:
                    ic_iter = iter(alm_ic_loader)
                    ic_p, idx_p = next(ic_iter)
                ic_p  = ic_p.to(device)
                idx_p = idx_p.to(device)
                out_p = model(ic_p).squeeze(-1) * mollifier_ic
                a_p   = ic_p[..., 0]
                l_pde = darcy_loss(out_p, a_p, f_rhs)
                h = (
                    pde_per_sample(out_p, a_p)
                    if ci == "pde"
                    else data_per_sample(out_d, u_true)
                )
                h      = h - eps_vec[idx_p]
                lam_b  = lam[idx_p].detach()
                loss_alm = (
                    uncon_scalar(l_data, l_pde)
                    + torch.mean(lam_b * h)
                    + 0.5 * mu_k * torch.mean(h**2)
                )
                inner_loss_trace.append(float(loss_alm.detach().cpu()))
                opt_alm.zero_grad()
                loss_alm.backward()
                opt_alm.step()
                if sch_alm is not None:
                    sch_alm.step()

            viol_sum   = 0.0
            viol_count = 0
            with torch.no_grad():
                if ci == "pde":
                    for ic, idx in alm_ic_loader:
                        ic  = ic.to(device)
                        idx = idx.to(device)
                        out = model(ic).squeeze(-1) * mollifier_ic
                        h   = pde_per_sample(out, ic[..., 0]) - eps_vec[idx]
                        lam[idx] = lam[idx] + mu_k * h
                        viol_sum   += float(h.sum().item())
                        viol_count += int(h.numel())
                else:
                    for ic, u_true, idx in alm_u_loader:
                        ic, u_true = ic.to(device), u_true.to(device)
                        idx = idx.to(device)
                        out = model(ic).squeeze(-1) * mollifier_u
                        h   = data_per_sample(out, u_true) - eps_vec[idx]
                        lam[idx] = lam[idx] + mu_k * h
                        viol_sum   += float(h.sum().item())
                        viol_count += int(h.numel())
            viol_mean = viol_sum / max(viol_count, 1)

            print(
                f"  [ALM] outer {i + 1}/{outer_iters}  Adam done  "
                f"wall={time.time() - t_outer:.1f}s",
                flush=True,
            )
            te, _, data_loss, physics_loss = print_run_metrics(
                f"ALM outer {i + 1}/{outer_iters}",
                model, train_u_loader, ic_loader, test_loader,
                device, lploss, mollifier_u, mollifier_ic,
                float(f_rhs), cell_t0, xy_w, f_w,
            )
            if test_loader is not None and not math.isnan(te):
                outer_test_l2.append(te)
                if te < best_test:
                    best_test  = te
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
            _print_lambda_stats(lam, i + 1, outer_iters, mu_cur)
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
            f"(saving final-outer weights; pass alm_save_best=True to restore best)",
            flush=True,
        )

    last50 = inner_loss_trace[-50:] if len(inner_loss_trace) >= 50 else list(inner_loss_trace)
    result: dict[str, Any] = {
        "inner_loss_trace": inner_loss_trace,
        "last50_inner_loss": last50,
        "n_inner_steps": len(inner_loss_trace),
        "outer_test_l2": outer_test_l2,
    }

    if alm_ckpt_path:
        _save_alm_ckpt(
            alm_ckpt_path, model, best_outer, best_test, lam, mu_cur,
            ci, inner_solver, uncon_weight, xy_w, f_w, f_rhs,
            last50, inner_loss_trace, outer_iters, inner_step, opt_alm,
        )

    return result


def _print_lambda_stats(lam: torch.Tensor, outer: int, total: int, mu: float) -> None:
    lam_np = lam.detach().cpu().numpy()
    if lam_np.size <= 16:
        lam_str = str(lam_np)
    else:
        lam_str = (
            f"mean={float(np.mean(lam_np)):.5g} std={float(np.std(lam_np)):.5g} "
            f"min={float(np.min(lam_np)):.5g} max={float(np.max(lam_np)):.5g} len={lam_np.size}"
        )
    print(f"    [ALM] outer {outer}/{total}  lambda={lam_str}  mu={mu:.6g}", flush=True)


def _save_alm_ckpt(
    path, model, best_outer, best_test, lam, mu_cur, ci, inner_solver,
    uncon_weight, xy_w, f_w, f_rhs, last50, inner_loss_trace,
    outer_iters, inner_step, opt_alm,
) -> None:
    try:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
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
        import torch as _torch
        _torch.save(ckpt, tmp)
        os.replace(tmp, path)
        print(f"  [ALM] saved checkpoint  {path}", flush=True)
    except Exception as e:
        print(f"  [ALM] WARN save ckpt: {e}", flush=True)
