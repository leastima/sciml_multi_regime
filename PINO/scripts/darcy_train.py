"""
Darcy PINO — single-cell training entry point (``train_one``).

Orchestrates data loading, model construction, and optimizer dispatch
(Adam / L-BFGS / ALM / NNCG / CL).
"""

from __future__ import annotations

import collections
import os
import time
from typing import Any

import numpy as np
import torch
import torch.utils.data

# ── Path setup ────────────────────────────────────────────────────────────────
import sys as _sys
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PINO_DIR = os.path.dirname(_SCRIPTS_DIR)
for _p in (_PINO_DIR, _SCRIPTS_DIR):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

from train_utils.datasets import DarcyFlow, DarcyIC, sample_data  # noqa: E402
from train_utils.losses import LpLoss, darcy_loss  # noqa: E402

from darcy_data import (  # noqa: E402
    N_FINE, SUB, PDE_SUB,
    DEFAULT_GRF_TAU, DEFAULT_GRF_ALPHA, DEFAULT_A_LOW,
    _darcy_mollifier, _load_torch_ckpt,
)
from darcy_model import (  # noqa: E402
    build_model,
    mean_test_rel_abs_errors,
    mean_train_data_pde,
    print_run_metrics,
)
from darcy_optimizers import run_alm, run_nncg, run_cl  # noqa: E402


def train_one(
    train_path: str,
    test_path: str,
    n_samples: int,
    steps: int,
    device,
    seed: int,
    f_loss: float = 1.0,
    xy_loss: float = 5.0,
    batchsize: int = 32,
    base_lr: float = 1e-3,
    progress_ckpt_path: str | None = None,
    ckpt_every: int = 2000,
    adam_ckpt_path: str | None = None,
    adam_save_path: str | None = None,
    f_rhs: float = 1.0,
    test_num: int = 200,
    opt: str = "adam",
    alm_ckpt_path: str | None = None,
    alm_outer_iters: int = 10,
    alm_inner_step: int = 100,
    cl_init_f: float = 1.0,
    cl_delta_f: float = 0.1,
    cl_target_f: float | None = None,
    cl_inner_step: int = 500,
    cl_progress_ckpt_path: str | None = None,
    cl_ckpt_every: int = 500,
    alm_cons_item: str = "pde",
    alm_uncon_weight: float = 1.0,
    alm_mu: float = 1.0,
    alm_rho: float = 1.02,
    alm_inner: str = "adam",
    alm_lbfgs_max_iter: int = 200,
    alm_lbfgs_chunks: int = 2,
    alm_pde_eps_slack: float = 0.0,
    alm_data_eps_slack: float = 0.0,
    alm_save_best: bool = False,
    alm_adam_lr: float | None = None,
    alm_adam_warmup_frac: float = 0.2,
    alm_adam_cosine: bool = True,
    nncg_steps: int = 5,
    lbfgs_max_iter: int = 10000,
    pde_sub: int | None = None,
    darcy_r: float | None = None,
    tau: float = DEFAULT_GRF_TAU,
    alpha: float = DEFAULT_GRF_ALPHA,
    a_low: float = DEFAULT_A_LOW,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    pde_sub_use = int(PDE_SUB if pde_sub is None else pde_sub)
    ds_train_u  = DarcyFlow(train_path, nx=N_FINE, sub=SUB,          offset=0, num=n_samples)
    ds_train_ic = DarcyIC( train_path, nx=N_FINE, sub=pde_sub_use,   offset=0, num=n_samples)
    ds_test     = DarcyFlow(test_path,  nx=N_FINE, sub=SUB,          offset=0, num=test_num)
    train_u_loader = torch.utils.data.DataLoader(
        ds_train_u, batch_size=batchsize, shuffle=True, drop_last=False
    )
    ic_loader = torch.utils.data.DataLoader(
        ds_train_ic, batch_size=batchsize, shuffle=True, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=batchsize, shuffle=False)

    model = build_model(device)
    if opt == "lbfgs":
        optimizer = torch.optim.LBFGS(
            model.parameters(), lr=1.0,
            max_iter=int(lbfgs_max_iter), history_size=50,
            line_search_fn="strong_wolfe",
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    # Optional base weights before CL (same key as ``*_adam.pt``).
    if opt == "cl" and adam_ckpt_path and os.path.isfile(adam_ckpt_path):
        try:
            adc = _load_torch_ckpt(adam_ckpt_path, device)
            sd  = adc.get("state_dict") or adc.get("model_state")
            if sd is None:
                raise KeyError("no state_dict / model_state")
            model.load_state_dict(sd)
            print(f"  [CL] loaded base init from  {adam_ckpt_path}", flush=True)
        except Exception as e:
            print(f"  [CL] WARN base ckpt load failed: {e}", flush=True)

    cl_final_loss_trace: list[float] = []
    if opt == "cl":
        run_cl(
            model=model,
            device=device,
            train_path=train_path,
            n_samples=n_samples,
            optimizer=optimizer,
            batchsize=batchsize,
            f_loss=f_loss,
            xy_loss=xy_loss,
            pde_sub_use=pde_sub_use,
            tau=tau,
            alpha=alpha,
            a_low=a_low,
            seed=seed,
            darcy_r=darcy_r,
            init_f=cl_init_f,
            delta_f=cl_delta_f,
            target_f=cl_target_f if cl_target_f is not None else f_rhs,
            inner_step=cl_inner_step,
            cl_progress_ckpt_path=cl_progress_ckpt_path,
            cl_ckpt_every=max(50, int(cl_ckpt_every)),
            final_loss_trace=cl_final_loss_trace,
        )

    milestones = [int(steps * frac) for frac in (0.2, 0.4, 0.6, 0.8)]
    scheduler  = (
        torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
        if opt != "lbfgs"
        else None
    )

    lploss      = LpLoss(size_average=True)
    mollifier_u  = _darcy_mollifier(ds_train_u.mesh, device)
    mollifier_ic = _darcy_mollifier(ds_train_ic.mesh, device)
    mollifier    = mollifier_u  # alias used by progress reporting

    print(
        f"  [data] DarcyFlow sub={SUB}  grid {ds_train_u.S}×{ds_train_u.S}  "
        f"[pde] DarcyIC pde_sub={pde_sub_use}  grid {ds_train_ic.S}×{ds_train_ic.S}  "
        f"(decoupled loaders, train_darcy style)",
        flush=True,
    )

    u_loader_iter    = sample_data(train_u_loader)
    ic_pde_loader_it = sample_data(ic_loader)
    step     = 0
    acc_data = acc_pde = acc_total = acc_n = 0
    recent_train_losses: collections.deque[float] = collections.deque(maxlen=50)
    cell_t0 = time.time()

    pretrain_reported = False
    pretrain_metrics: dict[str, float] | None = None

    def _snapshot_pretrain(label: str) -> None:
        nonlocal pretrain_reported, pretrain_metrics
        te, ta, d_l2, pde_r = print_run_metrics(
            label, model, train_u_loader, ic_loader, test_loader,
            device, lploss, mollifier_u, mollifier_ic, f_rhs, cell_t0,
            xy_loss, f_loss,
            cons_item=alm_cons_item if opt == "alm" else None,
        )
        pretrain_metrics = {
            "test_error":     float(te),
            "test_error_abs": float(ta),
            "data_l2":        float(d_l2),
            "pde_res":        float(pde_r),
        }
        pretrain_reported = True

    # ── Resume or warm-start from Adam checkpoint ─────────────────────────────
    if progress_ckpt_path and os.path.exists(progress_ckpt_path):
        try:
            pc = _load_torch_ckpt(progress_ckpt_path, device)
            model.load_state_dict(pc["model_state"])
            optimizer.load_state_dict(pc["opt_state"])
            if scheduler is not None:
                scheduler.load_state_dict(pc["sch_state"])
            step     = int(pc["step"])
            acc_data = float(pc["acc_data"])
            acc_pde  = float(pc["acc_pde"])
            acc_total = float(pc["acc_total"])
            acc_n    = int(pc["acc_n"])
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
            acc_data = acc_pde = acc_total = acc_n = 0
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
            sd  = adc.get("state_dict") or adc.get("model_state")
            if sd is None:
                raise KeyError("checkpoint has no state_dict / model_state")
            model.load_state_dict(sd)
            step      = steps
            avg_d     = float(adc.get("adam_avg_data", 0.0))
            avg_p     = float(adc.get("adam_avg_pde", 0.0))
            saved_steps = max(1, int(adc.get("adam_steps", steps)))
            acc_n     = saved_steps
            acc_data  = avg_d * saved_steps
            acc_pde   = avg_p * saved_steps
            acc_total = (xy_loss * avg_d + f_loss * avg_p) * saved_steps
            ckpt_steps = int(adc.get("adam_steps", saved_steps))
            ckpt_test  = adc.get("test_after_adam")
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

    # ── Main training loop ────────────────────────────────────────────────────
    if opt == "lbfgs":
        all_ic_u   = torch.stack([ds_train_u[i][0] for i in range(n_samples)]).to(device)
        all_u      = torch.stack([ds_train_u[i][1] for i in range(n_samples)]).to(device)
        all_ic_pde = torch.stack([ds_train_ic[i]   for i in range(n_samples)]).to(device)

        def closure():
            optimizer.zero_grad()
            out_u  = model(all_ic_u).squeeze(-1) * mollifier_u
            l_data = lploss(out_u, all_u)
            out_p  = model(all_ic_pde).squeeze(-1) * mollifier_ic
            l_pde  = darcy_loss(out_p, all_ic_pde[..., 0], f_rhs)
            loss   = xy_loss * l_data + f_loss * l_pde
            loss.backward()
            return loss

        while step < steps:
            optimizer.step(closure)
            with torch.no_grad():
                out_u  = model(all_ic_u).squeeze(-1) * mollifier_u
                l_data = lploss(out_u, all_u)
                out_p  = model(all_ic_pde).squeeze(-1) * mollifier_ic
                l_pde  = darcy_loss(out_p, all_ic_pde[..., 0], f_rhs)
                loss   = xy_loss * l_data + f_loss * l_pde
            acc_data  += float(l_data.item())
            acc_pde   += float(l_pde.item())
            acc_total += float(loss.item())
            acc_n     += 1
            recent_train_losses.append(float(loss.item()))
            step += 1
    else:
        while step < steps:
            if xy_loss > 0:
                ic_d, u_true = next(u_loader_iter)
                ic_d, u_true = ic_d.to(device), u_true.to(device)
                out_d  = model(ic_d).squeeze(-1) * mollifier_u
                l_data = lploss(out_d, u_true)
            else:
                l_data = torch.zeros(1, device=device)

            if f_loss > 0:
                ic_p = next(ic_pde_loader_it)
                ic_p = ic_p.to(device)
                out_p = model(ic_p).squeeze(-1) * mollifier_ic
                l_pde = darcy_loss(out_p, ic_p[..., 0], f_rhs)
            else:
                l_pde = 0.0

            loss = xy_loss * l_data + f_loss * l_pde
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            acc_data  += float(l_data.item())
            acc_pde   += float(l_pde.item() if torch.is_tensor(l_pde) else l_pde)
            acc_total += loss.item()
            acc_n     += 1
            recent_train_losses.append(float(loss.item()))
            step += 1

            if progress_ckpt_path and (step % ckpt_every == 0 or step == steps):
                try:
                    os.makedirs(os.path.dirname(progress_ckpt_path), exist_ok=True)
                    tmp = progress_ckpt_path + ".tmp"
                    torch.save(
                        {
                            "step": step,
                            "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                            "opt_state": optimizer.state_dict(),
                            "sch_state": scheduler.state_dict() if scheduler else {},
                            "acc_data": acc_data, "acc_pde": acc_pde,
                            "acc_total": acc_total, "acc_n": acc_n,
                            "torch_rng": torch.get_rng_state(),
                            "numpy_rng": np.random.get_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        },
                        tmp,
                    )
                    os.replace(tmp, progress_ckpt_path)
                except Exception as e:
                    print(f"  [ckpt] WARN save: {e}", flush=True)

    if not pretrain_reported and step >= steps and acc_n > 0:
        _snapshot_pretrain("after pretrain")

    # ── Post-training optimization phase ─────────────────────────────────────
    alm_stats  = None
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
        alm_stats = run_alm(
            model=model, device=device,
            train_u_loader=train_u_loader, ic_loader=ic_loader,
            ds_train_u=ds_train_u, ds_train_ic=ds_train_ic,
            n_samples=n_samples, f_rhs=f_rhs, base_lr=base_lr,
            mollifier_u=mollifier_u, mollifier_ic=mollifier_ic,
            lploss=lploss, xy_loss=xy_loss, f_loss=f_loss,
            test_loader=test_loader, cell_t0=cell_t0,
            cons_item=alm_cons_item, uncon_weight=alm_uncon_weight,
            mu=alm_mu, rho=alm_rho,
            outer_iters=alm_outer_iters, inner_step=alm_inner_step,
            alm_inner=alm_inner,
            alm_lbfgs_max_iter=alm_lbfgs_max_iter,
            alm_lbfgs_chunks=alm_lbfgs_chunks,
            alm_pde_eps_slack=alm_pde_eps_slack,
            alm_data_eps_slack=alm_data_eps_slack,
            alm_adam_lr=alm_adam_lr,
            alm_adam_warmup_frac=alm_adam_warmup_frac,
            alm_adam_cosine=alm_adam_cosine,
            alm_save_best=alm_save_best,
            alm_ckpt_path=alm_ckpt_path,
        )
    elif opt == "nncg":
        nncg_stats = run_nncg(
            model=model, device=device,
            ds_train_u=ds_train_u, ds_train_ic=ds_train_ic,
            n_samples=n_samples, f_rhs=f_rhs,
            mollifier_u=mollifier_u, mollifier_ic=mollifier_ic,
            lploss=lploss, xy_loss=xy_loss, f_loss=f_loss,
            test_loader=test_loader,
            nncg_steps=nncg_steps,
        )

    # ── Final metrics ─────────────────────────────────────────────────────────
    test_err, test_err_abs = mean_test_rel_abs_errors(
        model, test_loader, device, lploss, mollifier_u
    )
    if opt == "alm":
        report_data_l2, report_pde = mean_train_data_pde(
            model, train_u_loader, ic_loader, device,
            lploss, mollifier_u, mollifier_ic, f_rhs,
        )
    else:
        report_data_l2 = acc_data / max(1, acc_n)
        report_pde     = acc_pde  / max(1, acc_n)

    if recent_train_losses:
        lt50 = list(recent_train_losses)
        train_loss_last50_mean = float(sum(lt50) / len(lt50))
        train_loss_last50_min  = float(min(lt50))
    else:
        train_loss_last50_mean = float(acc_total / max(1, acc_n))
        train_loss_last50_min  = train_loss_last50_mean

    if opt == "cl" and cl_final_loss_trace:
        lt = cl_final_loss_trace[-50:]
        train_loss_last50_mean = float(sum(lt) / len(lt))
        train_loss_last50_min  = float(min(lt))

    if progress_ckpt_path and os.path.exists(progress_ckpt_path):
        try:
            os.remove(progress_ckpt_path)
        except Exception:
            pass

    # ── Save Adam (or post-train) checkpoint ──────────────────────────────────
    # ALM warm-starts from *_adam.pt — do not overwrite with post-ALM weights.
    save_path  = adam_save_path if opt == "alm" else (adam_save_path or adam_ckpt_path)
    ckpt_saved = False
    if save_path:
        try:
            d = os.path.dirname(os.path.abspath(save_path))
            if d:
                os.makedirs(d, exist_ok=True)
            torch.save(
                {
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "test_after_adam": test_err,
                    "test_after_adam_abs": test_err_abs,
                    "adam_avg_data": acc_data / max(1, acc_n),
                    "adam_avg_pde":  acc_pde  / max(1, acc_n),
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
        "train_loss":              acc_total / max(1, acc_n),
        "train_loss_last50_mean":  train_loss_last50_mean,
        "train_loss_last50_min":   train_loss_last50_min,
        "data_l2":                 report_data_l2,
        "pde_res":                 report_pde,
        "test_error":              test_err,
        "test_error_abs":          test_err_abs,
        "alm_last50_inner_loss":   (alm_stats  or {}).get("last50_inner_loss"),
        "alm_n_inner_steps":       (alm_stats  or {}).get("n_inner_steps"),
        "alm_outer_test_l2":       (alm_stats  or {}).get("outer_test_l2"),
        "nncg_loss_history":       (nncg_stats or {}).get("loss_history"),
        "pretrain_test_error":     (pretrain_metrics or {}).get("test_error"),
        "pretrain_test_error_abs": (pretrain_metrics or {}).get("test_error_abs"),
        "pretrain_data_l2":        (pretrain_metrics or {}).get("data_l2"),
        "pretrain_pde_res":        (pretrain_metrics or {}).get("pde_res"),
    }
