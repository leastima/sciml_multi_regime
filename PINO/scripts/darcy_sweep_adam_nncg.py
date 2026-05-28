#!/usr/bin/env python3
"""
Darcy PINO: Adam (minibatch) warm-start, then full-batch **L-BFGS** refinement.
NNCG supports optional **subsampled** grad/HVP via ``--nncg-subsample-size`` (requires ``--nncg-fb-chunk``).

Second phase uses first-order gradients only (no HVP / Nyström Newton-CG), which avoids
the large-n OOM that NNCG had.

Same loss as darcy_sweep.py train_one (PINO: xy_loss * L_data + f_loss * L_pde).

CLI flag ``--nncg-steps`` is retained as the **number of L-BFGS outer steps** (historical name).

Checkpoint / resume (optional):
  --checkpoint-dir DIR   unified state for Adam + L-BFGS (resume + periodic save)
  --ckpt_every N         save every N trainer steps (Adam steps and refine steps)
  --wall-seconds T       exit with status 99 before T seconds (segment budget)

Usage:
    cd sciml_multi_regime/PINO
    python scripts/darcy_sweep_adam_nncg.py ... --checkpoint-dir /path --ckpt_every 100 \\
        --wall-seconds 14100
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import signal
import sys
import time

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _SCRIPTS)

from train_utils.datasets import DarcyFlow
from train_utils.losses import LpLoss, darcy_loss

import darcy_sweep as ds
from nys_newton_cg import NysNewtonCG
from nys_newton_cg_chunked import (
    ChunkedNysNewtonCG,
    NncgLoaderState,
    NncgSubsampleState,
    make_chunked_grad_fn,
    make_chunked_hvp_fn,
    make_chunked_loss_fn,
)


N_FINE = ds.N_FINE
SUB = ds.SUB

CKPT_VERSION = 1
EXIT_RESUBMIT = 99


def build_model(device):
    return ds.build_model(device)


def _append_training_log(log_path: str | None, line: str) -> None:
    if not log_path:
        return
    try:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {line}\n")
    except Exception as e:
        print(f"  [log] WARN: {e}", flush=True)


def _pack_nncg(optim: "NysNewtonCG") -> dict:
    """Serialize NNCG tensors/state for checkpoint (CPU)."""
    _od = getattr(optim, "old_dir", None)
    return {
        "U": None if optim.U is None else optim.U.detach().cpu(),
        "S": None if optim.S is None else optim.S.detach().cpu(),
        "old_dir": None if _od is None else _od.detach().cpu(),
        "mu": float(optim.mu),
        "initial_mu": float(optim.initial_mu),
        "n_iters": int(optim.n_iters),
        "cholesky_failures": int(optim.cholesky_failures),
        "_numel_cache": optim._numel_cache,
        "rho": getattr(optim, "rho", None),
    }


def _unpack_nncg(optim: "NysNewtonCG", packed: dict, device: torch.device) -> None:
    optim.U = None if packed["U"] is None else packed["U"].to(device=device)
    optim.S = None if packed["S"] is None else packed["S"].to(device=device)
    if packed.get("old_dir") is not None:
        optim.old_dir = packed["old_dir"].to(device=device)
    optim.mu = packed["mu"]
    optim.initial_mu = packed["initial_mu"]
    optim.n_iters = packed["n_iters"]
    optim.cholesky_failures = packed["cholesky_failures"]
    optim._numel_cache = packed.get("_numel_cache")
    if packed.get("rho") is not None:
        r = packed["rho"]
        optim.rho = r.item() if isinstance(r, torch.Tensor) else r


def _should_stop(deadline: float | None, stop_flag: list[bool]) -> bool:
    if stop_flag[0]:
        return True
    if deadline is not None and time.time() >= deadline:
        return True
    return False


def _nncg_real_loss(loss: torch.Tensor) -> torch.Tensor:
    """FNO/FFT graphs can yield complex-dtyped scalars; NNCG needs real gradients."""
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


def train_adam_then_nncg(
    train_path,
    test_path,
    n_samples,
    adam_steps,
    nncg_steps,
    device,
    seed,
    loss_mode="pino",
    f_loss=1.0,
    xy_loss=5.0,
    batchsize=20,
    base_lr=1e-3,
    nncg_lr=1.0,
    nncg_rank=10,
    nncg_mu=1e-2,
    nncg_cg_tol=1e-5,
    nncg_cg_max_iters=100,
    nncg_precond_update_freq=1,
    nncg_chunk_size=1,
    nncg_fb_chunk=0,
    nncg_subsample_size=0,
    nncg_data_mode="preload",
    refine_optimizer="lbfgs",
    lbfgs_lr=1.0,
    lbfgs_max_iter=20,
    progress_ckpt_path=None,
    ckpt_every=100,
    checkpoint_path=None,
    log_path=None,
    wall_seconds=None,
    load_adam_ckpt_path=None,
    nncg_test_every=0,
    keep_unified_ckpt=False,
):
    """
    Returns dict including 'status': 'complete' | 'wall_timeout' | 'stopped'.
    wall_timeout / stopped: caller should sys.exit(EXIT_RESUBMIT) to get a new node.
    """
    stop_flag = [False]

    def _on_signal(signum, frame):
        stop_flag[0] = True
        print(f"  [signal] caught {signum}, will checkpoint and exit", flush=True)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    segment_start = time.time()
    deadline = None
    if wall_seconds is not None and int(wall_seconds) > 0:
        deadline = segment_start + float(wall_seconds)

    ro = str(refine_optimizer).lower().strip()
    if ro not in ("nncg", "lbfgs"):
        raise ValueError("refine_optimizer must be 'nncg' or 'lbfgs'")

    unified_ckpt = checkpoint_path
    # Legacy: Adam-only file used as unified path name base
    if unified_ckpt is None and progress_ckpt_path:
        unified_ckpt = progress_ckpt_path.replace("_progress.pt", "_adam_nncg.pt")
        if unified_ckpt == progress_ckpt_path:
            unified_ckpt = progress_ckpt_path + ".adam_nncg"

    def save_adam_ckpt(
        step,
        model,
        optimizer,
        scheduler,
        acc_data,
        acc_pde,
        acc_total,
        acc_n,
        data_iter_state=None,
    ):
        if not unified_ckpt:
            return
        try:
            os.makedirs(os.path.dirname(unified_ckpt) or ".", exist_ok=True)
            tmp = unified_ckpt + ".tmp"
            payload = {
                "version": CKPT_VERSION,
                "phase": "adam",
                "adam_step": step,
                "nncg_completed": 0,
                "adam_steps_target": adam_steps,
                "nncg_steps_target": nncg_steps,
                "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
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
                "test_after_adam": None,
                "nncg_pack": None,
                "meta": {
                    "loss_mode": loss_mode,
                    "f_loss": f_loss,
                    "xy_loss": xy_loss,
                    "seed": seed,
                    "n_samples": n_samples,
                    "nncg_lr": nncg_lr,
                    "nncg_rank": nncg_rank,
                    "nncg_mu": nncg_mu,
                    "nncg_cg_tol": nncg_cg_tol,
                    "nncg_cg_max_iters": nncg_cg_max_iters,
                    "nncg_precond_update_freq": nncg_precond_update_freq,
                    "nncg_chunk_size": nncg_chunk_size,
                },
            }
            if data_iter_state is not None:
                payload["data_iter_state"] = data_iter_state
            torch.save(payload, tmp)
            os.replace(tmp, unified_ckpt)
            _append_training_log(
                log_path,
                f"ckpt phase=adam adam_step={step}/{adam_steps} path={unified_ckpt}",
            )
        except Exception as e:
            print(f"  [ckpt] WARN save adam: {e}", flush=True)

        # Mirror legacy progress path if requested
        if progress_ckpt_path and unified_ckpt != progress_ckpt_path:
            try:
                os.makedirs(os.path.dirname(progress_ckpt_path) or ".", exist_ok=True)
                tmp2 = progress_ckpt_path + ".tmp"
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
                    tmp2,
                )
                os.replace(tmp2, progress_ckpt_path)
            except Exception as e:
                print(f"  [ckpt] WARN legacy progress: {e}", flush=True)

    def save_refine_ckpt(
        step_completed,
        model,
        optim_refine,
        test_after_adam_f,
        ropt: str,
    ):
        if not unified_ckpt:
            return
        try:
            tmp = unified_ckpt + ".tmp"
            n_pack = None
            lbfgs_sd = None
            if ropt == "nncg":
                n_pack = _pack_nncg(optim_refine)
            else:
                lbfgs_sd = optim_refine.state_dict()
            meta_base = {
                "loss_mode": loss_mode,
                "f_loss": f_loss,
                "xy_loss": xy_loss,
                "seed": seed,
                "n_samples": n_samples,
                "refine_optimizer": ropt,
                "nncg_lr": nncg_lr,
                "nncg_rank": nncg_rank,
                "nncg_mu": nncg_mu,
                "nncg_cg_tol": nncg_cg_tol,
                "nncg_cg_max_iters": nncg_cg_max_iters,
                "nncg_precond_update_freq": nncg_precond_update_freq,
                "nncg_chunk_size": nncg_chunk_size,
                "lbfgs_lr": lbfgs_lr,
                "lbfgs_max_iter": lbfgs_max_iter,
            }
            payload = {
                "version": CKPT_VERSION,
                "phase": "nncg",
                "adam_step": adam_steps,
                "nncg_completed": step_completed,
                "adam_steps_target": adam_steps,
                "nncg_steps_target": nncg_steps,
                "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "opt_state": None,
                "sch_state": None,
                "acc_data": None,
                "acc_pde": None,
                "acc_total": None,
                "acc_n": None,
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
                "test_after_adam": test_after_adam_f,
                "nncg_pack": n_pack,
                "lbfgs_state": lbfgs_sd,
                "meta": meta_base,
            }
            torch.save(payload, tmp)
            os.replace(tmp, unified_ckpt)
            tag = "nncg" if ropt == "nncg" else "lbfgs"
            _append_training_log(
                log_path,
                f"ckpt phase={tag} refine_step={step_completed}/{nncg_steps} "
                f"test_after_adam={test_after_adam_f:.6f}",
            )
        except Exception as e:
            print(f"  [ckpt] WARN save refine: {e}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    ds_train = DarcyFlow(train_path, nx=N_FINE, sub=SUB, offset=0, num=n_samples)
    ds_test = DarcyFlow(test_path, nx=N_FINE, sub=SUB, offset=0, num=200)
    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=batchsize, shuffle=True, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        ds_test, batch_size=batchsize, shuffle=False
    )

    model = build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    milestones = [int(adam_steps * f) for f in (0.2, 0.4, 0.6, 0.8)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5
    )
    lploss = LpLoss(size_average=True)

    mesh = ds_train.mesh.to(device)
    mollifier = (
        0.001
        * torch.sin(math.pi * mesh[..., 0])
        * torch.sin(math.pi * mesh[..., 1])
    ).unsqueeze(0)

    data_iter = iter(train_loader)
    step = 0
    acc_data = acc_pde = acc_total = acc_n = 0
    phase = "adam"
    nncg_completed = 0
    test_after_adam = None
    optim_refine = None
    ic_all = u_all = None
    warmstart_from_adam_file = False

    # --- Checkpoints: unified (resume) > Adam .pt (warmstart) > legacy Adam progress ---
    resume_handled = False
    if unified_ckpt and os.path.isfile(unified_ckpt):
        try:
            pc = torch.load(
                unified_ckpt, map_location=device, weights_only=False
            )
            model.load_state_dict(pc["model_state"])
            phase = pc.get("phase", "adam")
            if "torch_rng" in pc:
                torch.set_rng_state(pc["torch_rng"])
            if pc.get("numpy_rng") is not None:
                np.random.set_state(pc["numpy_rng"])
            if torch.cuda.is_available() and pc.get("cuda_rng") is not None:
                try:
                    torch.cuda.set_rng_state_all(pc["cuda_rng"])
                except Exception:
                    pass

            if phase == "adam":
                optimizer.load_state_dict(pc["opt_state"])
                scheduler.load_state_dict(pc["sch_state"])
                step = int(pc["adam_step"])
                acc_data = float(pc["acc_data"])
                acc_pde = float(pc["acc_pde"])
                acc_total = float(pc["acc_total"])
                acc_n = int(pc["acc_n"])
                print(
                    f"  [resume] unified ckpt: phase=adam step={step}/{adam_steps}",
                    flush=True,
                )
                _append_training_log(
                    log_path,
                    f"resume phase=adam adam_step={step}/{adam_steps}",
                )
                resume_handled = True
            elif phase == "nncg":
                step = adam_steps
                nncg_completed = int(pc.get("nncg_completed", 0))
                test_after_adam = float(pc["test_after_adam"])
                ro_resume = (pc.get("meta") or {}).get("refine_optimizer", "nncg")
                full_loader = torch.utils.data.DataLoader(
                    ds_train, batch_size=n_samples, shuffle=False, drop_last=False
                )
                ic_all, u_all = next(iter(full_loader))
                ic_all = ic_all.to(device)
                u_all = u_all.to(device)
                if str(ro_resume).lower() == "lbfgs":
                    optim_refine = torch.optim.LBFGS(
                        model.parameters(),
                        lr=lbfgs_lr,
                        max_iter=lbfgs_max_iter,
                        line_search_fn="strong_wolfe",
                    )
                    if pc.get("lbfgs_state"):
                        optim_refine.load_state_dict(pc["lbfgs_state"])
                else:
                    _dm = (nncg_data_mode or "preload").lower()
                    use_chunked = int(nncg_fb_chunk) > 0 or _dm in (
                        "loader_full",
                        "loader_minibatch",
                    )
                    NncgCls = ChunkedNysNewtonCG if use_chunked else NysNewtonCG
                    optim_refine = NncgCls(
                        model.parameters(),
                        lr=nncg_lr,
                        rank=nncg_rank,
                        mu=nncg_mu,
                        cg_tol=nncg_cg_tol,
                        cg_max_iters=nncg_cg_max_iters,
                        chunk_size=nncg_chunk_size,
                        line_search_fn=None if use_chunked else "armijo",
                    )
                    if pc.get("nncg_pack"):
                        _unpack_nncg(optim_refine, pc["nncg_pack"], device)
                ro = str(ro_resume).lower()
                print(
                    f"  [resume] unified ckpt: phase=refine ({ro_resume}) done={nncg_completed}/"
                    f"{nncg_steps}",
                    flush=True,
                )
                _append_training_log(
                    log_path,
                    f"resume phase=refine refine_opt={ro_resume} step={nncg_completed}/{nncg_steps}",
                )
                resume_handled = True
        except Exception as e:
            print(
                f"  [resume] WARN unified load failed: {e}; trying adam_ckpt next",
                flush=True,
            )
            phase = "adam"
            step = 0
            data_iter = iter(train_loader)
            nncg_completed = 0
            optim_refine = None

    if not resume_handled and load_adam_ckpt_path:
        if not os.path.isfile(load_adam_ckpt_path):
            raise FileNotFoundError(
                f"Adam warm-start checkpoint missing: {load_adam_ckpt_path}\n"
                "Generate with: python scripts/darcy_sweep.py ... --ckpt_dir DIR "
                f"(expects {os.path.basename(load_adam_ckpt_path)})"
            )
        try:
            pc = torch.load(
                load_adam_ckpt_path, map_location=device, weights_only=False
            )
            if isinstance(pc, dict) and "state_dict" in pc:
                model.load_state_dict(pc["state_dict"], strict=False)
            else:
                model.load_state_dict(pc, strict=False)
            step = adam_steps
            phase = "adam"
            warmstart_from_adam_file = True
            print(
                f"  [warmstart] Loaded Adam weights from {load_adam_ckpt_path}; "
                f"skipping Adam training ({adam_steps} steps).",
                flush=True,
            )
            _append_training_log(
                log_path,
                f"warmstart from {load_adam_ckpt_path} skip_adam_steps={adam_steps}",
            )
            resume_handled = True
        except Exception as e:
            raise RuntimeError(f"Failed to load Adam ckpt {load_adam_ckpt_path}") from e

    # Legacy-only resume (no unified file yet)
    if not resume_handled and progress_ckpt_path and os.path.exists(
        progress_ckpt_path
    ):
        try:
            pc = torch.load(
                progress_ckpt_path, map_location=device, weights_only=False
            )
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
            print(f"  [resume] Adam progress: step={step}/{adam_steps}", flush=True)
        except Exception as e:
            print(f"  [resume] WARN load failed: {e}; restarting Adam phase", flush=True)
            step = 0
            acc_data = acc_pde = acc_total = 0.0
            acc_n = 0

    model.train()

    # ===================== Adam phase =====================================
    while phase == "adam" and step < adam_steps:
        if _should_stop(deadline, stop_flag):
            save_adam_ckpt(
                step,
                model,
                optimizer,
                scheduler,
                acc_data,
                acc_pde,
                acc_total,
                acc_n,
            )
            return {
                "status": "stopped" if stop_flag[0] else "wall_timeout",
                "train_loss": acc_total / max(1, acc_n),
                "data_l2": acc_data / max(1, acc_n),
                "pde_res": acc_pde / max(1, acc_n),
                "test_error": float("nan"),
                "test_after_adam": float("nan"),
                "train_loss_nncg": float("nan"),
                "data_l2_nncg": float("nan"),
                "pde_res_nncg": float("nan"),
            }

        try:
            ic, u_true = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            ic, u_true = next(data_iter)

        ic = ic.to(device)
        u_true = u_true.to(device)

        out = model(ic).squeeze(-1)
        out = out * mollifier
        a = ic[..., 0]
        l_pde = darcy_loss(out, a)

        if loss_mode == "pinn":
            loss = f_loss * l_pde
            l_data = torch.tensor(0.0, device=device)
        else:
            l_data = lploss(out, u_true)
            loss = xy_loss * l_data + f_loss * l_pde

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc_data += l_data.item()
        acc_pde += l_pde.item()
        acc_total += loss.item()
        acc_n += 1
        step += 1

        if unified_ckpt and (step % max(1, ckpt_every) == 0 or step == adam_steps):
            save_adam_ckpt(
                step,
                model,
                optimizer,
                scheduler,
                acc_data,
                acc_pde,
                acc_total,
                acc_n,
            )
        elif progress_ckpt_path and not unified_ckpt:
            if step % max(1, ckpt_every) == 0 or step == adam_steps:
                try:
                    os.makedirs(os.path.dirname(progress_ckpt_path) or ".", exist_ok=True)
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

    if phase == "adam" and nncg_steps <= 0:
        if progress_ckpt_path and os.path.exists(progress_ckpt_path):
            try:
                os.remove(progress_ckpt_path)
            except Exception:
                pass
        if unified_ckpt and os.path.isfile(unified_ckpt) and not keep_unified_ckpt:
            try:
                os.remove(unified_ckpt)
            except Exception:
                pass
        model.eval()
        errs = []
        with torch.no_grad():
            for ic, u_true in test_loader:
                ic, u_true = ic.to(device), u_true.to(device)
                out = model(ic).squeeze(-1) * mollifier
                errs.append(lploss(out, u_true).item())
        test_err = float(np.mean(errs))
        no_adam_stats = warmstart_from_adam_file and acc_n == 0
        return {
            "status": "complete",
            "train_loss": float("nan")
            if no_adam_stats
            else acc_total / max(1, acc_n),
            "data_l2": float("nan") if no_adam_stats else acc_data / max(1, acc_n),
            "pde_res": float("nan") if no_adam_stats else acc_pde / max(1, acc_n),
            "test_error": test_err,
            "test_after_adam": test_err,
            "train_loss_nncg": float("nan"),
            "data_l2_nncg": float("nan"),
            "pde_res_nncg": float("nan"),
        }

    # Prepare second-phase optimizer (full batch) after Adam
    if phase == "adam":
        model.eval()
        errs_adam = []
        with torch.no_grad():
            for ic, u_true in test_loader:
                ic, u_true = ic.to(device), u_true.to(device)
                out = model(ic).squeeze(-1) * mollifier
                errs_adam.append(lploss(out, u_true).item())
        test_after_adam = float(np.mean(errs_adam))

        full_loader = torch.utils.data.DataLoader(
            ds_train, batch_size=n_samples, shuffle=False, drop_last=False
        )
        ic_all, u_all = next(iter(full_loader))
        ic_all = ic_all.to(device)
        u_all = u_all.to(device)

        if ro == "nncg":
            _dm = (nncg_data_mode or "preload").lower()
            use_chunked = int(nncg_fb_chunk) > 0 or _dm in (
                "loader_full",
                "loader_minibatch",
            )
            NncgCls = ChunkedNysNewtonCG if use_chunked else NysNewtonCG
            optim_refine = NncgCls(
                model.parameters(),
                lr=nncg_lr,
                rank=nncg_rank,
                mu=nncg_mu,
                cg_tol=nncg_cg_tol,
                cg_max_iters=nncg_cg_max_iters,
                chunk_size=nncg_chunk_size,
                line_search_fn="armijo",
            )
        else:
            optim_refine = torch.optim.LBFGS(
                model.parameters(),
                lr=lbfgs_lr,
                max_iter=lbfgs_max_iter,
                line_search_fn="strong_wolfe",
            )
        nncg_completed = 0
        msg_done = f"adam_phase_done test_after_adam={test_after_adam:.6f}; start {ro.upper()} "
        msg_done += (
            "(L-BFGS full-batch, first-order)"
            if ro == "lbfgs"
            else "(Nyström Newton-CG / NNCG)"
        )
        _append_training_log(log_path, msg_done)
        if unified_ckpt:
            save_refine_ckpt(0, model, optim_refine, test_after_adam, ro)

    assert ic_all is not None and optim_refine is not None

    nncg_subsample_state = None
    nncg_loader_state = None
    _data_mode = (nncg_data_mode or "preload").lower()
    if ro == "nncg" and int(nncg_subsample_size) > 0 and int(nncg_fb_chunk) <= 0:
        raise ValueError(
            "--nncg-subsample-size requires --nncg-fb-chunk > 0 (chunked NNCG path)"
        )
    if ro == "nncg" and _data_mode in ("loader_full", "loader_minibatch"):
        if int(nncg_fb_chunk) <= 0 and _data_mode == "loader_full":
            raise ValueError(
                "--nncg-data-mode loader_full requires --nncg-fb-chunk > 0"
            )
        lb = int(nncg_fb_chunk) if _data_mode == "loader_full" else int(batchsize)
        nncg_loader_state = NncgLoaderState(
            _data_mode,
            ds_train,
            int(ic_all.shape[0]),
            device,
            loader_batch_size=max(1, lb),
            minibatch_size=int(batchsize),
        )
        print(
            f"  [nncg] data_mode={_data_mode}  "
            f"loader_batch={nncg_loader_state.loader_batch_size}  "
            f"minibatch={nncg_loader_state.minibatch_size}",
            flush=True,
        )
        _append_training_log(
            log_path,
            f"nncg_data_mode={_data_mode} loader_batch={nncg_loader_state.loader_batch_size}",
        )
    elif (
        ro == "nncg"
        and int(nncg_fb_chunk) > 0
        and int(nncg_subsample_size) > 0
    ):
        nncg_subsample_state = NncgSubsampleState(
            int(ic_all.shape[0]), int(nncg_subsample_size), device
        )
        if nncg_subsample_state.enabled:
            print(
                f"  [nncg] subsampled grad/HVP: "
                f"{nncg_subsample_state.subsample_size}/{nncg_subsample_state.n_total} "
                f"per outer step (full-batch metrics unchanged)",
                flush=True,
            )
            _append_training_log(
                log_path,
                f"nncg_subsample={nncg_subsample_state.subsample_size}/"
                f"{nncg_subsample_state.n_total}",
            )

    model.train()

    def eval_fullbatch_train_loss() -> float:
        """Scalar PINO/PINN objective on full train batch (no grad); matches end-of-run metric."""
        model.eval()
        with torch.no_grad():
            out = model(ic_all).squeeze(-1) * mollifier
            a = ic_all[..., 0]
            l_pde_fb = darcy_loss(out, a)
            if loss_mode == "pinn":
                total_fb = (f_loss * l_pde_fb).item()
            else:
                total_fb = (
                    xy_loss * lploss(out, u_all) + f_loss * l_pde_fb
                ).item()
        model.train()
        return float(total_fb)

    def closure_nncg():
        optim_refine.zero_grad(set_to_none=True)
        out = model(ic_all).squeeze(-1) * mollifier
        a = ic_all[..., 0]
        l_pde = darcy_loss(out, a)
        if loss_mode == "pinn":
            loss = f_loss * l_pde
        else:
            l_data = lploss(out, u_all)
            loss = xy_loss * l_data + f_loss * l_pde
        loss = _nncg_real_loss(loss)
        grad_tuple = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_tuple = _nncg_real_grad_tuple(grad_tuple)
        return loss, grad_tuple

    def closure_lbfgs():
        optim_refine.zero_grad(set_to_none=True)
        out = model(ic_all).squeeze(-1) * mollifier
        a = ic_all[..., 0]
        l_pde = darcy_loss(out, a)
        if loss_mode == "pinn":
            loss = f_loss * l_pde
        else:
            l_data = lploss(out, u_all)
            loss = xy_loss * l_data + f_loss * l_pde
        loss = _nncg_real_loss(loss)
        loss.backward()
        return loss

    if isinstance(optim_refine, ChunkedNysNewtonCG):
        params_list = list(model.parameters())
        _chunked_grad_fn = make_chunked_grad_fn(
            model, ic_all, u_all, mollifier, lploss, darcy_loss,
            xy_loss, f_loss, params_list, int(nncg_fb_chunk),
            loss_mode=loss_mode,
            subsample_state=nncg_subsample_state,
            loader_state=nncg_loader_state,
        )
        _chunked_hvp_fn = make_chunked_hvp_fn(
            model, ic_all, u_all, mollifier, lploss, darcy_loss,
            xy_loss, f_loss, params_list, int(nncg_fb_chunk),
            loss_mode=loss_mode,
            subsample_state=nncg_subsample_state,
            loader_state=nncg_loader_state,
        )
        _chunked_loss_fn = make_chunked_loss_fn(
            model, ic_all, u_all, mollifier, lploss, darcy_loss,
            xy_loss, f_loss, int(nncg_fb_chunk),
            loss_mode=loss_mode,
            subsample_state=nncg_subsample_state,
            loader_state=nncg_loader_state,
        )
        optim_refine.attach_callbacks(
            grad_fn=_chunked_grad_fn,
            hvp_fn=_chunked_hvp_fn,
            loss_fn=_chunked_loss_fn,
        )

    def _eval_test_err():
        model.eval()
        errs = []
        with torch.no_grad():
            for ic, u_true in test_loader:
                ic, u_true = ic.to(device), u_true.to(device)
                out = model(ic).squeeze(-1) * mollifier
                errs.append(lploss(out, u_true).item())
        return float(np.mean(errs))

    k = nncg_completed
    while k < nncg_steps:
        if _should_stop(deadline, stop_flag):
            save_refine_ckpt(k, model, optim_refine, test_after_adam, ro)
            return {
                "status": "stopped" if stop_flag[0] else "wall_timeout",
                "train_loss": float("nan"),
                "data_l2": float("nan"),
                "pde_res": float("nan"),
                "test_error": float("nan"),
                "test_after_adam": test_after_adam,
                "train_loss_nncg": float("nan"),
                "data_l2_nncg": float("nan"),
                "pde_res_nncg": float("nan"),
            }

        if ro == "nncg":
            if isinstance(optim_refine, ChunkedNysNewtonCG):
                if nncg_subsample_state is not None:
                    nncg_subsample_state.refresh(k)
                if nncg_loader_state is not None:
                    nncg_loader_state.refresh_minibatch(k)
                if (k % max(1, int(nncg_precond_update_freq))) == 0:
                    optim_refine.update_preconditioner_chunked()
                optim_refine.step_chunked()
            else:
                if (k % max(1, int(nncg_precond_update_freq))) == 0:
                    _, grad_tuple = closure_nncg()
                    optim_refine.update_preconditioner(grad_tuple)
                optim_refine.step(closure_nncg)
        else:
            optim_refine.step(closure_lbfgs)

        k += 1
        if torch.cuda.is_available() and (k % 5 == 0):
            torch.cuda.empty_cache()

        _nn_prog = 10
        tag = "lbfgs" if ro == "lbfgs" else "nncg"
        if k == 1 or (k % _nn_prog == 0):
            fb = eval_fullbatch_train_loss()
            extra_test = ""
            test_every = int(nncg_test_every or 0)
            if test_every > 0 and (k == 1 or (k % test_every == 0)):
                te_curr = _eval_test_err()
                extra_test = f"  test={te_curr:.6f}"
                _append_training_log(
                    log_path,
                    f"{tag} step={k}/{nncg_steps} test={te_curr:.6f}",
                )
            print(
                f"  [{tag}] step {k}/{nncg_steps}  train_loss_fb={fb:.6g}{extra_test}",
                flush=True,
            )
            _append_training_log(
                log_path,
                f"{tag} step={k}/{nncg_steps} train_loss_fb={fb:.6g}",
            )

        if unified_ckpt and (k % max(1, ckpt_every) == 0 or k == nncg_steps):
            save_refine_ckpt(k, model, optim_refine, test_after_adam, ro)

    nncg_completed = k

    # Full-batch scalar losses (no graph)
    model.eval()
    with torch.no_grad():
        out = model(ic_all).squeeze(-1) * mollifier
        a = ic_all[..., 0]
        l_pde_fb = darcy_loss(out, a)
        if loss_mode == "pinn":
            total_fb = (f_loss * l_pde_fb).item()
            data_fb = float("nan")
        else:
            l_data_fb = lploss(out, u_all)
            data_fb = l_data_fb.item()
            total_fb = (xy_loss * l_data_fb + f_loss * l_pde_fb).item()

    test_err = _eval_test_err()

    if progress_ckpt_path and os.path.exists(progress_ckpt_path):
        try:
            os.remove(progress_ckpt_path)
        except Exception:
            pass
    if unified_ckpt and os.path.isfile(unified_ckpt) and not keep_unified_ckpt:
        try:
            os.remove(unified_ckpt)
        except Exception:
            pass

    _append_training_log(
        log_path,
        f"complete test_error={test_err:.6f} train_loss_nncg={total_fb:.6f}",
    )

    return {
        "status": "complete",
        "train_loss": total_fb,
        "data_l2": data_fb if loss_mode == "pino" else 0.0,
        "pde_res": l_pde_fb.item(),
        "test_error": test_err,
        "test_after_adam": test_after_adam,
        "train_loss_nncg": total_fb,
        "data_l2_nncg": data_fb,
        "pde_res_nncg": l_pde_fb.item(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--r", type=float, nargs="+", default=[2, 4, 8])
    p.add_argument("--n_samples", type=int, nargs="+", default=[100, 500, 1000])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--adam-steps", type=int, default=10000)
    p.add_argument("--nncg-steps", type=int, default=2000)
    p.add_argument("--batchsize", type=int, default=20, help="Adam minibatch size")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--loss_mode",
        type=str,
        default="pino",
        choices=["pino", "pinn"],
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="/pscratch/sd/w/wyx345/pino/sweep/darcy_adam_nncg",
    )
    p.add_argument("--progress_dir", type=str, default=None)
    p.add_argument("--ckpt_every", type=int, default=100)
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Unified Adam+NNCG checkpoints + logs (per-cell .pt and .log)",
    )
    p.add_argument(
        "--adam-ckpt-dir",
        type=str,
        default=None,
        help="Directory with darcy_sweep.py Adam checkpoints "
        "(files named r{n}_n{m}_s{s}_adam.pt); skip Adam training and run NNCG only.",
    )
    p.add_argument(
        "--wall-seconds",
        type=float,
        default=None,
        dest="wall_seconds",
        help="Segment time budget in seconds; exit 99 for resubmit. "
        "Env SEGMENT_WALL_SECONDS overrides if this is unset.",
    )
    p.add_argument("--shard_id", type=int, default=-1)
    p.add_argument("--n_shards", type=int, default=8)
    p.add_argument("--base-lr", type=float, default=1e-3)
    p.add_argument("--nncg-lr", type=float, default=1.0)
    p.add_argument("--nncg-rank", type=int, default=10)
    p.add_argument("--nncg-mu", type=float, default=1e-2)
    p.add_argument("--nncg-cg-tol", type=float, default=1e-5)
    p.add_argument("--nncg-cg-max-iters", type=int, default=100)
    p.add_argument("--nncg-precond-update-freq", type=int, default=1)
    p.add_argument("--nncg-chunk-size", type=int, default=1)
    p.add_argument(
        "--nncg-fb-chunk",
        type=int,
        default=0,
        dest="nncg_fb_chunk",
        help="Full-batch chunk size for NNCG grad/HVP. 0 disables (legacy retained-graph path; "
        "OOM-prone at large n). Set e.g. 50 to re-forward in chunks of 50 and avoid OOM.",
    )
    p.add_argument(
        "--nncg-subsample-size",
        type=int,
        default=0,
        dest="nncg_subsample_size",
        help="If >0 and < n_samples, each NNCG outer step uses a random subset of this "
        "many training points for grad/HVP/Armijo (unbiased mean). 0 = full batch. "
        "Requires --nncg-fb-chunk > 0.",
    )
    p.add_argument(
        "--nncg-data-mode",
        type=str,
        default="preload",
        choices=("preload", "loader_full", "loader_minibatch"),
        dest="nncg_data_mode",
        help="NNCG data pipeline: preload=ic_all tensor (default); "
        "loader_full=DataLoader microbatch accumulation (full-batch objective); "
        "loader_minibatch=one Adam-style minibatch per NNCG step.",
    )
    p.add_argument(
        "--nncg-test-every",
        type=int,
        default=0,
        dest="nncg_test_every",
        help="If >0, evaluate test_error every K NNCG steps and log it (for finding the sweet-spot step count).",
    )
    p.add_argument(
        "--refine-optimizer",
        type=str,
        default="lbfgs",
        choices=("lbfgs", "nncg"),
        help="Second phase after Adam: lbfgs (default, first-order, OOM-safe) or nncg (Nyström Newton-CG, HVP).",
    )
    p.add_argument(
        "--lbfgs-lr",
        type=float,
        default=1.0,
        help="LR for L-BFGS second phase (refine-optimizer lbfgs).",
    )
    p.add_argument(
        "--lbfgs-max-iter",
        type=int,
        default=20,
        help="max_iter per L-BFGS outer step (PyTorch LBFGS inner iterations).",
    )
    p.add_argument(
        "--keep-unified-ckpt",
        action="store_true",
        help="Do not delete the unified checkpoint after a successful run (for optimizer chains).",
    )
    args = p.parse_args()

    wall_s = args.wall_seconds
    if wall_s is None and os.environ.get("SEGMENT_WALL_SECONDS"):
        try:
            wall_s = float(os.environ["SEGMENT_WALL_SECONDS"])
        except ValueError:
            wall_s = None
    if wall_s is not None and wall_s <= 0:
        wall_s = None

    outdir = os.path.join(args.outdir, args.loss_mode)
    os.makedirs(outdir, exist_ok=True)
    if args.progress_dir:
        os.makedirs(args.progress_dir, exist_ok=True)
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.adam_ckpt_dir:
        os.makedirs(args.adam_ckpt_dir, exist_ok=True)
    csv_path = os.path.join(outdir, "results.csv")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    fieldnames = [
        "r",
        "n_samples",
        "seed",
        "adam_steps",
        "nncg_steps",
        "train_loss",
        "data_l2",
        "pde_res",
        "test_after_adam",
        "test_error",
        "elapsed_s",
    ]

    done = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                try:
                    done.add(
                        (
                            float(row["r"]),
                            int(row["n_samples"]),
                            int(row["seed"]),
                        )
                    )
                except (ValueError, TypeError, KeyError):
                    pass

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    print(
        f'Adam + refine({args.refine_optimizer}) sweep → {csv_path}  '
        f"(adam={args.adam_steps}, refine_steps={args.nncg_steps})",
        flush=True,
    )
    if wall_s:
        print(f"  segment wall: {wall_s:.0f}s", flush=True)

    # Match darcy_sweep.py: one MAT pool per r (max n), each cell uses first n_samples.
    n_pool_train = max(args.n_samples)
    n_pool_test = 200

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        total = len(args.r) * len(args.n_samples) * len(args.seeds)
        idx = 0
        for r in args.r:
            train_path = ds.ensure_data(r, seed=0, n_samples=n_pool_train, f=1.0)
            test_path = ds.ensure_data(r, seed=1, n_samples=n_pool_test, f=1.0)
            for n_samples in args.n_samples:
                for seed in args.seeds:
                    idx += 1
                    if (
                        args.shard_id >= 0
                        and ds.cell_shard(r, n_samples, seed, args.n_shards)
                        != args.shard_id
                    ):
                        continue
                    if (r, n_samples, seed) in done:
                        print(f"[{idx}/{total}] SKIP r={r} n={n_samples} seed={seed}")
                        continue

                    print(
                        f"\n[{idx}/{total}] r={r} n={n_samples} seed={seed} "
                        f"adam={args.adam_steps} refine={args.refine_optimizer} "
                        f"steps={args.nncg_steps}",
                        flush=True,
                    )
                    t0 = time.time()
                    tag = f"r{ds.r_sweep_token(r)}_n{n_samples}_s{seed}"
                    progress_ckpt = (
                        os.path.join(args.progress_dir, f"{tag}_progress.pt")
                        if args.progress_dir
                        else None
                    )
                    ckpt_path = (
                        os.path.join(args.checkpoint_dir, f"{tag}_adam_nncg.pt")
                        if args.checkpoint_dir
                        else None
                    )
                    log_path = (
                        os.path.join(args.checkpoint_dir, f"{tag}.log")
                        if args.checkpoint_dir
                        else None
                    )
                    load_adam = (
                        os.path.join(args.adam_ckpt_dir, f"{tag}_adam.pt")
                        if args.adam_ckpt_dir
                        else None
                    )
                    result = train_adam_then_nncg(
                        train_path,
                        test_path,
                        n_samples,
                        args.adam_steps,
                        args.nncg_steps,
                        device,
                        seed,
                        loss_mode=args.loss_mode,
                        progress_ckpt_path=progress_ckpt,
                        ckpt_every=args.ckpt_every,
                        checkpoint_path=ckpt_path,
                        log_path=log_path,
                        wall_seconds=wall_s,
                        load_adam_ckpt_path=load_adam,
                        batchsize=args.batchsize,
                        base_lr=args.base_lr,
                        nncg_lr=args.nncg_lr,
                        nncg_rank=args.nncg_rank,
                        nncg_mu=args.nncg_mu,
                        nncg_cg_tol=args.nncg_cg_tol,
                        nncg_cg_max_iters=args.nncg_cg_max_iters,
                        nncg_precond_update_freq=args.nncg_precond_update_freq,
                        nncg_chunk_size=args.nncg_chunk_size,
                        nncg_fb_chunk=args.nncg_fb_chunk,
                        nncg_subsample_size=args.nncg_subsample_size,
                        nncg_data_mode=args.nncg_data_mode,
                        nncg_test_every=args.nncg_test_every,
                        refine_optimizer=args.refine_optimizer,
                        lbfgs_lr=args.lbfgs_lr,
                        lbfgs_max_iter=args.lbfgs_max_iter,
                        keep_unified_ckpt=args.keep_unified_ckpt,
                    )
                    st = result.get("status", "complete")
                    if st in ("wall_timeout", "stopped"):
                        print(
                            f"  [{st}] checkpoint saved; exiting {EXIT_RESUBMIT} for next segment",
                            flush=True,
                        )
                        sys.exit(EXIT_RESUBMIT)

                    elapsed = time.time() - t0

                    print(
                        f'  → test_adam={result["test_after_adam"]:.4f}  '
                        f'test_final={result["test_error"]:.4f}  '
                        f'({elapsed:.0f}s)',
                        flush=True,
                    )

                    writer.writerow(
                        {
                            "r": r,
                            "n_samples": n_samples,
                            "seed": seed,
                            "adam_steps": args.adam_steps,
                            "nncg_steps": args.nncg_steps,
                            "train_loss": round(result["train_loss"], 6),
                            "data_l2": round(result["data_l2"], 6),
                            "pde_res": round(result["pde_res"], 6),
                            "test_after_adam": round(result["test_after_adam"], 6),
                            "test_error": round(result["test_error"], 6),
                            "elapsed_s": round(elapsed, 1),
                        }
                    )
                    csvfile.flush()

    print(f"\nDone. Results: {csv_path}")


if __name__ == "__main__":
    main()
