from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torchdiffeq
from scipy.integrate import solve_ivp


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pendulum(t, x, b=0.0, k=0.0, drag=0.0):
    theta, velocity = x
    theta_dot = velocity
    velocity_dot = -b * velocity - np.sin(theta) + drag * np.cos(k * t)
    return theta_dot, velocity_dot


def spherical_pendulum(v):
    x = np.sin(v[0]) * np.cos(v[1])
    y = np.sin(v[0]) * np.sin(v[1])
    z = -np.cos(v[0])
    return np.vstack((x, y, z))


def shallow(in_dim, hidden, out_dim, Act=torch.nn.Tanh):
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden),
        Act(),
        torch.nn.Linear(hidden, hidden),
        Act(),
        torch.nn.Linear(hidden, hidden),
        Act(),
        torch.nn.Linear(hidden, out_dim),
    )


class ShallowODE(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden=10, Act=torch.nn.Tanh):
        super().__init__()
        self.net = shallow(in_dim, hidden, out_dim, Act=Act)

    def forward(self, t, x):
        return self.net(x)


def numpy_to_torch(array, dtype=None, device=None):
    try:
        tensor = torch.from_numpy(array)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor
    except RuntimeError as exc:
        if "Numpy is not available" not in str(exc):
            raise
        return torch.tensor(array.tolist(), dtype=dtype, device=device)


def build_data(b, dt, dt_test, tmax, train_theta, test_theta, representation="embedded"):
    grid = np.arange(0, tmax, dt)
    grid_test = np.arange(0, tmax, dt_test)

    xtrain = solve_ivp(
        pendulum,
        (0, tmax),
        y0=np.array([train_theta, 0.0]),
        args=(b, 0.0),
        t_eval=grid,
    )
    xtest = solve_ivp(
        pendulum,
        (0, tmax),
        y0=np.array([test_theta, 0.0]),
        args=(b, 0.0),
        t_eval=grid_test,
    )

    dt_train = (xtrain.t[1:] - xtrain.t[:-1]).reshape(-1, 1)
    dt_test = (xtest.t[1:] - xtest.t[:-1]).reshape(-1, 1)

    if representation == "embedded":
        xtrain = spherical_pendulum(xtrain.y)
        xtest = spherical_pendulum(xtest.y)
    elif representation == "state":
        xtrain = xtrain.y
        xtest = xtest.y
    else:
        raise ValueError(f"Unsupported representation: {representation}")

    return xtrain, xtest, dt_train, dt_test


def build_loader(xtrain, dt_train, batch_size, device=None):
    dev = torch.device(device) if device is not None else torch.device("cpu")
    inputs_train = numpy_to_torch(xtrain[:, 0:-1].T, dtype=torch.float32, device=dev)
    targets_train = numpy_to_torch(xtrain[:, 1:].T, dtype=torch.float32, device=dev)
    idx = numpy_to_torch(np.arange(inputs_train.shape[0]), dtype=torch.long, device=dev)
    dt_train = numpy_to_torch(dt_train, dtype=torch.float32, device=dev)

    train_data = torch.utils.data.TensorDataset(
        *(inputs_train, targets_train, dt_train, idx)
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=False
    )
    return train_loader


def freeze_region_noise(inputs, sample_count, effective_radius):
    if effective_radius <= 0.0 or sample_count <= 1:
        return None
    return torch.empty(
        (sample_count - 1, *inputs.shape), dtype=inputs.dtype, device=inputs.device
    ).uniform_(-effective_radius, effective_radius)


def compute_batch_losses(
    model,
    inputs,
    targets,
    batch_idx,
    dt_step,
    method,
    physics_mode,
    damp_b,
    lamb,
    rho,
    region_samples,
    region_max_radius,
    region_gradient_variance,
    constraint_method="penalty",
    alm_dual_batch=None,
    alm_penalty=1.0,
    region_noise=None,
    alm_target="physics",
    alm_uncon_weight=1.0,
):
    preds = torchdiffeq.odeint(
        model,
        inputs,
        torch.tensor([0.0, dt_step], dtype=inputs.dtype, device=inputs.device),
        method=method,
    )[-1, :, :]
    loss_fn = torch.nn.MSELoss().to(inputs.device)
    L1 = loss_fn(targets, preds)

    if physics_mode == "none":
        L2 = torch.zeros((), dtype=preds.dtype, device=preds.device)
        L3 = torch.zeros((), dtype=preds.dtype, device=preds.device)
    elif physics_mode == "sphere":
        g = preds[:, 0] ** 2 + preds[:, 1] ** 2 + preds[:, 2] ** 2 - 1.0
        L2 = torch.mean(torch.abs(g))
        L3 = torch.mean(g**2)
    elif physics_mode in {"pinn", "pinn_region"}:
        if inputs.shape[1] != 2:
            raise ValueError(
                f"physics_mode='{physics_mode}' expects state representation with inputs=[theta, omega]."
            )

        if physics_mode == "pinn_region":
            sample_count = max(1, int(region_samples))
            effective_radius = float(
                np.clip(
                    rho / max(region_gradient_variance, 1e-6),
                    a_min=0.0,
                    a_max=region_max_radius,
                )
            )
            if region_noise is None:
                region_noise = freeze_region_noise(inputs, sample_count, effective_radius)
            if region_noise is not None:
                sampled_inputs = [inputs]
                for sample_noise in region_noise:
                    sampled_inputs.append(inputs + sample_noise)
                region_inputs = torch.cat(sampled_inputs, dim=0)
            else:
                region_inputs = inputs
        else:
            region_inputs = inputs

        theta = region_inputs[:, 0]
        omega = region_inputs[:, 1]
        f_pred = model(torch.tensor(0.0, dtype=inputs.dtype, device=inputs.device), region_inputs)
        residual_theta = f_pred[:, 0] - omega
        residual_omega = f_pred[:, 1] + damp_b * omega + torch.sin(theta)
        residual = torch.stack((residual_theta, residual_omega), dim=1)
        L2 = torch.mean(torch.abs(residual))
        L3 = (dt_step**2) * torch.mean(residual**2)
    elif physics_mode == "pinn_alm":
        # ALM variant of PINN: state representation (theta, omega), 2-D inputs.
        # Supports two targets for the ALM constraint:
        #   alm_target='physics': minimize L_data, constrain ODE residual = 0
        #   alm_target='data':    minimize L_physics, constrain pred = target
        if inputs.shape[1] != 2:
            raise ValueError("physics_mode='pinn_alm' expects state representation with inputs=[theta, omega].")
        theta = inputs[:, 0]
        omega = inputs[:, 1]
        f_pred = model(torch.tensor(0.0, dtype=inputs.dtype, device=inputs.device), inputs)
        residual_theta = f_pred[:, 0] - omega
        residual_omega = f_pred[:, 1] + damp_b * omega + torch.sin(theta)
        residual = torch.stack((residual_theta, residual_omega), dim=1)   # [N, 2]
        data_err = preds - targets                                          # [N, 2]
        L2 = torch.mean(torch.abs(residual))
        L3 = (dt_step**2) * torch.mean(residual**2)
    else:
        raise ValueError(f"Unsupported physics_mode: {physics_mode}")

    # pinn_alm is always ALM-based; bypass the generic constraint_method dispatch.
    # alm_uncon_weight scales the unconstrained objective (analogous to Darcy uncon_weight).
    if physics_mode == "pinn_alm":
        if alm_target == "physics":
            # Unconstrained: alm_uncon_weight * L_data  |  Constrained: ODE residual = 0
            g_out = residual                                                  # [N, 2]
            if alm_dual_batch is not None:
                dual_term = torch.mean(alm_dual_batch * g_out)
            else:
                dual_term = torch.zeros((), dtype=preds.dtype, device=preds.device)
            L = alm_uncon_weight * L1 + dual_term + 0.5 * alm_penalty * (dt_step**2) * torch.mean(g_out**2)
        else:  # alm_target == "data"
            # Unconstrained: alm_uncon_weight * L_physics  |  Constrained: pred = target
            g_out = data_err                                                  # [N, 2]
            if alm_dual_batch is not None:
                dual_term = torch.mean(alm_dual_batch * g_out)
            else:
                dual_term = torch.zeros((), dtype=preds.dtype, device=preds.device)
            L = alm_uncon_weight * L3 + dual_term + 0.5 * alm_penalty * torch.mean(g_out**2)
        return L1, L2, L3, L, g_out

    if constraint_method == "penalty":
        if lamb != 0.0 and physics_mode != "none":
            L = L1 + lamb * L3
        else:
            L = L1
    elif constraint_method == "alm":
        if physics_mode == "sphere":
            if alm_dual_batch is None:
                raise ValueError("constraint_method='alm' requires alm_dual_batch for physics_mode='sphere'.")
            dual_term = torch.mean(alm_dual_batch * g)
            L = L1 + dual_term + 0.5 * alm_penalty * L3
        else:
            raise ValueError(
                f"constraint_method='alm' (legacy) is not supported for physics_mode='{physics_mode}'. "
                "Use physics_mode='pinn_alm'."
            )
    else:
        raise ValueError(f"Unsupported constraint_method: {constraint_method}")

    g_out = g if physics_mode == "sphere" else None
    return L1, L2, L3, L, g_out


def loader_to_full_batch(data):
    inputs_all = []
    targets_all = []
    dt_all = []
    idx_all = []
    for inputs, targets, dt, idx in data:
        inputs_all.append(inputs)
        targets_all.append(targets)
        dt_all.append(dt)
        idx_all.append(idx)
    return (
        torch.cat(inputs_all, dim=0),
        torch.cat(targets_all, dim=0),
        torch.cat(dt_all, dim=0),
        torch.cat(idx_all, dim=0),
    )


def eval_pinn_alm_full_batch(
    model,
    data,
    *,
    method,
    damp_b,
    lamb,
    rho,
    region_samples,
    region_max_radius,
    region_gradient_variance,
    alm_target,
    alm_uncon_weight,
    alm_penalty,
    lam=None,
):
    """Full-train-loader ALM losses (no grad); used for init/outer-0/final logging."""
    model.eval()
    inp, tgt, dt_t, idx_t = loader_to_full_batch(data)
    dt_step = float(dt_t[0].item())
    batch_idx = idx_t.long()
    if lam is None:
        alm_dual = None
        lam_mean = lam_std = 0.0
    else:
        alm_dual = lam[batch_idx.cpu()].to(device=inp.device, dtype=inp.dtype)
        lam_np = lam.detach().cpu().numpy()
        lam_mean = float(lam_np.mean())
        lam_std = float(lam_np.std())
    with torch.no_grad():
        l1, l2, l3, lt, _ = compute_batch_losses(
            model=model,
            inputs=inp,
            targets=tgt,
            batch_idx=batch_idx,
            dt_step=dt_step,
            method=method,
            physics_mode="pinn_alm",
            damp_b=damp_b,
            lamb=lamb,
            rho=rho,
            region_samples=region_samples,
            region_max_radius=region_max_radius,
            region_gradient_variance=region_gradient_variance,
            alm_dual_batch=alm_dual,
            alm_penalty=alm_penalty,
            alm_target=alm_target,
            alm_uncon_weight=alm_uncon_weight,
        )
    return {
        "L_data": float(l1.detach().item()),
        "L_resid_l1": float(l2.detach().item()),
        "L_phys": float(l3.detach().item()),
        "L_total": float(lt.detach().item()),
        "lam_mean": lam_mean,
        "lam_std": lam_std,
    }


def eval_test_rollout_rel_l2(net, xtest, dt_test, train_dev) -> float:
    inputs_test = numpy_to_torch(xtest, dtype=torch.float32, device=train_dev)
    dt_test_t = numpy_to_torch(dt_test, dtype=torch.float32, device=train_dev)
    with torch.no_grad():
        _, _, _, l2 = rollout_metrics(inputs_test, model=net, dt=dt_test_t, method="euler")
    return float(l2)


def log_pinn_alm_metrics(
    prefix: str,
    tag: str,
    metrics: dict,
    mu: float,
    *,
    lam_note: str = "",
) -> None:
    """Print train-batch loss terms + test rollout rel-L2."""
    lam_part = (
        f"  lam(mean={metrics['lam_mean']:.4g} std={metrics['lam_std']:.4g})"
        if metrics.get("lam_mean", 0.0) != 0.0 or metrics.get("lam_std", 0.0) != 0.0
        else "  lam=0"
    )
    if lam_note:
        lam_part = f"  {lam_note}"
    err = metrics.get("test_rollout_l2")
    err_s = f"{err:.6g}" if err is not None and np.isfinite(err) else "nan"
    print(
        f"{prefix}[ALM] {tag}\n"
        f"  L_data={metrics['L_data']:.6g}  L_resid_l1={metrics['L_resid_l1']:.6g}  "
        f"L_phys={metrics['L_phys']:.6g}  L_total={metrics['L_total']:.6g}  mu={mu:g}"
        f"{lam_part}\n"
        f"  test_rollout_l2={err_s}",
        flush=True,
    )


def eval_pinn_alm_cell_metrics(
    net,
    data,
    xtest,
    dt_test,
    train_dev: torch.device,
    *,
    lam: torch.Tensor | None,
    eval_mu: float,
    **eval_kw,
) -> dict:
    m = eval_pinn_alm_full_batch(
        model=net,
        data=data,
        lam=lam,
        alm_penalty=eval_mu,
        **eval_kw,
    )
    m["test_rollout_l2"] = eval_test_rollout_rel_l2(net, xtest, dt_test, train_dev)
    return m


def compare_lbfgs_pretrain_vs_alm_trained(
    net,
    lbfgs_state: dict,
    data,
    xtest,
    dt_test,
    train_dev: torch.device,
    *,
    prefix: str,
    eval_mu: float,
    **eval_kw,
) -> dict:
    """Side-by-side: LBFGS init ckpt vs weights after ALM (eval at λ=0, same μ)."""
    final_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    def _eval(state: dict, label: str) -> dict:
        net.load_state_dict({k: v.to(train_dev) for k, v in state.items()})
        m = eval_pinn_alm_cell_metrics(
            net,
            data,
            xtest,
            dt_test,
            train_dev,
            lam=None,
            eval_mu=eval_mu,
            **eval_kw,
        )
        m["_label"] = label
        return m

    pre = _eval(lbfgs_state, "LBFGS (pretrain ckpt)")
    post = _eval(final_state, "ALM (after training)")
    net.load_state_dict({k: v.to(train_dev) for k, v in final_state.items()})

    keys = ("L_data", "L_resid_l1", "L_phys", "L_total", "test_rollout_l2")
    print(
        f"{prefix}[ALM] compare LBFGS pretrain vs ALM trained "
        f"(full train batch, λ=0, μ={eval_mu:g} for L_total)",
        flush=True,
    )
    for key in keys:
        a, b = pre[key], post[key]
        d = b - a
        winner = ""
        if key == "test_rollout_l2" or key.startswith("L_"):
            if np.isfinite(a) and np.isfinite(b):
                winner = "  ← ALM better" if b < a else ("  ← LBFGS better" if b > a else "  (tie)")
        print(f"  {key:<16}  LBFGS={a:.6g}  ALM={b:.6g}  delta={d:+.6g}{winner}", flush=True)

    te_pre, te_post = pre["test_rollout_l2"], post["test_rollout_l2"]
    if np.isfinite(te_pre) and np.isfinite(te_post) and te_pre > 0:
        print(
            f"  rollout ratio (ALM/LBFGS) = {te_post / te_pre:.4g}",
            flush=True,
        )
    summary = {"lbfgs_pretrain": pre, "alm_trained": post}
    return summary


def train_odenet(
    data,
    model,
    method="euler",
    learning_rate=1e-3,
    weight_decay=0.0,
    epochs=400,
    opti="Adam",
    lamb=0.0,
    rho=0.0,
    physics_mode="sphere",
    damp_b=0.0,
    optimizer_name="Adam",
    updates_per_epoch=None,
    region_samples=4,
    region_history=10,
    region_max_radius=0.05,
    nncg_rank=10,
    nncg_mu=1e-2,
    nncg_cg_tol=1e-5,
    nncg_cg_max_iters=100,
    nncg_precond_update_freq=1,
    nncg_switch_epoch=None,
    lbfgs_max_iter=10000,
    nncg_epochs=100,
    phase_ckpt_dir: Path | None = None,
    phase_ckpt_stem: str = "",
    constraint_method="penalty",
    alm_penalty=1.0,
    alm_penalty_growth=1.0,
    alm_target="physics",
    alm_uncon_weight=1.0,
    alm_outer_iters=10,
    alm_inner_step=100,
    alm_warmup_epochs=0,
    log_loss_every: int = 1,
    log_prefix: str = "",
):
    if optimizer_name == "Adam_LBFGS_NNCG":
        if physics_mode == "pinn_alm":
            raise ValueError("Adam_LBFGS_NNCG does not support physics_mode='pinn_alm'")
        updates_pe = max(1, int(updates_per_epoch or len(data)))
        return _train_adam_lbfgs_nncg_chain(
            data,
            model,
            method=method,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_epochs=int(epochs),
            nncg_epochs=int(nncg_epochs),
            lamb=lamb,
            rho=rho,
            physics_mode=physics_mode,
            damp_b=damp_b,
            updates_per_epoch=updates_pe,
            region_samples=region_samples,
            region_history=region_history,
            region_max_radius=region_max_radius,
            nncg_rank=nncg_rank,
            nncg_mu=nncg_mu,
            nncg_cg_tol=nncg_cg_tol,
            nncg_cg_max_iters=nncg_cg_max_iters,
            nncg_precond_update_freq=nncg_precond_update_freq,
            lbfgs_max_iter=lbfgs_max_iter,
            constraint_method=constraint_method,
            alm_penalty=alm_penalty,
            alm_penalty_growth=alm_penalty_growth,
            alm_target=alm_target,
            alm_uncon_weight=alm_uncon_weight,
            log_loss_every=log_loss_every,
            log_prefix=log_prefix,
            phase_ckpt_dir=phase_ckpt_dir,
            phase_ckpt_stem=phase_ckpt_stem,
        )

    def build_nncg():
        return NysNewtonCG(
            model.parameters(),
            lr=learning_rate,
            rank=nncg_rank,
            mu=nncg_mu,
            cg_tol=nncg_cg_tol,
            cg_max_iters=nncg_cg_max_iters,
            line_search_fn="armijo",
        )

    if optimizer_name == "Adam":
        optim = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
        current_optimizer_name = "Adam"
    elif optimizer_name == "SGD":
        optim = torch.optim.SGD(model.parameters(), learning_rate, weight_decay=weight_decay)
        current_optimizer_name = "SGD"
    elif optimizer_name == "NNCG":
        optim = build_nncg()
        current_optimizer_name = "NNCG"
    elif optimizer_name == "Adam_NNCG":
        optim = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
        current_optimizer_name = "Adam"
    elif optimizer_name == "LBFGS":
        optim = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=int(lbfgs_max_iter),
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        current_optimizer_name = "LBFGS"
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    losses1 = []
    losses2 = []
    losses3 = []
    total_losses = []

    lr_scheduler = None
    if current_optimizer_name in {"Adam", "SGD"}:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.995)

    if updates_per_epoch is None:
        updates_per_epoch = len(data)
    updates_per_epoch = max(1, int(updates_per_epoch))
    region_gradient_history = []
    region_gradient_variance = 1.0
    full_batch = loader_to_full_batch(data) if optimizer_name in {"NNCG", "Adam_NNCG", "LBFGS"} else None
    if constraint_method == "alm" and physics_mode != "sphere":
        raise ValueError("constraint_method='alm' (legacy) is only supported for physics_mode='sphere'.")
    full_batch_size = len(data.dataset)
    if physics_mode == "pinn_alm":
        # dual shape [N, 2]: per-sample, per-dim (residual or data_err, both 2-D for pendulum)
        alm_dual_state = torch.zeros(full_batch_size, 2, dtype=torch.float32)
    elif constraint_method == "alm" and physics_mode == "sphere":
        alm_dual_state = torch.zeros(full_batch_size, dtype=torch.float32)
    else:
        alm_dual_state = None

    # ── pinn_alm: structured two-loop ALM (mirroring darcy_sweep.alm) ──────────
    # outer_iters × inner_step Adam steps total; λ update + μ growth once per outer iter.
    if physics_mode == "pinn_alm":
        lam = alm_dual_state                 # [N, 2], starts at zero
        mu = float(alm_penalty)
        prefix = f"{log_prefix} " if log_prefix else ""
        use_lbfgs_inner = isinstance(optim, torch.optim.LBFGS)
        alm_full_batch = loader_to_full_batch(data) if use_lbfgs_inner else None

        # ── Optional pinn+penalty warmup before ALM ──
        warmup_losses1: list[float] = []
        warmup_losses3: list[float] = []
        warmup_total_losses: list[float] = []
        if alm_warmup_epochs > 0:
            if use_lbfgs_inner:
                assert alm_full_batch is not None
                wu_inp, wu_tgt, wu_dt, wu_idx = alm_full_batch
                wu_dt_step = float(wu_dt[0].item())
                wu_batch_idx = wu_idx.long()
                print(
                    f"{prefix}[ALM warmup] LBFGS+pinn penalty  steps={alm_warmup_epochs}  "
                    f"max_iter={lbfgs_max_iter}  lamb={lamb}",
                    flush=True,
                )
                for w_ep in range(int(alm_warmup_epochs)):
                    model.train()
                    wu_last: list[torch.Tensor] = []

                    def _wu_lbfgs_closure() -> torch.Tensor:
                        optim.zero_grad()
                        l1w, _, l3w, lw, _ = compute_batch_losses(
                            model=model,
                            inputs=wu_inp,
                            targets=wu_tgt,
                            batch_idx=wu_batch_idx,
                            dt_step=wu_dt_step,
                            method=method,
                            physics_mode="pinn",
                            damp_b=damp_b,
                            lamb=lamb,
                            rho=rho,
                            region_samples=region_samples,
                            region_max_radius=region_max_radius,
                            region_gradient_variance=1.0,
                            constraint_method="penalty",
                            alm_uncon_weight=1.0,
                        )
                        lw.backward()
                        wu_last[:] = [l1w, l3w, lw]
                        return lw

                    optim.step(_wu_lbfgs_closure)
                    warmup_losses1.append(float(wu_last[0].detach().item()))
                    warmup_losses3.append(float(wu_last[1].detach().item()))
                    warmup_total_losses.append(float(wu_last[2].detach().item()))
                    if log_loss_every > 0 and (w_ep + 1) % log_loss_every == 0:
                        print(
                            f"{prefix}[ALM warmup] step {w_ep + 1}/{alm_warmup_epochs}  "
                            f"L_data={warmup_losses1[-1]:.6g}  L_phys={warmup_losses3[-1]:.6g}",
                            flush=True,
                        )
            else:
                wu_optim = optim
                wu_scheduler = torch.optim.lr_scheduler.ExponentialLR(wu_optim, gamma=0.995)
                wu_steps = max(1, len(data))
                print(
                    f"{prefix}[ALM warmup] Adam+pinn penalty  epochs={alm_warmup_epochs}  "
                    f"steps_per_epoch={wu_steps}  lamb={lamb}",
                    flush=True,
                )
                for w_ep in range(int(alm_warmup_epochs)):
                    wu_data_iter = iter(data)
                    wu_l1 = wu_l3 = wu_lt = 0.0
                    for _ws in range(wu_steps):
                        try:
                            wu_inp, wu_tgt, wu_dt, wu_idx = next(wu_data_iter)
                        except StopIteration:
                            wu_data_iter = iter(data)
                            wu_inp, wu_tgt, wu_dt, wu_idx = next(wu_data_iter)
                        model.train()
                        wu_dt_step = float(wu_dt[0].item())
                        wu_optim.zero_grad()
                        L1w, _, L3w, Lw, _ = compute_batch_losses(
                            model=model,
                            inputs=wu_inp,
                            targets=wu_tgt,
                            batch_idx=wu_idx.long(),
                            dt_step=wu_dt_step,
                            method=method,
                            physics_mode="pinn",
                            damp_b=damp_b,
                            lamb=lamb,
                            rho=rho,
                            region_samples=region_samples,
                            region_max_radius=region_max_radius,
                            region_gradient_variance=1.0,
                            constraint_method="penalty",
                            alm_uncon_weight=1.0,
                        )
                        Lw.backward()
                        wu_optim.step()
                        wu_l1 += float(L1w.detach().item())
                        wu_l3 += float(L3w.detach().item())
                        wu_lt += float(Lw.detach().item())
                    wu_scheduler.step()
                    warmup_losses1.append(wu_l1 / wu_steps)
                    warmup_losses3.append(wu_l3 / wu_steps)
                    warmup_total_losses.append(wu_lt / wu_steps)
                    if log_loss_every > 0 and (w_ep + 1) % log_loss_every == 0:
                        print(
                            f"{prefix}[ALM warmup] epoch {w_ep + 1}/{alm_warmup_epochs}  "
                            f"L_data={warmup_losses1[-1]:.6g}  L_phys={warmup_losses3[-1]:.6g}",
                            flush=True,
                        )
            print(
                f"{prefix}[ALM warmup] done  final L_data={warmup_losses1[-1]:.6g}  "
                f"L_phys={warmup_losses3[-1]:.6g}",
                flush=True,
            )

        data_iter = iter(data)

        def _pinn_alm_next():
            nonlocal data_iter
            try:
                b = next(data_iter)
            except StopIteration:
                data_iter = iter(data)
                b = next(data_iter)
            return b

        def _pinn_alm_losses(inputs, targets, dt_step, batch_idx):
            return compute_batch_losses(
                model=model,
                inputs=inputs,
                targets=targets,
                batch_idx=batch_idx,
                dt_step=dt_step,
                method=method,
                physics_mode=physics_mode,
                damp_b=damp_b,
                lamb=lamb,
                rho=rho,
                region_samples=region_samples,
                region_max_radius=region_max_radius,
                region_gradient_variance=1.0,
                alm_dual_batch=lam[batch_idx.cpu()].to(device=inputs.device, dtype=inputs.dtype),
                alm_penalty=mu,
                alm_target=alm_target,
                alm_uncon_weight=alm_uncon_weight,
            )

        inner_note = (
            f"LBFGS full-batch max_iter={lbfgs_max_iter}"
            if use_lbfgs_inner
            else f"mini-batch steps={alm_inner_step}"
        )
        print(
            f"{prefix}[ALM] cons={alm_target}  uncon_weight={alm_uncon_weight}  "
            f"mu0={mu}  rho(growth)={alm_penalty_growth}  "
            f"outer={alm_outer_iters}  inner={inner_note}  "
            f"lam_dim={lam.shape}",
            flush=True,
        )
        init_metrics = eval_pinn_alm_full_batch(
            model,
            data,
            method=method,
            damp_b=damp_b,
            lamb=lamb,
            rho=rho,
            region_samples=region_samples,
            region_max_radius=region_max_radius,
            region_gradient_variance=region_gradient_variance,
            alm_target=alm_target,
            alm_uncon_weight=alm_uncon_weight,
            alm_penalty=mu,
            lam=lam,
        )
        log_pinn_alm_metrics(prefix, f"outer 0/{alm_outer_iters}", init_metrics, mu)

        losses1_alm: list[float] = []
        losses2_alm: list[float] = []
        losses3_alm: list[float] = []
        total_losses_alm: list[float] = []

        for outer in range(int(alm_outer_iters)):
            # ── λ update: λ += μ·g (data) or λ += μ·dt²·g (physics, matches penalty scaling) ──
            model.eval()
            if use_lbfgs_inner:
                assert alm_full_batch is not None
                inp, tgt, dt_t, idx_t = alm_full_batch
            else:
                inp, tgt, dt_t, idx_t = _pinn_alm_next()
            dt_step_o = float(dt_t[0].item())
            batch_idx_o = idx_t.long()
            with torch.no_grad():
                _, _, _, _, g_o = _pinn_alm_losses(inp, tgt, dt_step_o, batch_idx_o)
            if g_o is not None:
                g_upd = g_o.detach().cpu().float()
                if alm_target == "physics":
                    g_upd = (dt_step_o**2) * g_upd
                lam[batch_idx_o.cpu()] = lam[batch_idx_o.cpu()] + mu * g_upd
            mu = mu * float(alm_penalty_growth)

            # ── inner loop: inner_step optimizer steps with fixed λ, μ ─────────
            inner_l1 = inner_l2 = inner_l3 = inner_lt = 0.0
            inner_n = int(alm_inner_step)
            for _inner in range(inner_n):
                model.train()
                if use_lbfgs_inner:
                    assert alm_full_batch is not None
                    inp, tgt, dt_t, idx_t = alm_full_batch
                else:
                    inp, tgt, dt_t, idx_t = _pinn_alm_next()
                dt_step_i = float(dt_t[0].item())
                batch_idx_i = idx_t.long()
                if use_lbfgs_inner:
                    last_losses: list[torch.Tensor] = []

                    def _alm_lbfgs_closure() -> torch.Tensor:
                        optim.zero_grad()
                        l1, l2, l3, ltot, _ = _pinn_alm_losses(
                            inp, tgt, dt_step_i, batch_idx_i
                        )
                        ltot.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        last_losses[:] = [l1, l2, l3, ltot]
                        return ltot

                    optim.step(_alm_lbfgs_closure)
                    L1, L2, L3, L = (
                        last_losses[0],
                        last_losses[1],
                        last_losses[2],
                        last_losses[3],
                    )
                else:
                    optim.zero_grad()
                    L1, L2, L3, L, _ = _pinn_alm_losses(inp, tgt, dt_step_i, batch_idx_i)
                    L.backward()
                    optim.step()
                inner_l1 += float(L1.detach().item())
                inner_l2 += float(L2.detach().item())
                inner_l3 += float(L3.detach().item())
                inner_lt += float(L.detach().item())

            n = max(1, inner_n)
            losses1_alm.append(inner_l1 / n)
            losses2_alm.append(inner_l2 / n)
            losses3_alm.append(inner_l3 / n)
            total_losses_alm.append(inner_lt / n)

            if log_loss_every > 0 and (outer + 1) % log_loss_every == 0:
                lam_np = lam.numpy()
                print(
                    f"{prefix}[ALM] outer {outer + 1}/{alm_outer_iters}  "
                    f"L_data={losses1_alm[-1]:.6g}  L_phys={losses3_alm[-1]:.6g}  "
                    f"L_total={total_losses_alm[-1]:.6g}  mu={mu:.6g}  "
                    f"lam(mean={lam_np.mean():.4g} std={lam_np.std():.4g})",
                    flush=True,
                )

        lam_np = lam.numpy()
        print(
            f"{prefix}[ALM] done  outer={alm_outer_iters}  inner_step={alm_inner_step}  "
            f"mu_final={mu:.6g}  "
            f"lam mean={lam_np.mean():.5g} std={lam_np.std():.5g} "
            f"min={lam_np.min():.5g} max={lam_np.max():.5g}",
            flush=True,
        )
        # Concatenate warmup losses (if any) with ALM phase losses
        all_l1 = warmup_losses1 + losses1_alm
        all_l3 = warmup_losses3 + losses3_alm
        all_lt = warmup_total_losses + total_losses_alm
        all_l2 = [0.0] * len(warmup_losses1) + losses2_alm
        return (
            model,
            np.array(all_l1, dtype=float),
            np.array(all_l2, dtype=float),
            np.array(all_l3, dtype=float),
            np.array(all_lt, dtype=float),
        )
    # ── end pinn_alm ─────────────────────────────────────────────────────────────

    switch_epoch = None
    if optimizer_name == "Adam_NNCG":
        if nncg_switch_epoch is None:
            switch_epoch = max(1, epochs // 2)
        else:
            switch_epoch = max(0, int(nncg_switch_epoch))

    for epoch_idx in range(epochs):
        epoch_l1 = 0.0
        epoch_l2 = 0.0
        epoch_l3 = 0.0
        epoch_l = 0.0
        history_size = max(1, int(region_history))

        if optimizer_name == "Adam_NNCG" and current_optimizer_name == "Adam" and epoch_idx >= switch_epoch:
            optim = build_nncg()
            current_optimizer_name = "NNCG"
            lr_scheduler = None

        if current_optimizer_name == "NNCG":
            inputs, targets, dt, _idx = full_batch
            model.train()
            dt_step = float(dt[0].item())
            batch_idx = _idx.long()
            region_noise = None
            if physics_mode == "pinn_region":
                effective_radius = float(
                    np.clip(
                        rho / max(region_gradient_variance, 1e-6),
                        a_min=0.0,
                        a_max=region_max_radius,
                    )
                )
                region_noise = freeze_region_noise(inputs, max(1, int(region_samples)), effective_radius)

            def closure():
                optim.zero_grad()
                L1, L2, L3, L, _g = compute_batch_losses(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    batch_idx=batch_idx,
                    dt_step=dt_step,
                    method=method,
                    physics_mode=physics_mode,
                    damp_b=damp_b,
                    lamb=lamb,
                    rho=rho,
                    region_samples=region_samples,
                    region_max_radius=region_max_radius,
                    region_gradient_variance=region_gradient_variance,
                    constraint_method=constraint_method,
                    alm_dual_batch=(
                        alm_dual_state[batch_idx.cpu()].to(device=inputs.device, dtype=inputs.dtype)
                        if alm_dual_state is not None
                        else None
                    ),
                    alm_penalty=alm_penalty,
                    region_noise=region_noise,
                    alm_target=alm_target,
                    alm_uncon_weight=alm_uncon_weight,
                )
                grad_tuple = torch.autograd.grad(L, model.parameters(), create_graph=True)
                return L, grad_tuple

            if (epoch_idx % max(1, int(nncg_precond_update_freq))) == 0:
                _, grad_tuple = closure()
                optim.update_preconditioner(grad_tuple)

            loss_tensor, flat_grad = optim.step(closure)
            with torch.no_grad():
                L1, L2, L3, L, g = compute_batch_losses(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    batch_idx=batch_idx,
                    dt_step=dt_step,
                    method=method,
                    physics_mode=physics_mode,
                    damp_b=damp_b,
                    lamb=lamb,
                    rho=rho,
                    region_samples=region_samples,
                    region_max_radius=region_max_radius,
                    region_gradient_variance=region_gradient_variance,
                    constraint_method=constraint_method,
                    alm_dual_batch=(
                        alm_dual_state[batch_idx.cpu()].to(device=inputs.device, dtype=inputs.dtype)
                        if alm_dual_state is not None
                        else None
                    ),
                    alm_penalty=alm_penalty,
                    region_noise=region_noise,
                    alm_target=alm_target,
                    alm_uncon_weight=alm_uncon_weight,
                )
            if alm_dual_state is not None and g is not None:
                alm_dual_state[batch_idx.cpu()] = (
                    alm_dual_state[batch_idx.cpu()] + alm_penalty * g.detach().cpu().float()
                )
            if physics_mode == "pinn_region":
                flat_grad_np = flat_grad.cpu().numpy()
                region_gradient_history.append(flat_grad_np)
                region_gradient_history = region_gradient_history[-history_size:]
                grad_hist = np.array(region_gradient_history)
                region_gradient_variance = float(
                    (np.std(grad_hist, axis=0) / (np.mean(np.abs(grad_hist), axis=0) + 1e-6)).mean()
                )
                if not np.isfinite(region_gradient_variance) or region_gradient_variance <= 0.0:
                    region_gradient_variance = 1.0

            epoch_l1 = float(L1.detach().cpu().item())
            epoch_l2 = float(L2.detach().cpu().item())
            epoch_l3 = float(L3.detach().cpu().item())
            epoch_l = float(L.detach().cpu().item())
        elif current_optimizer_name == "LBFGS":
            inputs, targets, dt, _idx = full_batch
            model.train()
            dt_step = float(dt[0].item())
            batch_idx = _idx.long()

            def closure():
                optim.zero_grad()
                L1, L2, L3, L, _g = compute_batch_losses(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    batch_idx=batch_idx,
                    dt_step=dt_step,
                    method=method,
                    physics_mode=physics_mode,
                    damp_b=damp_b,
                    lamb=lamb,
                    rho=rho,
                    region_samples=region_samples,
                    region_max_radius=region_max_radius,
                    region_gradient_variance=region_gradient_variance,
                    constraint_method=constraint_method,
                    alm_dual_batch=(
                        alm_dual_state[batch_idx.cpu()].to(device=inputs.device, dtype=inputs.dtype)
                        if alm_dual_state is not None
                        else None
                    ),
                    alm_penalty=alm_penalty,
                    alm_target=alm_target,
                    alm_uncon_weight=alm_uncon_weight,
                )
                L.backward()
                return L

            optim.step(closure)
            with torch.no_grad():
                L1, L2, L3, L, g = compute_batch_losses(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    batch_idx=batch_idx,
                    dt_step=dt_step,
                    method=method,
                    physics_mode=physics_mode,
                    damp_b=damp_b,
                    lamb=lamb,
                    rho=rho,
                    region_samples=region_samples,
                    region_max_radius=region_max_radius,
                    region_gradient_variance=region_gradient_variance,
                    constraint_method=constraint_method,
                    alm_dual_batch=(
                        alm_dual_state[batch_idx.cpu()].to(device=inputs.device, dtype=inputs.dtype)
                        if alm_dual_state is not None
                        else None
                    ),
                    alm_penalty=alm_penalty,
                    alm_target=alm_target,
                    alm_uncon_weight=alm_uncon_weight,
                )
            if alm_dual_state is not None and g is not None:
                alm_dual_state[batch_idx.cpu()] = (
                    alm_dual_state[batch_idx.cpu()] + alm_penalty * g.detach().cpu().float()
                )
            epoch_l1 = float(L1.detach().cpu().item())
            epoch_l2 = float(L2.detach().cpu().item())
            epoch_l3 = float(L3.detach().cpu().item())
            epoch_l = float(L.detach().cpu().item())
        else:
            data_iter = iter(data)
            for _step in range(updates_per_epoch):
                try:
                    inputs, targets, dt, _idx = next(data_iter)
                except StopIteration:
                    data_iter = iter(data)
                    inputs, targets, dt, _idx = next(data_iter)

                model.train()
                dt_step = float(dt[0].item())
                batch_idx = _idx.long()
                optim.zero_grad()
                L1, L2, L3, L, g = compute_batch_losses(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    batch_idx=batch_idx,
                    dt_step=dt_step,
                    method=method,
                    physics_mode=physics_mode,
                    damp_b=damp_b,
                    lamb=lamb,
                    rho=rho,
                    region_samples=region_samples,
                    region_max_radius=region_max_radius,
                    region_gradient_variance=region_gradient_variance,
                    constraint_method=constraint_method,
                    alm_dual_batch=(
                        alm_dual_state[batch_idx.cpu()].to(device=inputs.device, dtype=inputs.dtype)
                        if alm_dual_state is not None
                        else None
                    ),
                    alm_penalty=alm_penalty,
                    alm_target=alm_target,
                    alm_uncon_weight=alm_uncon_weight,
                )
                L.backward()

                if physics_mode == "pinn_region":
                    grad_parts = []
                    for p in model.parameters():
                        if p.grad is not None:
                            grad_parts.append(p.grad.detach().reshape(-1))
                    if grad_parts:
                        flat_grad = torch.cat(grad_parts).cpu().numpy()
                        region_gradient_history.append(flat_grad)
                        region_gradient_history = region_gradient_history[-history_size:]
                        grad_hist = np.array(region_gradient_history)
                        region_gradient_variance = float(
                            (
                                np.std(grad_hist, axis=0)
                                / (np.mean(np.abs(grad_hist), axis=0) + 1e-6)
                            ).mean()
                        )
                        if not np.isfinite(region_gradient_variance) or region_gradient_variance <= 0.0:
                            region_gradient_variance = 1.0

                optim.step()
                if alm_dual_state is not None and g is not None:
                    alm_dual_state[batch_idx.cpu()] = (
                        alm_dual_state[batch_idx.cpu()] + alm_penalty * g.detach().cpu().float()
                    )
                epoch_l1 += float(L1.detach().cpu().item())
                epoch_l2 += float(L2.detach().cpu().item())
                epoch_l3 += float(L3.detach().cpu().item())
                epoch_l += float(L.detach().cpu().item())

            epoch_l1 /= updates_per_epoch
            epoch_l2 /= updates_per_epoch
            epoch_l3 /= updates_per_epoch
            epoch_l /= updates_per_epoch
            if lr_scheduler is not None:
                lr_scheduler.step()

        if alm_dual_state is not None:
            alm_penalty *= alm_penalty_growth

        losses1.append(epoch_l1)
        losses2.append(epoch_l2)
        losses3.append(epoch_l3)
        total_losses.append(epoch_l)

        if log_loss_every > 0 and (epoch_idx + 1) % log_loss_every == 0:
            prefix = f"{log_prefix} " if log_prefix else ""
            print(
                f"{prefix}epoch {epoch_idx + 1}/{epochs} "
                f"L_data={epoch_l1:.6g} L_phys={epoch_l3:.6g} L_total={epoch_l:.6g}",
                flush=True,
            )

    return (
        model,
        np.array(losses1, dtype=float),
        np.array(losses2, dtype=float),
        np.array(losses3, dtype=float),
        np.array(total_losses, dtype=float),
    )


def trajectory_metrics(data, model, dt, method="euler"):
    loss_fn = torch.nn.MSELoss(reduction="mean").to(data.device)
    with torch.no_grad():
        dt_step = float(dt[0].item()) if torch.is_tensor(dt) else float(np.asarray(dt[0]).item())
        preds = torchdiffeq.odeint(
            model,
            data.T[:-1, :],
            torch.tensor([0.0, dt_step], dtype=data.dtype, device=data.device),
            method=method,
        )[-1, :, :]
        targets = data.T[1:, :]
        mse = loss_fn(targets, preds)
        sse = torch.sum((targets - preds) ** 2)
        per_step = torch.norm(preds - targets, dim=1) / torch.clamp(
            torch.norm(targets, dim=1), min=1e-12
        )
    return float(mse.item()), float(sse.item()), per_step.cpu().numpy()


def rollout_predict_states(data, model, dt, method="euler"):
    """Autoregressive Euler rollout; returns ``preds`` with same shape as ``data``: (state_dim, T)."""
    with torch.no_grad():
        dt_step = float(dt[0].item()) if torch.is_tensor(dt) else float(np.asarray(dt[0]).item())
        pred_states = [data[:, 0]]
        time_grid = torch.tensor([0.0, dt_step], dtype=data.dtype, device=data.device)

        for _ in range(data.shape[1] - 1):
            next_state = torchdiffeq.odeint(
                model,
                pred_states[-1].unsqueeze(0),
                time_grid,
                method=method,
            )[-1, 0, :]
            pred_states.append(next_state)

        return torch.stack(pred_states, dim=1)


def mean_relative_l2_over_time_window(preds, data, physical_times, t_lo: float, t_hi: float) -> float:
    """Mean over time indices j with t_j in [t_lo, t_hi] of per-step relative L2 errors.

    Layout matches ``rollout_predict_states`` / ``numpy_to_torch(xtest)``: tensors are
    ``(state_dim, n_times)`` — one synthetic trajectory (columns are time).
    """
    if physical_times.shape[0] != data.shape[1]:
        raise ValueError(
            f"physical_times length {physical_times.shape[0]} != data columns {data.shape[1]}"
        )
    if preds.shape != data.shape:
        raise ValueError(f"preds shape {preds.shape} != data shape {data.shape}")
    rels: list[torch.Tensor] = []
    for j in range(data.shape[1]):
        tj = float(physical_times[j])
        if tj < t_lo - 1e-12 or tj > t_hi + 1e-12:
            continue
        pred_j = preds[:, j]
        true_j = data[:, j]
        diff = pred_j - true_j
        num = torch.norm(diff, p=2)
        den = torch.norm(true_j, p=2).clamp(min=1e-12)
        rels.append(num / den)
    if not rels:
        return float("nan")
    return float(torch.mean(torch.stack(rels)).item())


def rollout_metrics(data, model, dt, method="euler"):
    loss_fn = torch.nn.MSELoss(reduction="mean").to(data.device)
    with torch.no_grad():
        preds = rollout_predict_states(data, model, dt, method=method)
        error = preds[:, 1:] - data[:, 1:]          # [N, T, D]
        mse = loss_fn(preds[:, 1:].T, data[:, 1:].T)
        sse = torch.sum(error**2)
        # Standard relative L2 error (LpLoss.rel): each trajectory flattened to (T*D,),
        # ||pred - true||_2 / ||true||_2, then mean over samples. Typically in [0, 2].
        N = error.shape[0]
        diff_flat = error.reshape(N, -1)
        true_flat = data[:, 1:].reshape(N, -1)
        diff_norm = torch.norm(diff_flat, p=2, dim=1)
        true_norm = torch.norm(true_flat, p=2, dim=1).clamp(min=1e-12)
        l2_error = float(torch.mean(diff_norm / true_norm).item())
        per_step = torch.norm(error.T, dim=1) / torch.clamp(
            torch.norm(data[:, 1:].T, dim=1), min=1e-12
        )
    return float(mse.item()), float(sse.item()), per_step.cpu().numpy(), l2_error


def rollout_rel_l2_trajectory(
    model,
    *,
    b_data: float,
    dt_train: float,
    dt_test: float,
    tmax_eval: float,
    state_representation: str,
    device: torch.device,
) -> float:
    """Full-trajectory relative L2 (same definition as ``test_rollout_l2``) on a synthetic test orbit."""
    _, x_test, _, dt_test_arr = build_data(
        b=b_data,
        dt=dt_train,
        dt_test=dt_test,
        tmax=tmax_eval,
        train_theta=1.7,
        test_theta=2.8,
        representation=state_representation,
    )
    inputs = numpy_to_torch(x_test, dtype=torch.float32, device=device)
    dt_t = numpy_to_torch(dt_test_arr, dtype=torch.float32, device=device)
    _, _, _, l2 = rollout_metrics(inputs, model, dt_t, method="euler")
    return float(l2)


def _parse_float_csv(s: str) -> list[float]:
    s = (s or "").strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_csv(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(float(x.strip())) for x in s.split(",") if x.strip()]


def _column_indices_for_inv_b_labels(b_values: list[float], requested_inv_b: list[float]) -> list[int]:
    """Columns whose ``1/b`` axis value matches requested labels (within tolerance)."""
    inv_present = 1.0 / np.asarray(b_values, dtype=float)
    cols: list[int] = []
    seen: set[int] = set()
    for t in requested_inv_b:
        tv = float(t)
        dist = np.abs(inv_present - tv)
        j = int(np.argmin(dist))
        tol = max(1e-9, 1e-6 * max(1.0, abs(tv)))
        if float(dist[j]) > tol:
            print(
                f"[plot-inv-b-values] skipping 1/b={tv:g}: nearest in data is "
                f"{float(inv_present[j]):g} (Δ={float(dist[j]):.3g})",
                flush=True,
            )
            continue
        if j in seen:
            continue
        seen.add(j)
        cols.append(j)
    return cols


def fmt_tag(val: float) -> str:
    return f"{val:.3f}".rstrip("0").rstrip(".")


def build_pde_coeff_warmup_schedule(init_c: float, delta_c: float, target_c: float) -> list[float]:
    """Curriculum stages from ``init_c`` toward ``target_c`` in steps of magnitude ``delta_c``.

    For the damped pendulum, damping ``b`` is the PDE coefficient in ``pendulum`` / ``damp_b``.
    Step direction follows ``sign(target_c - init_c)``; the terminal stage is exactly ``target_c``.
    """
    if delta_c <= 0:
        raise ValueError("delta_c must be positive")
    if abs(init_c - target_c) < 1e-12:
        return [float(target_c)]
    direction = 1.0 if target_c > init_c else -1.0
    seq: list[float] = []
    cur = float(init_c)
    while True:
        seq.append(cur)
        if abs(cur - target_c) < 1e-12:
            break
        nxt = cur + direction * delta_c
        if direction > 0 and nxt >= target_c - 1e-12:
            if abs(seq[-1] - target_c) > 1e-12:
                seq.append(float(target_c))
            break
        if direction < 0 and nxt <= target_c + 1e-12:
            if abs(seq[-1] - target_c) > 1e-12:
                seq.append(float(target_c))
            break
        cur = nxt
    return seq


def _train_adam_lbfgs_nncg_chain(
    data,
    model,
    *,
    method: str,
    learning_rate: float,
    weight_decay: float,
    adam_epochs: int,
    nncg_epochs: int,
    lamb: float,
    rho: float,
    physics_mode: str,
    damp_b: float,
    updates_per_epoch: int,
    region_samples: int,
    region_history: int,
    region_max_radius: float,
    nncg_rank: int,
    nncg_mu: float,
    nncg_cg_tol: float,
    nncg_cg_max_iters: int,
    nncg_precond_update_freq: int,
    lbfgs_max_iter: int,
    constraint_method: str,
    alm_penalty: float,
    alm_penalty_growth: float,
    alm_target: str,
    alm_uncon_weight: float,
    log_loss_every: int,
    log_prefix: str,
    phase_ckpt_dir: Path | None,
    phase_ckpt_stem: str,
) -> tuple:
    """Adam (many epochs) → one LBFGS step → NNCG (many epochs); save phase-end ckpts."""

    def build_nncg():
        return NysNewtonCG(
            model.parameters(),
            lr=learning_rate,
            rank=nncg_rank,
            mu=nncg_mu,
            cg_tol=nncg_cg_tol,
            cg_max_iters=nncg_cg_max_iters,
            line_search_fn="armijo",
        )

    losses1: list[float] = []
    losses2: list[float] = []
    losses3: list[float] = []
    total_losses: list[float] = []

    # ── Phase 1: Adam ─────────────────────────────────────────────────────
    optim = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.995)
    data_iter = iter(data)
    prefix = f"{log_prefix} " if log_prefix else ""
    print(f"{prefix}[ALN] Adam phase: {adam_epochs} epochs", flush=True)
    for epoch_idx in range(int(adam_epochs)):
        epoch_l1 = epoch_l2 = epoch_l3 = epoch_l = 0.0
        for _step in range(updates_per_epoch):
            try:
                inputs, targets, dt, _idx = next(data_iter)
            except StopIteration:
                data_iter = iter(data)
                inputs, targets, dt, _idx = next(data_iter)
            model.train()
            dt_step = float(dt[0].item())
            batch_idx = _idx.long()
            optim.zero_grad()
            L1, L2, L3, L, _g = compute_batch_losses(
                model=model,
                inputs=inputs,
                targets=targets,
                batch_idx=batch_idx,
                dt_step=dt_step,
                method=method,
                physics_mode=physics_mode,
                damp_b=damp_b,
                lamb=lamb,
                rho=rho,
                region_samples=region_samples,
                region_max_radius=region_max_radius,
                region_gradient_variance=1.0,
                constraint_method=constraint_method,
                alm_penalty=alm_penalty,
                alm_target=alm_target,
                alm_uncon_weight=alm_uncon_weight,
            )
            L.backward()
            optim.step()
            epoch_l1 += float(L1.detach().item())
            epoch_l2 += float(L2.detach().item())
            epoch_l3 += float(L3.detach().item())
            epoch_l += float(L.detach().item())
        epoch_l1 /= updates_per_epoch
        epoch_l2 /= updates_per_epoch
        epoch_l3 /= updates_per_epoch
        epoch_l /= updates_per_epoch
        lr_scheduler.step()
        losses1.append(epoch_l1)
        losses2.append(epoch_l2)
        losses3.append(epoch_l3)
        total_losses.append(epoch_l)
        if log_loss_every > 0 and (epoch_idx + 1) % log_loss_every == 0:
            print(
                f"{prefix}[ALN Adam] epoch {epoch_idx + 1}/{adam_epochs} "
                f"L_data={epoch_l1:.6g} L_phys={epoch_l3:.6g} L_total={epoch_l:.6g}",
                flush=True,
            )

    full_batch = loader_to_full_batch(data)
    inputs, targets, dt, _idx = full_batch
    batch_idx = _idx.long()
    dt_step = float(dt[0].item())

    # ── Phase 2: one LBFGS step ───────────────────────────────────────────
    print(f"{prefix}[ALN] LBFGS phase: 1 step, max_iter={lbfgs_max_iter}", flush=True)
    lbfgs_optim = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=int(lbfgs_max_iter),
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def lbfgs_closure():
        lbfgs_optim.zero_grad()
        L1, L2, L3, L, _g = compute_batch_losses(
            model=model,
            inputs=inputs,
            targets=targets,
            batch_idx=batch_idx,
            dt_step=dt_step,
            method=method,
            physics_mode=physics_mode,
            damp_b=damp_b,
            lamb=lamb,
            rho=rho,
            region_samples=region_samples,
            region_max_radius=region_max_radius,
            region_gradient_variance=1.0,
            constraint_method=constraint_method,
            alm_penalty=alm_penalty,
            alm_target=alm_target,
            alm_uncon_weight=alm_uncon_weight,
        )
        L.backward()
        return L

    lbfgs_optim.step(lbfgs_closure)
    with torch.no_grad():
        L1, L2, L3, L, _g = compute_batch_losses(
            model=model,
            inputs=inputs,
            targets=targets,
            batch_idx=batch_idx,
            dt_step=dt_step,
            method=method,
            physics_mode=physics_mode,
            damp_b=damp_b,
            lamb=lamb,
            rho=rho,
            region_samples=region_samples,
            region_max_radius=region_max_radius,
            region_gradient_variance=1.0,
            constraint_method=constraint_method,
            alm_penalty=alm_penalty,
            alm_target=alm_target,
            alm_uncon_weight=alm_uncon_weight,
        )
    losses1.append(float(L1.detach().item()))
    losses2.append(float(L2.detach().item()))
    losses3.append(float(L3.detach().item()))
    total_losses.append(float(L.detach().item()))
    print(
        f"{prefix}[ALN LBFGS] L_data={losses1[-1]:.6g} L_phys={losses3[-1]:.6g} L_total={total_losses[-1]:.6g}",
        flush=True,
    )
    if phase_ckpt_dir is not None and phase_ckpt_stem:
        phase_ckpt_dir.mkdir(parents=True, exist_ok=True)
        lbfgs_ckpt = phase_ckpt_dir / f"{phase_ckpt_stem}_lbfgs_end.pt"
        torch.save(model.state_dict(), lbfgs_ckpt)
        print(f"{prefix}[ALN] saved {lbfgs_ckpt}", flush=True)

    # ── Phase 3: NNCG ─────────────────────────────────────────────────────
    print(f"{prefix}[ALN] NNCG phase: {nncg_epochs} epochs", flush=True)
    nncg_optim = build_nncg()
    for epoch_idx in range(int(nncg_epochs)):
        model.train()

        def closure():
            nncg_optim.zero_grad()
            L1, L2, L3, L, _g = compute_batch_losses(
                model=model,
                inputs=inputs,
                targets=targets,
                batch_idx=batch_idx,
                dt_step=dt_step,
                method=method,
                physics_mode=physics_mode,
                damp_b=damp_b,
                lamb=lamb,
                rho=rho,
                region_samples=region_samples,
                region_max_radius=region_max_radius,
                region_gradient_variance=1.0,
                constraint_method=constraint_method,
                alm_penalty=alm_penalty,
                alm_target=alm_target,
                alm_uncon_weight=alm_uncon_weight,
            )
            grad_tuple = torch.autograd.grad(L, model.parameters(), create_graph=True)
            return L, grad_tuple

        if (epoch_idx % max(1, int(nncg_precond_update_freq))) == 0:
            _, grad_tuple = closure()
            nncg_optim.update_preconditioner(grad_tuple)

        _loss_tensor, _flat_grad = nncg_optim.step(closure)
        with torch.no_grad():
            L1, L2, L3, L, _g = compute_batch_losses(
                model=model,
                inputs=inputs,
                targets=targets,
                batch_idx=batch_idx,
                dt_step=dt_step,
                method=method,
                physics_mode=physics_mode,
                damp_b=damp_b,
                lamb=lamb,
                rho=rho,
                region_samples=region_samples,
                region_max_radius=region_max_radius,
                region_gradient_variance=1.0,
                constraint_method=constraint_method,
                alm_penalty=alm_penalty,
                alm_target=alm_target,
                alm_uncon_weight=alm_uncon_weight,
            )
        losses1.append(float(L1.detach().item()))
        losses2.append(float(L2.detach().item()))
        losses3.append(float(L3.detach().item()))
        total_losses.append(float(L.detach().item()))
        if log_loss_every > 0 and (epoch_idx + 1) % log_loss_every == 0:
            print(
                f"{prefix}[ALN NNCG] epoch {epoch_idx + 1}/{nncg_epochs} "
                f"L_data={losses1[-1]:.6g} L_phys={losses3[-1]:.6g} L_total={total_losses[-1]:.6g}",
                flush=True,
            )

    if phase_ckpt_dir is not None and phase_ckpt_stem:
        nncg_ckpt = phase_ckpt_dir / f"{phase_ckpt_stem}_nncg_final.pt"
        torch.save(model.state_dict(), nncg_ckpt)
        print(f"{prefix}[ALN] saved {nncg_ckpt}", flush=True)

    return (
        model,
        np.array(losses1, dtype=float),
        np.array(losses2, dtype=float),
        np.array(losses3, dtype=float),
        np.array(total_losses, dtype=float),
    )


def _collect_pendulum_cell_metrics(
    *,
    net,
    xtrain,
    xtest,
    dt_train,
    dt_test,
    dt: float,
    tmax_train: float,
    tmax_test: float,
    b: float,
    b_target: float,
    b_data: float,
    inv_b_target: float,
    loss_l1: np.ndarray,
    loss_l3: np.ndarray,
    loss_total: np.ndarray,
    args: argparse.Namespace,
    train_dev: torch.device,
    state_representation: str,
    cl_schedule,
    curriculum_inv_b_schedule,
) -> dict:
    """Build the per-case JSON dict (train/test rollout metrics)."""
    inputs_train_torch = numpy_to_torch(xtrain, dtype=torch.float32, device=train_dev)
    train_mse, train_sse, _ = trajectory_metrics(
        inputs_train_torch, model=net, dt=dt_train, method="euler"
    )
    inputs_test_torch = numpy_to_torch(xtest, dtype=torch.float32, device=train_dev)
    test_mse, test_sse, _ = trajectory_metrics(
        inputs_test_torch, model=net, dt=dt_test, method="euler"
    )
    test_rollout_mse, test_rollout_sse, _, test_rollout_l2 = rollout_metrics(
        inputs_test_torch, model=net, dt=dt_test, method="euler"
    )
    report_dt = float(getattr(args, "rollout_report_test_dt", 0.05))
    t_span_lo = float(getattr(args, "rollout_report_t_span_lo", 0.0))
    t_span_hi = float(getattr(args, "rollout_report_t_span_hi", 10.0))
    test_rollout_l2_mean_rel_window = float("nan")
    test_rollout_l2_mean_rel_dt005_t0_20 = float("nan")
    if report_dt > 0.0:
        try:
            _, x_rep, _, dt_rep = build_data(
                b=b_data,
                dt=dt,
                dt_test=report_dt,
                tmax=tmax_test,
                train_theta=1.7,
                test_theta=2.8,
                representation=state_representation,
            )
            ncol = int(x_rep.shape[1])
            times_rep = np.arange(ncol, dtype=np.float64) * report_dt
            inputs_rep = numpy_to_torch(x_rep, dtype=torch.float32, device=train_dev)
            dt_rep_t = numpy_to_torch(dt_rep, dtype=torch.float32, device=train_dev)
            preds_rep = rollout_predict_states(inputs_rep, net, dt_rep_t, method="euler")
            test_rollout_l2_mean_rel_dt005_t0_20 = mean_relative_l2_over_time_window(
                preds_rep, inputs_rep, times_rep, 0.0, 20.0
            )
            if t_span_hi >= t_span_lo:
                test_rollout_l2_mean_rel_window = mean_relative_l2_over_time_window(
                    preds_rep, inputs_rep, times_rep, t_span_lo, t_span_hi
                )
        except Exception:
            pass
    test_rollout_l2_dt050_hor10 = float("nan")
    test_rollout_l2_dt005_hor15 = float("nan")
    test_rollout_l2_dt005_hor20 = float("nan")
    try:
        test_rollout_l2_dt050_hor10 = rollout_rel_l2_trajectory(
            net,
            b_data=b_data,
            dt_train=dt,
            dt_test=0.5,
            tmax_eval=10.0,
            state_representation=state_representation,
            device=train_dev,
        )
    except Exception:
        pass
    try:
        test_rollout_l2_dt005_hor15 = rollout_rel_l2_trajectory(
            net,
            b_data=b_data,
            dt_train=dt,
            dt_test=0.05,
            tmax_eval=15.0,
            state_representation=state_representation,
            device=train_dev,
        )
    except Exception:
        pass
    try:
        test_rollout_l2_dt005_hor20 = rollout_rel_l2_trajectory(
            net,
            b_data=b_data,
            dt_train=dt,
            dt_test=0.05,
            tmax_eval=20.0,
            state_representation=state_representation,
            device=train_dev,
        )
    except Exception:
        pass

    return {
        "b": b,
        "inv_b_target": float(inv_b_target),
        "b_physics": float(b_data),
        "b_target": float(b_target),
        "train_horizon": tmax_train,
        "test_horizon": tmax_test,
        "fixed_dt": dt,
        "lamb": args.lamb,
        "constraint_method": args.constraint_method,
        "alm_penalty": args.alm_penalty,
        "alm_penalty_growth": args.alm_penalty_growth,
        "alm_target": getattr(args, "alm_target", "physics"),
        "alm_uncon_weight": float(getattr(args, "alm_uncon_weight", 1.0)),
        "alm_outer_iters": int(getattr(args, "alm_outer_iters", 10)),
        "alm_inner_step": int(getattr(args, "alm_inner_step", 100)),
        "alm_warmup_epochs": int(getattr(args, "alm_warmup_epochs", 0)),
        "rho": args.rho,
        "physics_mode": args.physics_mode,
        "optimizer": args.optimizer,
        "cl_schedule_space": getattr(args, "cl_schedule_space", "b"),
        "nncg_switch_epoch": None if args.nncg_switch_epoch < 0 else args.nncg_switch_epoch,
        "nncg_epochs": int(getattr(args, "nncg_epochs", 0)),
        "num_steps": int(xtrain.shape[1] - 1),
        "epochs_ran": int(len(loss_total)),
        "curriculum_warmup": bool(args.cl_warmup),
        "curriculum_schedule_b": cl_schedule,
        "curriculum_inv_b_schedule": curriculum_inv_b_schedule,
        "cl_inner_epochs": int(args.cl_inner_epochs) if args.cl_warmup else None,
        "cl_init_coeff": float(args.cl_init_coeff) if args.cl_warmup and args.cl_schedule_space == "b" else None,
        "cl_delta_coeff": float(args.cl_delta_coeff) if args.cl_warmup and args.cl_schedule_space == "b" else None,
        "cl_inv_init": float(args.cl_inv_init) if args.cl_warmup and args.cl_schedule_space == "inv_b" else None,
        "cl_inv_delta": float(args.cl_inv_delta) if args.cl_warmup and args.cl_schedule_space == "inv_b" else None,
        "epochs_mode": str(getattr(args, "optimizer", "")),
        "train_data_loss": float(loss_l1[-1]),
        "train_physics_loss": float(loss_l3[-1]),
        "train_total_loss": float(loss_total[-1]),
        "train_data_loss_history": loss_l1.tolist(),
        "train_physics_loss_history": loss_l3.tolist(),
        "train_total_loss_history": loss_total.tolist(),
        "train_mse": train_mse,
        "train_sse": train_sse,
        "test_mse": test_mse,
        "test_sse": test_sse,
        "test_rollout_mse": test_rollout_mse,
        "test_rollout_sse": test_rollout_sse,
        "test_rollout_l2": test_rollout_l2,
        "test_rollout_l2_mean_rel_window": test_rollout_l2_mean_rel_window,
        "test_rollout_l2_mean_rel_dt005_t0_20": test_rollout_l2_mean_rel_dt005_t0_20,
        "rollout_report_test_dt": report_dt if report_dt > 0.0 else None,
        "rollout_report_t_span_lo": t_span_lo,
        "rollout_report_t_span_hi": t_span_hi,
        "test_rollout_l2_dt050_hor10": test_rollout_l2_dt050_hor10,
        "test_rollout_l2_dt005_hor15": test_rollout_l2_dt005_hor15,
        "test_rollout_l2_dt005_hor20": test_rollout_l2_dt005_hor20,
        "seed": int(args.seed),
    }


def build_uniform_schedule(init_c: float, target_c: float, outer_step: int) -> list[float]:
    """``outer_step`` equal segments on [init_c, target_c] → ``outer_step + 1`` knots (inclusive)."""
    init_c = float(init_c)
    target_c = float(target_c)
    outer_step = int(outer_step)
    if outer_step <= 0:
        raise ValueError("outer_step must be positive")
    if abs(init_c - target_c) < 1e-14:
        return [init_c]
    return [
        init_c + k * (target_c - init_c) / outer_step for k in range(outer_step + 1)
    ]


def execute_sweep_cell(
    *,
    i: int,
    j: int,
    horizon: float,
    b: float,
    out_dir: Path,
    args: argparse.Namespace,
    state_representation: str,
    state_dim: int,
    sweep_updates_per_epoch: int,
    force_train: bool = False,
) -> dict:
    """Train/eval one (horizon, b) cell; write case JSON. Returns scalars for matrices."""
    dt = args.fixed_dt
    tmax_train = horizon
    tmax_test = args.test_horizon
    b_target = args.cl_target_coeff if args.cl_target_coeff is not None else b
    b_data = b_target if args.cl_warmup else b
    inv_b_target = 1.0 / b_target

    seed_suffix = f"_seed{args.seed}" if getattr(args, "multi_seed_run", False) else ""
    case_name = f"b{fmt_tag(b)}_hor{fmt_tag(tmax_train)}{seed_suffix}.json"
    case_path = out_dir / case_name

    if getattr(args, "skip_existing_cases", False) and case_path.is_file() and not force_train:
        print(f"[skip existing case] {case_path}", flush=True)
        with case_path.open("r") as f:
            metrics = json.load(f)
        return {
            "i": i,
            "j": j,
            "train_data_loss": float(metrics["train_data_loss"]),
            "train_physics_loss": float(metrics["train_physics_loss"]),
            "train_total_loss": float(metrics["train_total_loss"]),
            "train_mse": float(metrics["train_mse"]),
            "train_sse": float(metrics["train_sse"]),
            "test_mse": float(metrics["test_mse"]),
            "test_sse": float(metrics["test_sse"]),
            "test_rollout_mse": float(metrics["test_rollout_mse"]),
            "test_rollout_sse": float(metrics["test_rollout_sse"]),
            "test_rollout_l2": float(metrics.get("test_rollout_l2", float("nan"))),
            "test_rollout_l2_mean_rel_window": float(
                metrics.get("test_rollout_l2_mean_rel_window", float("nan"))
            ),
            "test_rollout_l2_mean_rel_dt005_t0_20": float(
                metrics.get("test_rollout_l2_mean_rel_dt005_t0_20", float("nan"))
            ),
            "test_rollout_l2_dt050_hor10": float(
                metrics.get("test_rollout_l2_dt050_hor10", float("nan"))
            ),
            "test_rollout_l2_dt005_hor15": float(
                metrics.get("test_rollout_l2_dt005_hor15", float("nan"))
            ),
            "test_rollout_l2_dt005_hor20": float(
                metrics.get("test_rollout_l2_dt005_hor20", float("nan"))
            ),
        }

    print(
        f"[cell i={i} j={j}] b={b:g} 1/b(target)={inv_b_target:g} T_train={tmax_train:g} "
        f"{args.physics_mode} {args.optimizer}"
        + (f" | CL({args.cl_schedule_space})" if args.cl_warmup else ""),
        flush=True,
    )

    set_seed(args.seed)

    train_dev = torch.device(
        "cuda:0"
        if bool(getattr(args, "cuda", False)) and torch.cuda.is_available()
        else "cpu"
    )

    xtrain, _, dt_train, _ = build_data(
        b=b_data,
        dt=dt,
        dt_test=args.fixed_test_dt,
        tmax=tmax_train,
        train_theta=1.7,
        test_theta=2.8,
        representation=state_representation,
    )
    _, xtest, _, dt_test = build_data(
        b=b_data,
        dt=dt,
        dt_test=args.fixed_test_dt,
        tmax=tmax_test,
        train_theta=1.7,
        test_theta=2.8,
        representation=state_representation,
    )

    net = ShallowODE(in_dim=state_dim, hidden=args.hidden, out_dim=state_dim, Act=torch.nn.Tanh).to(
        train_dev
    )

    alm_log_prefix = f"[cell{i},{j} b={b:g} T_train={tmax_train:g}] "
    alm_lbfgs_snapshot: dict | None = None
    alm_eval_kw: dict | None = None
    if args.physics_mode == "pinn_alm":
        alm_eval_kw = dict(
            method="euler",
            damp_b=b_data,
            lamb=args.lamb,
            rho=args.rho,
            region_samples=args.region_samples,
            region_max_radius=args.region_max_radius,
            region_gradient_variance=1.0,
            alm_target=getattr(args, "alm_target", "physics"),
            alm_uncon_weight=getattr(args, "alm_uncon_weight", 1.0),
        )

    # Optional: load pretrained weights as initialisation (e.g. a saved Adam model).
    # If --alm-init-ckpt is a directory, look up <dir>/<case_stem>.pt for per-cell ckpt.
    alm_init_ckpt = getattr(args, "alm_init_ckpt", "").strip()
    if alm_init_ckpt:
        _root = Path(alm_init_ckpt)
        case_stem = case_path.stem
        if _root.is_dir():
            _p = _root / f"{case_stem}.pt"
            if not _p.is_file():
                _p_seed = _root / f"{case_stem}_seed{int(args.seed)}.pt"
                if _p_seed.is_file():
                    _p = _p_seed
        else:
            _p = _root
        if (
            not _p.is_file()
            and _root.is_dir()
            and args.physics_mode == "pinn_alm"
            and not force_train
            and not getattr(args, "no_alm_bootstrap", False)
        ):
            bootstrap_iters = int(getattr(args, "alm_bootstrap_lbfgs_max_iter", 1000))
            print(
                f"[cell i={i} j={j}] missing LBFGS init ckpt {_p.name} under {_root}; "
                f"running PINN+LBFGS pretrain (max_iter={bootstrap_iters}) then ALM.",
                flush=True,
            )
            pre_args = copy.copy(args)
            pre_args.physics_mode = "pinn"
            pre_args.optimizer = "LBFGS"
            pre_args.alm_init_ckpt = ""
            pre_args.skip_existing_cases = False
            pre_args.cl_warmup = False
            pre_args.epochs = 1
            pre_args.lbfgs_max_iter = bootstrap_iters
            pre_args.save_cell_ckpt = True
            execute_sweep_cell(
                i=i,
                j=j,
                horizon=horizon,
                b=b,
                out_dir=_root,
                args=pre_args,
                state_representation=state_representation,
                state_dim=state_dim,
                sweep_updates_per_epoch=sweep_updates_per_epoch,
                force_train=True,
            )
            if not _p.is_file():
                raise FileNotFoundError(
                    f"[cell i={i} j={j}] LBFGS pretrain did not produce {_p}"
                )
        if not _p.is_file():
            raise FileNotFoundError(
                f"[cell i={i} j={j}] --alm-init-ckpt resolved path does not exist: {_p}"
            )
        _ckpt = torch.load(_p, map_location=train_dev)
        if isinstance(_ckpt, dict) and "state_dict" in _ckpt:
            _ckpt = _ckpt["state_dict"]
        net.load_state_dict(_ckpt)
        print(f"[cell i={i} j={j}] loaded init weights from: {_p}", flush=True)
        if args.physics_mode == "pinn_alm" and alm_eval_kw is not None:
            alm_lbfgs_snapshot = {
                k: v.detach().cpu().clone() for k, v in net.state_dict().items()
            }
            _alm_loader = build_loader(
                xtrain, dt_train, args.batch_size, device=train_dev
            )
            _pre_mu = float(args.alm_penalty)
            _pre_metrics = eval_pinn_alm_cell_metrics(
                net,
                _alm_loader,
                xtest,
                dt_test,
                train_dev,
                lam=None,
                eval_mu=_pre_mu,
                **alm_eval_kw,
            )
            log_pinn_alm_metrics(
                alm_log_prefix,
                "after LBFGS ckpt load (before ALM)",
                _pre_metrics,
                _pre_mu,
                lam_note="reporting with λ=0",
            )

    cl_schedule = None
    curriculum_inv_b_schedule = None

    if args.cl_warmup:
        use_uniform = (
            getattr(args, "cl_outer_step", None) is not None
            and int(args.cl_outer_step) > 0
        )
        if args.cl_schedule_space == "inv_b":
            if use_uniform:
                inv_schedule = build_uniform_schedule(
                    float(args.cl_inv_init), float(inv_b_target), int(args.cl_outer_step)
                )
            else:
                inv_schedule = build_pde_coeff_warmup_schedule(
                    float(args.cl_inv_init), float(args.cl_inv_delta), float(inv_b_target)
                )
            curriculum_inv_b_schedule = inv_schedule
            cl_schedule = [1.0 / v for v in inv_schedule]
            print(
                f"  CL(inv_b): {len(cl_schedule)} stages × "
                f"{int(args.epochs) if len(cl_schedule) == 1 else int(args.cl_inner_epochs)} ep/stage "
                f"inv_b={inv_schedule}  (uniform={use_uniform})",
                flush=True,
            )
        else:
            if use_uniform:
                cl_schedule = build_uniform_schedule(
                    float(args.cl_init_coeff), float(b_target), int(args.cl_outer_step)
                )
            else:
                cl_schedule = build_pde_coeff_warmup_schedule(
                    float(args.cl_init_coeff), float(args.cl_delta_coeff), float(b_target)
                )
            curriculum_inv_b_schedule = [1.0 / bb for bb in cl_schedule]
            print(
                f"  CL(b): {len(cl_schedule)} stages × "
                f"{int(args.epochs) if len(cl_schedule) == 1 else int(args.cl_inner_epochs)} ep/stage "
                f"b={cl_schedule}  (uniform={use_uniform})",
                flush=True,
            )

        loss_chunks_l1 = []
        loss_chunks_l3 = []
        loss_chunks_tot = []
        ckpt_root = out_dir / "checkpoints"
        if args.cl_save_int_inv_checkpoints or bool(
            getattr(args, "cl_save_stage_checkpoints", False)
        ):
            ckpt_root.mkdir(parents=True, exist_ok=True)

        assert cl_schedule is not None and curriculum_inv_b_schedule is not None
        # Single-stage curriculum (e.g. target inv_b equals init): same physics as plain Adam
        # for that cell — use ``--epochs`` so compute budget matches non-CL baseline instead of
        # only ``--cl-inner-epochs`` (typically 40 vs 600).
        cl_stage_epochs = (
            int(args.epochs) if len(cl_schedule) == 1 else int(args.cl_inner_epochs)
        )
        for cl_idx, (cl_inv, cl_b) in enumerate(zip(curriculum_inv_b_schedule, cl_schedule)):
            is_last = cl_idx == len(cl_schedule) - 1
            cl_constraint = args.constraint_method if is_last else "penalty"
            cl_alm_pen = args.alm_penalty if is_last else 0.0
            cl_alm_growth = args.alm_penalty_growth if is_last else 1.0
            print(
                f"    stage {cl_idx + 1}/{len(cl_schedule)}: inv_b={cl_inv:g} b={cl_b:g}",
                flush=True,
            )
            cl_xtrain, _, cl_dt_train, _ = build_data(
                b=cl_b,
                dt=dt,
                dt_test=args.fixed_test_dt,
                tmax=tmax_train,
                train_theta=1.7,
                test_theta=2.8,
                representation=state_representation,
            )
            cl_loader = build_loader(cl_xtrain, cl_dt_train, args.batch_size, device=train_dev)
            net, l1, _, l3, lt = train_odenet(
                cl_loader,
                model=net,
                method="euler",
                learning_rate=args.learning_rate,
                weight_decay=0.0,
                epochs=int(cl_stage_epochs),
                optimizer_name=args.optimizer,
                lamb=args.lamb,
                rho=args.rho,
                physics_mode=args.physics_mode,
                constraint_method=cl_constraint,
                alm_penalty=cl_alm_pen,
                alm_penalty_growth=cl_alm_growth,
                alm_target=getattr(args, "alm_target", "physics"),
                alm_uncon_weight=getattr(args, "alm_uncon_weight", 1.0),
                alm_outer_iters=getattr(args, "alm_outer_iters", 10),
                alm_inner_step=getattr(args, "alm_inner_step", 100),
                alm_warmup_epochs=getattr(args, "alm_warmup_epochs", 0),
                damp_b=cl_b,
                updates_per_epoch=sweep_updates_per_epoch,
                region_samples=args.region_samples,
                region_history=args.region_history,
                region_max_radius=args.region_max_radius,
                nncg_rank=args.nncg_rank,
                nncg_mu=args.nncg_mu,
                nncg_cg_tol=args.nncg_cg_tol,
                nncg_cg_max_iters=args.nncg_cg_max_iters,
                nncg_precond_update_freq=args.nncg_precond_update_freq,
                nncg_switch_epoch=(
                    None if args.nncg_switch_epoch < 0 else args.nncg_switch_epoch
                ),
                lbfgs_max_iter=args.lbfgs_max_iter,
                log_loss_every=int(args.log_loss_every),
                log_prefix=(
                    f"[cell{i},{j} CL {cl_idx + 1}/{len(cl_schedule)} inv_b={cl_inv:g} T={tmax_train:g}]"
                ),
            )
            loss_chunks_l1.append(l1)
            loss_chunks_l3.append(l3)
            loss_chunks_tot.append(lt)

            if args.cl_save_int_inv_checkpoints and abs(cl_inv - round(cl_inv)) < 1e-8:
                inv_int = int(round(cl_inv))
                ckpt_path = (
                    ckpt_root
                    / f"invb{inv_int}_hor{fmt_tag(tmax_train)}_tinv{fmt_tag(inv_b_target)}_stage{cl_idx:02d}{seed_suffix}.pt"
                )
                torch.save(net.state_dict(), ckpt_path)
                print(f"      saved checkpoint {ckpt_path.name}", flush=True)

        loss_l1 = np.concatenate(loss_chunks_l1)
        loss_l3 = np.concatenate(loss_chunks_l3)
        loss_total = np.concatenate(loss_chunks_tot)
    else:
        phase_ckpt_dir: Path | None = None
        phase_ckpt_stem = ""
        if args.optimizer == "Adam_LBFGS_NNCG":
            phase_ckpt_dir = out_dir / "checkpoints"
            phase_ckpt_stem = case_path.stem
        train_loader = build_loader(xtrain, dt_train, args.batch_size, device=train_dev)
        net, loss_l1, _, loss_l3, loss_total = train_odenet(
            train_loader,
            model=net,
            method="euler",
            learning_rate=args.learning_rate,
            weight_decay=0.0,
            epochs=args.epochs,
            optimizer_name=args.optimizer,
            lamb=args.lamb,
            rho=args.rho,
            physics_mode=args.physics_mode,
            constraint_method=args.constraint_method,
            alm_penalty=args.alm_penalty,
            alm_penalty_growth=args.alm_penalty_growth,
            alm_target=getattr(args, "alm_target", "physics"),
            alm_uncon_weight=getattr(args, "alm_uncon_weight", 1.0),
            alm_outer_iters=getattr(args, "alm_outer_iters", 10),
            alm_inner_step=getattr(args, "alm_inner_step", 100),
            alm_warmup_epochs=getattr(args, "alm_warmup_epochs", 0),
            damp_b=b_data,
            updates_per_epoch=sweep_updates_per_epoch,
            region_samples=args.region_samples,
            region_history=args.region_history,
            region_max_radius=args.region_max_radius,
            nncg_rank=args.nncg_rank,
            nncg_mu=args.nncg_mu,
            nncg_cg_tol=args.nncg_cg_tol,
            nncg_cg_max_iters=args.nncg_cg_max_iters,
            nncg_precond_update_freq=args.nncg_precond_update_freq,
            nncg_switch_epoch=None if args.nncg_switch_epoch < 0 else args.nncg_switch_epoch,
            lbfgs_max_iter=args.lbfgs_max_iter,
            nncg_epochs=int(args.nncg_epochs),
            phase_ckpt_dir=phase_ckpt_dir,
            phase_ckpt_stem=phase_ckpt_stem,
            log_loss_every=int(args.log_loss_every),
            log_prefix=f"[cell{i},{j} b={b:g} T_train={tmax_train:g}]",
        )

    if args.physics_mode == "pinn_alm" and alm_eval_kw is not None:
        _alm_eval_loader = build_loader(
            xtrain, dt_train, args.batch_size, device=train_dev
        )
        _final_mu = float(args.alm_penalty)
        _final_metrics = eval_pinn_alm_cell_metrics(
            net,
            _alm_eval_loader,
            xtest,
            dt_test,
            train_dev,
            lam=None,
            eval_mu=_final_mu,
            **alm_eval_kw,
        )
        log_pinn_alm_metrics(
            alm_log_prefix,
            "after ALM training (final weights)",
            _final_metrics,
            _final_mu,
            lam_note="reporting with λ=0",
        )
        if alm_lbfgs_snapshot is not None:
            compare_lbfgs_pretrain_vs_alm_trained(
                net,
                alm_lbfgs_snapshot,
                _alm_eval_loader,
                xtest,
                dt_test,
                train_dev,
                prefix=alm_log_prefix,
                eval_mu=_final_mu,
                **alm_eval_kw,
            )

    if args.optimizer == "Adam_LBFGS_NNCG":
        lbfgs_dir = out_dir / "lbfgs_end"
        nncg_dir = out_dir / "nncg_final"
        lbfgs_dir.mkdir(parents=True, exist_ok=True)
        nncg_dir.mkdir(parents=True, exist_ok=True)
        adam_n = int(args.epochs)
        lbfgs_hist_end = adam_n + 1
        lbfgs_ckpt = (phase_ckpt_dir or (out_dir / "checkpoints")) / f"{phase_ckpt_stem}_lbfgs_end.pt"
        nncg_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(torch.load(lbfgs_ckpt, map_location=train_dev))
        lbfgs_metrics = _collect_pendulum_cell_metrics(
            net=net,
            xtrain=xtrain,
            xtest=xtest,
            dt_train=dt_train,
            dt_test=dt_test,
            dt=dt,
            tmax_train=tmax_train,
            tmax_test=tmax_test,
            b=b,
            b_target=b_target,
            b_data=b_data,
            inv_b_target=inv_b_target,
            loss_l1=loss_l1[:lbfgs_hist_end],
            loss_l3=loss_l3[:lbfgs_hist_end],
            loss_total=loss_total[:lbfgs_hist_end],
            args=args,
            train_dev=train_dev,
            state_representation=state_representation,
            cl_schedule=cl_schedule,
            curriculum_inv_b_schedule=curriculum_inv_b_schedule,
        )
        lbfgs_metrics["training_phase"] = "lbfgs_end"
        lbfgs_metrics["epochs_mode"] = f"Adam×{adam_n}+LBFGS×1"
        with (lbfgs_dir / case_name).open("w") as f:
            json.dump(lbfgs_metrics, f, indent=2)
        net.load_state_dict(nncg_state)
        metrics = _collect_pendulum_cell_metrics(
            net=net,
            xtrain=xtrain,
            xtest=xtest,
            dt_train=dt_train,
            dt_test=dt_test,
            dt=dt,
            tmax_train=tmax_train,
            tmax_test=tmax_test,
            b=b,
            b_target=b_target,
            b_data=b_data,
            inv_b_target=inv_b_target,
            loss_l1=loss_l1,
            loss_l3=loss_l3,
            loss_total=loss_total,
            args=args,
            train_dev=train_dev,
            state_representation=state_representation,
            cl_schedule=cl_schedule,
            curriculum_inv_b_schedule=curriculum_inv_b_schedule,
        )
        metrics["training_phase"] = "nncg_final"
        metrics["epochs_mode"] = (
            f"Adam×{adam_n}+LBFGS×1+NNCG×{int(args.nncg_epochs)}"
        )
        with (nncg_dir / case_name).open("w") as f:
            json.dump(metrics, f, indent=2)
        with case_path.open("w") as f:
            json.dump(metrics, f, indent=2)
        return {
            "i": i,
            "j": j,
            "train_data_loss": float(metrics["train_data_loss"]),
            "train_physics_loss": float(metrics["train_physics_loss"]),
            "train_total_loss": float(metrics["train_total_loss"]),
            "train_mse": float(metrics["train_mse"]),
            "train_sse": float(metrics["train_sse"]),
            "test_mse": float(metrics["test_mse"]),
            "test_sse": float(metrics["test_sse"]),
            "test_rollout_mse": float(metrics["test_rollout_mse"]),
            "test_rollout_sse": float(metrics["test_rollout_sse"]),
            "test_rollout_l2": float(metrics.get("test_rollout_l2", float("nan"))),
            "test_rollout_l2_mean_rel_window": float(
                metrics.get("test_rollout_l2_mean_rel_window", float("nan"))
            ),
            "test_rollout_l2_mean_rel_dt005_t0_20": float(
                metrics.get("test_rollout_l2_mean_rel_dt005_t0_20", float("nan"))
            ),
            "test_rollout_l2_dt050_hor10": float(
                metrics.get("test_rollout_l2_dt050_hor10", float("nan"))
            ),
            "test_rollout_l2_dt005_hor15": float(
                metrics.get("test_rollout_l2_dt005_hor15", float("nan"))
            ),
            "test_rollout_l2_dt005_hor20": float(
                metrics.get("test_rollout_l2_dt005_hor20", float("nan"))
            ),
        }

    inputs_train_torch = numpy_to_torch(xtrain, dtype=torch.float32, device=train_dev)
    train_mse, train_sse, _ = trajectory_metrics(
        inputs_train_torch, model=net, dt=dt_train, method="euler"
    )

    inputs_test_torch = numpy_to_torch(xtest, dtype=torch.float32, device=train_dev)
    test_mse, test_sse, _ = trajectory_metrics(
        inputs_test_torch, model=net, dt=dt_test, method="euler"
    )
    test_rollout_mse, test_rollout_sse, _, test_rollout_l2 = rollout_metrics(
        inputs_test_torch, model=net, dt=dt_test, method="euler"
    )

    # Finer test grid: mean relative L2 over a physical-time window (default [0,10] at dt=0.05).
    report_dt = float(getattr(args, "rollout_report_test_dt", 0.05))
    t_span_lo = float(getattr(args, "rollout_report_t_span_lo", 0.0))
    t_span_hi = float(getattr(args, "rollout_report_t_span_hi", 10.0))
    test_rollout_l2_mean_rel_window = float("nan")
    test_rollout_l2_mean_rel_dt005_t0_20 = float("nan")
    if report_dt > 0.0:
        try:
            _, x_rep, _, dt_rep = build_data(
                b=b_data,
                dt=dt,
                dt_test=report_dt,
                tmax=tmax_test,
                train_theta=1.7,
                test_theta=2.8,
                representation=state_representation,
            )
            ncol = int(x_rep.shape[1])
            times_rep = np.arange(ncol, dtype=np.float64) * report_dt
            inputs_rep = numpy_to_torch(x_rep, dtype=torch.float32, device=train_dev)
            dt_rep_t = numpy_to_torch(dt_rep, dtype=torch.float32, device=train_dev)
            preds_rep = rollout_predict_states(inputs_rep, net, dt_rep_t, method="euler")
            test_rollout_l2_mean_rel_dt005_t0_20 = mean_relative_l2_over_time_window(
                preds_rep, inputs_rep, times_rep, 0.0, 20.0
            )
            if t_span_hi >= t_span_lo:
                test_rollout_l2_mean_rel_window = mean_relative_l2_over_time_window(
                    preds_rep, inputs_rep, times_rep, t_span_lo, t_span_hi
                )
        except Exception as exc:
            print(
                f"[cell i={i} j={j}] rollout_report metric failed: {exc}",
                flush=True,
            )

    test_rollout_l2_dt050_hor10 = float("nan")
    test_rollout_l2_dt005_hor15 = float("nan")
    try:
        test_rollout_l2_dt050_hor10 = rollout_rel_l2_trajectory(
            net,
            b_data=b_data,
            dt_train=dt,
            dt_test=0.5,
            tmax_eval=10.0,
            state_representation=state_representation,
            device=train_dev,
        )
    except Exception as exc:
        print(
            f"[cell i={i} j={j}] aux rollout dt=0.5 hor=10 failed: {exc}",
            flush=True,
        )
    try:
        test_rollout_l2_dt005_hor15 = rollout_rel_l2_trajectory(
            net,
            b_data=b_data,
            dt_train=dt,
            dt_test=0.05,
            tmax_eval=15.0,
            state_representation=state_representation,
            device=train_dev,
        )
    except Exception as exc:
        print(
            f"[cell i={i} j={j}] aux rollout dt=0.05 hor=15 failed: {exc}",
            flush=True,
        )

    test_rollout_l2_dt005_hor20 = float("nan")
    try:
        test_rollout_l2_dt005_hor20 = rollout_rel_l2_trajectory(
            net,
            b_data=b_data,
            dt_train=dt,
            dt_test=0.05,
            tmax_eval=20.0,
            state_representation=state_representation,
            device=train_dev,
        )
    except Exception as exc:
        print(
            f"[cell i={i} j={j}] aux rollout dt=0.05 hor=20 failed: {exc}",
            flush=True,
        )

    metrics = {
        "b": b,
        "inv_b_target": float(inv_b_target),
        "b_physics": float(b_data),
        "b_target": float(b_target),
        "train_horizon": tmax_train,
        "test_horizon": tmax_test,
        "fixed_dt": dt,
        "lamb": args.lamb,
        "constraint_method": args.constraint_method,
        "alm_penalty": args.alm_penalty,
        "alm_penalty_growth": args.alm_penalty_growth,
        "alm_target": getattr(args, "alm_target", "physics"),
        "alm_uncon_weight": float(getattr(args, "alm_uncon_weight", 1.0)),
        "alm_outer_iters": int(getattr(args, "alm_outer_iters", 10)),
        "alm_inner_step": int(getattr(args, "alm_inner_step", 100)),
        "alm_warmup_epochs": int(getattr(args, "alm_warmup_epochs", 0)),
        "rho": args.rho,
        "physics_mode": args.physics_mode,
        "optimizer": args.optimizer,
        "cl_schedule_space": getattr(args, "cl_schedule_space", "b"),
        "nncg_switch_epoch": None if args.nncg_switch_epoch < 0 else args.nncg_switch_epoch,
        "num_steps": int(xtrain.shape[1] - 1),
        "epochs_ran": int(len(loss_total)),
        "curriculum_warmup": bool(args.cl_warmup),
        "curriculum_schedule_b": cl_schedule,
        "curriculum_inv_b_schedule": curriculum_inv_b_schedule,
        "cl_inner_epochs": int(args.cl_inner_epochs) if args.cl_warmup else None,
        "cl_init_coeff": float(args.cl_init_coeff) if args.cl_warmup and args.cl_schedule_space == "b" else None,
        "cl_delta_coeff": float(args.cl_delta_coeff) if args.cl_warmup and args.cl_schedule_space == "b" else None,
        "cl_inv_init": float(args.cl_inv_init) if args.cl_warmup and args.cl_schedule_space == "inv_b" else None,
        "cl_inv_delta": float(args.cl_inv_delta) if args.cl_warmup and args.cl_schedule_space == "inv_b" else None,
        "epochs_mode": (
            (
                f"1×{int(args.epochs)}"
                if len(cl_schedule) == 1
                else f"{len(cl_schedule)}×{int(args.cl_inner_epochs)}"
            )
            if args.cl_warmup and cl_schedule is not None
            else str(args.epochs)
        ),
        "train_data_loss": float(loss_l1[-1]),
        "train_physics_loss": float(loss_l3[-1]),
        "train_total_loss": float(loss_total[-1]),
        "train_data_loss_history": loss_l1.tolist(),
        "train_physics_loss_history": loss_l3.tolist(),
        "train_total_loss_history": loss_total.tolist(),
        "train_mse": train_mse,
        "train_sse": train_sse,
        "test_mse": test_mse,
        "test_sse": test_sse,
        "test_rollout_mse": test_rollout_mse,
        "test_rollout_sse": test_rollout_sse,
        "test_rollout_l2": test_rollout_l2,
        "test_rollout_l2_mean_rel_window": test_rollout_l2_mean_rel_window,
        "test_rollout_l2_mean_rel_dt005_t0_20": test_rollout_l2_mean_rel_dt005_t0_20,
        "rollout_report_test_dt": report_dt if report_dt > 0.0 else None,
        "rollout_report_t_span_lo": t_span_lo,
        "rollout_report_t_span_hi": t_span_hi,
        "test_rollout_l2_dt050_hor10": test_rollout_l2_dt050_hor10,
        "test_rollout_l2_dt005_hor15": test_rollout_l2_dt005_hor15,
        "test_rollout_l2_dt005_hor20": test_rollout_l2_dt005_hor20,
        "seed": int(args.seed),
    }
    with case_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    if getattr(args, "save_cell_ckpt", False):
        ckpt_name = case_path.stem + ".pt"   # same stem as JSON, e.g. b0.1_hor20_seed0.pt
        ckpt_save_path = out_dir / ckpt_name
        torch.save(net.state_dict(), ckpt_save_path)
        print(f"  [ckpt] saved model → {ckpt_save_path}", flush=True)


    return {
        "i": i,
        "j": j,
        "train_data_loss": float(loss_l1[-1]),
        "train_physics_loss": float(loss_l3[-1]),
        "train_total_loss": float(loss_total[-1]),
        "train_mse": train_mse,
        "train_sse": train_sse,
        "test_mse": test_mse,
        "test_sse": test_sse,
        "test_rollout_mse": test_rollout_mse,
        "test_rollout_sse": test_rollout_sse,
        "test_rollout_l2": float(metrics.get("test_rollout_l2", float("nan"))),
        "test_rollout_l2_mean_rel_window": float(
            metrics.get("test_rollout_l2_mean_rel_window", float("nan"))
        ),
        "test_rollout_l2_mean_rel_dt005_t0_20": float(
            metrics.get("test_rollout_l2_mean_rel_dt005_t0_20", float("nan"))
        ),
        "test_rollout_l2_dt050_hor10": float(
            metrics.get("test_rollout_l2_dt050_hor10", float("nan"))
        ),
        "test_rollout_l2_dt005_hor15": float(
            metrics.get("test_rollout_l2_dt005_hor15", float("nan"))
        ),
        "test_rollout_l2_dt005_hor20": float(
            metrics.get("test_rollout_l2_dt005_hor20", float("nan"))
        ),
    }


def _apply_cell_result_to_matrices(
    train_loss_matrix,
    test_error_matrix,
    train_sse_matrix,
    test_sse_matrix,
    train_data_loss_matrix,
    train_physics_loss_matrix,
    train_total_loss_matrix,
    test_rollout_error_matrix,
    test_rollout_sse_matrix,
    r: dict,
) -> None:
    i, j = r["i"], r["j"]
    train_loss_matrix[i, j] = r["train_mse"]
    # Match run_sweep_horizon_physics.py: phase-plot test error = rollout global rel-L2.
    test_error_matrix[i, j] = r.get("test_rollout_l2", r["test_mse"])
    train_sse_matrix[i, j] = r["train_sse"]
    test_sse_matrix[i, j] = r["test_sse"]
    train_data_loss_matrix[i, j] = r["train_data_loss"]
    train_physics_loss_matrix[i, j] = r["train_physics_loss"]
    train_total_loss_matrix[i, j] = r["train_total_loss"]
    test_rollout_error_matrix[i, j] = r["test_rollout_mse"]
    test_rollout_sse_matrix[i, j] = r["test_rollout_sse"]


def _parallel_cell_worker(task: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    args = argparse.Namespace(**task["args_dict"])
    return execute_sweep_cell(
        i=task["i"],
        j=task["j"],
        horizon=task["horizon"],
        b=task["b"],
        out_dir=Path(task["out_dir"]),
        args=args,
        state_representation=task["state_representation"],
        state_dim=task["state_dim"],
        sweep_updates_per_epoch=task["sweep_updates_per_epoch"],
        force_train=bool(task.get("force_train", False)),
    )


def _resolve_default_alm_init_ckpt() -> str:
    root = Path(__file__).resolve().parent
    for rel in ("lbfgs_ckpt_bank", "sweep_horizon_physics_lbfgs1000_5seeds/checkpoints"):
        p = root / rel
        if p.is_dir():
            return str(p)
    return ""


def _configure_pinn_alm_defaults(args) -> None:
    """Defaults aligned with scripts/run_horizon_pinn_alm_lbfgs_physics_8x8.sh (single-cell debug grid)."""
    if args.physics_mode != "pinn_alm":
        return

    if getattr(args, "no_alm_init_ckpt", False):
        args.alm_init_ckpt = ""
        print("[defaults] --no-alm-init-ckpt: skip LBFGS bank; use random init (+ warmup if set)", flush=True)
    elif not str(args.alm_init_ckpt).strip():
        ckpt = _resolve_default_alm_init_ckpt()
        if ckpt:
            args.alm_init_ckpt = ckpt
            print(f"[defaults] --alm-init-ckpt -> {ckpt}", flush=True)
        else:
            print(
                "[defaults] no lbfgs_ckpt_bank; random init (+ --alm-warmup-epochs if set)",
                flush=True,
            )

    if args.optimizer == "LBFGS":
        if int(args.alm_inner_step) != 1:
            args.alm_inner_step = 1
        if int(args.lbfgs_max_iter) >= 1000:
            args.lbfgs_max_iter = 100
            print(
                "[defaults] pinn_alm + LBFGS: --lbfgs-max-iter -> 100 per inner step",
                flush=True,
            )

    ib = _parse_float_csv(args.inv_b_values)
    hv = _parse_float_csv(args.horizon_values)
    if len(ib) * len(hv) == 1 and not args.write_case_json_only:
        args.write_case_json_only = True
        print(
            "[defaults] single grid cell: enabling --write-case-json-only (skip 1×1 heatmaps)",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep b vs training horizon for nonlinear pendulum with physics loss; "
            "optional curriculum warmup on damping coefficient b (--cl-warmup). "
            "Default grid: one cell (1/b=8, T_train=20, seed=0), pinn_alm+physics+LBFGS (8×8 sweep hyperparams)."
        )
    )
    parser.add_argument("--epochs", type=int, default=1,
                        help="Training epochs per grid cell when --cl-warmup is off (pinn_alm uses --alm-outer-iters). Use 600 for full Adam/LBFGS sweeps.")
    parser.add_argument(
        "--log-loss-every",
        type=int,
        default=10,
        help="Print L_data, L_phys, L_total every N ALM outer iters (0 disables). 8×8 sweep used 10.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="LBFGS",
        choices=["Adam", "SGD", "NNCG", "Adam_NNCG", "LBFGS", "Adam_LBFGS_NNCG"],
        help=(
            "Optimizer (default LBFGS for pinn_alm, same as 8×8 physics sweep). "
            "Inner: --alm-inner-step 1, --lbfgs-max-iter 100 per outer iter."
        ),
    )
    parser.add_argument(
        "--nncg-epochs",
        type=int,
        default=100,
        help="NNCG epochs after LBFGS when --optimizer Adam_LBFGS_NNCG.",
    )
    parser.add_argument(
        "--lbfgs-max-iter",
        type=int,
        default=100,
        help="For optimizer=LBFGS: max_iter in a single LBFGS step() (8×8 ALM inner default 100).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=str,
        default="3,4,5,6,7",
        help=(
            "Comma-separated random seeds: run the full grid once per seed and "
            "element-wise average matrices into one phase plot. When non-empty, overrides --seed."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="sweep_horizon_physics_pinn_alm_debug_ib8_hor020_uncon1_iter50",
    )
    parser.add_argument(
        "--skip-existing-cases",
        action="store_true",
        help=(
            "If the per-case JSON for this (horizon, b, seed) already exists under --out-dir, "
            "load it and skip training/eval for that cell."
        ),
    )
    parser.add_argument(
        "--save-cell-ckpt",
        action="store_true",
        help=(
            "After training each (horizon, b, seed) cell, save the model state_dict as "
            "<out_dir>/<case_stem>.pt (same name as the JSON, with .pt extension). "
            "Applies to all modes: Adam, pinn_alm, CL, etc. "
            "Use --alm-init-ckpt to load one of these files for ALM fine-tuning."
        ),
    )
    parser.add_argument(
        "--lamb",
        type=float,
        default=1.0,
        help="Physics penalty multiplier (lambda). Set 0.0 to disable physics loss.",
    )
    parser.add_argument(
        "--constraint-method",
        type=str,
        default="penalty",
        choices=["penalty", "alm"],
        help="Constraint handling for sphere physics. Default 'penalty' preserves prior experiment behavior.",
    )
    parser.add_argument(
        "--alm-penalty",
        type=float,
        default=1.0,
        help="Initial ALM penalty μ₀ (same as data-ALM production sweep).",
    )
    parser.add_argument(
        "--alm-penalty-growth",
        type=float,
        default=1.05,
        help="Per-outer multiplier for μ (data-ALM sweep default 1.05). Use 1.0 to keep fixed.",
    )
    parser.add_argument(
        "--alm-target",
        type=str,
        default="physics",
        choices=["physics", "data"],
        help=(
            "For physics_mode='pinn_alm': which loss is the ALM hard constraint. "
            "'physics' (default): minimize L_data, constrain ODE residual=0. "
            "'data': minimize L_physics, constrain pred=target."
        ),
    )
    parser.add_argument(
        "--alm-uncon-weight",
        type=float,
        default=1.0,
        help=(
            "For physics_mode='pinn_alm': scalar multiplier on the unconstrained objective "
            "(L_data for target=physics, L_physics for target=data). "
            "Analogous to uncon_weight in darcy_sweep.alm."
        ),
    )
    parser.add_argument(
        "--alm-outer-iters",
        type=int,
        default=50,
        help=(
            "For physics_mode='pinn_alm': number of outer ALM iterations (8×8 sweep used 100). "
            "Each outer iter updates λ once, then runs --alm-inner-step inner steps. "
        ),
    )
    parser.add_argument(
        "--alm-inner-step",
        type=int,
        default=1,
        help=(
            "For physics_mode='pinn_alm': inner steps per outer ALM iteration. "
            "Use 1 with --optimizer LBFGS (one full-batch solve per outer iter)."
        ),
    )
    parser.add_argument(
        "--alm-warmup-epochs",
        type=int,
        default=0,
        help=(
            "For physics_mode='pinn_alm': pinn+penalty warmup before ALM. "
            "Adam: epochs over the train loader. LBFGS: full-batch LBFGS steps (--lbfgs-max-iter each). "
            "Skip when --alm-init-ckpt loads a pretrained model. Loss histories are concatenated."
        ),
    )
    parser.add_argument(
        "--alm-init-ckpt",
        type=str,
        default="",
        help=(
            "For physics_mode='pinn_alm': path to a pretrained LBFGS .pt (state_dict) or a directory "
            "of per-cell files <case_stem>.pt (e.g. lbfgs_ckpt_bank/). "
            "If a directory entry is missing, PINN+LBFGS pretrain runs for that cell (--alm-bootstrap-lbfgs-max-iter), "
            "saves <case_stem>.pt there, then ALM continues. Use --no-alm-bootstrap to fail instead. "
            "Build the full bank first: bash scripts/run_horizon_lbfgs_ckpt_bank_cl.sh"
        ),
    )
    parser.add_argument(
        "--alm-bootstrap-lbfgs-max-iter",
        type=int,
        default=1000,
        help=(
            "When a per-cell file under --alm-init-ckpt is missing: one full-batch LBFGS step() "
            "with this max_iter (PINN penalty, same b/T as the ALM cell)."
        ),
    )
    parser.add_argument(
        "--no-alm-bootstrap",
        action="store_true",
        help="Do not auto-train missing LBFGS ckpts; raise if <dir>/<case_stem>.pt is absent.",
    )
    parser.add_argument(
        "--no-alm-init-ckpt",
        action="store_true",
        help=(
            "Do not load lbfgs_ckpt_bank (even if present). Each cell: random init, "
            "optional --alm-warmup-epochs PINN+LBFGS, then ALM."
        ),
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=1.0,
        help="Base region radius for physics_mode='pinn_region'. Effective radius is trust-calibrated and clipped.",
    )
    parser.add_argument(
        "--physics-mode",
        type=str,
        default="pinn_alm",
        choices=["none", "sphere", "pinn", "pinn_region", "pinn_alm"],
        help=(
            "Physics regularizer: none, sphere constraint, pointwise PINN residual, "
            "region-averaged PINN residual, or PINN-ALM (ALM-constrained PINN, "
            "see --alm-target for constraint direction)."
        ),
    )
    parser.add_argument(
        "--region-samples",
        type=int,
        default=4,
        help="Monte Carlo samples per training state for physics_mode='pinn_region' (includes the original point).",
    )
    parser.add_argument(
        "--region-history",
        type=int,
        default=10,
        help="Number of recent gradient vectors used for trust-region calibration in physics_mode='pinn_region'.",
    )
    parser.add_argument(
        "--region-max-radius",
        type=float,
        default=0.05,
        help="Maximum perturbation radius for physics_mode='pinn_region'.",
    )
    parser.add_argument("--fixed-dt", type=float, default=0.05)
    parser.add_argument("--nncg-rank", type=int, default=10)
    parser.add_argument("--nncg-mu", type=float, default=1e-2)
    parser.add_argument("--nncg-cg-tol", type=float, default=1e-5)
    parser.add_argument("--nncg-cg-max-iters", type=int, default=100)
    parser.add_argument("--nncg-precond-update-freq", type=int, default=1)
    parser.add_argument(
        "--nncg-switch-epoch",
        type=int,
        default=-1,
        help="Epoch at which Adam_NNCG switches from Adam to NNCG. If < 0, uses half the total epochs.",
    )
    parser.add_argument(
        "--updates-per-epoch",
        type=int,
        default=0,
        help=(
            "Optimizer steps per epoch for first-order methods. "
            "If <= 0, uses the maximum batches-per-epoch across the sweep so low-sample cases are not undertrained."
        ),
    )
    parser.add_argument("--test-horizon", type=float, default=20.0)
    parser.add_argument(
        "--fixed-test-dt",
        type=float,
        default=0.05,
        help="Test trajectory dt (match --fixed-dt for rollout metrics aligned with training).",
    )
    parser.add_argument(
        "--rollout-report-test-dt",
        type=float,
        default=0.05,
        help=(
            "If > 0, after training compute test_rollout_l2_mean_rel_window: mean over trajectories "
            "and over physical times in [--rollout-report-t-span-lo, --rollout-report-t-span-hi], "
            "using per-timestep relative L2 ||e||/||y|| on a synthetic test trajectory at this dt "
            "(independent of --fixed-test-dt)."
        ),
    )
    parser.add_argument(
        "--rollout-report-t-span-lo",
        type=float,
        default=0.0,
        help="Lower endpoint (inclusive) for rollout_report mean relative-L2 window.",
    )
    parser.add_argument(
        "--rollout-report-t-span-hi",
        type=float,
        default=10.0,
        help="Upper endpoint (inclusive) for rollout_report mean relative-L2 window.",
    )
    parser.add_argument(
        "--boundary-percentile",
        type=float,
        default=75.0,
        help="Boundary contour percentile (0-100) used for plotting overlays.",
    )
    parser.add_argument(
        "--load-results",
        type=str,
        default="",
        help="Path to a sweep_results.json to skip training and only plot.",
    )
    parser.add_argument(
        "--plot-inv-b-values",
        type=str,
        default="",
        help=(
            "Comma-separated 1/b axis values to keep when plotting (subset / reorder). "
            "Requires --load-results; warns and skips labels not present in the JSON grid."
        ),
    )
    parser.add_argument(
        "--b-values",
        type=str,
        default="",
        help="Comma-separated damping b values. If empty, uses the default 12x12 linspace grid.",
    )
    parser.add_argument(
        "--horizon-values",
        type=str,
        default="0.5,1,2,5,8,10,12,15,20",
        # default="0.5",
        help=(
            "Comma-separated training horizons t_max. If empty, uses linspace 1..20 (12 points). "
            "Default matches the 8×8 LBFGS/ALM sweep grid."
        ),
    )
    parser.add_argument(
        "--cl-warmup",
        action="store_true",
        help=(
            "Curriculum warmup on damping b: train from --cl-init-coeff toward the target "
            "coefficient in steps of --cl-delta-coeff, --cl-inner-epochs per stage. "
            "When --cl-target-coeff is omitted, the target is this sweep cell's b."
        ),
    )
    parser.add_argument(
        "--cl-init-coeff",
        type=float,
        default=None,
        help="Initial PDE coefficient (damping b) for curriculum. Larger b is typically easier.",
    )
    parser.add_argument(
        "--cl-delta-coeff",
        type=float,
        default=None,
        help="Absolute coefficient step toward the target each curriculum stage.",
    )
    parser.add_argument(
        "--cl-target-coeff",
        type=float,
        default=None,
        help="Terminal curriculum coefficient. Default: use the grid b of the current cell.",
    )
    parser.add_argument(
        "--cl-inner-epochs",
        type=int,
        default=None,
        help="Optimizer epochs per curriculum stage when --cl-warmup is set.",
    )
    parser.add_argument(
        "--cl-schedule-space",
        type=str,
        default="b",
        choices=["b", "inv_b"],
        help=(
            "Curriculum axis: 'b' steps damping b by --cl-delta-coeff; "
            "'inv_b' steps 1/b by --cl-inv-delta then uses b=1/inv_b each stage."
        ),
    )
    parser.add_argument(
        "--cl-inv-init",
        type=float,
        default=None,
        help="Initial 1/b when --cl-schedule-space inv_b.",
    )
    parser.add_argument(
        "--cl-inv-delta",
        type=float,
        default=None,
        help="Step size on the 1/b axis when --cl-schedule-space inv_b.",
    )
    parser.add_argument(
        "--inv-b-values",
        type=str,
        default="1,2,4,6,8,10,12,15",
        # default="10",
        help=(
            "Comma-separated 1/b grid points (sets b=1/inv for each). Overrides --b-values when non-empty. "
            "Default matches the 8×8 LBFGS/ALM sweep grid."
        ),
    )
    parser.add_argument(
        "--only-inv-b-values",
        type=str,
        default="",
        help=(
            "Comma-separated 1/b labels (e.g. 6). When non-empty, run only those columns of the "
            "full grid defined by --inv-b-values (ordering unchanged). Automatically enables "
            "--write-case-json-only so sweep_results.json / heatmaps are not overwritten with NaNs. "
            "Default empty = run all columns in --inv-b-values."
        ),
    )
    parser.add_argument(
        "--write-case-json-only",
        action="store_true",
        help=(
            "After running cells, skip sweep_results.json, matrix .npys, heatmaps, and loss-curve plots."
        ),
    )
    parser.add_argument(
        "--cl-save-int-inv-checkpoints",
        action="store_true",
        help="After each CL stage, save model.pt when current 1/b is (near) an integer.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel CPU worker processes for the sweep grid (1 = sequential).",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Train/eval on CUDA (single visible GPU per process; pair with CUDA_VISIBLE_DEVICES).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=-1,
        help="Run only grid cells with linear index ≡ shard-index (mod shard-count). Disabled if < 0.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=0,
        help="Modulus for sharding (must be > 0 when shard-index >= 0).",
    )
    args = parser.parse_args()
    _configure_pinn_alm_defaults(args)

    only_ib_csv = _parse_float_csv(args.only_inv_b_values)
    if only_ib_csv and not args.write_case_json_only:
        args.write_case_json_only = True
        print(
            "[only-inv-b-values] enabling --write-case-json-only "
            "(partial grid must not overwrite sweep aggregates).",
            flush=True,
        )

    if _parse_float_csv(args.plot_inv_b_values) and not (args.load_results or "").strip():
        parser.error("--plot-inv-b-values requires --load-results")

    if int(args.shard_index) >= 0:
        if int(args.shard_count) <= 0:
            parser.error("--shard-index requires --shard-count > 0")
        if int(args.shard_index) >= int(args.shard_count):
            parser.error("--shard-index must be strictly less than --shard-count")

    seeds_list = _parse_int_csv(args.seeds)
    if not seeds_list:
        seeds_list = [int(args.seed)]
    args.multi_seed_run = len(seeds_list) > 1

    if args.cl_warmup:
        if args.cl_inner_epochs is None:
            parser.error("--cl-warmup requires --cl-inner-epochs")
        if args.cl_inner_epochs < 1:
            parser.error("--cl-inner-epochs must be >= 1")
        if args.cl_schedule_space == "inv_b":
            if args.cl_inv_init is None or args.cl_inv_delta is None:
                parser.error("--cl-schedule-space inv_b requires --cl-inv-init and --cl-inv-delta")
            if args.cl_inv_delta <= 0:
                parser.error("--cl-inv-delta must be positive")
        else:
            if args.cl_init_coeff is None or args.cl_delta_coeff is None:
                parser.error("--cl-schedule-space b requires --cl-init-coeff and --cl-delta-coeff")
            if args.cl_delta_coeff <= 0:
                parser.error("--cl-delta-coeff must be positive")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_label = "Number of Samples"
    state_representation = (
        "state" if args.physics_mode in {"pinn", "pinn_region", "pinn_alm"} else "embedded"
    )
    state_dim = 2 if state_representation == "state" else 3
    if args.constraint_method == "alm" and args.physics_mode != "sphere":
        raise ValueError("constraint_method='alm' (legacy) is only supported for physics_mode='sphere'. Use physics_mode='pinn_alm' for ALM with PINN.")
    if args.physics_mode == "pinn_alm" and getattr(args, "alm_target", "physics") not in {"physics", "data"}:
        raise ValueError("--alm-target must be 'physics' or 'data'.")

    if args.load_results:
        with Path(args.load_results).open("r") as f:
            loaded = json.load(f)
        b_values = loaded["b_values"]
        horizon_values = loaded["horizon_values"]
        train_loss_matrix = np.array(loaded["train_loss_matrix"], dtype=float)
        test_error_matrix = np.array(loaded["test_error_matrix"], dtype=float)
        train_sse_matrix = np.array(loaded["train_sse_matrix"], dtype=float)
        test_sse_matrix = np.array(loaded["test_sse_matrix"], dtype=float)
        train_data_loss_matrix = np.array(
            loaded.get("train_data_loss_matrix", train_loss_matrix), dtype=float
        )
        train_physics_loss_matrix = np.array(
            loaded.get("train_physics_loss_matrix", np.zeros_like(train_loss_matrix)), dtype=float
        )
        train_total_loss_matrix = np.array(
            loaded.get("train_total_loss_matrix", train_loss_matrix), dtype=float
        )
        test_rollout_error_matrix = np.array(
            loaded.get("test_rollout_error_matrix", test_error_matrix), dtype=float
        )
        test_rollout_sse_matrix = np.array(
            loaded.get("test_rollout_sse_matrix", test_sse_matrix), dtype=float
        )
        dt = float(loaded.get("fixed_dt", args.fixed_dt))
        args.physics_mode = loaded.get("physics_mode", args.physics_mode)
        args.constraint_method = loaded.get("constraint_method", args.constraint_method)
        args.alm_penalty = float(loaded.get("alm_penalty", args.alm_penalty))
        args.alm_penalty_growth = float(loaded.get("alm_penalty_growth", args.alm_penalty_growth))
        if args.constraint_method == "alm" and args.physics_mode != "sphere":
            raise ValueError("Loaded results have constraint_method='alm' but physics_mode is not 'sphere'. Use physics_mode='pinn_alm' for ALM with PINN.")
    else:
        bv = _parse_float_csv(args.b_values)
        hv = _parse_float_csv(args.horizon_values)
        ib = _parse_float_csv(args.inv_b_values)
        if ib:
            b_values = [1.0 / float(x) for x in ib]
        elif bv:
            b_values = bv
        else:
            b_values = np.linspace(1.0, 0.1, 12).tolist()
        if hv:
            horizon_values = hv
        else:
            horizon_values = np.linspace(1.0, 20.0, 12).tolist()

        shape = (len(horizon_values), len(b_values))
        max_pairs = max(1, len(np.arange(0, max(horizon_values), args.fixed_dt)) - 1)
        max_batches_per_epoch = int(np.ceil(max_pairs / args.batch_size))
        default_updates_per_epoch = max(1, max_batches_per_epoch)
        sweep_updates_per_epoch = (
            int(args.updates_per_epoch)
            if int(args.updates_per_epoch) > 0
            else default_updates_per_epoch
        )
        print(f"Using updates_per_epoch={sweep_updates_per_epoch}")

        use_shard = int(args.shard_index) >= 0 and int(args.shard_count) > 0
        partial_cols = bool(only_ib_csv)
        fillv = float("nan") if (use_shard or partial_cols) else 0.0
        only_ib_set = {round(float(x), 4) for x in only_ib_csv} if only_ib_csv else None

        stack_tl = []
        stack_te = []
        stack_trsse = []
        stack_tesse = []
        stack_tdl = []
        stack_tpl = []
        stack_ttl = []
        stack_tre = []
        stack_trsse2 = []

        for si, seed in enumerate(seeds_list):
            args.seed = int(seed)
            print(
                f"\n========== sweep seed={seed} ({si + 1}/{len(seeds_list)}) ==========\n",
                flush=True,
            )
            train_loss_matrix = np.full(shape, fillv, dtype=float)
            test_error_matrix = np.full(shape, fillv, dtype=float)
            train_sse_matrix = np.full(shape, fillv, dtype=float)
            test_sse_matrix = np.full(shape, fillv, dtype=float)
            train_data_loss_matrix = np.full(shape, fillv, dtype=float)
            train_physics_loss_matrix = np.full(shape, fillv, dtype=float)
            train_total_loss_matrix = np.full(shape, fillv, dtype=float)
            test_rollout_error_matrix = np.full(shape, fillv, dtype=float)
            test_rollout_sse_matrix = np.full(shape, fillv, dtype=float)

            tasks = []
            for i, horizon in enumerate(horizon_values):
                for j, b in enumerate(b_values):
                    if only_ib_set is not None:
                        inv_lbl = round(1.0 / float(b), 4)
                        if inv_lbl not in only_ib_set:
                            continue
                    tasks.append(
                        {
                            "i": i,
                            "j": j,
                            "horizon": float(horizon),
                            "b": float(b),
                            "out_dir": str(out_dir.resolve()),
                            "args_dict": vars(args),
                            "state_representation": state_representation,
                            "state_dim": state_dim,
                            "sweep_updates_per_epoch": sweep_updates_per_epoch,
                        }
                    )

            if use_shard:
                si_s = int(args.shard_index)
                sc_s = int(args.shard_count)
                tasks = [
                    t for t in tasks if (t["i"] * len(b_values) + t["j"]) % sc_s == si_s
                ]
                print(
                    f"[shard {si_s}/{sc_s}] seed={seed}: running {len(tasks)} cells "
                    f"(of {shape[0] * shape[1]})",
                    flush=True,
                )

            n_workers = max(1, int(args.workers))
            if n_workers <= 1:
                for task in tasks:
                    r = execute_sweep_cell(
                        i=task["i"],
                        j=task["j"],
                        horizon=task["horizon"],
                        b=task["b"],
                        out_dir=out_dir,
                        args=args,
                        state_representation=state_representation,
                        state_dim=state_dim,
                        sweep_updates_per_epoch=sweep_updates_per_epoch,
                        force_train=False,
                    )
                    _apply_cell_result_to_matrices(
                        train_loss_matrix,
                        test_error_matrix,
                        train_sse_matrix,
                        test_sse_matrix,
                        train_data_loss_matrix,
                        train_physics_loss_matrix,
                        train_total_loss_matrix,
                        test_rollout_error_matrix,
                        test_rollout_sse_matrix,
                        r,
                    )
            else:
                nw = min(n_workers, len(tasks))
                print(f"Parallel sweep: {nw} worker processes, {len(tasks)} cells", flush=True)
                with ProcessPoolExecutor(max_workers=nw) as pool:
                    results = list(pool.map(_parallel_cell_worker, tasks))
                for r in results:
                    _apply_cell_result_to_matrices(
                        train_loss_matrix,
                        test_error_matrix,
                        train_sse_matrix,
                        test_sse_matrix,
                        train_data_loss_matrix,
                        train_physics_loss_matrix,
                        train_total_loss_matrix,
                        test_rollout_error_matrix,
                        test_rollout_sse_matrix,
                        r,
                    )

            stack_tl.append(train_loss_matrix.copy())
            stack_te.append(test_error_matrix.copy())
            stack_trsse.append(train_sse_matrix.copy())
            stack_tesse.append(test_sse_matrix.copy())
            stack_tdl.append(train_data_loss_matrix.copy())
            stack_tpl.append(train_physics_loss_matrix.copy())
            stack_ttl.append(train_total_loss_matrix.copy())
            stack_tre.append(test_rollout_error_matrix.copy())
            stack_trsse2.append(test_rollout_sse_matrix.copy())

        train_loss_matrix = np.nanmean(np.stack(stack_tl, axis=0), axis=0)
        test_error_matrix = np.nanmean(np.stack(stack_te, axis=0), axis=0)
        train_sse_matrix = np.nanmean(np.stack(stack_trsse, axis=0), axis=0)
        test_sse_matrix = np.nanmean(np.stack(stack_tesse, axis=0), axis=0)
        train_data_loss_matrix = np.nanmean(np.stack(stack_tdl, axis=0), axis=0)
        train_physics_loss_matrix = np.nanmean(np.stack(stack_tpl, axis=0), axis=0)
        train_total_loss_matrix = np.nanmean(np.stack(stack_ttl, axis=0), axis=0)
        test_rollout_error_matrix = np.nanmean(np.stack(stack_tre, axis=0), axis=0)
        test_rollout_sse_matrix = np.nanmean(np.stack(stack_trsse2, axis=0), axis=0)

        print(
            f"\n[seeds] averaged matrices over seeds={seeds_list} "
            f"({len(horizon_values)}×{len(b_values)} grid)\n",
            flush=True,
        )

    if getattr(args, "write_case_json_only", False) and not (args.load_results or "").strip():
        print(
            "[write-case-json-only] Done writing per-cell JSON/checkpoints; "
            "skipping sweep_results.json, matrix exports, and figures.",
            flush=True,
        )
        return

    if not args.load_results:
        dt = args.fixed_dt

    plot_ib = _parse_float_csv(args.plot_inv_b_values)
    if plot_ib:
        cols = _column_indices_for_inv_b_labels(b_values, plot_ib)
        if not cols:
            raise ValueError(
                "--plot-inv-b-values: no columns matched the loaded grid "
                "(check 1/b labels vs sweep_results.json b_values)."
            )
        b_values = [b_values[j] for j in cols]
        train_loss_matrix = train_loss_matrix[:, cols]
        test_error_matrix = test_error_matrix[:, cols]
        train_sse_matrix = train_sse_matrix[:, cols]
        test_sse_matrix = test_sse_matrix[:, cols]
        train_data_loss_matrix = train_data_loss_matrix[:, cols]
        train_physics_loss_matrix = train_physics_loss_matrix[:, cols]
        train_total_loss_matrix = train_total_loss_matrix[:, cols]
        test_rollout_error_matrix = test_rollout_error_matrix[:, cols]
        test_rollout_sse_matrix = test_rollout_sse_matrix[:, cols]
        kept = [1.0 / float(b_values[k]) for k in range(len(cols))]
        print(
            f"[plot-inv-b-values] plotting {len(cols)} columns in order {kept} (requested {plot_ib})",
            flush=True,
        )

    nt_values = [t / dt for t in horizon_values]

    inv_b_values = 1.0 / np.array(b_values, dtype=float)
    x_order = np.argsort(inv_b_values)
    inv_b_sorted = inv_b_values[x_order].tolist()
    train_loss_plot = train_loss_matrix[:, x_order]
    test_error_plot = test_error_matrix[:, x_order]
    train_sse_plot = train_sse_matrix[:, x_order]
    test_sse_plot = test_sse_matrix[:, x_order]
    train_data_loss_plot = train_data_loss_matrix[:, x_order]
    train_physics_loss_plot = train_physics_loss_matrix[:, x_order]
    train_total_loss_plot = train_total_loss_matrix[:, x_order]
    test_rollout_error_plot = test_rollout_error_matrix[:, x_order]
    test_rollout_sse_plot = test_rollout_sse_matrix[:, x_order]

    np.save(out_dir / "train_loss_matrix.npy", train_loss_matrix)
    np.save(out_dir / "test_error_matrix.npy", test_error_matrix)
    np.save(out_dir / "train_sse_matrix.npy", train_sse_matrix)
    np.save(out_dir / "test_sse_matrix.npy", test_sse_matrix)
    np.save(out_dir / "train_data_loss_matrix.npy", train_data_loss_matrix)
    np.save(out_dir / "train_physics_loss_matrix.npy", train_physics_loss_matrix)
    np.save(out_dir / "train_total_loss_matrix.npy", train_total_loss_matrix)
    np.save(out_dir / "test_rollout_error_matrix.npy", test_rollout_error_matrix)
    np.save(out_dir / "test_rollout_sse_matrix.npy", test_rollout_sse_matrix)

    sweep_payload = {
        "b_values": b_values,
        "horizon_values": horizon_values,
        "fixed_dt": args.fixed_dt,
        "test_horizon": args.test_horizon,
        "lamb": args.lamb,
        "constraint_method": args.constraint_method,
        "alm_penalty": args.alm_penalty,
        "alm_penalty_growth": args.alm_penalty_growth,
        "alm_target": getattr(args, "alm_target", "physics"),
        "alm_uncon_weight": getattr(args, "alm_uncon_weight", 1.0),
        "alm_outer_iters": getattr(args, "alm_outer_iters", 10),
        "alm_inner_step": getattr(args, "alm_inner_step", 100),
        "rho": args.rho,
        "physics_mode": args.physics_mode,
        "optimizer": args.optimizer,
        "nncg_switch_epoch": None if args.nncg_switch_epoch < 0 else args.nncg_switch_epoch,
        "train_loss_matrix": train_loss_matrix.tolist(),
        "test_error_matrix": test_error_matrix.tolist(),
        "train_sse_matrix": train_sse_matrix.tolist(),
        "test_sse_matrix": test_sse_matrix.tolist(),
        "train_data_loss_matrix": train_data_loss_matrix.tolist(),
        "train_physics_loss_matrix": train_physics_loss_matrix.tolist(),
        "train_total_loss_matrix": train_total_loss_matrix.tolist(),
        "test_rollout_error_matrix": test_rollout_error_matrix.tolist(),
        "test_rollout_sse_matrix": test_rollout_sse_matrix.tolist(),
    }
    if not args.load_results:
        sweep_payload["seeds_used_for_average"] = seeds_list
        sweep_payload["multi_seed_averaged"] = len(seeds_list) > 1
        sweep_payload["cuda_used"] = bool(getattr(args, "cuda", False))
        if int(args.shard_index) >= 0:
            sweep_payload["shard_index"] = int(args.shard_index)
            sweep_payload["shard_count"] = int(args.shard_count)
    if args.cl_warmup:
        sweep_payload["curriculum_warmup"] = True
        sweep_payload["cl_schedule_space"] = args.cl_schedule_space
        sweep_payload["cl_inner_epochs"] = args.cl_inner_epochs
        if args.cl_schedule_space == "inv_b":
            sweep_payload["cl_inv_init"] = args.cl_inv_init
            sweep_payload["cl_inv_delta"] = args.cl_inv_delta
        else:
            sweep_payload["cl_init_coeff"] = args.cl_init_coeff
            sweep_payload["cl_delta_coeff"] = args.cl_delta_coeff
        if args.cl_target_coeff is not None:
            sweep_payload["cl_target_coeff"] = args.cl_target_coeff
        sweep_payload["cl_target_coeff_note"] = (
            "Per cell: terminal coefficient is --cl-target-coeff when set, otherwise the grid b for that cell."
        )
        sweep_payload["cl_save_int_inv_checkpoints"] = bool(args.cl_save_int_inv_checkpoints)
        sweep_payload["workers"] = int(args.workers)
    else:
        sweep_payload["curriculum_warmup"] = False

    if plot_ib:
        sweep_payload["plot_inv_b_values_requested"] = plot_ib

    with (out_dir / "sweep_results.json").open("w") as f:
        json.dump(sweep_payload, f, indent=2)

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
