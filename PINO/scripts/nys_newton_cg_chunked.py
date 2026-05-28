"""Chunked-HVP variant of Nyström Newton-CG.

Mathematically identical to ``NysNewtonCG`` but every Hessian-vector product
and the full-batch gradient are computed by re-forwarding the model on
**chunks** of the data, instead of relying on a single retained graph for the
whole batch.

This removes the dominant memory cost for full-batch second-order training of
PINOs at large ``n`` (e.g. ``n=500`` + FNO rank-10 ran into ~80 GB peak HVP
memory on A100). With chunked HVP, peak memory is dominated by ONE chunk's
forward graph, so memory ~ ``O(chunk_size)`` not ``O(n)``.

Usage:
    nncg = ChunkedNysNewtonCG(
        model.parameters(),
        rank=2, mu=1e-2,
        cg_tol=1e-5, cg_max_iters=100,
    )
    grad_fn = make_chunked_grad_fn(model, ic_all, u_all, mol, lploss,
                                   xy_loss, f_loss, chunk_size)
    hvp_fn  = make_chunked_hvp_fn(model, ic_all, u_all, mol, lploss,
                                  xy_loss, f_loss, chunk_size)
    nncg.attach_callbacks(grad_fn=grad_fn, hvp_fn=hvp_fn)

    for k in range(refine_steps):
        if k % precond_update_freq == 0:
            nncg.update_preconditioner_chunked()
        loss = nncg.step_chunked()
"""
from __future__ import annotations

from functools import reduce
from typing import Callable, Iterator, Optional, Tuple

import torch

from nys_newton_cg import (
    NysNewtonCG,
    _apply_nys_precond_inv,
    _nystrom_pcg,
    _safe_eigh_with_jitter,
    _safe_eigvalsh,
    _split_flat_like_params,
)


GradFn = Callable[[], Tuple[torch.Tensor, torch.Tensor]]
HvpFn = Callable[[torch.Tensor], torch.Tensor]
LossFn = Callable[[], torch.Tensor]


class NncgSubsampleState:
    """Fixed random train indices for one NNCG outer step (grad, HVP, Armijo)."""

    def __init__(
        self,
        n_total: int,
        subsample_size: int,
        device: torch.device,
    ) -> None:
        self.n_total = int(n_total)
        ss = int(subsample_size)
        if ss <= 0 or ss >= self.n_total:
            self.enabled = False
            self.subsample_size = self.n_total
        else:
            self.enabled = True
            self.subsample_size = ss
        self.device = device
        self.active_idx: Optional[torch.Tensor] = None
        self.last_step = -1

    def refresh(self, step: int) -> None:
        """Resample indices for outer step ``step`` (call once per NNCG iteration)."""
        self.last_step = int(step)
        if not self.enabled:
            self.active_idx = None
            return
        g = torch.Generator(device=self.device)
        g.manual_seed(int(step) + 10_007 * int(self.n_total))
        self.active_idx = torch.randperm(
            self.n_total, generator=g, device=self.device
        )[: self.subsample_size]

    @property
    def n_active(self) -> int:
        return self.subsample_size if self.enabled else self.n_total


class NncgLoaderState:
    """DataLoader-backed batches for NNCG (Adam-style data pipeline).

    * ``loader_full``: microbatch accumulation over the full train set (full-batch
      objective, memory-friendly vs preloading ``ic_all``).
    * ``loader_minibatch``: one random minibatch per outer NNCG step (like Adam).
    """

    MODES = ("loader_full", "loader_minibatch")

    def __init__(
        self,
        mode: str,
        dataset,
        n_total: int,
        device: torch.device,
        loader_batch_size: int,
        minibatch_size: int = 20,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"NncgLoaderState mode must be one of {self.MODES}")
        self.mode = mode
        self.dataset = dataset
        self.n_total = int(n_total)
        self.device = device
        self.loader_batch_size = max(1, int(loader_batch_size))
        self.minibatch_size = max(1, int(minibatch_size))
        self._ic_mb: Optional[torch.Tensor] = None
        self._u_mb: Optional[torch.Tensor] = None

    def refresh_minibatch(self, step: int) -> None:
        """Sample one minibatch for ``loader_minibatch`` mode."""
        if self.mode != "loader_minibatch":
            return
        g = torch.Generator()
        g.manual_seed(int(step) + 10_007 * self.n_total)
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.minibatch_size,
            shuffle=True,
            generator=g,
            drop_last=False,
        )
        ic, u = next(iter(loader))
        self._ic_mb = ic.to(self.device)
        self._u_mb = u.to(self.device)

    def iter_batches(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, float]]:
        """Yield ``(ic, u, weight)`` where weights sum to 1 over the active set."""
        if self.mode == "loader_minibatch":
            if self._ic_mb is None or self._u_mb is None:
                raise RuntimeError(
                    "NncgLoaderState: call refresh_minibatch(step) before iter_batches()"
                )
            yield self._ic_mb, self._u_mb, 1.0
            return
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.loader_batch_size,
            shuffle=False,
            drop_last=False,
        )
        for ic, u in loader:
            ic = ic.to(self.device)
            u = u.to(self.device)
            w = float(ic.shape[0]) / float(self.n_total)
            yield ic, u, w

    @property
    def n_active(self) -> int:
        if self.mode == "loader_minibatch":
            if self._ic_mb is not None:
                return int(self._ic_mb.shape[0])
            return self.minibatch_size
        return self.n_total


def _iter_index_chunks(
    n_total: int,
    chunk_size: int,
    device: torch.device,
    active_idx: Optional[torch.Tensor],
) -> Iterator[Tuple[torch.Tensor, int]]:
    """Yield ``(row_indices, count)`` groups; weights use ``count / n_active``."""
    chunk_size = max(1, min(int(chunk_size), n_total))
    if active_idx is None:
        for s in range(0, n_total, chunk_size):
            e = min(s + chunk_size, n_total)
            yield torch.arange(s, e, device=device, dtype=torch.long), e - s
    else:
        m = int(active_idx.numel())
        for s in range(0, m, chunk_size):
            e = min(s + chunk_size, m)
            yield active_idx[s:e], e - s


def _chunked_loss_scalar(
    out_c: torch.Tensor,
    u_c: torch.Tensor,
    ic_c: torch.Tensor,
    lploss,
    darcy_loss_fn,
    xy_loss: float,
    f_loss: float,
    loss_mode: str,
) -> torch.Tensor:
    if loss_mode == "pino":
        l_pde = darcy_loss_fn(out_c, ic_c[..., 0])
        l_data = lploss(out_c, u_c)
        return xy_loss * l_data + f_loss * l_pde
    if loss_mode == "data":
        return xy_loss * lploss(out_c, u_c)
    if loss_mode == "pde":
        return f_loss * darcy_loss_fn(out_c, ic_c[..., 0])
    raise ValueError(f"Unknown loss_mode {loss_mode!r}")


class ChunkedNysNewtonCG(NysNewtonCG):
    """``NysNewtonCG`` that uses caller-provided chunked grad_fn / hvp_fn.

    All numerics (Nyström sketch, Cholesky / eigh fallbacks, PCG, Armijo) are
    inherited unchanged. We only override the grad/HVP plumbing so we never
    need a retained full-batch graph.

    Without line search, the Newton step ``θ ← θ - lr * d`` is unsafe at low
    Nyström rank: the local quadratic model is a poor approximation of the
    true Hessian and ``d`` may overshoot, so we provide a chunked Armijo
    line search that uses the supplied ``loss_fn`` to evaluate trial loss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._grad_fn: Optional[GradFn] = None
        self._hvp_fn: Optional[HvpFn] = None
        self._loss_fn: Optional[LossFn] = None

    def attach_callbacks(
        self,
        grad_fn: GradFn,
        hvp_fn: HvpFn,
        loss_fn: Optional[LossFn] = None,
    ) -> None:
        self._grad_fn = grad_fn
        self._hvp_fn = hvp_fn
        self._loss_fn = loss_fn

    def update_preconditioner_chunked(self) -> None:
        """Build the rank-``self.rank`` Nyström approximation of the Hessian
        using ``self._hvp_fn`` (chunked) instead of vmap on a retained graph."""
        if self._hvp_fn is None:
            raise RuntimeError(
                "ChunkedNysNewtonCG: call attach_callbacks(grad_fn, hvp_fn) first"
            )

        params_list = self._params_list
        device = params_list[0].device
        dtype = (
            torch.float64 if self.use_double else params_list[0].dtype
        )

        p = self._numel()
        Phi = torch.randn((self.rank, p), device=device, dtype=dtype) / (p ** 0.5)
        Phi = torch.linalg.qr(Phi.t(), mode="reduced")[0].t().contiguous()

        Y_rows = []
        for i in range(self.rank):
            Hv = self._hvp_fn(Phi[i])
            if torch.is_complex(Hv):
                Hv = torch.real(Hv)
            if Hv.dtype != dtype:
                Hv = Hv.to(dtype=dtype)
            Y_rows.append(Hv.detach())
        Y = torch.stack(Y_rows, dim=0)

        base_shift = torch.finfo(Y.dtype).eps
        shift = base_shift
        Y_shifted = Y + shift * Phi

        choleskytarget = torch.mm(Y_shifted, Phi.t())
        choleskytarget = (choleskytarget + choleskytarget.T) / 2

        C = None
        attempt = 0
        max_attempts = 5
        while C is None and attempt < max_attempts:
            try:
                if attempt == 0:
                    C = torch.linalg.cholesky(choleskytarget)
                else:
                    eigs = _safe_eigvalsh(choleskytarget)
                    min_eig = eigs[0]
                    if min_eig <= 0:
                        adaptive_jitter = (
                            torch.abs(min_eig)
                            + base_shift
                            + self.jitter_factor * eigs[-1]
                        )
                    else:
                        adaptive_jitter = (
                            self.jitter_factor * eigs[-1] * (10 ** attempt)
                        )
                    jitter = adaptive_jitter * torch.eye(
                        choleskytarget.shape[0],
                        device=choleskytarget.device,
                        dtype=choleskytarget.dtype,
                    )
                    shift = shift + adaptive_jitter
                    C = torch.linalg.cholesky(choleskytarget + jitter)
            except torch.linalg.LinAlgError:
                attempt += 1
                self.cholesky_failures += 1
                if attempt >= max_attempts:
                    eigs, eigvectors = _safe_eigh_with_jitter(
                        choleskytarget, float(base_shift), self.verbose,
                    )
                    min_eig = torch.min(eigs)
                    if min_eig <= 0:
                        eig_shift = (
                            torch.abs(min_eig)
                            + base_shift
                            + self.jitter_factor * torch.max(eigs)
                        )
                    else:
                        eig_shift = self.jitter_factor * torch.max(eigs)
                    eigs_shifted = torch.clamp(eigs + eig_shift, min=base_shift)
                    shift = shift + eig_shift
                    rebuilt = (
                        eigvectors @ torch.diag(eigs_shifted) @ eigvectors.T
                    )
                    rebuilt = (rebuilt + rebuilt.T) / 2
                    try:
                        C = torch.linalg.cholesky(rebuilt)
                    except torch.linalg.LinAlgError:
                        rb64 = rebuilt.detach().to(dtype=torch.float64)
                        C64 = torch.linalg.cholesky(
                            rb64
                            + 1e-6
                            * torch.eye(
                                rb64.shape[0],
                                device=rb64.device,
                                dtype=torch.float64,
                            )
                        )
                        C = C64.to(dtype=rebuilt.dtype)
                    if self.dynamic_damping and self.mu < self.max_mu:
                        self.mu = min(self.mu * 10, self.max_mu)

        if C is None:
            d = choleskytarget.shape[0]
            eye = torch.eye(d, device=choleskytarget.device, dtype=choleskytarget.dtype)
            tr = torch.trace(choleskytarget)
            lam = (torch.nan_to_num(tr.abs() / d, nan=1.0) + 1.0).detach()
            for mag in (1e-4, 1e-2, 1.0, 1e2, 1e4):
                try:
                    C = torch.linalg.cholesky(choleskytarget + (lam * mag) * eye)
                    break
                except torch.linalg.LinAlgError:
                    continue
            if C is None:
                raise RuntimeError(
                    "ChunkedNysNewtonCG.update_preconditioner_chunked: Cholesky failed"
                )

        try:
            B = torch.linalg.solve_triangular(C, Y_shifted, upper=False, left=True)
        except Exception:
            B = torch.linalg.solve_triangular(
                C.to("cpu"), Y_shifted.to("cpu"), upper=False, left=True,
            ).to(C.device)

        # GPU SVD (cusolver) can fail with INVALID_VALUE on ill-conditioned / NaN B.
        try:
            if not torch.isfinite(B).all():
                B = torch.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
            _, S, UT = torch.linalg.svd(B, full_matrices=False)
        except (torch.linalg.LinAlgError, RuntimeError):
            B64 = B.detach().to(device="cpu", dtype=torch.float64)
            if not torch.isfinite(B64).all():
                B64 = torch.nan_to_num(B64, nan=0.0, posinf=0.0, neginf=0.0)
            _, S, UT = torch.linalg.svd(B64, full_matrices=False)
            S = S.to(device=B.device, dtype=B.dtype)
            UT = UT.to(device=B.device, dtype=B.dtype)
        self.U = UT.t()
        self.S = torch.clamp(torch.square(S) - shift, min=0.0)
        self.rho = self.S[-1]

    def _apply_step(self, d: torch.Tensor, t: float) -> None:
        """In-place: θ ← θ - t * d (where d is flat, split per-param)."""
        for group in self.param_groups:
            ls = 0
            for p in group["params"]:
                np_ = torch.numel(p)
                dp = d[ls : ls + np_].reshape(p.shape)
                ls += np_
                p.data.add_(-dp, alpha=t)

    def _undo_step(self, d: torch.Tensor, t: float) -> None:
        for group in self.param_groups:
            ls = 0
            for p in group["params"]:
                np_ = torch.numel(p)
                dp = d[ls : ls + np_].reshape(p.shape)
                ls += np_
                p.data.add_(dp, alpha=t)

    def _chunked_armijo(
        self,
        d: torch.Tensor,
        g: torch.Tensor,
        f0: float,
        lr: float,
        alpha: float = 0.1,
        beta: float = 0.5,
        max_iters: int = 25,
        min_t: float = 1e-8,
    ) -> Tuple[float, float]:
        """Backtracking Armijo line search using ``self._loss_fn``.

        Returns ``(t, f1)``: accepted step size and loss after the step. The
        parameters are left at ``θ - t*d`` on return. If no Armijo-feasible
        ``t`` is found within ``max_iters`` halvings or below ``min_t``, we
        accept the smallest ``t`` tried (best-effort)."""
        if self._loss_fn is None:
            self._apply_step(d, lr)
            return float(lr), float("nan")

        gd = float(torch.dot(g, d).item())
        t = float(lr)
        best_t = t
        best_f = float("inf")

        for k in range(max_iters):
            self._apply_step(d, t)
            try:
                f1_t = self._loss_fn()
                f1 = float(f1_t.item() if torch.is_tensor(f1_t) else f1_t)
            except Exception:
                f1 = float("inf")
            if not (f1 != f1):  # finite check ignoring NaN
                if f1 < best_f:
                    best_f = f1
                    best_t = t
            if f1 <= f0 - alpha * t * gd and (f1 == f1) and f1 != float("inf"):
                return t, f1
            self._undo_step(d, t)
            t *= beta
            if t < min_t:
                break

        # No feasible t found; apply the best one we saw (or the smallest).
        if best_f < float("inf"):
            self._apply_step(d, best_t)
            return best_t, best_f
        # Worst case: take a tiny step
        self._apply_step(d, min_t)
        return float(min_t), float("nan")

    def step_chunked(self) -> torch.Tensor:
        """Same as ``NysNewtonCG.step`` but uses ``self._grad_fn`` (chunked)
        to obtain the full-batch gradient and ``self._hvp_fn`` (chunked) to
        run NyströmPCG.

        If ``self._loss_fn`` is attached and ``self.line_search_fn ==
        'armijo'``, performs chunked backtracking Armijo line search. With
        line search, lr is the initial step size (try lr=1.0 for "Newton-like"
        steps); without, lr is the fixed step size and may need to be small
        (e.g. 0.1 or 0.01) at low Nyström rank to avoid divergence.

        Returns the loss tensor at the start of the step (scalar)."""
        if self._grad_fn is None or self._hvp_fn is None:
            raise RuntimeError(
                "ChunkedNysNewtonCG: call attach_callbacks(grad_fn, hvp_fn) first"
            )
        if self.U is None or self.S is None:
            raise RuntimeError(
                "ChunkedNysNewtonCG: call update_preconditioner_chunked() first"
            )

        if self.n_iters == 0:
            self.old_dir = torch.zeros(
                self._numel(), device=self._params[0].device,
            )

        loss, g = self._grad_fn()
        if torch.is_complex(g):
            g = torch.real(g)

        d = _nystrom_pcg(
            self._hvp_fn,
            g,
            self.old_dir,
            self.mu,
            self.U,
            self.S,
            self.rank,
            self.cg_tol,
            self.cg_max_iters,
        )
        # Guard: if d is not a descent direction, fall back to negative grad.
        if torch.dot(d, g) <= 0:
            d = g.clone()
        self.old_dir = d

        lr = float(self.param_groups[0]["lr"])
        if self.line_search_fn == "armijo" and self._loss_fn is not None:
            f0 = float(loss.item() if torch.is_tensor(loss) else loss)
            t, _ = self._chunked_armijo(d, g, f0, lr)
        else:
            self._apply_step(d, lr)
            t = lr
        # Track step size for diagnostics
        self.state[0] = self.state.get(0, {})
        self.state[0]["t"] = t

        self.n_iters += 1
        return loss


def _flatten_params(params_list):
    return torch.cat([p.detach().reshape(-1) for p in params_list])


def _split_v_to_param_dtypes(v_flat, params_list):
    """Split flat vector into per-parameter tensors with matching dtype/device."""
    out = []
    offset = 0
    for p in params_list:
        n = p.numel()
        chunk = v_flat[offset : offset + n].reshape_as(p)
        if chunk.dtype != p.dtype:
            chunk = chunk.to(dtype=p.dtype)
        if chunk.device != p.device:
            chunk = chunk.to(device=p.device)
        out.append(chunk)
        offset += n
    assert offset == v_flat.numel()
    return out


def make_chunked_grad_fn(
    model,
    ic_all: torch.Tensor,
    u_all: torch.Tensor,
    mol: torch.Tensor,
    lploss,
    darcy_loss_fn,
    xy_loss: float,
    f_loss: float,
    params_list,
    chunk_size: int,
    loss_mode: str = "pino",
    subsample_state: Optional[NncgSubsampleState] = None,
    loader_state: Optional[NncgLoaderState] = None,
) -> GradFn:
    """Build ``(loss_total, g_flat)`` over full batch or a fixed subsample per step."""
    n = int(ic_all.shape[0]) if ic_all is not None else (
        loader_state.n_total if loader_state is not None else 0
    )
    device = (
        ic_all.device
        if ic_all is not None
        else (loader_state.device if loader_state is not None else params_list[0].device)
    )

    def grad_fn():
        for p in params_list:
            if p.grad is not None:
                p.grad = None
        loss_total = torch.zeros((), device=params_list[0].device)
        n_seen = 0
        if loader_state is not None:
            n_active = loader_state.n_active
            batch_iter = loader_state.iter_batches()
        else:
            n_active = subsample_state.n_active if subsample_state is not None else n
            active_idx = subsample_state.active_idx if subsample_state is not None else None
            batch_iter = (
                (ic_all[idx], u_all[idx], float(cnt) / float(n_active))
                for idx, cnt in _iter_index_chunks(n, chunk_size, device, active_idx)
            )
        for ic_c, u_c, w in batch_iter:
            out_c = model(ic_c).squeeze(-1) * mol
            loss_c = _chunked_loss_scalar(
                out_c, u_c, ic_c, lploss, darcy_loss_fn, xy_loss, f_loss, loss_mode,
            )
            if torch.is_complex(loss_c):
                loss_c = torch.real(loss_c)
            (loss_c * w).backward()
            with torch.no_grad():
                loss_total = loss_total + loss_c.detach() * w
            n_seen += int(ic_c.shape[0])
            del out_c, loss_c
        assert n_seen == n_active
        g_flat = torch.cat(
            [
                (p.grad.detach().reshape(-1)
                 if p.grad is not None
                 else torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
                for p in params_list
            ]
        )
        if torch.is_complex(g_flat):
            g_flat = torch.real(g_flat)
        return loss_total.detach(), g_flat

    return grad_fn


def make_chunked_loss_fn(
    model,
    ic_all: torch.Tensor,
    u_all: torch.Tensor,
    mol: torch.Tensor,
    lploss,
    darcy_loss_fn,
    xy_loss: float,
    f_loss: float,
    chunk_size: int,
    loss_mode: str = "pino",
    subsample_state: Optional[NncgSubsampleState] = None,
    loader_state: Optional[NncgLoaderState] = None,
) -> LossFn:
    """Scalar loss for Armijo (same subsample as grad/HVP when ``subsample_state`` set)."""
    n = int(ic_all.shape[0]) if ic_all is not None else (
        loader_state.n_total if loader_state is not None else 0
    )
    device = (
        ic_all.device
        if ic_all is not None
        else (loader_state.device if loader_state is not None else model.parameters().__next__().device)
    )

    def loss_fn():
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                dev = ic_all.device if ic_all is not None else device
                total = torch.zeros((), device=dev)
                if loader_state is not None:
                    batch_iter = loader_state.iter_batches()
                    n_active = loader_state.n_active
                else:
                    n_active = subsample_state.n_active if subsample_state is not None else n
                    active_idx = subsample_state.active_idx if subsample_state is not None else None
                    batch_iter = (
                        (ic_all[idx], u_all[idx], float(cnt) / float(n_active))
                        for idx, cnt in _iter_index_chunks(n, chunk_size, device, active_idx)
                    )
                n_seen = 0
                for ic_c, u_c, w in batch_iter:
                    out_c = model(ic_c).squeeze(-1) * mol
                    loss_c = _chunked_loss_scalar(
                        out_c, u_c, ic_c, lploss, darcy_loss_fn,
                        xy_loss, f_loss, loss_mode,
                    )
                    if torch.is_complex(loss_c):
                        loss_c = torch.real(loss_c)
                    total = total + loss_c * w
                    n_seen += int(ic_c.shape[0])
                assert n_seen == (loader_state.n_active if loader_state is not None else n_active)
                return total.detach()
        finally:
            if was_training:
                model.train()

    return loss_fn


def make_chunked_hvp_fn(
    model,
    ic_all: torch.Tensor,
    u_all: torch.Tensor,
    mol: torch.Tensor,
    lploss,
    darcy_loss_fn,
    xy_loss: float,
    f_loss: float,
    params_list,
    chunk_size: int,
    loss_mode: str = "pino",
    subsample_state: Optional[NncgSubsampleState] = None,
    loader_state: Optional[NncgLoaderState] = None,
) -> HvpFn:
    """``hvp_fn(v) -> Hv`` on full batch or the active subsample (chunked HVP)."""
    n = int(ic_all.shape[0]) if ic_all is not None else (
        loader_state.n_total if loader_state is not None else 0
    )
    device = (
        ic_all.device
        if ic_all is not None
        else (loader_state.device if loader_state is not None else params_list[0].device)
    )

    def hvp_fn(v_flat: torch.Tensor) -> torch.Tensor:
        v_parts = _split_v_to_param_dtypes(v_flat, params_list)
        Hv_total = None
        if loader_state is not None:
            batch_iter = loader_state.iter_batches()
        else:
            n_active = subsample_state.n_active if subsample_state is not None else n
            active_idx = subsample_state.active_idx if subsample_state is not None else None
            batch_iter = (
                (ic_all[idx], u_all[idx], float(cnt) / float(n_active))
                for idx, cnt in _iter_index_chunks(n, chunk_size, device, active_idx)
            )
        for ic_c, u_c, w in batch_iter:
            out_c = model(ic_c).squeeze(-1) * mol
            loss_c = _chunked_loss_scalar(
                out_c, u_c, ic_c, lploss, darcy_loss_fn, xy_loss, f_loss, loss_mode,
            )
            if torch.is_complex(loss_c):
                loss_c = torch.real(loss_c)
            gcs = torch.autograd.grad(
                loss_c * w, params_list, create_graph=True,
            )
            Hv_c = torch.autograd.grad(
                gcs, params_list, grad_outputs=v_parts, retain_graph=False,
            )
            Hv_c_flat = torch.cat(
                [h.reshape(-1).detach() for h in Hv_c]
            )
            if torch.is_complex(Hv_c_flat):
                Hv_c_flat = torch.real(Hv_c_flat)
            Hv_total = Hv_c_flat if Hv_total is None else Hv_total + Hv_c_flat
            del out_c, loss_c, gcs, Hv_c, Hv_c_flat
        return Hv_total

    return hvp_fn
