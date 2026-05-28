import os

import torch
from torch.optim import Optimizer
from torch.func import vmap
from functools import reduce

import numpy as np


def _symmetric_numpy(A: torch.Tensor) -> np.ndarray:
    Ac = A.detach().to(device="cpu", dtype=torch.float64)
    M = (Ac + Ac.T) * 0.5
    return M.numpy()


def _safe_eigvalsh(A: torch.Tensor) -> torch.Tensor:
    """Hermitian eigvals; GPU / torch CPU may fail — fall back to NumPy."""
    try:
        return torch.linalg.eigvalsh(A)
    except Exception:
        try:
            A64 = A.detach().to(device="cpu", dtype=torch.float64)
            return torch.linalg.eigvalsh(A64).to(device=A.device, dtype=A.dtype)
        except Exception:
            w = np.linalg.eigvalsh(_symmetric_numpy(A))
            return torch.from_numpy(w).to(device=A.device, dtype=A.dtype)


def _safe_eigh(A: torch.Tensor):
    """Hermitian eigh: torch → torch CPU64 → NumPy (symmetrized)."""
    dev, dt = A.device, A.dtype
    try:
        return torch.linalg.eigh(A)
    except Exception:
        pass
    try:
        A64 = A.detach().to(device="cpu", dtype=torch.float64)
        w, Q = torch.linalg.eigh(A64)
        return w.to(device=dev, dtype=dt), Q.to(device=dev, dtype=dt)
    except Exception:
        pass
    M = _symmetric_numpy(A)
    w, Q = np.linalg.eigh(M)
    w_t = torch.from_numpy(w).to(device=dev, dtype=dt)
    Q_t = torch.from_numpy(Q).to(device=dev, dtype=dt)
    return w_t, Q_t


def _safe_eigh_with_jitter(A: torch.Tensor, base_eps: float, verbose: bool):
    """Repeated eigh with increasing diagonal jitter (ill-conditioned PSD sketches)."""
    d = A.shape[0]
    eye = torch.eye(d, device=A.device, dtype=A.dtype)
    for mag in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2):
        Aj = A + (base_eps + mag) * eye
        Aj = (Aj + Aj.T) / 2
        try:
            return _safe_eigh(Aj)
        except Exception as e:
            if verbose:
                print(f"Warning: eigh retry mag={mag}: {e}")
            continue
    # Last resort: diagonal PSD (identity directions)
    if verbose:
        print("Warning: eigh failed completely; using diagonal PSD fallback for this step.")
    z = torch.ones(d, device=A.device, dtype=A.dtype) * (base_eps + 1e-3)
    I = torch.eye(d, device=A.device, dtype=A.dtype)
    return z, I


def _split_flat_like_params(flat: torch.Tensor, params) -> list:
    """Views matching param shapes; ``flat.numel()`` must equal sum of param numels."""
    out = []
    offset = 0
    for p in params:
        n = p.numel()
        out.append(flat[offset : offset + n].reshape_as(p))
        offset += n
    assert offset == flat.numel()
    return out


def _match_hvp_grad_output(gc: torch.Tensor, vc: torch.Tensor) -> torch.Tensor:
    """``grad_outputs`` must match ``gc`` dtype for nested ``autograd.grad``."""
    vc = vc.to(device=gc.device)
    if gc.dtype == vc.dtype:
        return vc
    if gc.is_complex():
        if vc.is_complex():
            return vc.to(dtype=gc.dtype)
        base = vc.to(dtype=gc.real.dtype)
        return torch.complex(base, torch.zeros_like(base)).to(dtype=gc.dtype)
    if vc.is_complex():
        return torch.real(vc).to(dtype=gc.dtype)
    return vc.to(dtype=gc.dtype)


def _armijo(f, x, gx, dx, t, alpha=0.1, beta=0.5):
    """Line search to find a step size that satisfies the Armijo condition."""
    f0 = f(x, 0, dx)
    f1 = f(x, t, dx)
    while f1 > f0 + alpha * t * gx.dot(dx):
        t *= beta
        f1 = f(x, t, dx)
    return t

def _apply_nys_precond_inv(U, S_mu_inv, mu, lambd_r, x):
    """Applies the inverse of the Nystrom approximation of the Hessian to a vector."""
    z = U.T @ x
    z = (lambd_r + mu) * (U @ (S_mu_inv * z)) + (x - U @ z)
    return z

def _nystrom_pcg(hess, b, x, mu, U, S, r, tol, max_iters):
    """Solves a positive-definite linear system using NyströmPCG.

    `Frangella et al. Randomized Nyström Preconditioning. 
    SIAM Journal on Matrix Analysis and Applications, 2023.
    <https://epubs.siam.org/doi/10.1137/21M1466244>`"""
    lambd_r = S[r - 1]
    S_mu_inv = (S + mu) ** (-1)

    resid = b - (hess(x) + mu * x)
    with torch.no_grad():
        z = _apply_nys_precond_inv(U, S_mu_inv, mu, lambd_r, resid)
        p = z.clone()

    i = 0

    while torch.norm(resid) > tol and i < max_iters:
        v = hess(p) + mu * p
        with torch.no_grad():
            alpha = torch.dot(resid, z) / torch.dot(p, v)
            x += alpha * p

            rTz = torch.dot(resid, z)
            resid -= alpha * v
            z = _apply_nys_precond_inv(U, S_mu_inv, mu, lambd_r, resid)
            beta = torch.dot(resid, z) / rTz

            p = z + beta * p

        i += 1

    if torch.norm(resid) > tol:
        print(f"Warning: PCG did not converge to tolerance. Tolerance was {tol} but norm of residual is {torch.norm(resid)}")

    return x

class NysNewtonCG(Optimizer):
    """Implementation of NysNewtonCG, a damped Newton-CG method that uses Nyström preconditioning.
    
    `Rathore et al. Challenges in Training PINNs: A Loss Landscape Perspective.
    Preprint, 2024. <https://arxiv.org/abs/2402.01868>`

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    TODO: This optimizer is currently a beta version. 

    Our implementation is inspired by the PyTorch implementation of `L-BFGS 
    <https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS>`.
    
    The parameters rank and mu will probably need to be tuned for your specific problem.
    If the optimizer is running very slowly, you can try one of the following:
    - Increase the rank (this should increase the accuracy of the Nyström approximation in PCG)
    - Reduce cg_tol (this will allow PCG to terminate with a less accurate solution)
    - Reduce cg_max_iters (this will allow PCG to terminate after fewer iterations)

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1.0)
        rank (int, optional): rank of the Nyström approximation (default: 10)
        mu (float, optional): damping parameter (default: 0.01)
        chunk_size (int, optional): number of Hessian-vector products to be computed in parallel (default: 1)
        cg_tol (float, optional): tolerance for PCG (default: 1e-5)
        cg_max_iters (int, optional): maximum number of PCG iterations (default: 1000)
        line_search_fn (str, optional): either 'armijo' or None (default: None)
        verbose (bool, optional): verbosity (default: False)
        use_double (bool, optional): use float64 precision for better numerical stability (default: False)
        dynamic_damping (bool, optional): automatically increase mu when encountering numerical issues (default: True)
        max_mu (float, optional): maximum value for mu when using dynamic damping (default: 1e2)
        jitter_factor (float, optional): relative jitter to add to diagonal for numerical stability (default: 1e-6)
    
    """
    def __init__(self, params, lr=1.0, rank=10, mu=0.01, chunk_size=1,
                 cg_tol=1e-5, cg_max_iters=1000, line_search_fn=None, verbose=False,
                 use_double=False, dynamic_damping=True, max_mu=1e2, jitter_factor=1e-6):
        defaults = dict(lr=lr, rank=rank, chunk_size=chunk_size, mu=mu, cg_tol=cg_tol,
                        cg_max_iters=cg_max_iters, line_search_fn=line_search_fn)
        self.rank = rank
        self.mu = mu
        self.initial_mu = mu  # Store initial mu for dynamic damping
        self.chunk_size = chunk_size
        self.cg_tol = cg_tol
        self.cg_max_iters = cg_max_iters
        self.line_search_fn = line_search_fn
        self.verbose = verbose
        self.use_double = use_double
        self.dynamic_damping = dynamic_damping
        self.max_mu = max_mu
        self.jitter_factor = jitter_factor
        self.U = None
        self.S = None
        self.n_iters = 0
        self.cholesky_failures = 0  # Track failures for diagnostics
        super(NysNewtonCG, self).__init__(params, defaults)

        if len(self.param_groups) > 1:
            raise ValueError(
                "NysNewtonCG doesn't currently support per-parameter options (parameter groups)")

        if self.line_search_fn is not None and self.line_search_fn != 'armijo':
            raise ValueError("NysNewtonCG only supports Armijo line search")

        self._params = self.param_groups[0]['params']
        self._params_list = list(self._params)
        self._numel_cache = None

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns (i) the loss and (ii) gradient w.r.t. the parameters.
            The closure can compute the gradient w.r.t. the parameters by calling torch.autograd.grad on the loss with create_graph=True.
        """
        if self.n_iters == 0:
            # Store the previous direction for warm starting PCG
            self.old_dir = torch.zeros(
                self._numel(), device=self._params[0].device)

        # NOTE: The closure must return both the loss and the gradient
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss, grad_tuple = closure()

        g = torch.cat([grad.reshape(-1) for grad in grad_tuple if grad is not None])
        if torch.is_complex(g):
            g = torch.real(g)

        # One step update
        for group_idx, group in enumerate(self.param_groups):
            def hvp_temp(x):
                return self._hvp(g, self._params_list, x)

            # Calculate the Newton direction
            d = _nystrom_pcg(hvp_temp, g, self.old_dir,
                             self.mu, self.U, self.S, self.rank, self.cg_tol, self.cg_max_iters)

            # Store the previous direction for warm starting PCG
            self.old_dir = d

            # Check if d is a descent direction
            if torch.dot(d, g) <= 0:
                print("Warning: d is not a descent direction")

            if self.line_search_fn == 'armijo':
                x_init = self._clone_param()

                def obj_func(x, t, dx):
                    self._add_grad(t, dx)
                    loss = float(closure()[0])
                    self._set_param(x)
                    return loss

                # Use -d for convention
                t = _armijo(obj_func, x_init, g, -d, group['lr'])
            else:
                t = group['lr']

            self.state[group_idx]['t'] = t

            # update parameters
            ls = 0
            for p in group['params']:
                np = torch.numel(p)
                dp = d[ls:ls+np].reshape(p.shape)
                ls += np
                p.data.add_(-dp, alpha=t)

        self.n_iters += 1

        return loss, g

    def update_preconditioner(self, grad_tuple):
        """Update the Nystrom approximation of the Hessian with robust error handling.

        Args:
            grad_tuple (tuple): tuple of Tensors containing the gradients of the loss w.r.t. the parameters. 
            This tuple can be obtained by calling torch.autograd.grad on the loss with create_graph=True.
        """

        # Flatten and concatenate the gradients
        gradsH = torch.cat([gradient.reshape(-1)
                           for gradient in grad_tuple if gradient is not None])
        if torch.is_complex(gradsH):
            gradsH = torch.real(gradsH)

        # Use double precision if requested
        if self.use_double and gradsH.dtype != torch.float64:
            gradsH = gradsH.double()

        # Generate test matrix (NOTE: This is transposed test matrix)
        p = gradsH.shape[0]
        dtype = torch.float64 if self.use_double else gradsH.dtype
        Phi = torch.randn((self.rank, p), device=gradsH.device, dtype=dtype) / (p ** 0.5)
        Phi = torch.linalg.qr(Phi.t(), mode='reduced')[0].t()

        # functorch vmap over rank HVPs spikes VRAM on large PINO models; sequential is slower but safer.
        use_seq = os.environ.get("NNCG_SEQ_HVP", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "",
        )
        if use_seq:
            Y_rows: list[torch.Tensor] = []
            for ri in range(self.rank):
                vi = Phi[ri]
                yrow = self._hvp(gradsH, self._params_list, vi)
                Y_rows.append(yrow)
                del yrow
            Y = torch.stack(Y_rows, dim=0)
        else:
            Y = self._hvp_vmap(gradsH, self._params_list)(Phi)
        if torch.is_complex(Y):
            Y = torch.real(Y)

        # Calculate initial shift based on machine precision
        base_shift = torch.finfo(Y.dtype).eps
        shift = base_shift
        Y_shifted = Y + shift * Phi

        # Calculate Phi^T * H * Phi (w/ shift) for Cholesky
        choleskytarget = torch.mm(Y_shifted, Phi.t())
        
        # Check symmetry (for debugging)
        if self.verbose:
            symmetry_error = torch.norm(choleskytarget - choleskytarget.T, p='fro')
            print(f'Symmetry check: Frobenius norm of choleskytarget - choleskytarget.T = {symmetry_error}')
        
        # Force symmetry to handle numerical errors
        choleskytarget = (choleskytarget + choleskytarget.T) / 2

        # Attempt Cholesky with progressive fallback strategies
        C = None
        attempt = 0
        max_attempts = 5
        
        while C is None and attempt < max_attempts:
            try:
                if attempt == 0:
                    # First attempt: standard Cholesky
                    C = torch.linalg.cholesky(choleskytarget)
                else:
                    # Subsequent attempts: add more damping
                    if self.verbose:
                        print(f"Cholesky attempt {attempt}: Adding jitter...")
                    
                    # Compute eigenvalues to diagnose the issue
                    eigs = _safe_eigvalsh(choleskytarget)
                    min_eig = eigs[0]
                    
                    if self.verbose:
                        print(f"Minimum eigenvalue: {min_eig}, Maximum eigenvalue: {eigs[-1]}")
                        print(f"Condition number estimate: {eigs[-1] / max(eigs[0], 1e-16)}")
                    
                    # Calculate adaptive jitter
                    if min_eig <= 0:
                        # Add enough to make all eigenvalues positive
                        adaptive_jitter = torch.abs(min_eig) + base_shift + self.jitter_factor * eigs[-1]
                    else:
                        # Even if positive, may need more for numerical stability
                        adaptive_jitter = self.jitter_factor * eigs[-1] * (10 ** attempt)
                    
                    if self.verbose:
                        print(f"Adding jitter: {adaptive_jitter}")
                    
                    jitter = adaptive_jitter * torch.eye(choleskytarget.shape[0], 
                                                         device=choleskytarget.device, 
                                                         dtype=choleskytarget.dtype)
                    choleskytarget_jittered = choleskytarget + jitter
                    shift = shift + adaptive_jitter  # Update total shift
                    
                    C = torch.linalg.cholesky(choleskytarget_jittered)
                    
            except torch.linalg.LinAlgError as e:
                attempt += 1
                self.cholesky_failures += 1
                
                if attempt >= max_attempts:
                    # Final fallback: use eigendecomposition with clipped eigenvalues
                    if self.verbose:
                        print(f"All Cholesky attempts failed. Using eigendecomposition fallback.")
                    
                    eigs, eigvectors = _safe_eigh_with_jitter(
                        choleskytarget,
                        float(base_shift),
                        self.verbose,
                    )
                    
                    # Clip negative eigenvalues and add margin
                    min_eig = torch.min(eigs)
                    if min_eig <= 0:
                        eig_shift = torch.abs(min_eig) + base_shift + self.jitter_factor * torch.max(eigs)
                    else:
                        eig_shift = self.jitter_factor * torch.max(eigs)
                    
                    eigs_shifted = torch.clamp(eigs + eig_shift, min=base_shift)
                    shift = shift + eig_shift
                    
                    if self.verbose:
                        print(f"Eigenvalue shift applied: {eig_shift}")
                        print(f"Shifted eigenvalues range: [{torch.min(eigs_shifted)}, {torch.max(eigs_shifted)}]")
                    
                    # Reconstruct with positive eigenvalues
                    choleskytarget_fixed = eigvectors @ torch.diag(eigs_shifted) @ eigvectors.T
                    choleskytarget_fixed = (choleskytarget_fixed + choleskytarget_fixed.T) / 2
                    try:
                        C = torch.linalg.cholesky(choleskytarget_fixed)
                    except torch.linalg.LinAlgError:
                        cf64 = choleskytarget_fixed.detach().to(dtype=torch.float64)
                        C64 = torch.linalg.cholesky(cf64 + 1e-6 * torch.eye(cf64.shape[0], device=cf64.device, dtype=torch.float64))
                        C = C64.to(dtype=choleskytarget_fixed.dtype)
                    
                    # Optionally increase mu for future iterations if dynamic damping is enabled
                    if self.dynamic_damping and self.mu < self.max_mu:
                        old_mu = self.mu
                        self.mu = min(self.mu * 10, self.max_mu)
                        if self.verbose:
                            print(f"Dynamic damping: Increased mu from {old_mu} to {self.mu}")

        if C is None:
            d = choleskytarget.shape[0]
            eye = torch.eye(
                d, device=choleskytarget.device, dtype=choleskytarget.dtype
            )
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
                    "NysNewtonCG.update_preconditioner: Cholesky failed after all fallbacks"
                )

        # Solve triangular system
        try:
            B = torch.linalg.solve_triangular(C, Y_shifted, upper=False, left=True)
        except:
            # Fallback for numerical issues (PyTorch bug workaround)
            if self.verbose:
                print("Warning: solve_triangular failed, using CPU fallback")
            B = torch.linalg.solve_triangular(C.to('cpu'), Y_shifted.to('cpu'), 
                                             upper=False, left=True).to(C.device)
            
        # B = V * S * U^T b/c we have been using transposed sketch
        _, S, UT = torch.linalg.svd(B, full_matrices=False)
        self.U = UT.t()
        self.S = torch.clamp(torch.square(S) - shift, min=0.0)

        self.rho = self.S[-1]

        if self.verbose:
            print(f'Approximate eigenvalues = {self.S}')
            if torch.any(self.S < 1e-10):
                print(f"Warning: Very small eigenvalues detected. Minimum: {torch.min(self.S)}")
            print(f"Condition number of preconditioner: {self.S[0] / max(self.S[-1], 1e-16)}")

    def _hvp_vmap(self, grad_params, params):
        return vmap(lambda v: self._hvp(grad_params, params, v), in_dims=0, chunk_size=self.chunk_size)

    def _hvp(self, grad_params, params, v):
        """Hessian-vector product using nested grads; dtype-aligned per-parameter ``grad_outputs``."""
        gcs = _split_flat_like_params(grad_params, params)
        vcs = _split_flat_like_params(v, params)
        vcs = [_match_hvp_grad_output(gc, vc) for gc, vc in zip(gcs, vcs)]
        Hv = torch.autograd.grad(
            tuple(gcs),
            tuple(params),
            grad_outputs=tuple(vcs),
            retain_graph=True,
        )
        Hv = tuple(Hvi.detach() for Hvi in Hv)
        flat = torch.cat([Hvi.reshape(-1) for Hvi in Hv])
        if torch.is_complex(flat):
            flat = torch.real(flat)
        return flat

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # Avoid in-place operation by creating a new tensor
            p.data = p.data.add(
                update[offset:offset + numel].reshape_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            # Replace the .data attribute of the tensor
            p.data = pdata.data

    def get_diagnostics(self):
        """Get diagnostic information about the optimizer state.
        
        Returns:
            dict: Dictionary containing diagnostic information
        """
        diagnostics = {
            'current_mu': self.mu,
            'initial_mu': self.initial_mu,
            'cholesky_failures': self.cholesky_failures,
            'n_iters': self.n_iters,
        }
        
        if self.S is not None:
            diagnostics['eigenvalues'] = self.S.detach().cpu().numpy().tolist()
            diagnostics['condition_number'] = (self.S[0] / max(self.S[-1], 1e-16)).item()
            diagnostics['min_eigenvalue'] = self.S[-1].item()
            diagnostics['max_eigenvalue'] = self.S[0].item()
        
        return diagnostics
    
    def reset_damping(self):
        """Reset damping parameter to initial value."""
        self.mu = self.initial_mu
        if self.verbose:
            print(f"Reset mu to initial value: {self.mu}")