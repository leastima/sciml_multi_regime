"""
NeuralODE — pendulum simulation and data-loading utilities.

Provides:
  - ``pendulum``        : ODE right-hand side for the nonlinear pendulum
  - ``spherical_pendulum`` : embedded (S²) representation
  - ``build_data``      : integrate ODE and return train/test arrays
  - ``build_loader``    : wrap arrays in a DataLoader
  - ``numpy_to_torch``  : safe numpy → torch conversion
"""

from __future__ import annotations

import numpy as np
import torch
import torch.utils.data
from scipy.integrate import solve_ivp


# ──────────────────────────────────────────────────────────────────────────────
# ODE dynamics
# ──────────────────────────────────────────────────────────────────────────────

def pendulum(t: float, x, b: float = 0.0, k: float = 0.0, drag: float = 0.0):
    """Nonlinear pendulum: θ̈ = -b θ̇ - sin θ + drag·cos(k t)."""
    theta, velocity = x
    theta_dot    = velocity
    velocity_dot = -b * velocity - np.sin(theta) + drag * np.cos(k * t)
    return theta_dot, velocity_dot


def spherical_pendulum(v: np.ndarray) -> np.ndarray:
    """Map (θ, φ) → (x, y, z) on S² for embedded representation."""
    x = np.sin(v[0]) * np.cos(v[1])
    y = np.sin(v[0]) * np.sin(v[1])
    z = -np.cos(v[0])
    return np.vstack((x, y, z))


# ──────────────────────────────────────────────────────────────────────────────
# Data construction
# ──────────────────────────────────────────────────────────────────────────────

def build_data(
    b: float,
    dt: float,
    dt_test: float,
    tmax: float,
    train_theta: float,
    test_theta: float,
    representation: str = "embedded",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the pendulum ODE and return train/test trajectory arrays.

    Args:
        b           : damping coefficient
        dt          : training time-step
        dt_test     : test time-step
        tmax        : total integration time
        train_theta : initial angle for training trajectory
        test_theta  : initial angle for test trajectory
        representation: ``"embedded"`` (S²) or ``"state"`` (θ, ω)

    Returns:
        ``(xtrain, xtest, dt_train, dt_test)`` as numpy arrays.
        ``xtrain`` shape: ``(d, T_train)`` where d=3 for embedded, d=2 for state.
    """
    grid      = np.arange(0, tmax, dt)
    grid_test = np.arange(0, tmax, dt_test)

    sol_train = solve_ivp(
        pendulum, (0, tmax), y0=np.array([train_theta, 0.0]),
        args=(b, 0.0), t_eval=grid,
    )
    sol_test = solve_ivp(
        pendulum, (0, tmax), y0=np.array([test_theta, 0.0]),
        args=(b, 0.0), t_eval=grid_test,
    )

    dt_train_ = (sol_train.t[1:] - sol_train.t[:-1]).reshape(-1, 1)
    dt_test_  = (sol_test.t[1:]  - sol_test.t[:-1]).reshape(-1, 1)

    if representation == "embedded":
        xtrain = spherical_pendulum(sol_train.y)
        xtest  = spherical_pendulum(sol_test.y)
    elif representation == "state":
        xtrain = sol_train.y
        xtest  = sol_test.y
    else:
        raise ValueError(f"Unsupported representation: {representation!r}")

    return xtrain, xtest, dt_train_, dt_test_


def build_loader(
    xtrain: np.ndarray,
    dt_train: np.ndarray,
    batch_size: int,
    device=None,
) -> torch.utils.data.DataLoader:
    """Wrap training trajectory into a DataLoader with ``(inputs, targets, dt, idx)``."""
    dev = torch.device(device) if device is not None else torch.device("cpu")
    inputs  = numpy_to_torch(xtrain[:, 0:-1].T, dtype=torch.float32, device=dev)
    targets = numpy_to_torch(xtrain[:, 1:].T,   dtype=torch.float32, device=dev)
    idx     = numpy_to_torch(np.arange(inputs.shape[0]), dtype=torch.long, device=dev)
    dts     = numpy_to_torch(dt_train, dtype=torch.float32, device=dev)
    ds      = torch.utils.data.TensorDataset(inputs, targets, dts, idx)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def numpy_to_torch(
    array: np.ndarray,
    dtype=None,
    device=None,
) -> torch.Tensor:
    """Convert a numpy array to a torch Tensor (handles CUDA unavailability)."""
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
