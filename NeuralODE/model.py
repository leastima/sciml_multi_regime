"""
NeuralODE — ShallowODE model and seed utility.

ShallowODE wraps a 3-layer MLP as a continuous-time ODE RHS ``f(t, x)``
for use with ``torchdiffeq.odeint``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Architecture
# ──────────────────────────────────────────────────────────────────────────────

def shallow(
    in_dim: int,
    hidden: int,
    out_dim: int,
    Act=torch.nn.Tanh,
) -> torch.nn.Sequential:
    """3-hidden-layer MLP with activation ``Act`` (default Tanh)."""
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
    """NODE dynamics: ``dx/dt = net(x)`` (time-homogeneous shallow MLP).

    Args:
        in_dim : state dimension (2 for state repr, 3 for embedded repr)
        out_dim: same as in_dim
        hidden : width of each hidden layer (default 10)
        Act    : activation function class (default ``Tanh``)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int = 10,
        Act=torch.nn.Tanh,
    ):
        super().__init__()
        self.net = shallow(in_dim, hidden, out_dim, Act=Act)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
