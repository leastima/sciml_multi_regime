"""
Darcy PINO optimizer modules: ALM, NNCG, CL.

Each module exposes a single ``run_*`` function with explicit parameter
signatures (no ``inspect.currentframe()`` frame injection).
"""

from .alm import run_alm
from .nncg import run_nncg
from .cl import run_cl

__all__ = ["run_alm", "run_nncg", "run_cl"]
