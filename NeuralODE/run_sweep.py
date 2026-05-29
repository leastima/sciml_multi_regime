#!/usr/bin/env python3
"""
NeuralODE sweep — entry point.

Delegates to the full implementation in ``run_sweep_horizon_physics_cl.py``.
See README.md for usage examples and argument documentation.

Usage:
    python run_sweep.py --help
    python run_sweep.py --optimizer LBFGS --physics-mode pinn_alm \\
        --inv-b 8 --horizon 20 --seed 0 --epochs 1 --alm-outer-iters 20
"""

import os
import sys

# Ensure this directory is on the path so relative imports resolve.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_sweep_horizon_physics import main  # noqa: E402

if __name__ == "__main__":
    main()
