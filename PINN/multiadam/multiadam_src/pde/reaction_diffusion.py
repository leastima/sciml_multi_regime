import deepxde as dde
import numpy as np
import torch
import scipy.interpolate
import os  # Added for file existence check

from . import baseclass

class ReactionDiffusion1D(baseclass.BaseTimePDE):
    def __init__(self, alpha=5.0, zeta=None, tau=4.0, n_colloc=100, domain_length=2*np.pi):
        super().__init__()
        self.alpha = alpha  # Growth rate
        self.tau = tau      # Spreading speed
        
        # Default zeta (initial sharpness): 1/(2 * (pi/4)^2)
        if zeta is None:
            self.zeta = 1.0 / (2.0 * (np.pi / 4.0) ** 2)
        else:
            self.zeta = zeta
            
        self.n_colloc = n_colloc
        
        # Set output dimension
        self.output_dim = 1
        
        # Domain setup - using 2π for periodic-like behavior
        self.geom = dde.geometry.Interval(0, domain_length)
        timedomain = dde.geometry.TimeDomain(0, 1)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)
        self.bbox = [0, domain_length, 0, 1]
        
        # Reaction-Diffusion PDE: u_t = tau * u_xx + alpha * u * (1 - u)
        def reaction_diffusion_pde(x, u):
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            u_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            
            # Reaction-diffusion equation: u_t - tau*u_xx - alpha*u*(1-u) = 0
            return u_t - tau * u_xx - self.alpha * u * (1 - u)

        self.pde = reaction_diffusion_pde
        self.set_pdeloss()
        
        # Gaussian initial condition: u(x, 0) = exp(-zeta * (x - pi)^2)
        def gaussian_ic(x):
            return np.exp(-self.zeta * (x[:, 0:1] - np.pi) ** 2)
        
        # Define a function to check if a point is on the periodic boundary
        def boundary_periodic(x, on_boundary):
            return on_boundary and np.isclose(x[0], 0)

        self.add_bcs([{
            'component': 0,
            'function': gaussian_ic,
            'bc': lambda _, on_initial: on_initial,
            'type': 'ic'
        }, {
            'component': 0,
            'component_x': 0,  # The x-axis (axis 0) is periodic
            'bc': boundary_periodic,
            'type': 'periodic'
        }])
        
        # Training points
        self.training_points(domain=n_colloc)
        
        # Dynamically set reference data path based on alpha and tau (using float formatting to match .0)
        ref_datapath = f'ref/rd_alpha{self.alpha:.1f}_tau{self.tau:.1f}.dat'
        if os.path.exists(ref_datapath):
            self.load_ref_data(ref_datapath, t_transpose=False)  # Set to False for standard [x, t, u] format (no COMSOL parsing)
            self._init_interpolator()
        else:
            print(f"Warning: Reference data file '{ref_datapath}' not found for alpha={self.alpha}, tau={self.tau}. Falling back to placeholder solution (inaccurate L2RE). Generate the file using generate_rd_ref.py.")

    def _init_interpolator(self):
        if self.ref_data is not None:
            points = self.ref_data[:, :2]  # [x, t]
            values = self.ref_data[:, 2]  # [u], flatten to 1D for interpolator
            self.interpolator = scipy.interpolate.LinearNDInterpolator(points, values, fill_value=0.0)
            self.interpolator_nearest = scipy.interpolate.NearestNDInterpolator(points, values)

    def solution(self, x):
        if hasattr(self, 'interpolator'):
            u = self.interpolator(x)
            nan_mask = np.isnan(u)
            if np.any(nan_mask):
                u[nan_mask] = self.interpolator_nearest(x[nan_mask])
            return u[:, np.newaxis]  # Ensure (N,1) shape
        else:
            # Placeholder approximation (not accurate; use for testing only)
            return np.exp(-self.zeta * (x[:, 0:1] - np.pi) ** 2)  # Initial condition, (N,1)

    def gen_testdata(self):
        nx, nt = 256, 101
        x_values = np.linspace(self.bbox[0], self.bbox[1], nx)
        t_values = np.linspace(self.bbox[2], self.bbox[3], nt)
        X, T = np.meshgrid(x_values, t_values)
        test_x = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        return test_x

class Reaction1D(baseclass.BaseTimePDE):
    def __init__(self, alpha=5.0, zeta=None, n_colloc=100, domain_length=2*np.pi):
        super().__init__()
        self.alpha = alpha  # Growth rate
        
        # Default zeta (initial sharpness): 1/(2 * (pi/4)^2)
        if zeta is None:
            self.zeta = 1.0 / (2.0 * (np.pi / 4.0) ** 2)
        else:
            self.zeta = zeta
            
        self.n_colloc = n_colloc
        
        # Set output dimension
        self.output_dim = 1
        
        # Domain setup
        self.geom = dde.geometry.Interval(0, domain_length)
        timedomain = dde.geometry.TimeDomain(0, 1)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)
        self.bbox = [0, domain_length, 0, 1]
        
        # Pure Reaction PDE: u_t = alpha * u * (1 - u)
        def reaction_pde(x, u):
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            
            # Pure reaction equation: u_t - alpha*u*(1-u) = 0
            return u_t - self.alpha * u * (1 - u)
        
        self.pde = reaction_pde
        self.set_pdeloss()
        
        # Gaussian initial condition: u(x, 0) = exp(-zeta * (x - pi)^2)
        def gaussian_ic(x):
            return np.exp(-self.zeta * (x[:, 0:1] - np.pi) ** 2)
        
        self.add_bcs([{
            'component': 0,
            'function': gaussian_ic,
            'bc': lambda _, on_initial: on_initial,
            'type': 'ic'
        }])
        
        # Training points
        self.training_points(domain=n_colloc)

    def solution(self, x):
        u0 = np.exp(-self.zeta * (x[:, 0:1] - np.pi) ** 2)
        exp_term = np.exp(-self.alpha * x[:, 1:2])
        return u0 / (u0 + (1 - u0) * exp_term)

    def gen_testdata(self):
        nx, nt = 256, 101
        x_values = np.linspace(self.bbox[0], self.bbox[1], nx)
        t_values = np.linspace(self.bbox[2], self.bbox[3], nt)
        X, T = np.meshgrid(x_values, t_values)
        test_x = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        return test_x

class Diffusion1D(baseclass.BaseTimePDE):
    def __init__(self, tau=4.0, zeta=None, n_colloc=100, domain_length=2*np.pi):
        super().__init__()
        self.tau = tau      # Spreading speed
        
        # Default zeta (initial sharpness): 1/(2 * (pi/4)^2)
        if zeta is None:
            self.zeta = 1.0 / (2.0 * (np.pi / 4.0) ** 2)
        else:
            self.zeta = zeta
            
        self.n_colloc = n_colloc
        
        # Set output dimension
        self.output_dim = 1
        
        # Domain setup
        self.geom = dde.geometry.Interval(0, domain_length)
        timedomain = dde.geometry.TimeDomain(0, 1)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)
        self.bbox = [0, domain_length, 0, 1]
        
        # Pure Diffusion PDE: u_t = tau * u_xx
        def diffusion_pde(x, u):
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            
            # Pure diffusion equation: u_t - tau*u_xx = 0
            return u_t - self.tau * u_xx
        
        self.pde = diffusion_pde
        self.set_pdeloss()
        
        # Gaussian initial condition: u(x, 0) = exp(-zeta * (x - pi)^2)
        def gaussian_ic(x):
            return np.exp(-self.zeta * (x[:, 0:1] - np.pi) ** 2)
        
        # Define a function to check if a point is on the periodic boundary
        def boundary_periodic(x, on_boundary):
            return on_boundary and np.isclose(x[0], 0)

        self.add_bcs([{
            'component': 0,
            'function': gaussian_ic,
            'bc': lambda _, on_initial: on_initial,
            'type': 'ic'
        }, {
            'component': 0,
            'component_x': 0,  # The x-axis (axis 0) is periodic
            'bc': boundary_periodic,
            'type': 'periodic'
        }])
        
        # Training points
        self.training_points(domain=n_colloc)

    def solution(self, x):
        L = self.bbox[1] - self.bbox[0]
        u = np.zeros((len(x), 1))  # Updated shape to (N, 1) for consistency
        k_max = 20  # Number of images for convergence
        for k in range(-k_max, k_max + 1):
            denom = 4 * self.tau * x[:, 1:2] * self.zeta + 1
            exp_term = np.exp(-self.zeta * (x[:, 0:1] - np.pi - k * L) ** 2 / denom)
            prefactor = np.sqrt(1 / denom)
            u += prefactor * exp_term
        return u

    def gen_testdata(self):
        nx, nt = 256, 101
        x_values = np.linspace(self.bbox[0], self.bbox[1], nx)
        t_values = np.linspace(self.bbox[2], self.bbox[3], nt)
        X, T = np.meshgrid(x_values, t_values)
        test_x = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        return test_x
