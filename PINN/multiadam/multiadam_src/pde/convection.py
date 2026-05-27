import deepxde as dde
import numpy as np
import torch

from . import baseclass

class Convection1D(baseclass.BaseTimePDE):
    def __init__(self, beta=1.0, n_colloc=100, domain_length=1.0):
        super().__init__()
        self.beta = beta
        self.n_colloc = n_colloc
        
        # Set output dimension
        self.output_dim = 1
        
        # Domain setup
        self.geom = dde.geometry.Interval(0, domain_length)
        timedomain = dde.geometry.TimeDomain(0, 1)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)
        self.bbox = [0, domain_length, 0, 1]
        
        # Simple PDE definition
        def convection_pde(x, u):
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            u_x = dde.grad.jacobian(u, x, i=0, j=0)
            return u_t + self.beta * u_x
        
        self.pde = convection_pde
        self.set_pdeloss()
        
        # Define a function to check if a point is on the periodic boundary
        def boundary_periodic(x, on_boundary):
            return on_boundary and np.isclose(x[0], 0)

        # Add Initial Condition AND Periodic Boundary Condition
        self.add_bcs([{
            'component': 0,
            'function': lambda x: np.sin(2 * np.pi * x[:, 0:1] / domain_length), 
            'bc': lambda _, on_initial: on_initial,
            'type': 'ic'
        }, {
            'component': 0,
            'component_x': 0,  
            'bc': boundary_periodic, 
            'type': 'periodic'
        }])
        
        # Training points
        self.training_points(domain=n_colloc)
        
        # Add to data creation (assuming baseclass has a create_data or similar; add solution here if needed)
        # If baseclass sets self.data = dde.data.PDE(...), update it to include solution=self.solution

    def solution(self, x):
        """Exact solution (numpy array)."""
        L = self.bbox[1] - self.bbox[0]
        return np.sin(2 * np.pi * (x[:, 0:1] - self.beta * x[:, 1:2]) / L)

    def gen_testdata(self):
        """Generate test points (x, t) as numpy array."""
        nx, nt = 256, 101  # Dense grid for accurate L2RE
        x_values = np.linspace(self.bbox[0], self.bbox[1], nx)
        t_values = np.linspace(self.bbox[2], self.bbox[3], nt)
        X, T = np.meshgrid(x_values, t_values)
        test_x = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        return test_x