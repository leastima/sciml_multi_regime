import os
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt

from src.models import PINN
from src.train_utils import *
from src.pyhessian import hessian
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib.colors import Normalize

def set_requires_grad(x_tuple, t_tuple):
    x_tuple = tuple(v.clone().detach().requires_grad_(True) for v in x_tuple)
    t_tuple = tuple(v.clone().detach().requires_grad_(True) for v in t_tuple)
    return x_tuple, t_tuple

def make_loss_closure(model, x, t, predict, loss_func):
    def closure():
        preds = predict(x, t, model)
        loss_res, loss_bc, loss_ic = loss_func(x, t, preds)
        return loss_res + loss_bc + loss_ic
    return closure

def normalize_direction(direction, params):
    """Filter-wise normalization (Li et al., 2018)"""
    normed = []
    for d, p in zip(direction, params):
        if p.ndim <= 1:
            scale = torch.norm(p)
            norm = torch.norm(d)
            normed.append(d * (scale / (norm + 1e-10)))
        else:
            shape = [p.shape[0]] + [1]*(p.ndim-1)
            p_norm = torch.norm(p.view(p.shape[0], -1), dim=1).view(shape)
            d_norm = torch.norm(d.view(d.shape[0], -1), dim=1).view(shape)
            normed.append(d * (p_norm / (d_norm + 1e-10)))
    return normed

def orthogonalize(d1, d2):
    dot = sum(torch.sum(a*b) for a,b in zip(d1,d2))
    norm = sum(torch.sum(a*a) for a in d1)
    coeff = dot / (norm + 1e-10)
    return [b - coeff*a for a,b in zip(d1,d2)]

def set_params(model, base_params, d1=None, d2=None, alpha=0.0, beta=0.0):
    for idx, p in enumerate(model.parameters()):
        p.data = base_params[idx].clone()
        if d1 is not None:
            p.data += alpha * d1[idx]
        if d2 is not None:
            p.data += beta * d2[idx]

def loss_landscape_1d(model, predict, loss_func, x, t,
                      radius=0.5, points=101, log_scale=True,
                      device="cpu", save_path=None):
    x, t = set_requires_grad(x, t)
    model.to(device)
    params = [p.detach().clone() for p in model.parameters()]

    hess = hessian(model, predict, loss_func, data=(x, t), device=device)
    eigvals, eigvecs, _ = hess.eigenvalues(max_num_iter=100, top_n=1)
    d = normalize_direction(eigvecs[0], params)

    loss_closure = make_loss_closure(model, x, t, predict, loss_func)
    alphas = np.linspace(-radius, radius, points)
    Z = np.zeros(points)

    for i, a in enumerate(alphas):
        set_params(model, params, d1=d, alpha=a)
        l = loss_closure().item()
        Z[i] = np.log(l + 1e-12) if log_scale else l

    set_params(model, params, alpha=0.0)

    plt.figure(figsize=(6,4))
    plt.plot(alphas, Z)
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\log L$" if log_scale else "loss")
    plt.title("1D Loss Landscape (Li et al. 2018)")
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def loss_landscape_2d(model, predict, loss_func, x, t,
                      radius=0.5, grid=41, log_scale=True,
                      device="cpu", save_path=None):
    x, t = set_requires_grad(x, t)
    model.to(device)
    params = [p.detach().clone() for p in model.parameters()]

    hess = hessian(model, predict, loss_func, data=(x, t), device=device)
    eigvals, eigvecs, _ = hess.eigenvalues(max_num_iter=100, top_n=2)
    d1 = normalize_direction(eigvecs[0], params)
    d2 = normalize_direction(eigvecs[1], params)
    d2 = orthogonalize(d1, d2)

    loss_closure = make_loss_closure(model, x, t, predict, loss_func)
    alphas = np.linspace(-radius, radius, grid)
    betas = np.linspace(-radius, radius, grid)
    Z = np.zeros((grid, grid))

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            set_params(model, params, d1=d1, d2=d2, alpha=a, beta=b)
            l = loss_closure().item()
            Z[j, i] = np.log(l + 1e-12) if log_scale else l

    set_params(model, params, alpha=0.0, beta=0.0)

    A, B = np.meshgrid(alphas, betas)
    plt.figure(figsize=(6,5))
    cp = plt.contourf(A, B, Z, levels=50, cmap="plasma")
    plt.colorbar(cp)
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\beta$")
    plt.title("2D Loss Landscape (Hessian top-2 directions)")
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def loss_landscape_3d(
        model, predict, loss_func, x, t,
        radius=0.5, grid=401, log_scale=True,
        device="cpu", save_path=None,
        elev=30, azim=135, cmap="RdYlBu_r"):
    """
    Visualize the loss landscape in 3D (smooth Figure 1 style).
    - Uses top-2 Hessian eigenvectors as directions.
    - Applies filter-wise normalization for each parameter tensor.
    - Plots a smooth 3D surface with contour projection on the bottom plane.
    """

    model.to(device)

    # Move inputs to device
    if isinstance(x, tuple):
        x = tuple(xi.to(device) for xi in x)
    else:
        x = x.to(device)
    if isinstance(t, tuple):
        t = tuple(ti.to(device) for ti in t)
    else:
        t = t.to(device)

    # Enable gradient tracking
    x, t = set_requires_grad(x, t)

    # Save original model parameters
    orig_params = [p.detach().clone() for p in model.parameters()]

    # Compute Hessian top-2 eigenvectors
    hess = hessian(model, predict, loss_func, data=(x, t), device=device)
    eigvals, eigvecs, _ = hess.eigenvalues(max_num_iter=100, top_n=2)
    d1, d2 = eigvecs[0], eigvecs[1]

    # Make d2 orthogonal to d1
    d2 = orthonormalization(d2, [d1])

    # ---- Filter-wise normalization ----
    for i, p in enumerate(orig_params):
        if p.ndim > 1:  # weight matrices / conv kernels
            d1[i] = d1[i] / (d1[i].norm() + 1e-12) * p.norm()
            d2[i] = d2[i] / (d2[i].norm() + 1e-12) * p.norm()
        else:  # bias or batchnorm parameters
            if d1[i].norm() > 0:
                d1[i] = d1[i] / d1[i].norm() * (p.norm() + 1e-12)
            if d2[i].norm() > 0:
                d2[i] = d2[i] / d2[i].norm() * (p.norm() + 1e-12)

    # Alpha-beta grid
    alphas = np.linspace(-radius, radius, grid)
    betas  = np.linspace(-radius, radius, grid)
    Z = np.zeros((grid, grid))

    # Loss closure for evaluation
    loss_closure = make_loss_closure(model, x, t, predict, loss_func)

    # Scan the 2D grid in direction space
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            for p, dp1, dp2, orig in zip(model.parameters(), d1, d2, orig_params):
                p.data.copy_(orig + a*dp1 + b*dp2)
            l = loss_closure().item()
            Z[j, i] = np.log(l + 1e-12) if log_scale else l

    # Restore original model parameters
    for p, orig in zip(model.parameters(), orig_params):
        p.data.copy_(orig)

    # Meshgrid for plotting
    A, B = np.meshgrid(alphas, betas)

    # ---- Normalize colors to avoid sudden jumps ----
    norm = Normalize(vmin=np.percentile(Z, 5), vmax=np.percentile(Z, 95))

    # ---- Plot (smooth 3D version) ----
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Main smooth surface
    surf = ax.plot_surface(
        A, B, Z,
        rstride=1, cstride=1,
        cmap=cmap, norm=norm,
        linewidth=0, antialiased=True,
        shade=False, alpha=0.95
    )

    # # Contour projection on the bottom plane
    # ax.contourf(A, B, Z, zdir='z', offset=Z.min(),
    #             cmap=cmap, norm=norm, levels=100, alpha=0.8)
    # ax.contour(A, B, Z, zdir='z', offset=Z.min(),
    #            colors='black', levels=50, linewidths=0.3)

    # Camera view
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    # Save or show the figure
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help='initial seed')
    parser.add_argument('--type', type=str, default='1D')
    parser.add_argument('--pde', type=str,
                        default='convection', help='PDE type')
    parser.add_argument('--pde_params', nargs='+', type=str,
                        default=None, help='PDE coefficients')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='number of layers of the neural net')
    parser.add_argument('--num_neurons', type=int, default=50,
                        help='number of neurons per layer')
    parser.add_argument('--loss', type=str, default='mse',
                        help='type of loss function')
    parser.add_argument('--num_res', type=int, default=10000,
                        help='number of sampled residual points')
    parser.add_argument('--set_idx', type=int, default=0, help='the index of dataset')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.pde == 'convection':
        model_path = os.path.join(
        './saved_models', 
        f'system_{args.pde}', 
        f'N_f_{args.num_res}', 
        f'beta_{args.pde_params[1]}',
        f'set_{args.set_idx}',
        f'seed_{args.seed}.pt'
        )
    model = PINN(in_dim=2, hidden_dim=args.num_neurons, out_dim=1, num_layer=args.num_layers).to(device)
    model.load_state_dict(torch.load(model_path))
    if args.pde == 'convection':
        training_set = os.path.join(
        './dataset',
        f'system_{args.pde}',
        f'N_f_{args.num_res}',
        f'beta_{float(args.pde_params[1])}',
        f'train_{args.set_idx}.pt'
    )

    x_range, t_range, loss_func, pde_coefs = get_pde(args.pde, args.pde_params, args.loss)

    data = torch.load(training_set, map_location=device)
    x = (data['x_res'], data['x_left'], data['x_upper'], data['x_lower'])
    t = (data['t_res'], data['t_left'], data['t_upper'], data['t_lower'])
    data_params = data['data_params']
    if args.pde == 'convection':
        save_path = os.path.join(
            './loss_landscape', 
            f'system_{args.pde}', 
            f'N_f_{args.num_res}',
            f'beta_{args.pde_params[-1]}',
            f'set_{args.set_idx}',
            f'{args.type}_loss_landscape.png'
        )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if args.type == '1D':
        loss_landscape_1d(
            model=model,
            predict=predict,
            loss_func=loss_func,
            x=x,
            t=t,
            radius=0.5,      
            points=101,      
            log_scale=True,  
            device=device,    
            save_path=save_path   
        )
    elif args.type == '2D':
        loss_landscape_2d(
            model=model,
            predict=predict,
            loss_func=loss_func,
            x=x,
            t=t,
            radius=0.5,      
            grid=41,      
            log_scale=True,  
            device=device,    
            save_path=save_path   
        )
    elif args.type == '3D':
        loss_landscape_3d(
            model=model,
            predict=predict,
            loss_func=loss_func,
            x=x,
            t=t,
            radius=0.5,      
            grid=401,      
            log_scale=True,  
            device=device,    
            save_path=save_path   
        )

if __name__ == "__main__":
    main()