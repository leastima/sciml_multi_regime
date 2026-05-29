import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import math

from matplotlib.colors import LightSource
from matplotlib.colors import Normalize


def orthonormalize(v, basis_list):
    v_proj = sum([(v*b).sum()/(b*b).sum() * b for b in basis_list])
    v = v - v_proj
    return v

def log_formatter(z, pos):
    return r"$10^{%.1f}$" % (z / np.log(10))

def update_bn_stats(model, data, train_loader, device, full_dataset=True):
    """
    重新校准 BatchNorm 统计量（论文级正确版本）

    参数：
    - data: 单 batch 数据（当 full_dataset=False 时使用）
    - train_loader: DataLoader（当 full_dataset=True 时使用）
    - full_dataset: 是否用整个数据集估计 BN（强烈建议 True）
    """

    # -------------------------------
    # 1. 检查是否存在 BN
    # -------------------------------
    bn_layers = [m for m in model.modules()
                 if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]

    if len(bn_layers) == 0:
        model.eval()
        return

    model.train()

    # -------------------------------
    # 2. reset running stats（关键！）
    # -------------------------------
    for m in bn_layers:
        m.running_mean.zero_()
        m.running_var.fill_(1)
        m.num_batches_tracked.zero_()

    # -------------------------------
    # 3. 保存并修改 momentum（关键！）
    # -------------------------------
    momenta = {}
    for m in bn_layers:
        momenta[m] = m.momentum
        m.momentum = None   # cumulative moving average

    # -------------------------------
    # 4. 前向传播更新统计量
    # -------------------------------
    with torch.no_grad():
        if full_dataset:
            assert train_loader is not None, "full_dataset=True 时必须提供 train_loader"
            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(device)
                model(x)
        else:
            if isinstance(data, (list, tuple)):
                x = data[0]
            else:
                x = data
            x = x.to(device)
            model(x)

    # -------------------------------
    # 5. 恢复 momentum
    # -------------------------------
    for m in bn_layers:
        m.momentum = momenta[m]

    model.eval()

def plot_loss_landscape_1d(model, loss_func, inputs, labels, d1,
                           alpha_range=(-0.5, 0.5), points=101,
                           log_scale=True, device='cuda', save_path=None):
    model.eval()
    device = torch.device(device)
    model.to(device)
    inputs, labels = inputs.to(device), labels.to(device)

    d1 = [v.to(device) for v in d1]

    base_params = [p.detach().clone() for p in model.parameters()]

    alphas = np.linspace(alpha_range[0], alpha_range[1], points)
    losses = []

    for a in alphas:
        for p, base, v in zip(model.parameters(), base_params, d1):
            p.data = base.clone() + a * v

        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_func(outputs, labels).item()

        losses.append(np.log(loss + 1e-12) if log_scale else loss)

    plt.figure(figsize=(6, 4))
    plt.plot(alphas, losses)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\log \mathcal{L}$' if log_scale else 'loss')
    plt.title('1D Loss Landscape')

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, '1D_loss_landscape.png'),
                    dpi=300, bbox_inches='tight')
    plt.show()

    for p, base in zip(model.parameters(), base_params):
        p.data = base.clone()


def plot_loss_landscape_2d(model, loss_func, inputs, labels, d1, d2,
                           alpha_range=(-0.5, 0.5), beta_range=(-0.5, 0.5),
                           grid=41, log_scale=True, device='cuda', save_path=None):
    model.eval()
    device = torch.device(device)
    model.to(device)
    inputs, labels = inputs.to(device), labels.to(device)

    d1 = [v.to(device) for v in d1]
    d2 = [v.to(device) for v in d2]

    base_params = [p.detach().clone() for p in model.parameters()]

    alphas = np.linspace(alpha_range[0], alpha_range[1], grid)
    betas = np.linspace(beta_range[0], beta_range[1], grid)
    Z = np.zeros((grid, grid))

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            for p, base, v1, v2 in zip(model.parameters(), base_params, d1, d2):
                p.data = base.clone() + a * v1 + b * v2

            with torch.no_grad():
                outputs = model(inputs)
                loss = loss_func(outputs, labels).item()

            loss_val = max(loss, 1e-12)
            Z[j, i] = np.log(loss_val) if log_scale else loss_val

    Z = np.nan_to_num(Z, nan=np.min(Z))

    A, B = np.meshgrid(alphas, betas)
    plt.figure(figsize=(6, 5))
    cp = plt.contourf(A, B, Z, levels=50, cmap='viridis', vmin=Z.min(), vmax=Z.max())
    plt.colorbar(cp)
    plt.xlabel(r'$\varepsilon_1$')
    plt.ylabel(r'$\varepsilon_2$')
    plt.title('2D Loss Landscape')

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, '2D_loss_landscape.png'),
                    dpi=300, bbox_inches='tight')

    plt.show()

    for p, base in zip(model.parameters(), base_params):
        p.data = base.clone()


def plot_loss_landscape_3d(model, loss_func, inputs, labels, d1, d2,
                           alpha_range=(-0.5, 0.5), beta_range=(-0.5, 0.5),
                           grid=401, log_scale=True, device='cuda',
                           elev=30, azim=135, save_path=None, train_loader=None):
    """
    Plot the loss landscape with:
    - Direction orthogonalization
    - Filter-wise normalization
    - Color normalization
    - Contour projection
    """

    # -------------------------------
    #  Preparation
    # -------------------------------
    model.eval()
    device = torch.device(device)
    model.to(device)
    inputs, labels = inputs.to(device), labels.to(device)

    # Copy directions to device
    d1 = [v.to(device) for v in d1]
    d2 = [v.to(device) for v in d2]

    # Save base parameters
    base_params = [p.detach().clone() for p in model.parameters()]

    # ===============================================================
    # 2) Filter-wise Normalization (Garipov et al.)
    # ===============================================================
    d2 = [orthonormalize(v2, [v1]) for v1, v2 in zip(d1, d2)]

    for i, p in enumerate(base_params):
        if p.ndim > 1:  # weights
            scale = p.norm() + 1e-10
            d1[i] = d1[i] / (d1[i].norm() + 1e-10) * scale
            d2[i] = d2[i] / (d2[i].norm() + 1e-10) * scale
        else:  # bias / batchnorm
            if d1[i].norm() > 0:
                d1[i] = d1[i] / (d1[i].norm() + 1e-12) * (p.norm() + 1e-12)
            if d2[i].norm() > 0:
                d2[i] = d2[i] / (d2[i].norm() + 1e-12) * (p.norm() + 1e-12)

    # -------------------------------
    #  alpha-beta Grid Evaluation
    # -------------------------------
    alphas = np.linspace(alpha_range[0], alpha_range[1], grid)
    betas = np.linspace(beta_range[0], beta_range[1], grid)
    Z = np.zeros((grid, grid))

    now = 0
    data = (inputs, labels)

    final_loss = loss_func(model(inputs), labels).item()
    print(loss)

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            for p, base, v1, v2 in zip(model.parameters(), base_params, d1, d2):
                p.data = base + a * v1 + b * v2

            update_bn_stats(model, data, train_loader, device, full_dataset=False)

            with torch.no_grad():
                loss = loss_func(model(inputs), labels).item()

            loss_val = max(loss, 1e-12)
            Z[j, i] = np.log10(loss_val) if log_scale else loss_val

            if now % 1000 == 0:
                print('finish :', now)
            now = now + 1

            if abs(a) < 1e-12 and abs(b) < 1e-12:
                Z[i, j] = np.log10(final_loss) if log_scale else final_loss
                print(Z[i, j])

    # Restore params
    for p, base in zip(model.parameters(), base_params):
        p.data.copy_(base)


    # -------------------------------
    # 3) Color Normalization
    # -------------------------------
    norm = Normalize(vmin=np.percentile(Z, 5),
                     vmax=np.percentile(Z, 95))

    A, B = np.meshgrid(alphas, betas)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Surface
    surf = ax.plot_surface(
        A, B, Z,
        cmap='RdYlBu_r', norm=norm,
        linewidth=0, antialiased=True, shade=False
    )

    from matplotlib.ticker import MaxNLocator

    # 在 ax.view_init 之后添加
    # 限制 X, Y, Z 轴最多只显示 5 个刻度
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax.set_xlabel(r'$\varepsilon_1$')
    ax.set_ylabel(r'$\varepsilon_2$')
    ax.view_init(elev=elev, azim=azim)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        print('plot saved to', os.path.join(save_path, '3D_loss_landscape.pdf'))
        plt.savefig(os.path.join(save_path, '3D_loss_landscape_random.pdf'),
                    dpi=300, bbox_inches='tight', transparent=True)

    plt.show()