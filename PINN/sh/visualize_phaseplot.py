import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
import os
import json
import argparse

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import StrMethodFormatter
from scipy.stats import pearsonr, spearmanr


def get_pde_folder(pde, hc, set_idx, Nf, beta):
    if pde == 'convection':
        ckpt_folder = f'output/system_' + pde + f'/N_f_{Nf}/beta_{float(beta)}/set_{set_idx}/{hc}'
    elif pde == 'reaction':
        ckpt_folder = f'output/system_' + pde + f'/N_f_{Nf}/rho_{float(beta)}/set_{set_idx}/{hc}'
    elif pde == 'reaction_diffusion':
        ckpt_folder = f'output/system_' + pde + f'/N_f_{Nf}/nu_2.0_rho_{float(beta)}/set_{set_idx}/{hc}'
    elif pde == 'wave':
        ckpt_folder = f'output/system_' + pde + f'/N_f_{Nf}/beta_2.0_c_{float(beta)}/set_{set_idx}/{hc}'
    return ckpt_folder


def get_metric_file(ckpt_folder, metric, hc):
    if metric == 'cka':
        file_name = os.path.join(ckpt_folder, 'cka_metric.npy')
    elif 'mc' in metric:
        file_name = os.path.join(ckpt_folder, 'mode_connectivity.npy')
    elif metric == 'EIR':
        file_name = os.path.join(ckpt_folder, 'EIR.npy')
    elif 'cos' in metric:
        file_name = os.path.join(ckpt_folder, 'cos.npy')
    else:
        file_name = os.path.join(ckpt_folder, "base_metrics.npy")
    return file_name


def aggregate_function_mean_nonzero(metrics):
    metrics = np.array(metrics, dtype=float)
    metrics[metrics == 0] = np.nan
    return np.nanmean(metrics)


def get_metric_val(metric_file, metric):
    if metric == 'training_loss':
        result = np.load(metric_file, allow_pickle=True).item()
        return np.nanmean(result['loss'])
    elif metric == 'test_error':
        result = np.load(metric_file, allow_pickle=True).item()
        return np.nanmean(result['test/l2re'])
    elif metric == 'loss_res':
        result = np.load(metric_file, allow_pickle=True).item()
        return np.nanmean(result['loss_res'])
    elif metric == 'loss_bc':
        result = np.load(metric_file, allow_pickle=True).item()
        return np.nanmean(result['loss_bc'])
    elif metric == 'loss_ic':
        result = np.load(metric_file, allow_pickle=True).item()
        return np.nanmean(result['loss_ic'])
    elif metric == 'log_hessian_trace':
        result = np.load(metric_file, allow_pickle=True).item()
        return np.log(np.nanmean(result['hessian_trace']))
    elif metric == 'log_hessian_eigenvalue':
        result = np.load(metric_file, allow_pickle=True).item()
        return np.log(np.nanmean(result['hessian_top_eigenvalue']))
    elif metric == 'cka':
        result = np.load(metric_file, allow_pickle=True).item()
        cka_matrix = result['linear_cka_matrix']
        return aggregate_function_mean_nonzero(cka_matrix)
    elif metric == 'lmc':
        result = np.load(metric_file, allow_pickle=True).item()
        print(metric_file, result.keys())
        mc_matrix = result['lmc_loss_total']
        return aggregate_function_mean_nonzero(mc_matrix)
    elif metric == 'mc':
        result = np.load(metric_file, allow_pickle=True).item()
        mc_matrix = result['mc_loss_total']
        return aggregate_function_mean_nonzero(mc_matrix)


def vminmax(metric):
    if metric == 'training_loss':
        vmin, vmax = 0, 0.1
    elif metric == 'test_error':
        vmin, vmax = 0, 1
    elif metric == 'log_hessian_trace':
        vmin, vmax = 8, 13
    elif metric == 'log_hessian_eigenvalue':
        vmin, vmax = 8, 13
    elif metric == 'cka':
        vmin, vmax = 0.2, 1.0
    elif metric == 'lmc':
        vmin, vmax = -10000, 0
    elif metric == 'mc':
        vmin, vmax = -100, 0
    elif metric == 'EIR':
        vmin, vmax = 0, 0.3
    elif 'cos' in metric:
        vmin, vmax = -0.1, 1
    return vmin, vmax


def plot_2Dphase(phase2D, x_label_list, y_label_list, metric, metric_title, pde, hc, show_values):
    x_label_list = [int(x) for x in x_label_list]

    ticks_fontsize = 24
    label_fontsize = 32
    plt.rcParams['xtick.labelsize'] = ticks_fontsize
    plt.rcParams['ytick.labelsize'] = ticks_fontsize
    plt.rcParams['axes.labelsize'] = label_fontsize

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    heatmap = ax.imshow(phase2D, aspect='auto', cmap='viridis_r', origin='lower')

    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.ax.tick_params(labelsize=ticks_fontsize)
    from matplotlib.ticker import FormatStrFormatter
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.set_xticks(np.arange(len(x_label_list)))
    if pde == 'wave':
        ax.set_xticklabels([str(x ** 2) for x in x_label_list])
    else:
        ax.set_xticklabels([str(x) for x in x_label_list])

    ax.set_yticks(np.arange(len(y_label_list)))
    ax.set_yticklabels([str(y) for y in y_label_list])

    x_label = ''
    if pde == 'convection':
        x_label = r'$\beta$'
    elif pde == 'reaction' or pde == 'reaction_diffusion':
        x_label = r'$\rho$'
    elif pde == 'wave':
        x_label = r'$c^2$'

    ax.set_xlabel("PDE Settings", fontsize=label_fontsize)
    ax.set_ylabel('Number of Samples', fontsize=label_fontsize)

    # ✅ 如果启用显示值
    if show_values:
        for i in range(len(y_label_list)):
            for j in range(len(x_label_list)):
                val = phase2D[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2e}", ha='center', va='center',
                            color='black', fontsize=14)

    plt.tight_layout()

    save_dir = 'output/phase_plot/' + pde + '/' + hc
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pdf_path = os.path.join(save_dir, f"{metric}.pdf")
    plt.savefig(pdf_path, bbox_inches='tight')

    # ✅ JSON 同名导出
    json_path = os.path.splitext(pdf_path)[0] + ".json"
    json_data = {
        "title": f"{pde.upper()} {metric_title}",
        "x_axis": x_label_list,
        "y_axis": y_label_list,
        "data_matrix": phase2D.tolist(),
        "units": {
            "x": "PDE Parameter",
            "y": "Number of Samples",
            "data": metric_title
        },
        "notes": f"Generated from {pde} with hyperparameter {hc}."
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)


parser = argparse.ArgumentParser()
parser.add_argument('--hc', type=str, default='none', help='hc method')
parser.add_argument('--set_idx', type=int, default=0, help='the index of dataset')
parser.add_argument('--show_values', type=int, default=0, help='show the value')
args = parser.parse_args()

hc = args.hc
set_idx = args.set_idx
show_values = args.show_values

# ✅ 定义所有 PDE 的配置
if hc == 'alm':
    pde_configs = {
        "convection": {
            "beta_list": [5, 10, 15, 20, 25, 30, 50, 60, 70, 80, 100, 150],
            "Nf_list": [10, 50, 100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000],
        },
        "reaction": {
            "beta_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "Nf_list": [5, 10, 50, 100, 250, 500, 1000, 2000, 5000, 10000, 20000],
        },
        "wave": {
            "beta_list": [0.1, 0.5, 1, 2, 3, 4, 5, 6],
            "Nf_list": [5, 10, 50, 100, 250, 500, 1000, 2000, 5000, 10000],
        },
        "reaction_diffusion": {
            "beta_list": [1, 5, 10, 15, 20, 25, 30],
            "Nf_list": [5, 10, 50, 100, 500, 1000, 2000, 5000, 10000],
        },
    }
elif hc == 'none':
    pde_configs = {
        "convection": {
            "beta_list": [5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 70, 100, 150],
            "Nf_list": [10, 50, 100, 150, 200, 250, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000],
        },
        "reaction": {
            "beta_list": [1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12],
            "Nf_list": [2, 5, 10, 50, 100, 250, 500, 1000, 2000, 5000, 10000, 20000],
        },
        "wave": {
            "beta_list": [0.1, 0.5, 1, 2, 3, 4, 5, 6],
            "Nf_list": [5, 10, 50, 100, 250, 500, 1000, 2000, 5000, 10000],
        },
        "reaction_diffusion": {
            "beta_list": [1, 5, 10, 15, 20, 25, 30],
            "Nf_list": [5, 10, 50, 100, 500, 1000, 2000, 5000, 10000],
        },
    }

# ✅ 你想要绘制的指标
metric_list = ['training_loss', 'test_error', 'log_hessian_trace', 'log_hessian_eigenvalue', 'cka', 'lmc', 'mc']
metric_title_list = ['Training loss', 'Test error', 'Log Hessian trace', 'Log Hessian eigenvalue', 'CKA Similarity',
                     'Linear Mode Connectivity', 'Mode Connectivity']
metric_list = ['training_loss', 'test_error', 'loss_res', 'loss_bc', 'loss_ic']
metric_title_list = ['Training loss', 'Test error', 'loss_res', 'loss_bc', 'loss_ic']

# ✅ 主循环：遍历每个 PDE 设置
for pde, config in pde_configs.items():
    beta_list = config["beta_list"]
    Nf_list = config["Nf_list"]

    print(f"\n=== Processing PDE: {pde} ===")

    for metric, metric_title in zip(metric_list, metric_title_list):
        lenx = len(beta_list)
        leny = len(Nf_list)
        phase2D = np.full((leny, lenx), np.nan)  # ✅ 初始化为 NaN

        for j, beta in enumerate(beta_list):
            for i, Nf in enumerate(Nf_list):
                ckpt_folder = get_pde_folder(pde, hc, set_idx, Nf, beta)
                metric_file = get_metric_file(ckpt_folder, metric, hc)

                if not os.path.exists(metric_file):
                    # print(f"[WARN] Missing file: {metric_file}")
                    continue  # 保持 NaN

                try:
                    phase2D[i][j] = get_metric_val(metric_file, metric)
                except Exception as e:
                    # print(f"[ERROR] Failed to read {metric_file}: {e}")
                    phase2D[i][j] = np.nan

        print(f"Saving {metric} heatmap for {pde}")
        plot_2Dphase(phase2D, beta_list, Nf_list, metric, metric_title, pde, hc, show_values)