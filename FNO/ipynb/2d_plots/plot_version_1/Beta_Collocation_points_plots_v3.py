#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plot_boundary_utils import (
    build_boundary_overlays,
    merge_overlays,
)


# Regime overlay configuration (shared by train/test plots) \n
REGIME_NAMES = {1: "I", 2: "II", 3: "III"}
REGIME_LABEL_STYLE = {"fontsize": 16, "fontweight": "normal", "color": "white", "family": "serif"}
REGIME_LINE_STYLE = {"color": "white", "linewidth": 2.0, "alpha": 0.9}
REGIME_POINT_LABEL_STYLE = {"fontsize": 8, "color": "white"}
TEXT_PATH_EFFECTS = [PathEffects.withStroke(linewidth=2.5, foreground="black")]
DRAW_BOUNDARIES = False  # set True to enable boundary search lines
# Default label positions (fractions of the grid: x is columns, y is rows)
REGIME_POSITION_FRAC_DEFAULT = {1: (0.2, 0.8), 2: (0.8, 0.8), 3: (0.8, 0.2)}
# Optional overrides per func
REGIME_POSITION_FRAC_MAP = {
    "convection": {1: (0.2, 0.6), 2: (0.87, 0.8), 3: (0.8, 0.2)},
    "reaction": {1: (0.15, 0.6), 2: (0.8, 0.6), 3: (0.6, 0.05)},
    "wave": {1: (0.25, 0.6), 2: (0.85, 0.6), 3: (0.6, 0.02)},
}
# Default regime curves in fractional grid coordinates (x: columns, y: rows)
# Leave empty to disable unless explicitly provided per func.
REGIME_CURVES_FRAC_DEFAULT = []
# Optional overrides per func
REGIME_CURVES_FRAC_MAP = {
    # "convection": [ [(0.67, 1.0), (0.67, 0.5)], [(0.67, 0.5), (1.0, 0.5)] , [(0.67, 0.5), (0.2, 0.0)]  ], 
    # "reaction": [ [(0.33, 1.0), (0.33, 0.25)], [(0.33, 0.25), (1.0, 0.25)] , [(0.33, 0.25), (0.0, 0.0)]  ],
    # "wave": [ [(0.60, 1.0), (0.60, 0.20)], [(0.60, 0.2), (1.0, 0.2)] , [(0.60, 0.2), (0.2, 0.0)]  ] 
 
}
# Threshold to decide "yellow" region when deriving boundaries (normalized 0-1)
REGIME_YELLOW_THRESHOLD = 0.05
# Polygonal boundary extraction (train/test) configuration
BOUNDARY_SMOOTHING_SIGMA = 1.5
BOUNDARY_THRESHOLD_RATIO = 0.3
BOUNDARY_THRESHOLD_RATIO_MAP = {
    "convection": 0.05,
    "reaction": 0.3,
    "wave": 0.3,
}
BOUNDARY_THRESHOLD_RATIO_METRIC_MAP = {
    "convection": {"training_loss": 0.2, "test_error": 0.15},
    "reaction": {"training_loss": 0.25, "test_error": 0.25},
    "wave": {"training_loss": 0.35, "test_error": 0.25},
}
BOUNDARY_USE_BINARY_MASK = False
BOUNDARY_LINE_STYLE = {"color": "white", "linewidth": 1.5, "alpha": 0.9, "linestyle": "-"}
BOUNDARY_LINE_STYLE_TRAIN = {"color": "white", "linewidth": 2.5, "alpha": 0.9, "linestyle": "-"}
BOUNDARY_LINE_STYLE_TEST = {"color": "white", "linewidth": 2.5, "alpha": 0.9, "linestyle": "--"}
BOUNDARY_FILL_ENABLED = False
BOUNDARY_FILL_STYLE = {"color": "crimson", "alpha": 0.15}
BOUNDARY_POINT_STYLE = {"color": "crimson", "marker": "o", "markersize": 6, "linestyle": "None"}
BOUNDARY_FUSE_ENABLED = True
BOUNDARY_FUSE_THRESHOLD = 1.0
BOUNDARY_DRAW_FUSED_LINE = False
FUSED_BOUNDARY_LINE_STYLE = {"color": "white", "linewidth": 2.2, "alpha": 0.95, "linestyle": "--"}
BOUNDARY_CONFIG = {
    "smoothing_sigma": BOUNDARY_SMOOTHING_SIGMA,
    "threshold_ratio": BOUNDARY_THRESHOLD_RATIO,
    "use_binary_mask": BOUNDARY_USE_BINARY_MASK,
    "line_style_default": BOUNDARY_LINE_STYLE,
    "line_style_map": {
        "training_loss": BOUNDARY_LINE_STYLE_TRAIN,
        "test_error": BOUNDARY_LINE_STYLE_TEST,
    },
    "fill_enabled": BOUNDARY_FILL_ENABLED,
    "fill_style": BOUNDARY_FILL_STYLE,
    "fuse_enabled": BOUNDARY_FUSE_ENABLED,
    "fuse_threshold": BOUNDARY_FUSE_THRESHOLD,
}


def get_regime_positions(func):
    """Return per-func label positions if configured; otherwise defaults."""
    return REGIME_POSITION_FRAC_MAP.get(func, REGIME_POSITION_FRAC_DEFAULT)


def get_regime_curves(func):
    """Return per-func curve definitions if configured; otherwise defaults."""
    return REGIME_CURVES_FRAC_MAP.get(func, REGIME_CURVES_FRAC_DEFAULT)


def vminmax(metric, func):
    """Return color scale bounds for known metric/dataset combos."""
    vmin = vmax = None

    if func == "convection":
        if metric == "training_loss":
            vmin, vmax = 0, 0.025
        elif metric == "test_error":
            vmin, vmax = 0, 2.0
        elif metric == "log_hessian_trace":
            vmin, vmax = 9.0, 13.0
        elif metric == "log_hessian_eigenvalue":
            vmin, vmax = 9.0, 13.0
        elif metric == "cka":
            vmin, vmax = 0.35, 1.0
        elif metric == "mc":
            vmin, vmax = -100, 10
        elif metric == "EIR":
            vmin, vmax = 0, 0.3
        elif "cos" in metric:
            vmin, vmax = -0.1, 1
    elif func == "reaction":
        if metric == "training_loss":
            vmin, vmax = 0, 0.2
        elif metric == "test_error":
            vmin, vmax = 0, 1.2
        elif metric == "log_hessian_trace":
            vmin, vmax = 9.9, 12
        elif metric == "log_hessian_eigenvalue":
            vmin, vmax = 9.4, 12
        elif metric == "cka":
            vmin, vmax = 0.35, 1.0
        elif metric == "mc":
            vmin, vmax = -175, 175
        elif metric == "EIR":
            vmin, vmax = 0, 0.3
        elif "cos" in metric:
            vmin, vmax = -0.1, 1
    elif func == "wave":
        if metric == "training_loss":
            vmin, vmax = 0, 0.025
        elif metric == "test_error":
            vmin, vmax = 0, 1.2
        elif metric == "log_hessian_trace":
            vmin, vmax = 9.9, 12
        elif metric == "log_hessian_eigenvalue":
            vmin, vmax = 9.4, 12
        elif metric == "cka":
            vmin, vmax = 0.35, 1.0
        elif metric == "mc":
            vmin, vmax = -175, 175
        elif metric == "EIR":
            vmin, vmax = 0, 0.3
        elif "cos" in metric:
            vmin, vmax = -0.1, 1

    return vmin, vmax


def truncated_cmap(cmap_name, minval=0.0, maxval=1.0, n=256):
    """Return a copy of cmap_name trimmed to [minval, maxval] to avoid harsh ends."""
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(f"{cmap_name}_trunc", colors)


def _interp_index(fraction, length):
    if length <= 1:
        return 0.0
    return fraction * (length - 1)


def _format_axis_value(idx_float, labels):
    if not labels:
        return f"{idx_float:.2f}"
    n = len(labels)
    if idx_float <= 0:
        val = labels[0]
    elif idx_float >= n - 1:
        val = labels[-1]
    else:
        lower_i = int(np.floor(idx_float))
        upper_i = int(np.ceil(idx_float))
        frac = idx_float - lower_i
        lower = labels[lower_i]
        upper = labels[upper_i]
        try:
            lower_f = float(lower)
            upper_f = float(upper)
            val = lower_f + frac * (upper_f - lower_f)
        except (TypeError, ValueError):
            val = lower
    try:
        return f"{float(val):g}"
    except (TypeError, ValueError):
        return str(val)


def _find_boundary_points(mask_yellow, direction="top"):
    """Find boundary points scanning from top-right or bottom-right until yellow is hit."""
    n_rows, n_cols = mask_yellow.shape
    points = []
    row_iter = range(n_rows) if direction == "top" else range(n_rows - 1, -1, -1)
    for r in row_iter:
        boundary_c = None
        for c in range(n_cols - 1, -1, -1):
            if mask_yellow[r, c]:
                boundary_c = c
                break
        if boundary_c is not None:
            points.append((boundary_c, r))
    return points


def build_regime_overlay_from_shape(shape, pos_frac=None, curves_frac=None, x_labels=None, y_labels=None):
    """Place regime labels and curves using fixed relative positions on the grid."""
    if not shape or len(shape) != 2:
        return None
    n_rows, n_cols = shape
    pos_frac = pos_frac or REGIME_POSITION_FRAC_DEFAULT
    curves_frac = curves_frac or []
    labels = {}
    for regime_id, (fx, fy) in pos_frac.items():
        x = fx * (n_cols - 1 if n_cols > 1 else 0)
        y = fy * (n_rows - 1 if n_rows > 1 else 0)
        labels[regime_id] = (x, y)

    lines = []
    for curve in curves_frac:
        xs, ys = [], []
        for fx, fy in curve:
            x_idx = _interp_index(fx, n_cols)
            y_idx = _interp_index(fy, n_rows)
            xs.append(x_idx)
            ys.append(y_idx)
        if xs and ys:
            lines.append({"x": xs, "y": ys})

    return {"labels": labels, "lines": lines}


def build_boundary_lines(data_matrix, x_labels, y_labels, vmin, vmax):
    """Derive regime boundary lines from data by scanning for yellow regions."""
    if vmin is None or vmax is None or vmin == vmax:
        return []
    arr = np.array(data_matrix, dtype=float)
    norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    mask_yellow = norm <= REGIME_YELLOW_THRESHOLD

    lines = []
    for direction in ("top", "bottom"):
        pts = _find_boundary_points(mask_yellow, direction=direction)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        lines.append({"x": xs, "y": ys})
    return lines


def plot_2d_phase(
    phase2d,
    x_label_list,
    y_label_list,
    metric,
    metric_title,
    func,
    use_data_range=False,
    log_scale=False,
    regime_overlay=None,
    boundary_data=None,
):
    x_label_list = [
        int(x) if isinstance(x, (int, float)) and x >= 1 else x for x in x_label_list
    ]

    # Treat missing values as masked so they show up as empty cells
    phase2d = np.ma.masked_invalid(np.array(phase2d, dtype=float))
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    x_idx, y_idx = np.meshgrid(np.arange(len(x_label_list)), np.arange(len(y_label_list)))
    vmin, vmax = vminmax(metric, func)
    data_min = np.nanmin(np.ma.filled(phase2d, np.nan))
    data_max = np.nanmax(np.ma.filled(phase2d, np.nan))

    if use_data_range or vmin is None or vmax is None:
        vmin = data_min
        vmax = data_max

    if vmin == vmax:
        vmax = vmin + 1e-8

    # Pre-smooth the matrix to reduce stochastic noise before plotting
    phase2d_filled = np.ma.filled(phase2d, np.nan)
    if phase2d_filled.shape[0] > 5 and phase2d_filled.shape[1] > 5:
        phase2d_smooth = gaussian_filter(phase2d_filled, sigma=0.1)
    else:
        phase2d_smooth = phase2d_filled
    phase2d_smooth = np.ma.array(phase2d_smooth, mask=np.isnan(phase2d_filled))

    norm = None
    if log_scale and (vmin <= 0 or vmax <= 0):
        print(f"[{func}] {metric} 含有非正数值，无法使用对数坐标，回退为线性刻度。")
        log_scale = False

    if log_scale:
        safe_vmin = max(vmin, 1e-8)
        levels = np.logspace(np.log10(safe_vmin), np.log10(vmax), 100)
        norm = LogNorm(vmin=safe_vmin, vmax=vmax)
    else:
        levels = np.linspace(vmin, vmax, 100)

    current_cmap = "magma"
    if metric == "mc":
        current_cmap = "seismic"

    pos = ax.contourf(
        x_idx,
        y_idx,
        phase2d_smooth,
        levels=levels,
        cmap=current_cmap,
        norm=norm,
        extend="both",
        antialiased=False
    )
    
    for collection in pos.collections:
        collection.set_rasterized(True)
    # ax.contour(x_idx, y_idx, phase2d_smooth, levels=levels[::10], colors="white", alpha=0.1, linewidths=0.5)

    # Make each cell square
    ax.set_aspect("equal", adjustable="box")

    ax.set_xticks(np.arange(len(x_label_list)))
    ax.set_xticklabels([str(x) for x in x_label_list], fontsize=10)

    ax.set_yticks(np.arange(len(y_label_list)))
    ax.set_yticklabels([str(y) for y in y_label_list], fontsize=10)
    ax.tick_params(which="both", width=0)

    if func == "convection":
        xlabel = r"$\beta$"
    elif func == "reaction":
        xlabel = r"$\rho$"
    elif func == "wave":
        xlabel = r"$c$"
    else:
        xlabel = "Parameter"

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Collocation Points", fontsize=16)
    ax.set_title(metric_title, fontsize=16)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(pos, cax=cax)
    cbar.ax.tick_params(labelsize=10)

    def format_tick(x, _):
        abs_x = abs(x)
        if abs_x < 1:
            return f"{x:.3f}"
        if abs_x < 10:
            return f"{x:.2f}"
        if abs_x < 100:
            return f"{x:.1f}"
        return f"{x:.0f}"

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_tick))

    # --- NEW BOUNDARY LOGIC START ---
    # Use smoothed training/test matrices so contours align with the rendered heatmap.
    def _prepare_smooth_matrix(raw):
        arr = np.array(raw, dtype=float)
        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            return None
        fill_val = np.nanmean(arr[finite_mask])
        arr[~finite_mask] = fill_val
        return gaussian_filter(arr, sigma=0.1)

    train_smooth = test_smooth = None
    if boundary_data:
        if "training_loss" in boundary_data:
            train_smooth = _prepare_smooth_matrix(boundary_data["training_loss"])
        if "test_error" in boundary_data:
            test_smooth = _prepare_smooth_matrix(boundary_data["test_error"])

    # --- FUSED BOUNDARY LOGIC (WITH BUFFER) ---
    # Base thresholds; adjust per dataset/colorbar if needed
    train_thresh = 0.004
    test_thresh = 0.23
    # Relaxation buffers to avoid over-masking
    train_buffer = 0.002
    test_buffer = 0.2

    if boundary_data and "training_loss" in boundary_data and "test_error" in boundary_data:
        t_raw = np.array(boundary_data["training_loss"], dtype=float)
        e_raw = np.array(boundary_data["test_error"], dtype=float)

        t_mat = gaussian_filter(t_raw, sigma=0.1)
        e_mat = gaussian_filter(e_raw, sigma=0.1)

        # Dashed I/II boundary (test error), masked out of high-loss (Regime III) region
        e_masked = np.ma.masked_where(t_mat > (train_thresh + train_buffer), e_mat)
        ax.contour(
            x_idx,
            y_idx,
            e_masked,
            levels=[test_thresh],
            colors="white",
            linewidths=2.5,
            linestyles="dashed",
            alpha=0.9,
        )

        # Solid II/III boundary (training loss), masked out of low-error (Regime I) region
        t_masked = np.ma.masked_where(e_mat < (test_thresh - test_buffer), t_mat)
        if t_masked.max() > train_thresh and t_masked.min() < train_thresh:
            ax.contour(
                x_idx,
                y_idx,
                t_masked,
                levels=[train_thresh],
                colors="white",
                linewidths=2.5,
                linestyles="solid",
                alpha=0.9,
            )
        else:
            print(f"Warning: Training Loss threshold {train_thresh} is out of range for smoothed data.")
    # --- NEW BOUNDARY LOGIC END ---

    if regime_overlay:
        # Old boundary lines are intentionally disabled to avoid misalignment with smoothed contours.
        # for line in regime_overlay.get("lines", []):
        #     x_pts = np.array(line.get("x", []))
        #     y_pts = np.array(line.get("y", []))
        #
        #     if len(x_pts) > 3:
        #         t = np.arange(len(x_pts))
        #         t_new = np.linspace(t.min(), t.max(), 300)
        #
        #         spl_x = make_interp_spline(t, x_pts, k=3)
        #         spl_y = make_interp_spline(t, y_pts, k=3)
        #
        #         x_smooth = spl_x(t_new)
        #         y_smooth = spl_y(t_new)
        #
        #         ax.plot(x_smooth, y_smooth, color="white", linestyle="--", linewidth=2.5, alpha=0.8)
        #     else:
        #         ax.plot(x_pts, y_pts, color="white", linestyle="--", linewidth=2.5, alpha=0.8)
        for regime_id, (x_pos, y_pos) in regime_overlay.get("labels", {}).items():
            name = REGIME_NAMES.get(regime_id)
            if not name:
                continue
            txt = ax.text(x_pos, y_pos, name, ha="center", va="center", **REGIME_LABEL_STYLE)
            txt.set_path_effects(TEXT_PATH_EFFECTS)

    save_dir = f"/jumbo/yaoqingyang/yuanzhehu/neuraloperators-TL-scaling/ipynb/2d_plots/data/2d_plots/{func}"
    os.makedirs(save_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{metric}.pdf", bbox_inches='tight')
    plt.close(fig)


def print_phase_data(phase2d, x_labels, y_labels, metric, func):
    print(f"[{func}] {metric} 数值：")
    header = "y/x\t" + "\t".join(str(x) for x in x_labels)
    print(header)
    for y_val, row in zip(y_labels, phase2d):
        row_str = "\t".join(f"{v:.5f}" if np.isfinite(v) else "nan" for v in row)
        print(f"{y_val}\t{row_str}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot 2D phase metrics.")
    parser.add_argument(
        "--print-grid",
        action="store_true",
        help="打印每个2D网格的数值矩阵，便于分析。",
    )
    parser.add_argument(
        "--use-data-range",
        action="store_true",
        help="强制颜色轴起点/终点使用当前2D网格的最小值和最大值。",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="颜色轴使用对数刻度（仅当数据全部为正时有效）。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = {
        "convection": "/jumbo/yaoqingyang/yuanzhehu/neuraloperators-TL-scaling/ipynb/2d_plots/data/xiaopeng/experiment_data_convection.json",
        "reaction": "/jumbo/yaoqingyang/yuanzhehu/neuraloperators-TL-scaling/ipynb/2d_plots/data/xiaopeng/experiment_data_reaction.json",
        "wave": "/jumbo/yaoqingyang/yuanzhehu/neuraloperators-TL-scaling/ipynb/2d_plots/data/xiaopeng/experiment_data_wave.json",
    }

    metric_list = ["training_loss", "test_error", 'log_hessian_trace', 'log_hessian_eigenvalue', 'cka', 'mc']
    metric_title_list = ["Training Loss", "Test Error", "Log Hessian Trace", "Log Hessian Eigenvalue", "CKA Similarity", "Mode Connectivity"]

    for func, json_file_path in datasets.items():
        print(f"\n处理 {func} 数据集...")

        with open(json_file_path, "r") as f:
            data = json.load(f)

        x_axis = data["x_axis"]
        y_axis = data["y_axis"]

        regime_overlay = None
        if "training_loss" in data and "test_error" in data:
            positions = get_regime_positions(func)
            curves = get_regime_curves(func)
            regime_overlay = build_regime_overlay_from_shape(
                np.array(data["training_loss"]).shape,
                pos_frac=positions,
                curves_frac=curves,
                x_labels=x_axis,
                y_labels=y_axis,
            )
            if DRAW_BOUNDARIES:
                t_vmin, t_vmax = vminmax("training_loss", func)
                boundary_lines = build_boundary_lines(
                    data["training_loss"],
                    x_axis,
                    y_axis,
                    t_vmin,
                    t_vmax,
                )
                regime_overlay["lines"].extend(boundary_lines)

        boundary_config = dict(BOUNDARY_CONFIG)
        boundary_config["threshold_ratio"] = BOUNDARY_THRESHOLD_RATIO_MAP.get(func, BOUNDARY_THRESHOLD_RATIO)
        if func in BOUNDARY_THRESHOLD_RATIO_METRIC_MAP:
            boundary_config["threshold_ratio_map"] = BOUNDARY_THRESHOLD_RATIO_METRIC_MAP[func]
        boundary_overlays, combined_train_test_overlay, fused_overlay, mapping_info = build_boundary_overlays(
            data,
            config=boundary_config,
            metric_keys=("training_loss", "test_error"),
        )
        if mapping_info:
            print(f"[{func}] 融合映射: train_idx={mapping_info.get('train_indices')} -> test_idx={mapping_info.get('test_indices')}")

        for metric, metric_title in zip(metric_list, metric_title_list):
            phase2d = np.array(data[metric])
            if args.print_grid:
                print_phase_data(phase2d, x_axis, y_axis, metric, func)
            metric_overlay = (
                combined_train_test_overlay
                if metric in ("training_loss", "test_error")
                else boundary_overlays.get(metric)
            )
            overlay = merge_overlays(regime_overlay, metric_overlay)
            plot_2d_phase(
                phase2d,
                x_axis,
                y_axis,
                metric,
                metric_title,
                func,
                use_data_range=args.use_data_range,
                log_scale=args.log_scale,
                regime_overlay=overlay,
                boundary_data=data,
            )

        print(f"{func} 的所有图表已生成并保存到 2d_plots/{func}/")

    print("\n所有数据集的图表已生成完毕！")


if __name__ == "__main__":
    main()
