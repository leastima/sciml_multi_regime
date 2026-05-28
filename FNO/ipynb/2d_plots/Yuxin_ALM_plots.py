import sys
import os
import json
import glob
import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
from plot_boundary_utils import build_boundary_overlays, merge_overlays, truncated_cmap


REGIME_NAMES = {1: "I", 2: "II", 3: "III"}
REGIME_LABEL_STYLE = {"fontsize": 16, "fontweight": "normal", "color": "white", "family": "serif"}
TEXT_PATH_EFFECTS = [PathEffects.withStroke(linewidth=2.5, foreground="black")]
# Default label positions (fractions of the grid: x is columns, y is rows)
REGIME_POSITION_FRAC_DEFAULT = {1: (0.2, 0.8), 2: (0.8, 0.8), 3: (0.8, 0.2)}
# Optional overrides per func
REGIME_POSITION_FRAC_MAP = {
    "convection": {1: (0.2, 0.6), 2: (0.87, 0.8), 3: (0.6, 0.2)},
    "reaction": {1: (0.15, 0.6), 2: (0.8, 0.6), 3: (0.3, 0.09)},
    "wave": {1: (0.25, 0.6), 2: (0.85, 0.6), 3: (0.6, 0.09)},
    "reaction_diffusion": {1: (0.1, 0.6), 2: (0.8, 0.6), 3: (0.3, 0.09)},
}

# Boundary style and per-equation parameters
BOUNDARY_LINE_STYLE = {"color": "white", "linewidth": 1.5, "alpha": 0.9, "linestyle": "-"}
BOUNDARY_LINE_STYLE_TRAIN = {"color": "white", "linewidth": 2.5, "alpha": 0.9, "linestyle": "-"}
BOUNDARY_LINE_STYLE_TEST = {"color": "white", "linewidth": 2.5, "alpha": 0.9, "linestyle": "--"}

BOUNDARY_BASE_CONFIG = {
    "use_binary_mask": False,
    "line_style_default": BOUNDARY_LINE_STYLE,
    "line_style_map": {
        "training_loss": BOUNDARY_LINE_STYLE_TRAIN,
        "test_error": BOUNDARY_LINE_STYLE_TEST,
    },
    "fuse_enabled": True,
    "fuse_threshold": 1.0,
}

BOUNDARY_DEFAULT_CONFIG = {
    "smoothing_sigma": 1.5,
    "threshold_ratio": 0.3,
    "threshold_ratio_map": {},
}

BOUNDARY_CONFIG_PRESETS = {
    "convection": {
        "smoothing_sigma": 1.5,
        "threshold_ratio": 0.05,
        "threshold_ratio_map": {"training_loss": 0.2, "test_error": 0.15},
    },
    "reaction": {
        "smoothing_sigma": 1.5,
        "threshold_ratio": 0.3,
        "threshold_ratio_map": {"training_loss": 0.25, "test_error": 0.25},
    },
    "wave": {
        "smoothing_sigma": 1.5,
        "threshold_ratio": 0.3,
        "threshold_ratio_map": {"training_loss": 0.35, "test_error": 0.25},
    },
    "reaction_diffusion": {
        "smoothing_sigma": 1.5,
        "threshold_ratio": 0.3,
        "threshold_ratio_map": {"training_loss": 0.25, "test_error": 0.16},
    },
}

FUSED_BOUNDARY_DEFAULT = {
    "train_thresh": 0.004,
    "test_thresh": 0.23,
    "train_buffer": 0.002,
    "test_buffer": 0.2,
    "smooth_sigma": 0.1,
}

FUSED_BOUNDARY_PARAMS = {
    "convection": {
        "train_thresh": 0.06,
        "test_thresh": 0.23,
        "train_buffer": 0.1,
        "test_buffer": 0.3,
        "smooth_sigma": 0.7,
    },
    "reaction": {
        "train_thresh": 0.0001,
        "test_thresh": 0.5,
        "train_buffer": 0.004,
        "test_buffer": 0.2,
        "smooth_sigma": 0.7,
    },
    "wave": {
        "train_thresh": 0.004,
        "test_thresh": 0.23,
        "train_buffer": 0.002,
        "test_buffer": 0.2,
        "smooth_sigma": 0.7,
    },
}


def get_regime_positions(func):
    """Return per-func label positions if configured; otherwise defaults."""
    return REGIME_POSITION_FRAC_MAP.get(func, REGIME_POSITION_FRAC_DEFAULT)


def vminmax(metric, func):
    """Return color scale bounds for known metric/dataset combos."""
    vmin = vmax = None

    if func == "convection":
        if metric == "training_loss":
            vmin, vmax = 0, 0.35
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
            vmin, vmax = 0, 0.0015
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
    elif func == "reaction_diffusion":
        if metric == "training_loss":
            vmin, vmax = 0, 0.025
        elif metric == "test_error":
            vmin, vmax = 0, 1.2
        elif metric == "log_hessian_trace":
            vmin, vmax = 10.0, 12.5
        elif metric == "log_hessian_eigenvalue":
            vmin, vmax = 10.0, 12.5
        elif metric == "cka":
            vmin, vmax = 0.35, 1.0
        elif metric == "mc":
            vmin, vmax = -0.5, 0.5
        elif metric == "EIR":
            vmin, vmax = 0, 0.3
        elif "cos" in metric:
            vmin, vmax = -0.1, 1

    return vmin, vmax



def build_regime_overlay_from_shape(shape, pos_frac=None):
    """Place regime labels using fixed relative positions on the grid."""
    if not shape or len(shape) != 2:
        return None
    n_rows, n_cols = shape
    pos_frac = pos_frac or REGIME_POSITION_FRAC_DEFAULT
    labels = {}
    for regime_id, (fx, fy) in pos_frac.items():
        x = fx * (n_cols - 1 if n_cols > 1 else 0)
        y = fy * (n_rows - 1 if n_rows > 1 else 0)
        labels[regime_id] = (x, y)

    return {"labels": labels, "lines": []}


def plot_2Dphase(
    phase2D,
    x_label_list,
    y_label_list,
    metric,
    metric_title,
    func,
    method,
    regime_overlay=None,
    boundary_data=None,
    use_data_range=False,
    log_scale=False,
):
    """Plot 2D heatmap using the Beta v4 contourf-based renderer."""
    x_label_list = [
        int(x) if isinstance(x, (int, float)) and x >= 1 else x for x in x_label_list
    ]

    # Treat missing values as masked so they show up as empty cells
    phase2d = np.ma.masked_invalid(np.array(phase2D, dtype=float))
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

    # current_cmap = "viridis_r"
    # current_cmap = "magma"
    current_cmap = truncated_cmap("viridis_r", minval=0.0, maxval=0.8) #"viridis_r"

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
        antialiased=False,
    )

    if hasattr(pos, "set_rasterized"):
        pos.set_rasterized(True)
    else:
        for collection in getattr(pos, "collections", []):
            collection.set_rasterized(True)

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
    elif func == "reaction_diffusion":
        xlabel = r"$\rho$"
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

    fused_params = {**FUSED_BOUNDARY_DEFAULT, **FUSED_BOUNDARY_PARAMS.get(func, {})}
    smooth_sigma = fused_params["smooth_sigma"]

    # --- FUSED BOUNDARY LOGIC (WITH BUFFER) ---
    train_thresh = fused_params["train_thresh"]
    test_thresh = fused_params["test_thresh"]
    # Relaxation buffers to avoid over-masking
    train_buffer = fused_params["train_buffer"]
    test_buffer = fused_params["test_buffer"]

    boundary_shapes_ok = False
    if boundary_data and "training_loss" in boundary_data and "test_error" in boundary_data:
        t_raw = np.array(boundary_data["training_loss"], dtype=float)
        e_raw = np.array(boundary_data["test_error"], dtype=float)
        boundary_shapes_ok = (
            t_raw.shape == e_raw.shape == phase2d_smooth.shape
        )
        if not boundary_shapes_ok:
            print(
                f"[{func}/{metric}] Boundary shapes mismatch, skip drawing: "
                f"phase {phase2d_smooth.shape}, train {t_raw.shape}, test {e_raw.shape}"
            )

    if boundary_shapes_ok:
        t_mat = gaussian_filter(t_raw, sigma=smooth_sigma)
        e_mat = gaussian_filter(e_raw, sigma=smooth_sigma)

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

    if regime_overlay:
        for regime_id, (x_pos, y_pos) in regime_overlay.get("labels", {}).items():
            name = REGIME_NAMES.get(regime_id)
            if not name:
                continue
            txt = ax.text(x_pos, y_pos, name, ha="center", va="center", **REGIME_LABEL_STYLE)
            txt.set_path_effects(TEXT_PATH_EFFECTS)

    # Save to PDE-specific folder with method prefix
    save_dir = f"/jumbo/yaoqingyang/yuanzhehu/neuraloperators-TL-scaling/ipynb/2d_plots/data/2d_plots/{func}"
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{method}_{metric}.pdf"

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}", bbox_inches="tight")
    print(f"Saved: {func}/{filename}")
    plt.close()
    




###main


# Base directory for yuxin's JSON files
base_dir = '/jumbo/yaoqingyang/yuanzhehu/neuraloperators-TL-scaling/ipynb/2d_plots/data/yuxin/jsons'

# PDE types
pde_types = ['reaction_diffusion', 'convection', 'reaction', 'wave'] 

# Metric names and their display titles
metric_mapping = {
    'training_loss': 'Training Loss',
    'test_error': 'Test Error',
    # 'log_hessian_trace': 'Log Hessian Trace',
    # 'log_hessian_eigenvalue': 'Log Hessian Eigenvalue',
    # 'cka': 'CKA Similarity',
    # # 'lmc': 'Mode Connectivity',
    # 'mc': 'Mode Connectivity'
}

# Process each PDE type
for pde in pde_types:
    pde_dir = os.path.join(base_dir, pde)
    print(f"\n{'='*60}")
    print(f"Processing {pde.upper()}")
    print(f"{'='*60}")
    
    # Find all JSON files in this PDE directory
    json_files = glob.glob(os.path.join(pde_dir, '*.json'))
    
    if not json_files:
        print(f"No JSON files found in {pde_dir}")
        continue

    method_cache = {}
    
    # Process each JSON file
    for json_file in sorted(json_files):
        filename = os.path.basename(json_file)
        # Parse filename: method_metric.json
        name_parts = filename.replace('.json', '').split('_', 1)
        if len(name_parts) != 2:
            print(f"Skipping {filename}: unexpected format")
            continue
        
        method, metric = name_parts
        
        # Get metric title
        metric_title = metric_mapping.get(metric, metric.replace('_', ' ').title())
        
        # Load JSON data
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            x_axis = data['x_axis']
            y_axis = data['y_axis']
            # Do NOT transpose - data_matrix is already in correct format
            # data_matrix[row][col] where row=y_index, col=x_index
            data_matrix = np.array(data['data_matrix'])

            method_metrics = method_cache.setdefault(method, {})
            method_metrics[metric] = {
                "data_matrix": data_matrix,
                "x_axis": x_axis,
                "y_axis": y_axis,
                "metric_title": metric_title,
            }
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    for method, metrics in method_cache.items():
        regime_overlay = None
        if "training_loss" in metrics and "test_error" in metrics:
            positions = get_regime_positions(pde)
            regime_overlay = build_regime_overlay_from_shape(
                metrics["training_loss"]["data_matrix"].shape,
                pos_frac=positions,
            )

        boundary_overlays = {}
        combined_train_test_overlay = None
        boundary_data = None
        if "training_loss" in metrics and "test_error" in metrics:
            boundary_data = {
                "training_loss": metrics["training_loss"]["data_matrix"],
                "test_error": metrics["test_error"]["data_matrix"],
            }
            boundary_config = {
                **BOUNDARY_BASE_CONFIG,
                **BOUNDARY_DEFAULT_CONFIG,
                **BOUNDARY_CONFIG_PRESETS.get(pde, {}),
            }
            boundary_overlays, combined_train_test_overlay, fused_overlay, mapping_info = build_boundary_overlays(
                boundary_data,
                config=boundary_config,
                metric_keys=("training_loss", "test_error"),
            )
            if mapping_info:
                print(f"[{pde}/{method}] 融合映射: train_idx={mapping_info.get('train_indices')} -> test_idx={mapping_info.get('test_indices')}")

        for metric, meta in metrics.items():
            metric_overlay = (
                combined_train_test_overlay
                if metric in ("training_loss", "test_error")
                else boundary_overlays.get(metric)
            )
            overlay = merge_overlays(regime_overlay, metric_overlay)
            plot_2Dphase(
                meta["data_matrix"],
                meta["x_axis"],
                meta["y_axis"],
                metric,
                meta["metric_title"],
                pde,
                method,
                regime_overlay=overlay,
                boundary_data=boundary_data,
            )
    
    print(f"\nCompleted {pde}!")

print("\n" + "="*60)
print("All plots have been generated!")
print("="*60)
