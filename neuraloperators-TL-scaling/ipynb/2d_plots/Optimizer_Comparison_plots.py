import os
import sys
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
from plot_boundary_utils import build_boundary_overlays, merge_overlays, truncated_cmap


# Regime overlay configuration (label-only)
REGIME_NAMES = {1: "I", 2: "II", 3: "III"}
REGIME_LABEL_STYLE = {"fontsize": 16, "fontweight": "normal", "color": "white", "family": "serif"}
TEXT_PATH_EFFECTS = [PathEffects.withStroke(linewidth=2.5, foreground="black")]

# Default label positions (fractions of the grid: x is columns, y is rows)
REGIME_POSITION_FRAC_DEFAULT = {1: (0.2, 0.8), 2: (0.8, 0.8), 3: (0.8, 0.2)}
# Optional overrides per optimizer
REGIME_POSITION_FRAC_MAP = {
    "nncg": {1: (0.2, 0.8), 2: (0.8, 0.8), 3: (0.7, 0.3)},
    "ropinn": {1: (0.2, 0.6), 2: (0.88, 0.8), 3: (0.7, 0.3)},
    "multiadam": {1: (0.2, 0.8), 2: (0.8, 0.8), 3: (0.8, 0.2)},
}


def get_regime_positions_for_optimizer(optimizer_name):
    """Return per-optimizer label positions if configured; otherwise defaults."""
    key = (optimizer_name or "").lower()
    return REGIME_POSITION_FRAC_MAP.get(key, REGIME_POSITION_FRAC_DEFAULT)


# Boundary extraction configuration (for build_boundary_overlays)
BOUNDARY_SMOOTHING_SIGMA = 1
BOUNDARY_THRESHOLD_RATIO = 0.3
BOUNDARY_THRESHOLD_RATIO_MAP = {
    "nncg": 0.3,
    "ropinn": 0.25,
    "multiadam": 0.35,
}
BOUNDARY_THRESHOLD_RATIO_METRIC_MAP = {
    "nncg": {"training_loss": 0.03, "test_error": 0.03},
    "ropinn": {"training_loss": 0.3, "test_error": 0.1},
    "multiadam": {"training_loss": 0.35, "test_error": 0.2},
}
BOUNDARY_USE_BINARY_MASK = False
BOUNDARY_LINE_STYLE = {"color": "white", "linewidth": 1.5, "alpha": 0.9, "linestyle": "-"}
BOUNDARY_LINE_STYLE_TRAIN = {"color": "white", "linewidth": 2.5, "alpha": 0.9, "linestyle": "-"}
BOUNDARY_LINE_STYLE_TEST = {"color": "white", "linewidth": 2.5, "alpha": 0.9, "linestyle": "--"}
BOUNDARY_FILL_ENABLED = False
BOUNDARY_FILL_STYLE = {"color": "crimson", "alpha": 0.15}
BOUNDARY_FUSE_ENABLED = True
BOUNDARY_FUSE_THRESHOLD = 1.0
BOUNDARY_CONFIG = {
    "smoothing_sigma": BOUNDARY_SMOOTHING_SIGMA,
    "threshold_ratio": BOUNDARY_THRESHOLD_RATIO,
    "use_binary_mask": BOUNDARY_USE_BINARY_MASK,
    "line_style_default": BOUNDARY_LINE_STYLE,
    "line_style_map": {
        "training_loss": BOUNDARY_LINE_STYLE_TRAIN,
        "test_error": BOUNDARY_LINE_STYLE_TEST,
        "test_loss": BOUNDARY_LINE_STYLE_TEST,
    },
    "fill_enabled": BOUNDARY_FILL_ENABLED,
    "fill_style": BOUNDARY_FILL_STYLE,
    "fuse_enabled": BOUNDARY_FUSE_ENABLED,
    "fuse_threshold": BOUNDARY_FUSE_THRESHOLD,
}

# Fused boundary thresholds used when drawing contour lines directly on smoothed train/test maps
FUSED_BOUNDARY_DEFAULT = {
    "train_thresh": 0.004,
    "test_thresh": 0.23,
    "train_buffer": 0.002,
    "test_buffer": 0.2,
    "smooth_sigma": 0.7,
}

FUSED_BOUNDARY_PARAMS = {
    "nncg": {
        "train_thresh": 0.004,
        "test_thresh": 0.25,
        "train_buffer": 0.002,
        "test_buffer": 0.2,
        "smooth_sigma": 0.7,
    },
    "ropinn": {
        "train_thresh": 0.004,
        "test_thresh": 0.25,
        "train_buffer": 0.002,
        "test_buffer": 0.2,
        "smooth_sigma": 0.7,
    },
    "multiadam": {
        "train_thresh": 0.004,
        "test_thresh": 0.25,
        "train_buffer": 0.002,
        "test_buffer": 0.2,
        "smooth_sigma": 0.7,
    },
}


def vminmax(metric, optimizer_name):
    """Return color scale bounds for known metric/optimizer combos."""
    vmin = vmax = None

    metric = (metric or "").lower()
    optimizer_name = (optimizer_name or "").lower()

    if optimizer_name == "multiadam":
        if "training_loss" in metric:
            vmin, vmax = 0.01, 0.5
        elif "test_error" in metric:
            vmin, vmax = 0.8, 2
    elif optimizer_name == "ropinn":
        if "training_loss" in metric:
            vmin, vmax = 0, 0.025
        elif "test_error" in metric:
            vmin, vmax = 0, 2
    elif optimizer_name == "nncg":
        if "training_loss" in metric:
            vmin, vmax = 0, 0.025
        elif "test_error" in metric:
            vmin, vmax = 0, 2
            
        # if "success" in metric:
        #     vmin, vmax = 0, 100
        # elif "improvement" in metric and "training_loss" in metric:
        #     vmin, vmax = -1200, 1200
        # elif "improvement" in metric and ("test_error" in metric or "l2" in metric):
        #     vmin, vmax = -70, 70
        # elif "phase1" in metric and "training_loss" in metric:
        #     vmin, vmax = 0, 0.02
        # elif "phase2" in metric and "training_loss" in metric:
        #     vmin, vmax = 0, 0.02
        # elif "phase1" in metric and ("test_error" in metric or "l2" in metric):
        #     vmin, vmax = 0, 4
        # elif "phase2" in metric and ("test_error" in metric or "l2" in metric):
        #     vmin, vmax = 0, 4

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



def plot_2Dphase_optimizer(
    phase2D,
    x_label_list,
    y_label_list,
    title,
    optimizer_name,
    metric_name,
    regime_overlay=None,
    boundary_data=None,
):
    """Plot optimizer phase diagrams using the Beta-style contourf rendering."""
    x_label_list = [
        int(x) if isinstance(x, (int, float)) and x >= 1 else x for x in x_label_list
    ]

    phase2d = np.ma.masked_invalid(np.array(phase2D, dtype=float))
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    x_idx, y_idx = np.meshgrid(np.arange(len(x_label_list)), np.arange(len(y_label_list)))

    title_lower = title.lower()
    metric_key = metric_name
    if "phase 1" in title_lower:
        metric_key = f"phase1_{metric_key}"
    elif "phase 2" in title_lower:
        metric_key = f"phase2_{metric_key}"
    if "improvement" in title_lower or "improvement" in metric_key:
        metric_key = f"{metric_key}_improvement"
    elif "success" in title_lower or "success" in metric_key:
        metric_key = "success_rate"

    vmin, vmax = vminmax(metric_key, optimizer_name)
    data_min = np.nanmin(np.ma.filled(phase2d, np.nan))
    data_max = np.nanmax(np.ma.filled(phase2d, np.nan))
    if vmin is None or vmax is None:
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
    metric_lower = metric_name.lower()
    # current_cmap = "viridis_r"
    # current_cmap = "magma"
    current_cmap = truncated_cmap("viridis_r", minval=0.0, maxval=0.8) #"viridis_r"


    # Diverging colormap for improvement metrics
    if "improvement" in metric_lower:
        max_abs = max(abs(data_min), abs(data_max))
        vmin = -max_abs if vmin is None else min(vmin, -max_abs)
        vmax = max_abs if vmax is None else max(vmax, max_abs)
        current_cmap = "seismic"
    elif "success" in metric_lower:
        current_cmap = "viridis"

    levels = np.linspace(vmin, vmax, 100)

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
    for collection in pos.collections:
        collection.set_rasterized(True)

    # Make each cell square
    ax.set_aspect("equal", adjustable="box")

    ax.set_xticks(np.arange(len(x_label_list)))
    ax.set_xticklabels([str(x) for x in x_label_list], fontsize=10)

    ax.set_yticks(np.arange(len(y_label_list)))
    ax.set_yticklabels([str(y) for y in y_label_list], fontsize=10)
    ax.tick_params(which="both", width=0)

    ax.set_xlabel(r"$\beta$", fontsize=16)
    ax.set_ylabel("Collocation Points", fontsize=16)
    ax.set_title(title, fontsize=16)

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

    # --- FUSED BOUNDARY LOGIC ---
    fused_params = {**FUSED_BOUNDARY_DEFAULT, **FUSED_BOUNDARY_PARAMS.get((optimizer_name or "").lower(), {})}
    smooth_sigma = fused_params["smooth_sigma"]

    train_thresh = fused_params["train_thresh"]
    test_thresh = fused_params["test_thresh"]
    train_buffer = fused_params["train_buffer"]
    test_buffer = fused_params["test_buffer"]

    if boundary_data and "training_loss" in boundary_data and ("test_error" in boundary_data or "test_loss" in boundary_data):
        t_raw = np.array(boundary_data["training_loss"], dtype=float)
        e_raw = np.array(boundary_data.get("test_error", boundary_data.get("test_loss")), dtype=float)

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
    # --- END BOUNDARY LOGIC ---

    if regime_overlay:
        for regime_id, (x_pos, y_pos) in regime_overlay.get("labels", {}).items():
            name = REGIME_NAMES.get(regime_id)
            if not name:
                continue
            txt = ax.text(x_pos, y_pos, name, ha="center", va="center", **REGIME_LABEL_STYLE)
            txt.set_path_effects(TEXT_PATH_EFFECTS)

    # Save to convection folder with optimizer prefix
    save_dir = "/jumbo/yaoqingyang/yuanzhehu/neuraloperators-TL-scaling/ipynb/2d_plots/data/2d_plots/convection"
    os.makedirs(save_dir, exist_ok=True)

    # Create filename: optimizer_metric.pdf
    filename = f"{optimizer_name.lower()}_{metric_name}.pdf"

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}", bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()
    
    










# Load optimizer comparison data
json_file_path = '/jumbo/yaoqingyang/yuanzhehu/neuraloperators-TL-scaling/ipynb/2d_plots/data/xiaokun/optimizer_comparison_data.json'

with open(json_file_path, 'r') as f:
    data = json.load(f)

# Mapping from dataset titles to metric names for filenames
def get_metric_name(title):
    """Extract metric name from title for filename"""
    title_lower = title.lower()
    
    # Define metric keywords and their file names
    if 'training loss' in title_lower:
        return 'training_loss'
    elif 'test loss' in title_lower:
        return 'test_loss'
    elif 'test error' in title_lower or 'test l2' in title_lower:
        return 'test_error'
    elif 'improvement' in title_lower and 'loss' in title_lower:
        return 'loss_improvement'
    elif 'improvement' in title_lower and ('l2' in title_lower or 'error' in title_lower):
        return 'error_improvement'
    elif 'success rate' in title_lower:
        return 'success_rate'
    else:
        # Default: use simplified title
        return title.lower().replace(' ', '_').replace('(', '').replace(')', '')

def clean_title(title, optimizer_name):
    """Remove optimizer name and PDE name from title for display"""
    # Remove optimizer name (case insensitive)
    title_clean = title.replace(optimizer_name, '').strip()
    
    # Remove common prefixes
    prefixes_to_remove = ['PINN', 'pinn', 'convection', 'Convection']
    for prefix in prefixes_to_remove:
        title_clean = title_clean.replace(prefix, '').strip()
    
    # Clean up multiple spaces
    import re
    title_clean = re.sub(r'\s+', ' ', title_clean)
    
    return title_clean

# Process each optimizer
for optimizer_data in data['optimizers']:
    optimizer_name = optimizer_data['optimizer']
    print(f"\n{'='*60}")
    print(f"Processing optimizer: {optimizer_name}")
    print(f"{'='*60}")

    processed_datasets = []
    metric_cache = {}

    # Process each dataset for this optimizer
    for dataset in optimizer_data['datasets']:
        title = dataset['title']
        x_axis = dataset['x_axis']
        y_axis = dataset['y_axis']
        # IMPORTANT: Transpose the data matrix
        # Original: data_matrix[row][col] where row=y_index, col=x_index
        # Need: data_matrix[row][col] where row=y_index (for pcolormesh Y axis)
        data_matrix = np.array(dataset['data_matrix']).T  # Transpose!

        # Clean title to remove optimizer name
        title_clean = clean_title(title, optimizer_name)

        # Get metric name for filename
        metric_name = get_metric_name(title)

        entry = {
            "title": title_clean,
            "metric_name": metric_name,
            "data_matrix": data_matrix,
            "x_axis": x_axis,
            "y_axis": y_axis,
        }
        processed_datasets.append(entry)
        if metric_name not in metric_cache:
            metric_cache[metric_name] = entry

    regime_overlay = None
    train_entry = metric_cache.get("training_loss")
    test_entry = metric_cache.get("test_error") or metric_cache.get("test_loss")
    boundary_overlays = {}
    combined_train_test_overlay = None
    if train_entry and test_entry:
        positions = get_regime_positions_for_optimizer(optimizer_name)
        regime_overlay = build_regime_overlay_from_shape(
            train_entry["data_matrix"].shape,
            pos_frac=positions,
        )
        test_key = "test_error" if test_entry["metric_name"] != "test_loss" else "test_loss"
        boundary_data = {
            "training_loss": train_entry["data_matrix"],
            test_key: test_entry["data_matrix"],
        }
        boundary_config = dict(BOUNDARY_CONFIG)
        boundary_config["threshold_ratio"] = BOUNDARY_THRESHOLD_RATIO_MAP.get(
            (optimizer_name or "").lower(), BOUNDARY_THRESHOLD_RATIO
        )
        opt_key = (optimizer_name or "").lower()
        if opt_key in BOUNDARY_THRESHOLD_RATIO_METRIC_MAP:
            boundary_config["threshold_ratio_map"] = BOUNDARY_THRESHOLD_RATIO_METRIC_MAP[opt_key]
        boundary_overlays, combined_train_test_overlay, fused_overlay, mapping_info = build_boundary_overlays(
            boundary_data,
            config=boundary_config,
            metric_keys=("training_loss", test_key),
        )
        if mapping_info:
            print(f"[{optimizer_name}] 融合映射: train_idx={mapping_info.get('train_indices')} -> test_idx={mapping_info.get('test_indices')}")

    for entry in processed_datasets:
        metric_overlay = combined_train_test_overlay if entry["metric_name"] in ("training_loss", "test_error", "test_loss") else None
        overlay = merge_overlays(regime_overlay, metric_overlay)
        plot_2Dphase_optimizer(
            entry["data_matrix"],
            entry["x_axis"],
            entry["y_axis"],
            entry["title"],
            optimizer_name,
            entry["metric_name"],
            regime_overlay=overlay,
            boundary_data=boundary_data if train_entry and test_entry else None,
        )

    print(f"\nCompleted {optimizer_name}!")

print("\n" + "="*60)
print("All optimizer comparison plots have been generated!")
print("="*60)
