import copy
import numpy as np
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm  

def _fill_invalid_with_mean(data):
    """Replace non-finite entries with the finite mean to keep smoothing stable."""
    arr = np.array(data, dtype=float)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return arr
    finite_mean = np.nanmean(arr[finite_mask])
    arr[~finite_mask] = finite_mean
    return arr

def truncated_cmap(cmap_name, minval=0.0, maxval=1.0, n=256):
    """Return a copy of cmap_name trimmed to [minval, maxval] to avoid harsh ends."""
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(f"{cmap_name}_trunc", colors)


def generate_smooth_polygon(
    data_matrix,
    smoothing_sigma=1.0,
    threshold_ratio=0.2,
    use_binary_mask=False,
):
    """
    Smooth data with edge padding and extract a contour polygon around the peak.
    Padding the array before smoothing avoids artificial dips at the boundary so contours
    can run flush into the border instead of folding back prematurely.
    """
    arr = _fill_invalid_with_mean(data_matrix)
    if arr.size == 0:
        return None

    # Pad the array so high-value regions near the border are preserved during smoothing
    pad_width = 0
    if smoothing_sigma and smoothing_sigma > 0:
        pad_width = int(np.ceil(3 * smoothing_sigma))
    if pad_width > 0:
        padded = np.pad(arr, pad_width=((pad_width, pad_width), (pad_width, pad_width)), mode="edge")
    else:
        padded = arr

    smoothed = gaussian_filter(padded, sigma=smoothing_sigma) if smoothing_sigma and smoothing_sigma > 0 else padded
    max_idx_padded = np.unravel_index(np.nanargmax(smoothed, axis=None), smoothed.shape)
    max_val = smoothed[max_idx_padded]
    if not np.isfinite(max_val) or max_val <= 0:
        return None

    threshold_val = max_val * threshold_ratio
    if use_binary_mask:
        # Build binary mask then find its boundary at 0.5 to avoid altering the base heatmap
        binary_mask = (smoothed >= threshold_val).astype(float)
        contours = measure.find_contours(binary_mask, 0.5)
    else:
        contours = measure.find_contours(smoothed, threshold_val)

    if not contours:
        return None

    # Convert peak location back to the unpadded canvas for downstream plotting/selection
    peak_r = max_idx_padded[0] - pad_width
    peak_c = max_idx_padded[1] - pad_width
    peak_r_clipped = int(np.clip(peak_r, 0, arr.shape[0] - 1))
    peak_c_clipped = int(np.clip(peak_c, 0, arr.shape[1] - 1))

    target_polygon = None
    shifted_contours = []
    for contour in contours:
        if contour.shape[0] < 2:
            continue
        shifted = contour.copy()
        if pad_width > 0:
            shifted[:, 0] -= pad_width
            shifted[:, 1] -= pad_width
        shifted_contours.append(shifted)
        # Use the shifted coordinates (row, col order) for containment check
        if Path(shifted).contains_point((peak_r, peak_c)):
            target_polygon = shifted
            break

    if target_polygon is None and shifted_contours:
        target_polygon = max(shifted_contours, key=lambda c: c.shape[0])

    if target_polygon is None:
        return None

    # Keep only the portion of the path that lies on the original canvas so it "cuts off" cleanly at the border.
    n_rows, n_cols = arr.shape
    inside_mask = (
        (target_polygon[:, 0] >= -0.5)
        & (target_polygon[:, 0] <= n_rows - 0.5)
        & (target_polygon[:, 1] >= -0.5)
        & (target_polygon[:, 1] <= n_cols - 0.5)
    )
    if np.any(inside_mask):
        target_polygon = target_polygon[inside_mask]

    if target_polygon.shape[0] < 2:
        return None

    return {
        "polygon": target_polygon,
        "start_point": (peak_c_clipped, peak_r_clipped),  # convert to (x, y) for plotting
    }


def extract_boundary_polygon(
    data_matrix,
    smoothing_sigma=1.0,
    threshold_ratio=0.2,
    use_binary_mask=False,
):
    """
    Backwards-compatible wrapper that delegates to generate_smooth_polygon.
    """
    return generate_smooth_polygon(
        data_matrix,
        smoothing_sigma=smoothing_sigma,
        threshold_ratio=threshold_ratio,
        use_binary_mask=use_binary_mask,
    )


def polygon_to_overlay(polygon_dict, line_style):
    """Convert raw contour output to plotting overlay entries."""
    if not polygon_dict:
        return None
    polygon = polygon_dict.get("polygon")
    start_point = polygon_dict.get("start_point")
    if polygon is None:
        return None
    return {
        "lines": [],
        "labels": {},
        "polygons": [{"x": polygon[:, 1], "y": polygon[:, 0], "style": line_style}],
        "points": [start_point] if start_point is not None else [],
        "fused_lines": [],
    }


def merge_overlays(base_overlay, extra_overlay):
    """Combine regime overlays with boundary overlays without mutating inputs."""
    if not base_overlay and not extra_overlay:
        return None
    merged = {"lines": [], "labels": {}, "polygons": [], "points": [], "fused_lines": []}
    for overlay in (base_overlay, extra_overlay):
        if not overlay:
            continue
        merged["lines"].extend(overlay.get("lines", []))
        merged["polygons"].extend(overlay.get("polygons", []))
        merged["points"].extend(overlay.get("points", []))
        merged["labels"].update(overlay.get("labels", {}))
        merged["fused_lines"].extend(overlay.get("fused_lines", []))
    return merged


def compute_fused_boundary(poly_a_xy, poly_b_xy, proximity_threshold):
    """
    Compute midpoint curve where two polygons are within a proximity threshold.
    Poly coords are expected as Nx2 arrays in (x, y) order.
    """
    if poly_a_xy is None or poly_b_xy is None:
        return None
    if len(poly_a_xy) == 0 or len(poly_b_xy) == 0:
        return None
    if not np.isfinite(proximity_threshold) or proximity_threshold <= 0:
        return None

    tree_b = cKDTree(poly_b_xy)
    dists, idxs = tree_b.query(poly_a_xy)
    mask = np.isfinite(dists) & (dists <= proximity_threshold)
    if not np.any(mask):
        return None
    midpoints = 0.5 * (poly_a_xy[mask] + poly_b_xy[idxs[mask]])
    a_indices = np.nonzero(mask)[0]
    b_indices = idxs[mask]
    return {"midpoints": midpoints, "a_indices": a_indices, "b_indices": b_indices}


def build_fused_overlay(
    train_poly_dict,
    test_poly_dict,
    proximity_threshold,
    train_line_style,
):
    """Build overlay for fused boundary between train/test polygons."""
    if not train_poly_dict or not test_poly_dict:
        return None
    train_poly = train_poly_dict.get("polygon")
    test_poly = test_poly_dict.get("polygon")
    if train_poly is None or test_poly is None:
        return None

    # Convert (row, col) to (x, y)
    train_xy = np.column_stack((train_poly[:, 1], train_poly[:, 0]))
    test_xy = np.column_stack((test_poly[:, 1], test_poly[:, 0]))
    fused_info = compute_fused_boundary(train_xy, test_xy, proximity_threshold)
    if not fused_info:
        return None
    fused_points = fused_info["midpoints"]
    if fused_points is None or fused_points.size == 0:
        return None

    patched_train_xy = np.array(train_xy, copy=True)
    if fused_info.get("a_indices") is not None:
        patched_train_xy[fused_info["a_indices"]] = fused_points
    return {
        "lines": [],
        "labels": {},
        "polygons": [{"x": patched_train_xy[:, 0], "y": patched_train_xy[:, 1], "style": train_line_style}],
        "points": [],
        "fused_lines": [{"x": fused_points[:, 0], "y": fused_points[:, 1]}],
        "mapping": {
            "train_indices": fused_info.get("a_indices", []),
            "test_indices": fused_info.get("b_indices", []),
        },
    }


def build_boundary_overlays(data, config, metric_keys=("training_loss", "test_error")):
    """
    Build boundary overlays and fused overlays for the given metrics.
    Returns (boundary_overlays, combined_train_test_overlay, fused_overlay, mapping_info)
    """
    boundary_overlays = {}
    boundary_raw_polygons = {}
    line_style_map = config.get("line_style_map", {})
    default_style = config.get("line_style_default")
    mapping_info = None
    threshold_ratio_map = config.get("threshold_ratio_map", {})
    default_threshold = config.get("threshold_ratio", 0.2)

    for metric_key in metric_keys:
        if metric_key in data:
            threshold_ratio = threshold_ratio_map.get(metric_key, default_threshold)
            poly_dict = extract_boundary_polygon(
                data[metric_key],
                smoothing_sigma=config.get("smoothing_sigma", 1.0),
                threshold_ratio=threshold_ratio,
                use_binary_mask=config.get("use_binary_mask", False),
            )
            style = line_style_map.get(metric_key, default_style)
            boundary_overlay = polygon_to_overlay(poly_dict, line_style=style) if poly_dict else None
            if poly_dict:
                boundary_raw_polygons[metric_key] = poly_dict
            if boundary_overlay:
                boundary_overlays[metric_key] = boundary_overlay

    fused_overlay = None
    if config.get("fuse_enabled"):
        fused_overlay = build_fused_overlay(
            boundary_raw_polygons.get("training_loss"),
            boundary_raw_polygons.get("test_error"),
            proximity_threshold=config.get("fuse_threshold", 10.0),
            train_line_style=line_style_map.get("training_loss", default_style),
        )
        if fused_overlay and fused_overlay.get("mapping"):
            mapping_info = fused_overlay["mapping"]
        if fused_overlay and boundary_overlays.get("training_loss") and fused_overlay.get("polygons"):
            # Replace train polygon with patched version in downstream overlays
            boundary_overlays["training_loss"]["polygons"] = fused_overlay["polygons"]

    combined_train_test_overlay = merge_overlays(
        boundary_overlays.get("training_loss"),
        boundary_overlays.get("test_error"),
    )
    combined_train_test_overlay = merge_overlays(combined_train_test_overlay, fused_overlay)

    return boundary_overlays, combined_train_test_overlay, fused_overlay, mapping_info
