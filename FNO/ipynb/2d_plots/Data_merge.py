from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Iterable, List, Mapping, MutableSequence, Sequence

# Allowed axis values provided by the user
ALLOWED_X: set[float] = {float(v) for v in [5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 70]}
ALLOWED_Y: set[float] = {float(v) for v in [100, 150, 200, 250, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]}

# Wave-specific allowed axes
WAVE_ALLOWED_X: set[float] = {float(v) for v in [0.1, 0.5, 1, 2, 3, 4, 5, 6]}
WAVE_ALLOWED_Y: set[float] = {float(v) for v in [50, 100, 250, 500, 1000, 2000, 5000, 10000]}


def _indices_to_keep(values: Sequence[float], allowed: set[float]) -> List[int]:
    """Return indices for items that are present in the allowed set (float compare)."""
    return [idx for idx, val in enumerate(values) if float(val) in allowed]


def _filter_matrix(matrix: Sequence[Sequence], keep_rows: Iterable[int], keep_cols: Iterable[int]) -> List[List]:
    keep_rows_set = set(keep_rows)
    keep_cols_list = list(keep_cols)
    return [
        [row[col] for col in keep_cols_list]
        for r_idx, row in enumerate(matrix)
        if r_idx in keep_rows_set
    ]


def _validate_matrix_shape(matrix: Sequence[Sequence], expected_rows: int, expected_cols: int, name: str) -> None:
    if len(matrix) != expected_rows:
        raise ValueError(f"{name} expected {expected_rows} rows, found {len(matrix)}")
    for r_idx, row in enumerate(matrix):
        if len(row) != expected_cols:
            raise ValueError(f"{name} row {r_idx} expected {expected_cols} columns, found {len(row)}")


def clean_experiment_json(
    input_path: Path,
    output_path: Path | None = None,
    *,
    allowed_x: set[float] | None = None,
    allowed_y: set[float] | None = None,
) -> dict:
    """
    Drop rows/columns whose x_axis or y_axis are not in the allowed lists.
    Returns a small summary describing what changed.
    """
    data_path = Path(input_path)
    data = json.loads(data_path.read_text())

    x_axis = data.get("x_axis", [])
    y_axis = data.get("y_axis", [])

    allowed_x = allowed_x or ALLOWED_X
    allowed_y = allowed_y or ALLOWED_Y

    keep_x = _indices_to_keep(x_axis, allowed_x)
    keep_y = _indices_to_keep(y_axis, allowed_y)

    cleaned: dict = {
        "x_axis": [x_axis[i] for i in keep_x],
        "y_axis": [y_axis[i] for i in keep_y],
    }

    # Assert original matrices align with the provided axes so we do not silently drop mismatched data.
    expected_rows = len(y_axis)
    expected_cols = len(x_axis)

    for key, value in data.items():
        if key in ("x_axis", "y_axis"):
            continue

        if isinstance(value, list) and value and isinstance(value[0], list):
            _validate_matrix_shape(value, expected_rows, expected_cols, key)
            cleaned[key] = _filter_matrix(value, keep_rows=keep_y, keep_cols=keep_x)
        elif isinstance(value, list):
            # 1D arrays fall back to row- or column-based filtering if the length matches.
            if len(value) == expected_rows:
                cleaned[key] = [value[i] for i in keep_y]
            elif len(value) == expected_cols:
                cleaned[key] = [value[i] for i in keep_x]
            else:
                cleaned[key] = value
        else:
            cleaned[key] = value

    out_path = output_path or data_path
    Path(out_path).write_text(json.dumps(cleaned, indent=2))

    return {
        "input": str(data_path),
        "output": str(out_path),
        "dropped_x": len(x_axis) - len(keep_x),
        "dropped_y": len(y_axis) - len(keep_y),
        "kept_x": len(keep_x),
        "kept_y": len(keep_y),
    }


def _build_axis_union(primary_axis: Sequence[float], secondary_axis: Sequence[float]) -> List[float]:
    """Keep the primary axis order, append any missing values from the secondary axis."""
    seen = {float(v) for v in primary_axis}
    merged = list(primary_axis)
    for val in secondary_axis:
        if float(val) not in seen:
            merged.append(val)
            seen.add(float(val))
    return merged


def _empty_matrix(rows: int, cols: int) -> List[List[None]]:
    return [[None for _ in range(cols)] for _ in range(rows)]


def _sort_axes_and_reorder_matrices(data: Mapping) -> Mapping:
    """
    Sort axes in ascending order and reorder all matrices/vectors to match.
    This keeps consistency between x_axis and y_axis ordering.
    """
    x_axis = data.get("x_axis", [])
    y_axis = data.get("y_axis", [])

    x_order = sorted(range(len(x_axis)), key=lambda i: float(x_axis[i]))
    y_order = sorted(range(len(y_axis)), key=lambda i: float(y_axis[i]))

    # If already sorted, avoid unnecessary work.
    if x_order == list(range(len(x_axis))) and y_order == list(range(len(y_axis))):
        return data

    def reorder_matrix(matrix: Sequence[Sequence]) -> List[List]:
        return [[matrix[r][c] for c in x_order] for r in y_order]

    def reorder_vector(vector: Sequence) -> List:
        if len(vector) == len(x_axis):
            return [vector[i] for i in x_order]
        if len(vector) == len(y_axis):
            return [vector[i] for i in y_order]
        return list(vector)

    reordered: dict = {"x_axis": [x_axis[i] for i in x_order], "y_axis": [y_axis[i] for i in y_order]}

    for key, value in data.items():
        if key in {"x_axis", "y_axis"}:
            continue
        if isinstance(value, list) and value and isinstance(value[0], list):
            reordered[key] = reorder_matrix(value)
        elif isinstance(value, list):
            reordered[key] = reorder_vector(value)
        else:
            reordered[key] = deepcopy(value)

    return reordered


def _fill_matrix(
    target: MutableSequence[MutableSequence],
    source_matrix: Sequence[Sequence],
    x_axis: Sequence[float],
    y_axis: Sequence[float],
    x_index: Mapping[float, int],
    y_index: Mapping[float, int],
) -> None:
    """Copy values from the source matrix into the target grid using axis lookups."""
    if not source_matrix:
        return
    _validate_matrix_shape(source_matrix, len(y_axis), len(x_axis), "source_matrix")
    for y_i, y_val in enumerate(y_axis):
        for x_i, x_val in enumerate(x_axis):
            val = source_matrix[y_i][x_i]
            if val is None:
                continue
            target[y_index[float(y_val)]][x_index[float(x_val)]] = val


def merge_aggregated_into_experiment(
    aggregated: Mapping,
    experiment: Mapping,
    *,
    train_key: str = "training_loss",
    test_key: str = "test_error",
    aggregated_train_key: str = "train_loss_matrix",
    aggregated_test_key: str = "test_l2re_matrix",
) -> Mapping:
    """
    Merge aggregated train/test matrices into experiment data.
    Keeps existing experiment values, fills missing cells with aggregated values,
    and pads other metrics with null so every matrix has the same dimensions.
    """
    agg_x = aggregated.get("x_axis", [])
    agg_y = aggregated.get("y_axis", [])
    exp_x = experiment.get("x_axis", [])
    exp_y = experiment.get("y_axis", [])

    merged_x = _build_axis_union(agg_x, exp_x)
    merged_y = _build_axis_union(agg_y, exp_y)

    x_index = {float(v): idx for idx, v in enumerate(merged_x)}
    y_index = {float(v): idx for idx, v in enumerate(merged_y)}

    def new_grid() -> List[List[None]]:
        return _empty_matrix(len(merged_y), len(merged_x))

    merged: dict = {"x_axis": merged_x, "y_axis": merged_y}

    # training_loss
    merged_train = new_grid()
    _fill_matrix(merged_train, experiment.get(train_key, []), exp_x, exp_y, x_index, y_index)
    _fill_matrix(merged_train, aggregated.get(aggregated_train_key, []), agg_x, agg_y, x_index, y_index)
    merged[train_key] = merged_train

    # test_error
    merged_test = new_grid()
    _fill_matrix(merged_test, experiment.get(test_key, []), exp_x, exp_y, x_index, y_index)
    _fill_matrix(merged_test, aggregated.get(aggregated_test_key, []), agg_x, agg_y, x_index, y_index)
    merged[test_key] = merged_test

    # Other experiment metrics, padded to the merged grid.
    for key, value in experiment.items():
        if key in {"x_axis", "y_axis", train_key, test_key}:
            continue
        if isinstance(value, list) and value and isinstance(value[0], list):
            grid = new_grid()
            _fill_matrix(grid, value, exp_x, exp_y, x_index, y_index)
            merged[key] = grid
        else:
            # Non-matrix fields are copied verbatim.
            merged[key] = deepcopy(value)

    return _sort_axes_and_reorder_matrices(merged)


def save_json(data: Mapping, path: Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2))


def merge_all() -> None:
    base_dir = Path(__file__).parent / "data" / "xiaopeng"
    aggregated = json.loads((base_dir / "aggregated_metrics.json").read_text())

    tasks = [
        # ("beta", base_dir / "experiment_data_convection.json"),
        ("c", base_dir / "experiment_data_wave.json"),
        # ("rho", base_dir / "experiment_data_reaction.json"),
    ]

    for agg_key, experiment_path in tasks:
        experiment_data = json.loads(experiment_path.read_text())
        
        merged = merge_aggregated_into_experiment(aggregated[agg_key], experiment_data)
        save_json(merged, experiment_path)
        print(f"Merged aggregated '{agg_key}' into {experiment_path}")
        
        if experiment_path.name == "experiment_data_wave.json":
            # Enforce wave-specific axis whitelist before merging.
            summary = clean_experiment_json(
                experiment_path,
                output_path=experiment_path,
                allowed_x=WAVE_ALLOWED_X,
                allowed_y=WAVE_ALLOWED_Y,
            )
            # Reload after cleaning so merge sees trimmed axes.
            experiment_data = json.loads(experiment_path.read_text())
            print(
                f"Cleaned wave axes for {experiment_path} "
                f"(dropped x: {summary['dropped_x']}, dropped y: {summary['dropped_y']})"
            )
        elif experiment_path.name == "experiment_data_convection.json":
            # Enforce wave-specific axis whitelist before merging.
            summary = clean_experiment_json(
                experiment_path,
                output_path=experiment_path,
                allowed_x=ALLOWED_X,
                allowed_y=ALLOWED_Y,
            )
            # Reload after cleaning so merge sees trimmed axes.
            experiment_data = json.loads(experiment_path.read_text())
            print(
                f"Cleaned wave axes for {experiment_path} "
                f"(dropped x: {summary['dropped_x']}, dropped y: {summary['dropped_y']})"
            )
            


if __name__ == "__main__":
    merge_all()
