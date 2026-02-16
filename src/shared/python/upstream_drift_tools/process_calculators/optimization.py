"""Optimization helpers for the advanced plots tab."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, TypedDict, cast

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import differential_evolution, minimize

from .analysis_utils import evaluate_output


@dataclass
class OptimizationHistoryEntry:
    """Container tracking the progress of each optimization iteration."""

    iteration: int
    objective: float
    parameters: dict[str, float]


class OptimizationResults(TypedDict):
    """Results from an Adam-based optimization session."""

    best_output: float
    best_parameters: dict[str, float]
    best_state: dict[str, float]
    best_composition: dict[str, float]
    history: list[OptimizationHistoryEntry]
    final_parameters: dict[str, float]
    iterations: int


def _build_override_mapping(
    parameter_names: Sequence[str],
    values: Sequence[float],
) -> dict[str, float]:
    """Create a mapping of parameter names to their associated values."""

    override: dict[str, float] = {}
    for name, value in zip(parameter_names, values, strict=False):
        if name in {"Temperature", "O2/Feed Ratio", "Steam/Feed Ratio", "Pressure"}:
            override[name] = value
    return override


def run_adam_optimization(
    engine: Any,
    analysis_params: dict[str, object],
    manual_hhv: float,
    parameter_configs: Sequence[dict[str, Any]],
    *,
    maximize: bool,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    gradient_step: float,
    max_iterations: int,
    tolerance: float,
    gradient_tolerance: float | None = None,
) -> OptimizationResults:
    """Run an Adam-based search across operating parameters.

    Parameters
    ----------
    engine:
        Calculation engine capable of evaluating the thermodynamic outputs.
    analysis_params:
        Dictionary containing ``"base_params"`` with the baseline operating
        point and ``"output_variable"`` describing the objective to optimize.
    manual_hhv:
        Manually specified higher heating value supplied by the user [Btu/lb].
    parameter_configs:
        Sequence of dictionaries describing each parameter to optimize. Every
        entry must provide ``"name"`` (parameter label), ``"min"``/
        ``"max"`` bounds, and an ``"initial"`` starting guess.
    maximize:
        ``True`` to maximize the output, ``False`` to minimize.
    learning_rate, beta1, beta2, epsilon:
        Adam optimizer hyperparameters as defined by Kingma & Ba (2014).
    gradient_step:
        Finite difference step size used to approximate gradients within the
        bounded search domain.
    max_iterations:
        Maximum number of Adam iterations to perform.
    tolerance:
        Convergence tolerance applied to the L2 norm of successive parameter
        updates.
    gradient_tolerance:
        Optional convergence tolerance for the gradient norm. Defaults to the
        same value as ``tolerance`` when omitted.

    Returns
    -------
    Dict[str, object]
        Dictionary containing the best objective value, parameter set, resolved
        state, composition, the per-iteration history, the final parameter
        values, and the number of executed iterations.
    """

    if not parameter_configs:
        raise ValueError("At least one parameter must be provided for optimization")

    parameter_names = [cfg["name"] for cfg in parameter_configs]
    lower_bounds = np.array([cfg["min"] for cfg in parameter_configs], dtype=float)
    upper_bounds = np.array([cfg["max"] for cfg in parameter_configs], dtype=float)
    values = np.array([cfg["initial"] for cfg in parameter_configs], dtype=float)

    m = np.zeros_like(values)
    v = np.zeros_like(values)

    best_output = -np.inf if maximize else np.inf
    best_parameters: dict[str, float] = {}
    best_state: dict[str, float] = {}
    best_composition: dict[str, float] = {}
    history: list[OptimizationHistoryEntry] = []

    previous_values = values.copy()

    base_params = cast(dict[str, float], analysis_params["base_params"])
    output_name = cast(str, analysis_params["output_variable"])

    if gradient_tolerance is None:
        gradient_tolerance = tolerance

    for iteration in range(1, max_iterations + 1):
        overrides = _build_override_mapping(parameter_names, values.tolist())
        objective, composition, state = evaluate_output(
            engine,
            base_params,
            manual_hhv,
            output_name,
            overrides,
        )

        if not np.isfinite(objective):
            # Treat invalid evaluations as non-improving points.
            objective = -np.inf if maximize else np.inf
            composition = {}
            state = {}

        if np.isfinite(objective) and (
            (maximize and objective > best_output)
            or (not maximize and objective < best_output)
        ):
            best_output = objective
            best_parameters = overrides.copy()
            best_state = state
            best_composition = composition

        history.append(
            OptimizationHistoryEntry(
                iteration=iteration,
                objective=objective,
                parameters=overrides.copy(),
            )
        )

        gradient = np.zeros_like(values)
        if np.isfinite(objective):
            for index, cfg in enumerate(parameter_configs):
                lower = float(cfg["min"])
                upper = float(cfg["max"])
                if np.isclose(lower, upper):
                    continue

                step = max(gradient_step, (upper - lower) * 1e-3)
                if step <= 0:
                    continue

                current_value = values[index]
                at_lower = np.isclose(current_value, lower)
                at_upper = np.isclose(current_value, upper)

                if at_lower and at_upper:
                    continue

                if at_lower and not at_upper:
                    forward_value = np.clip(current_value + step, lower, upper)
                    if np.isclose(forward_value, current_value):
                        continue
                    forward = values.copy()
                    forward[index] = forward_value
                    overrides_forward = _build_override_mapping(
                        parameter_names, forward.tolist()
                    )
                    forward_objective, _, _ = evaluate_output(
                        engine,
                        base_params,
                        manual_hhv,
                        output_name,
                        overrides_forward,
                    )
                    if not np.isfinite(forward_objective):
                        continue
                    gradient[index] = (forward_objective - objective) / (
                        forward_value - current_value
                    )
                    continue

                if at_upper and not at_lower:
                    backward_value = np.clip(current_value - step, lower, upper)
                    if np.isclose(backward_value, current_value):
                        continue
                    backward = values.copy()
                    backward[index] = backward_value
                    overrides_backward = _build_override_mapping(
                        parameter_names, backward.tolist()
                    )
                    backward_objective, _, _ = evaluate_output(
                        engine,
                        base_params,
                        manual_hhv,
                        output_name,
                        overrides_backward,
                    )
                    if not np.isfinite(backward_objective):
                        continue
                    gradient[index] = (objective - backward_objective) / (
                        current_value - backward_value
                    )
                    continue

                plus = values.copy()
                minus = values.copy()
                plus[index] = np.clip(current_value + step, lower, upper)
                minus[index] = np.clip(current_value - step, lower, upper)

                if np.isclose(plus[index], minus[index]):
                    continue

                overrides_plus = _build_override_mapping(parameter_names, plus.tolist())
                overrides_minus = _build_override_mapping(
                    parameter_names, minus.tolist()
                )

                value_plus, _, _ = evaluate_output(
                    engine,
                    base_params,
                    manual_hhv,
                    output_name,
                    overrides_plus,
                )
                value_minus, _, _ = evaluate_output(
                    engine,
                    base_params,
                    manual_hhv,
                    output_name,
                    overrides_minus,
                )

                if not (np.isfinite(value_plus) and np.isfinite(value_minus)):
                    continue

                gradient[index] = (value_plus - value_minus) / (
                    plus[index] - minus[index]
                )

        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm < gradient_tolerance:
            break

        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient**2)

        m_hat = m / (1 - beta1**iteration)
        v_hat = v / (1 - beta2**iteration)

        direction = 1.0 if maximize else -1.0
        update = direction * learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        values = values + update
        values = np.clip(values, lower_bounds, upper_bounds)

        if np.linalg.norm(values - previous_values) < tolerance:
            break

        previous_values = values.copy()

    return {
        "best_output": best_output,
        "best_parameters": best_parameters,
        "best_state": best_state,
        "best_composition": best_composition,
        "history": history,
        "final_parameters": _build_override_mapping(parameter_names, values.tolist()),
        "iterations": len(history),
    }


# =============================================================================
# SURFACE OPTIMIZATION FUNCTIONS (Interpolation-based)
# =============================================================================

OPTIMIZATION_PENALTY_VALUE: Final[float] = 1e10


def find_optimal_on_surface(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    method: str = "Grid Search",
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    callback: Any | None = None,
) -> dict[str, Any]:
    """
    Find the optimal (maximum) point on a surface defined by grid data.

    Args:
        x_grid: 1D or 2D array of X coordinates.
        y_grid: 1D or 2D array of Y coordinates.
        z_grid: 2D array of Z values (shape matching grid).
        method: Optimization method ("Grid Search", "L-BFGS-B", "Differential Evolution").
        bounds: Optional ((min_x, max_x), (min_y, max_y)). Calculated from grid if None.
        callback: Optional callback(evaluations, total) for progress tracking.

    Returns:
        Dictionary containing 'optimal_x', 'optimal_y', 'optimal_z', etc.
    """
    # Prepare data for interpolation
    # Ensure 1D unique sorted arrays for RegularGridInterpolator
    x_vals = np.unique(x_grid) if x_grid.ndim > 1 else x_grid

    y_vals = np.unique(y_grid) if y_grid.ndim > 1 else y_grid

    # z_grid should be (x, y) or (y, x) depending on meshgrid convention
    # Usually meshgrid(x, y) produces shape (len(y), len(x)).
    # RegularGridInterpolator expects (len(x), len(y)) if x is first dim.
    # The AdvancedPlotsTab passed Z.T, implying Z was (y, x).
    # We will try to infer or assume Z corresponds to shape (len(y_vals), len(x_vals)).

    # If Z shape is (len(y), len(x)), we transpose for Interp((x, y), Z.T)
    if z_grid.shape == (len(y_vals), len(x_vals)):
        interpolator = RegularGridInterpolator(
            (x_vals, y_vals),
            z_grid.T,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
    elif z_grid.shape == (len(x_vals), len(y_vals)):
        interpolator = RegularGridInterpolator(
            (x_vals, y_vals),
            z_grid,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
    else:
        raise ValueError(f"Z grid shape {z_grid.shape} does not match X/Y dimensions.")

    if bounds is None:
        bounds = (
            (float(x_vals.min()), float(x_vals.max())),
            (float(y_vals.min()), float(y_vals.max())),
        )

    # Evaluation wrapper
    def objective(p: Any) -> float:  # Maximization -> Negative for minimization
        try:
            val = interpolator(p)
            # Handle potential 1-element array return
            if np.ndim(val) > 0:
                val = val.item()

            if np.isnan(val):
                return OPTIMIZATION_PENALTY_VALUE
            return -float(val)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            return OPTIMIZATION_PENALTY_VALUE

    if method == "Grid Search":
        return _run_grid_search(interpolator, bounds, callback)
    elif method == "L-BFGS-B":
        return _run_lbfgsb(objective, bounds, callback)
    elif method == "Differential Evolution":
        return _run_differential_evolution(objective, bounds, callback)
    else:
        raise ValueError(f"Unknown optimization method: {method}")


def _run_grid_search(interpolator: Any, bounds: Any, callback: Any) -> dict:
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    # Use 15x15 grid (225 points)
    n_x, n_y = 15, 15
    xs = np.linspace(x_min, x_max, n_x)
    ys = np.linspace(y_min, y_max, n_y)

    best_x, best_y, best_z = None, None, float("-inf")
    total_points = n_x * n_y
    count = 0

    for x in xs:
        for y in ys:
            count += 1
            if callback and count % 10 == 0:
                callback(count, total_points)

            try:
                val = interpolator([x, y])
                if np.ndim(val) > 0:
                    val = val.item()
                z = float(val)

                if not np.isnan(z) and z > best_z:
                    best_z = z
                    best_x = x
                    best_y = y
            except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                pass

    return {
        "success": best_x is not None,
        "optimal_x": best_x,
        "optimal_y": best_y,
        "optimal_z": best_z,
        "evaluations": count,
    }


def _run_lbfgsb(objective: Any, bounds: Any, callback: Any) -> dict:
    x_mean = (bounds[0][0] + bounds[0][1]) / 2
    y_mean = (bounds[1][0] + bounds[1][1]) / 2
    x0 = [x_mean, y_mean]

    eval_count = [0]

    def tracked_obj(p: Any) -> float:
        eval_count[0] += 1
        if callback and eval_count[0] % 5 == 0:
            callback(eval_count[0], 100)  # Unknown total
        return float(objective(p))

    res = minimize(
        tracked_obj,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 100, "ftol": 1e-6},
    )

    return {
        "success": res.success,
        "optimal_x": res.x[0],
        "optimal_y": res.x[1],
        "optimal_z": -res.fun,
        "evaluations": eval_count[0],
        "message": res.message,
    }


def _run_differential_evolution(objective: Any, bounds: Any, callback: Any) -> dict:
    eval_count = [0]

    def tracked_obj(p: Any) -> float:
        eval_count[0] += 1
        if callback and eval_count[0] % 10 == 0:
            callback(eval_count[0], 100)
        return float(objective(p))

    res = differential_evolution(
        tracked_obj, bounds, maxiter=50, popsize=10, atol=0.01, tol=0.01
    )

    return {
        "success": res.success,
        "optimal_x": res.x[0],
        "optimal_y": res.x[1],
        "optimal_z": -res.fun,
        "evaluations": eval_count[0],
        "message": res.message,
    }
