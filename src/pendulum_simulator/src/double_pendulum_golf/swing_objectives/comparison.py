"""Controlled comparison of the competing swing objectives.

Solves the same downswing once per objective and then scores **every** resulting
swing against **every** objective. The diagonal of that matrix is 100% by
construction; what the comparison is actually about is how close the off-diagonal
entries get, because that is the difference between "these mechanisms are
alternative routes to speed" and "these mechanisms are what speed is made of".

Two guards make the table trustworthy:

* Each swing must lead its own column. If it does not, the solver returned a
  local optimum and nothing in the table can be relied on.
* Torque saturation is reported per swing. Objectives that agree only because
  every one of them pinned the torques to their bounds have not agreed about
  anything.

Closes #4770.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from double_pendulum_golf.swing_objectives.downswing import (
    DownswingConfig,
    DownswingOptimizer,
    DownswingResult,
)
from double_pendulum_golf.swing_objectives.objectives import (
    SWING_OBJECTIVES,
    get_objective,
)

__all__ = [
    "COMPARISON_SCHEMA_VERSION",
    "SwingComparison",
    "compare_objectives",
    "cross_evaluation_matrix",
    "comparison_to_payload",
    "comparison_from_payload",
]

FloatArray = npt.NDArray[np.float64]

#: Versioned wire identity. Consumers must reject any other value.
COMPARISON_SCHEMA_VERSION = "1.0.0"

_PERCENT = 100.0
_MIN_OBJECTIVES = 2
_SATURATION_ATOL = 1e-6

#: Two swings whose torque profiles differ by less than this fraction of the
#: torque budget (RMS) are treated as the same swing. Below it the feasible set
#: has collapsed and the objectives were never given a choice to make.
_DEGENERACY_RMS_FRACTION = 1e-3


@dataclass(frozen=True, slots=True)
class SwingComparison:
    """Result of optimizing one downswing against several objectives.

    Attributes:
        objective_keys: Objectives compared, in table order.
        raw_values: ``raw_values[row][column]`` is the value objective *column*
            attains on the swing optimized for objective *row*, in that
            objective's own units.
        matrix: The same table normalized so each column's best is 100.
        torque_saturation: Per swing, the fraction of samples at the hub and
            wrist torque bounds.
        swing_distance: Symmetric matrix of RMS torque differences between
            swings, as a fraction of the torque budget. Near-zero everywhere
            means the objectives all had to produce the same swing.
        diagnostics: Per swing, the solver evidence a reader needs to decide
            whether to trust the row. Held as plain data rather than derived
            from :attr:`results` so a comparison restored from the wire remains
            fully self-describing.
        results: Optimization result per objective key. Empty on a comparison
            restored from a payload, which carries the tables but not the
            trajectories.
    """

    objective_keys: tuple[str, ...]
    raw_values: Mapping[str, Mapping[str, float]]
    matrix: FloatArray
    torque_saturation: Mapping[str, FloatArray]
    swing_distance: FloatArray
    diagnostics: Mapping[str, Mapping[str, float | bool | int]]
    results: Mapping[str, DownswingResult] = field(default_factory=dict)

    @property
    def max_swing_distance(self) -> float:
        """Largest RMS torque difference between any two compared swings."""
        return float(np.max(self.swing_distance))

    @property
    def is_degenerate(self) -> bool:
        """Whether every objective was forced to the same swing.

        When the constraints pin the trajectory — a downswing close to the
        golfer's minimum duration is the usual cause — the feasible set collapses
        and every objective returns the identical answer. The resulting table of
        100% entries looks like unanimous agreement between the mechanisms but is
        an artifact of the configuration, so callers must check this before
        reporting agreement as a finding.
        """
        return self.max_swing_distance < _DEGENERACY_RMS_FRACTION


def _validated_keys(objective_keys: Sequence[str] | None) -> tuple[str, ...]:
    """Resolve and validate the objective list.

    Pre: every key names a registered objective.
    Post: at least two distinct keys are returned, in the requested order.
    """
    keys = tuple(objective_keys) if objective_keys is not None else tuple(SWING_OBJECTIVES)
    for key in keys:
        get_objective(key)
    if len(set(keys)) < _MIN_OBJECTIVES:
        raise ValueError(f"a comparison needs at least two distinct objectives, got {keys}")
    return keys


def _torque_saturation(result: DownswingResult, config: DownswingConfig) -> FloatArray:
    """Fraction of samples sitting on each torque bound."""
    limits = config.torque_limit_vector
    at_bound = np.abs(np.abs(result.torques) - limits) < _SATURATION_ATOL
    fractions: FloatArray = np.mean(at_bound, axis=0)
    return fractions


def cross_evaluation_matrix(
    raw_values: Mapping[str, Mapping[str, float]], objective_keys: Sequence[str]
) -> FloatArray:
    """Normalize a raw score table so each column's best becomes 100.

    Args:
        raw_values: ``raw_values[row][column]`` in each objective's own units.
        objective_keys: Table order for both axes.

    Returns:
        Square percentage matrix.

    Pre: every row/column pair is present and finite, and each column has a
    strictly positive best value.
    Post: the returned matrix is finite with a maximum of 100 per column.
    """
    raw = np.array(
        [[raw_values[row][column] for column in objective_keys] for row in objective_keys],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(raw)):
        raise ValueError("cross-evaluation values must all be finite")

    column_best = raw.max(axis=0)
    if np.any(column_best <= 0.0):
        raise ValueError(
            "every objective must have a positive best value to normalize against; "
            f"got column maxima {column_best}"
        )
    matrix: FloatArray = raw / column_best * _PERCENT
    return matrix


def compare_objectives(
    config: DownswingConfig, objective_keys: Sequence[str] | None = None
) -> SwingComparison:
    """Solve the same downswing for each objective and cross-score the results.

    Args:
        config: The conditions held fixed across every objective.
        objective_keys: Objectives to compare. Defaults to all five.

    Returns:
        The completed comparison.

    Raises:
        KeyError: If an objective key is not registered.
        ValueError: If fewer than two distinct objectives are requested.

    Pre: ``config`` validated at construction.
    Post: every reported swing is the optimizer's answer for its own objective.
    """
    keys = _validated_keys(objective_keys)
    optimizer = DownswingOptimizer(config)

    results = {key: optimizer.solve(key) for key in keys}
    raw_values = {
        row: {column: get_objective(column).evaluate(results[row].signals) for column in keys}
        for row in keys
    }
    return SwingComparison(
        objective_keys=keys,
        raw_values=raw_values,
        matrix=cross_evaluation_matrix(raw_values, keys),
        torque_saturation={key: _torque_saturation(results[key], config) for key in keys},
        swing_distance=_swing_distance_matrix(results, keys, config),
        diagnostics={key: _diagnostics(results[key]) for key in keys},
        results=results,
    )


def _swing_distance_matrix(
    results: Mapping[str, DownswingResult],
    objective_keys: Sequence[str],
    config: DownswingConfig,
) -> FloatArray:
    """RMS torque difference between every pair of swings, budget-normalized."""
    budget = float(np.mean(config.torque_limit_vector))
    profiles = np.array([results[key].torques for key in objective_keys])
    count = len(objective_keys)
    distance = np.zeros((count, count), dtype=np.float64)
    for row in range(count):
        for column in range(row + 1, count):
            rms = float(np.sqrt(np.mean((profiles[row] - profiles[column]) ** 2)))
            distance[row, column] = distance[column, row] = rms / budget
    return distance


def _diagnostics(result: DownswingResult) -> dict[str, float | bool | int]:
    """Solver evidence for one row of the comparison."""
    return {
        "objective_value": float(result.objective_value),
        "success": bool(result.success),
        "max_defect": float(result.max_defect),
        "max_slew_violation": float(result.max_slew_violation),
        "iterations": int(result.iterations),
    }


def comparison_to_payload(comparison: SwingComparison) -> dict[str, Any]:
    """Serialize a comparison to a versioned, JSON-safe payload.

    Trajectories are deliberately excluded: the wire carries the comparison, not
    the swings, so it stays small enough to embed in a report.
    """
    keys = list(comparison.objective_keys)
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "objective_keys": keys,
        "units": {key: get_objective(key).units for key in keys},
        "raw_values": {
            row: {column: float(comparison.raw_values[row][column]) for column in keys}
            for row in keys
        },
        "matrix": [[float(value) for value in row] for row in comparison.matrix],
        "torque_saturation": {
            key: [float(value) for value in comparison.torque_saturation[key]] for key in keys
        },
        "swing_distance": [
            [float(value) for value in row] for row in comparison.swing_distance
        ],
        "is_degenerate": bool(comparison.is_degenerate),
        "diagnostics": {key: dict(comparison.diagnostics[key]) for key in keys},
    }


def _require_schema_version(payload: Mapping[str, Any]) -> None:
    """Reject payloads that are unversioned or from a different schema."""
    version = payload.get("schema_version")
    if version != COMPARISON_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {COMPARISON_SCHEMA_VERSION!r}, got {version!r}"
        )


def comparison_from_payload(payload: Mapping[str, Any]) -> SwingComparison:
    """Rebuild a comparison from a payload, failing closed on any drift.

    The optimization results themselves are not reconstructed — a payload carries
    the comparison, not the trajectories — so ``results`` is empty on a restored
    object and the numeric tables are authoritative.

    Raises:
        ValueError: If the schema version, key list, or matrix shape is wrong.
    """
    _require_schema_version(payload)
    keys = tuple(payload.get("objective_keys") or ())
    if len(set(keys)) < _MIN_OBJECTIVES:
        raise ValueError(f"payload must name at least two objectives, got {keys}")
    for key in keys:
        get_objective(key)

    matrix = np.asarray(payload.get("matrix", []), dtype=np.float64)
    if matrix.shape != (len(keys), len(keys)):
        raise ValueError(
            f"matrix must be {len(keys)}x{len(keys)} to match objective_keys, "
            f"got shape {matrix.shape}"
        )

    raw_values = payload.get("raw_values", {})
    saturation = payload.get("torque_saturation", {})
    diagnostics = payload.get("diagnostics", {})
    distance = np.asarray(payload.get("swing_distance", []), dtype=np.float64)
    if distance.shape != (len(keys), len(keys)):
        raise ValueError(
            f"swing_distance must be {len(keys)}x{len(keys)}, got {distance.shape}"
        )
    return SwingComparison(
        objective_keys=keys,
        raw_values={
            row: {column: float(raw_values[row][column]) for column in keys} for row in keys
        },
        matrix=matrix,
        torque_saturation={key: np.asarray(saturation[key], dtype=np.float64) for key in keys},
        swing_distance=distance,
        diagnostics={key: dict(diagnostics[key]) for key in keys},
    )
