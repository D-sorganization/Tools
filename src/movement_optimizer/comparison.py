# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Trial comparison storage and metrics computation.

Allows users to store multiple optimization runs and compare them
side-by-side with overlaid plots and summary metrics.

Design Principles:
    DBC  -- preconditions documented and checked.
    DRY  -- metrics extraction is a single function.
    LoD  -- store only exposes add/get/clear.
"""

from __future__ import annotations

import copy
import dataclasses
import threading
from typing import Any

import numpy as np

from .constants import trapezoid
from .trajectory import OptimizationResult


def _frozen_array(value: np.ndarray) -> np.ndarray:
    """Return an isolated read-only copy of an array.

    Arrays are copied unconditionally before setting ``writeable=False``
    so that neither caller flag-resetting after ingest nor consumer
    flag-resetting after retrieval can alias and mutate stored data.
    """
    frozen = np.array(value, copy=True)
    frozen.flags.writeable = False
    return frozen


def _snapshot_result(result: OptimizationResult) -> OptimizationResult:
    """Return a copy of ``result`` whose NumPy fields are frozen private copies."""
    snapshot = copy.copy(result)
    for field in dataclasses.fields(snapshot):
        value = getattr(snapshot, field.name)
        if isinstance(value, np.ndarray):
            setattr(snapshot, field.name, _frozen_array(value))
    return snapshot


def _snapshot_trial(
    name: str,
    result: OptimizationResult,
    body_params: dict[str, Any],
    bar_mass: float,
) -> dict[str, Any]:
    """Return an isolated trial record.

    Trajectory arrays are copied unconditionally and frozen (``writeable=False``),
    ensuring that neither the caller nor a ``get_trials()`` consumer can mutate
    stored data in place even if NumPy array writeability flags are toggled back.
    """
    return {
        "name": name,
        "result": _snapshot_result(result),
        "body_params": copy.deepcopy(body_params),
        "bar_mass": bar_mass,
    }


class ComparisonStore:
    """Thread-safe store for named optimization trials.

    Each trial contains a name, OptimizationResult, body params, and bar mass.
    """

    def __init__(self) -> None:
        """Initialise an empty comparison store with a reentrant lock."""
        self._lock = threading.Lock()
        self._trials: list[dict[str, Any]] = []

    def add_trial(
        self,
        name: str,
        result: OptimizationResult,
        body_params: dict[str, Any],
        bar_mass: float,
    ) -> None:
        """Add a named trial to the comparison list.

        Preconditions:
            name is a non-empty string.
            result is a valid OptimizationResult.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Trial name must be a non-empty string.")
        if not isinstance(result, OptimizationResult):
            raise TypeError("result must be an OptimizationResult.")
        snapshot = _snapshot_trial(name, result, body_params, bar_mass)
        with self._lock:
            self._trials.append(snapshot)

    def get_trials(self) -> list[dict[str, Any]]:
        """Return defensive snapshots of the stored trials list."""
        with self._lock:
            return [
                _snapshot_trial(
                    t["name"],
                    t["result"],
                    t["body_params"],
                    t["bar_mass"],
                )
                for t in self._trials
            ]

    def clear(self) -> None:
        """Remove all stored trials."""
        with self._lock:
            self._trials.clear()


def comparison_metrics(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute summary metrics for each trial.

    Returns a list of dicts, one per trial, each containing:
        name: trial name
        peak_torques: list of 3 floats (max abs torque per joint, Nm)
        total_work: float (net-power-magnitude integral, integral of |sum(power)| dt, in J; not metabolic cost)
        abs_joint_work_j: float (summed absolute joint work, integral of sum(|power|) dt, in J; not metabolic cost)
        com_sway_cm: float (horizontal COM range in cm)

    Preconditions:
        Each trial dict has keys: name, result, body_params, bar_mass.
    """
    if not trials:
        return []

    metrics = []
    for trial in trials:
        r: OptimizationResult = trial["result"]
        peak_torques = [float(np.max(np.abs(r.torques[:, j]))) for j in range(3)]
        # Net-power-magnitude integral: sums joint powers per timestep first
        # (capturing inter-joint power transfer) before integrating |P_net|.
        total_work = float(trapezoid(np.abs(np.sum(r.power, axis=1)), r.t))
        # Summed absolute joint work: integrates sum_j |P_j(t)| without
        # inter-joint cancellation. Neither metric is metabolic cost.
        abs_joint_work_j = float(trapezoid(np.sum(np.abs(r.power), axis=1), r.t))
        com_sway_cm = float(r.com_horizontal_range_cm)

        metrics.append(
            {
                "name": trial["name"],
                "peak_torques": peak_torques,
                "total_work": total_work,
                "abs_joint_work_j": abs_joint_work_j,
                "com_sway_cm": com_sway_cm,
            }
        )

    return metrics
