"""
Reusable data extractor for simulation results.

Provides a registry of named data series that can be extracted from any
simulation result object.  Used by the pop-out chart data selector and
the data table module.

Design by Contract
------------------
- extract_series() returns (values, unit_label) or raises KeyError.
- list_available_series() returns all valid series names for a result type.
- All returned arrays are shape (N,) and finite.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry of extractable data series
# ---------------------------------------------------------------------------

# Each entry: (description, unit_label, extractor_function)
# extractor_function(result) -> np.ndarray, shape (N,)

_DOUBLE_SERIES: dict[str, tuple[str, str]] = {
    "time": ("Time", "s"),
    "theta1": ("Shoulder angle θ₁", "rad"),
    "phi": ("Wrist angle φ", "rad"),
    "dtheta1": ("Shoulder velocity θ̇₁", "rad/s"),
    "dphi": ("Wrist velocity φ̇", "rad/s"),
    "torque_shoulder": ("Shoulder torque", "N·m"),
    "torque_wrist": ("Wrist torque", "N·m"),
    "total_torque_shoulder": ("Total shoulder torque", "N·m"),
    "total_torque_wrist": ("Total wrist torque", "N·m"),
    "kinetic_energy": ("Kinetic energy", "J"),
    "potential_energy": ("Potential energy", "J"),
    "total_energy": ("Total energy", "J"),
    "wrist_speed": ("Wrist speed", "m/s"),
    "tip_speed": ("Tip speed", "m/s"),
    "accel_shoulder": ("Shoulder acceleration", "rad/s²"),
    "accel_wrist": ("Wrist acceleration", "rad/s²"),
    "coriolis_shoulder": ("Coriolis term (shoulder)", "N·m"),
    "coriolis_wrist": ("Coriolis term (wrist)", "N·m"),
    "gravity_shoulder": ("Gravity term (shoulder)", "N·m"),
    "gravity_wrist": ("Gravity term (wrist)", "N·m"),
    "friction_shoulder": ("Friction torque (shoulder)", "N·m"),
    "friction_wrist": ("Friction torque (wrist)", "N·m"),
    "base_force_x": ("Base force Fx", "N"),
    "base_force_y": ("Base force Fy", "N"),
    "base_force_mag": ("Base force magnitude", "N"),
}


def _extract_double(result: Any, key: str) -> np.ndarray:
    """Extract a named data series from a double pendulum result.

    Preconditions:
        result is a SimulationResult with n_steps >= 1.
        key is in _DOUBLE_SERIES.
    Postconditions:
        Returns shape (N,) finite array.
    """
    n = result.n_steps

    if key == "time":
        return np.asarray(result.t, dtype=float)
    if key == "theta1":
        return result.states[:, 0]
    if key == "phi":
        return result.states[:, 1]
    if key == "dtheta1":
        return result.states[:, 2]
    if key == "dphi":
        return result.states[:, 3]

    if key == "torque_shoulder":
        return np.array([result.torques_at(i)[0] for i in range(n)], dtype=float)
    if key == "torque_wrist":
        return np.array([result.torques_at(i)[1] for i in range(n)], dtype=float)

    if key == "total_torque_shoulder":
        data = result.all_total_torques()
        return data[:, 0]
    if key == "total_torque_wrist":
        data = result.all_total_torques()
        return data[:, 1]

    if key in ("kinetic_energy", "potential_energy", "total_energy"):
        energies = result.all_energies()
        return energies[key.replace("_energy", "")]

    if key == "wrist_speed":
        return np.array(
            [result.joint_velocities_at(i)["wrist_speed"] for i in range(n)],
            dtype=float,
        )
    if key == "tip_speed":
        return np.array(
            [result.joint_velocities_at(i)["tip_speed"] for i in range(n)],
            dtype=float,
        )

    if key == "accel_shoulder":
        return result.all_accelerations()[:, 0]
    if key == "accel_wrist":
        return result.all_accelerations()[:, 1]

    if key == "coriolis_shoulder":
        return np.array([result.coriolis_at(i)[0] for i in range(n)], dtype=float)
    if key == "coriolis_wrist":
        return np.array([result.coriolis_at(i)[1] for i in range(n)], dtype=float)

    if key == "gravity_shoulder":
        return np.array([result.gravity_at(i)[0] for i in range(n)], dtype=float)
    if key == "gravity_wrist":
        return np.array([result.gravity_at(i)[1] for i in range(n)], dtype=float)

    if key == "friction_shoulder":
        return result.all_friction_torques()[:, 0]
    if key == "friction_wrist":
        return result.all_friction_torques()[:, 1]

    if key == "base_force_x":
        return np.array([result.base_force_at(i)["fx"] for i in range(n)], dtype=float)
    if key == "base_force_y":
        return np.array([result.base_force_at(i)["fy"] for i in range(n)], dtype=float)
    if key == "base_force_mag":
        return np.array(
            [result.base_force_at(i)["magnitude"] for i in range(n)], dtype=float
        )

    raise KeyError(f"Unknown series key: {key!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_available_series(model_type: str = "double") -> list[tuple[str, str, str]]:
    """List all available data series for a model type.

    Returns
    -------
    list of (key, description, unit_label)
    """
    registry = _DOUBLE_SERIES  # extend for triple/golfer later
    return [(k, desc, unit) for k, (desc, unit) in registry.items()]


def extract_series(
    result: Any,
    key: str,
    model_type: str = "double",
) -> tuple[np.ndarray, str, str]:
    """Extract a named data series from a simulation result.

    Parameters
    ----------
    result : SimulationResult (or triple/golfer variant)
    key : str
        Series name (e.g., "torque_shoulder", "tip_speed").
    model_type : str
        "double", "triple", or "golfer".

    Returns
    -------
    (values, description, unit_label)

    Raises
    ------
    KeyError if key is not recognized.
    """
    desc, unit = _DOUBLE_SERIES[key]
    values = _extract_double(result, key)
    assert values.ndim == 1, f"Expected 1-D array, got shape {values.shape}"
    return values, desc, unit
