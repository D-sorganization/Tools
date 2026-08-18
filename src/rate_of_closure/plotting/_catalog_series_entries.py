"""Swing, kinetics, and flight series catalog rows."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from rate_of_closure.plotting._catalog_entry_types import CatalogRow
from rate_of_closure.plotting.catalog_contract import Extractor
from rate_of_closure.simulation.kinetics import KineticsSeries, kinetics_for_run
from rate_of_closure.simulation.session import SimulationRun


def _speed_series(vectors: np.ndarray) -> np.ndarray:
    """Return row-wise Euclidean speed as a float array."""
    speeds: np.ndarray = np.asarray(np.linalg.norm(vectors, axis=1), dtype=float)
    return speeds


def _kinetics_series(picker: Callable[[KineticsSeries], np.ndarray]) -> Extractor:
    """Return an extractor that represents unavailable kinetics with NaNs."""

    def _extract(run: SimulationRun) -> np.ndarray:
        series = kinetics_for_run(run)
        if series is None:
            missing: np.ndarray = np.full(run.swing_times.shape[0], np.nan)
            return missing
        values: np.ndarray = np.asarray(picker(series), dtype=float)
        return values

    return _extract


SWING_ROWS: tuple[CatalogRow, ...] = (
    (
        "time_s",
        "Swing Time",
        "s",
        lambda run: np.asarray(run.swing_times, float),
    ),
    (
        "x_m",
        "Clubhead X (Target Line)",
        "m",
        lambda run: np.asarray(run.swing_positions[:, 0], float),
    ),
    (
        "y_m",
        "Clubhead Y (Up)",
        "m",
        lambda run: np.asarray(run.swing_positions[:, 1], float),
    ),
    (
        "z_m",
        "Clubhead Z (Right)",
        "m",
        lambda run: np.asarray(run.swing_positions[:, 2], float),
    ),
    (
        "speed_mps",
        "Clubhead Speed",
        "m/s",
        lambda run: _speed_series(run.swing_twists[:, 3:]),
    ),
    (
        "angular_speed_dps",
        "Clubhead Angular Speed",
        "deg/s",
        lambda run: np.degrees(_speed_series(run.swing_twists[:, :3])),
    ),
)

KINETICS_ROWS: tuple[CatalogRow, ...] = (
    (
        "shoulder_torque_nm",
        "Shoulder Net Torque",
        "N·m",
        _kinetics_series(lambda series: series.torque_inertial_nm[:, 0]),
    ),
    (
        "wrist_torque_nm",
        "Wrist Net Torque",
        "N·m",
        _kinetics_series(lambda series: series.torque_inertial_nm[:, 1]),
    ),
    (
        "shoulder_gravity_torque_nm",
        "Shoulder Gravity Torque",
        "N·m",
        _kinetics_series(lambda series: series.torque_gravity_nm[:, 0]),
    ),
    (
        "wrist_gravity_torque_nm",
        "Wrist Gravity Torque",
        "N·m",
        _kinetics_series(lambda series: series.torque_gravity_nm[:, 1]),
    ),
    (
        "shoulder_damping_torque_nm",
        "Shoulder Damping Torque",
        "N·m",
        _kinetics_series(lambda series: series.torque_damping_nm[:, 0]),
    ),
    (
        "wrist_damping_torque_nm",
        "Wrist Damping Torque",
        "N·m",
        _kinetics_series(lambda series: series.torque_damping_nm[:, 1]),
    ),
    (
        "shoulder_ztcf_torque_nm",
        "Shoulder ZTCF Inertial Torque",
        "N·m",
        _kinetics_series(lambda series: series.ztcf_inertial_torque_nm[:, 0]),
    ),
    (
        "wrist_ztcf_torque_nm",
        "Wrist ZTCF Inertial Torque",
        "N·m",
        _kinetics_series(lambda series: series.ztcf_inertial_torque_nm[:, 1]),
    ),
    (
        "shoulder_power_w",
        "Shoulder Power",
        "W",
        _kinetics_series(lambda series: series.power_w[:, 0]),
    ),
    (
        "wrist_power_w",
        "Wrist Power",
        "W",
        _kinetics_series(lambda series: series.power_w[:, 1]),
    ),
    (
        "shoulder_force_n",
        "Shoulder Reaction Force",
        "N",
        _kinetics_series(lambda series: series.force_magnitude_n("shoulder")),
    ),
    (
        "wrist_force_n",
        "Wrist Reaction Force",
        "N",
        _kinetics_series(lambda series: series.force_magnitude_n("wrist")),
    ),
    (
        "clubhead_force_n",
        "Clubhead Force",
        "N",
        _kinetics_series(lambda series: series.force_magnitude_n("clubhead")),
    ),
    (
        "shoulder_ztcf_force_n",
        "Shoulder ZTCF Reaction Force",
        "N",
        _kinetics_series(lambda series: series.ztcf_force_magnitude_n("shoulder")),
    ),
    (
        "wrist_ztcf_force_n",
        "Wrist ZTCF Reaction Force",
        "N",
        _kinetics_series(lambda series: series.ztcf_force_magnitude_n("wrist")),
    ),
    (
        "clubhead_ztcf_force_n",
        "Clubhead ZTCF Force",
        "N",
        _kinetics_series(lambda series: series.ztcf_force_magnitude_n("clubhead")),
    ),
)

FLIGHT_ROWS: tuple[CatalogRow, ...] = (
    (
        "time_s",
        "Flight Time",
        "s",
        lambda run: np.asarray(run.flight_times, float),
    ),
    (
        "x_m",
        "Downrange Distance",
        "m",
        lambda run: np.asarray(run.flight_positions[:, 0], float),
    ),
    (
        "y_m",
        "Height",
        "m",
        lambda run: np.asarray(run.flight_positions[:, 1], float),
    ),
    (
        "z_m",
        "Lateral (Right of Target)",
        "m",
        lambda run: np.asarray(run.flight_positions[:, 2], float),
    ),
    (
        "speed_mps",
        "Ball Speed",
        "m/s",
        lambda run: _speed_series(run.flight_velocities),
    ),
)
