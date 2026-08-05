"""Run data export: CSV time series and JSON summary/params.

One :class:`~rate_of_closure.simulation.session.SimulationRun` becomes:

* a CSV with one row per sample — swing rows (clubhead position and
  speed) followed by flight rows (ball position and speed), phase-tagged
  so the two series stay distinguishable in a flat file; and
* a JSON document carrying the request parameters, the delivery and
  launch summaries, and both time series.

All values are SI + app frame (x target, y up, z right), matching the
in-memory record; display units are a UI concern.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from rate_of_closure._contracts import require
from rate_of_closure.simulation.session import SimulationRun

__all__ = ["CSV_COLUMNS", "run_to_json_dict", "series_rows", "write_csv", "write_json"]

#: CSV column order (kept stable for downstream consumers).
CSV_COLUMNS: tuple[str, ...] = (
    "phase",
    "t_s",
    "x_m",
    "y_m",
    "z_m",
    "speed_mps",
)


def series_rows(
    run: SimulationRun,
) -> list[tuple[str, float, float, float, float, float]]:
    """Flatten the swing and flight series into phase-tagged rows."""
    rows: list[tuple[str, float, float, float, float, float]] = []
    for t, pos, twist in zip(
        run.swing_times, run.swing_positions, run.swing_twists, strict=True
    ):
        rows.append(
            (
                "swing",
                float(t),
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                float(np.linalg.norm(twist[3:])),
            )
        )
    t0 = run.impact_time_s
    for t, pos, vel in zip(
        run.flight_times, run.flight_positions, run.flight_velocities, strict=True
    ):
        rows.append(
            (
                "flight",
                t0 + float(t),
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                float(np.linalg.norm(vel)),
            )
        )
    return rows


def write_csv(run: SimulationRun, path: str | Path) -> None:
    """Write the run's time series as CSV.

    Args:
        run: The simulation run to export.
        path: Destination file path.
    """
    require(isinstance(run, SimulationRun), "run must be a SimulationRun", run)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(series_rows(run))


def run_to_json_dict(run: SimulationRun) -> dict[str, Any]:
    """The run as a JSON-serialisable dictionary.

    Args:
        run: The simulation run to export.

    Returns:
        Parameters, summaries, and both time series; non-finite floats
        are rendered as ``None`` for strict-JSON consumers.
    """
    require(isinstance(run, SimulationRun), "run must be a SimulationRun", run)
    config = run.config
    scenario = config.scenario

    def _clean(value: float) -> float | None:
        return float(value) if math.isfinite(value) else None

    return {
        "format": "rate_of_closure.simulation_run/1",
        "parameters": {
            "source_kind": config.source_kind,
            "club": config.club.name,
            "flight_model": config.flight_model,
            "impact_time_s": run.impact_time_s,
            "swing_duration_s": config.swing_duration_s,
            "plane_tilts_deg": {
                "yaw": config.plane.yaw_deg,
                "side_tilt": config.plane.side_tilt_deg,
                "forward_tilt": config.plane.forward_tilt_deg,
            },
            "scenario": {
                "clubhead_speed_mph": scenario.clubhead_speed_mph,
                "omega_plane_dps": scenario.omega_plane_dps,
                "omega_shaft_dps": scenario.omega_shaft_dps,
                "lie_angle_deg": scenario.lie_angle_deg,
                "com_to_face_mm": scenario.com_to_face_mm,
                "impact_offset_toe_mm": scenario.impact_offset_toe_mm,
                "impact_offset_high_mm": scenario.impact_offset_high_mm,
                "contact_duration_us": scenario.contact_duration_us,
            },
        },
        "delivery": {
            "clubhead_speed_mps": float(np.linalg.norm(run.delivery.clubhead_velocity)),
            "spin_loft_deg": run.delivery.spin_loft_deg,
            "face_to_path_deg": run.delivery.face_to_path_deg,
            "spin_axis_tilt_deg": run.delivery.spin_axis_tilt_deg,
        },
        "launch": {key: _clean(value) for key, value in run.launch.items()},
        "series": {
            "columns": list(CSV_COLUMNS),
            "rows": [list(row) for row in series_rows(run)],
            "swing_joints_app_m": run.swing_joints.tolist(),
        },
    }


def write_json(run: SimulationRun, path: str | Path) -> None:
    """Write the run's summary + series as a JSON document.

    Args:
        run: The simulation run to export.
        path: Destination file path.
    """
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(run_to_json_dict(run), handle, indent=2)
