"""Run data export: CSV time series and JSON summary/params.

One :class:`~rate_of_closure.simulation.session.SimulationRun` becomes:

* a CSV with one row per sample — swing rows (clubhead position and
  speed) followed by flight rows (ball position and speed), phase-tagged
  so the two series stay distinguishable in a flat file; and
* an optional long-form torque CSV with one row per swing sample and
  stable joint ID (empty for sources without applied-torque histories); and
* a JSON document carrying the request parameters, the delivery and
  launch summaries, and both time series.

All values are SI + app frame (x target, y up, z right), matching the
in-memory record; display units are a UI concern.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from rate_of_closure._contracts import require
from rate_of_closure.simulation.manual_delivery import ManualDeliveryConfig
from rate_of_closure.simulation.screw_analysis import analyze_twist
from rate_of_closure.simulation.session import SimulationRun
from rate_of_closure.simulation.target_persistence import (
    TARGET_CSV_COLUMNS,
    default_spatial_target,
    simulation_document_format,
    spatial_target_from_simulation_document,
    target_csv_values,
    target_document_fields,
)
from shared.python.swing_sim.ball_setup import BallSetup
from shared.python.swing_sim.solver import SpatialTarget

__all__ = [
    "CSV_COLUMNS",
    "SCREW_CSV_COLUMNS",
    "TARGET_CSV_COLUMNS",
    "TORQUE_CSV_COLUMNS",
    "ball_setup_from_json_dict",
    "manual_delivery_from_json_dict",
    "run_to_json_dict",
    "series_rows",
    "screw_series_rows",
    "spatial_target_from_simulation_document",
    "torque_series_rows",
    "write_csv",
    "write_json",
    "write_screw_csv",
    "write_torque_csv",
]

#: CSV column order (kept stable for downstream consumers).
CSV_COLUMNS: tuple[str, ...] = (
    "phase",
    "t_s",
    "x_m",
    "y_m",
    "z_m",
    "speed_mps",
    "is_fixed_ball_contact",
    "impact_occurred",
    "impact_time_s",
    "candidate_time_s",
    "closest_approach_m",
    "contact_margin_m",
)

TORQUE_CSV_COLUMNS: tuple[str, ...] = ("t_s", "joint_id", "applied_torque_nm")

SCREW_CSV_COLUMNS: tuple[str, ...] = (
    "t_s",
    "motion_kind",
    "angular_rate_rad_s",
    "pitch_m_rad",
    "axial_speed_m_s",
    "r_isa_m",
    "axis_x",
    "axis_y",
    "axis_z",
    "axis_point_x_m",
    "axis_point_y_m",
    "axis_point_z_m",
    "orbital_vx_m_s",
    "orbital_vy_m_s",
    "orbital_vz_m_s",
    "axial_vx_m_s",
    "axial_vy_m_s",
    "axial_vz_m_s",
    "reconstruction_residual_m_s",
)

_CURRENT_VERSION = 5
_CURRENT_BALL_SETUP_FIELDS = frozenset(
    ("support_mode", "tee_height_m", "height_reference", "ball_center_m")
)
_CURRENT_MANUAL_DELIVERY_FIELDS = frozenset(
    (
        "attack_angle_deg",
        "club_path_deg",
        "forward_shaft_lean_deg",
        "shaft_axis_datum",
    )
)


def _current_parameters(
    data: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return strict current-v5 parameters; older inputs stay migratable."""
    version, _is_web = simulation_document_format(data)
    if version != _CURRENT_VERSION:
        return None
    parameters = data.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError(
            f"simulation schema version {_CURRENT_VERSION} requires parameters"
        )
    return parameters


def _require_current_block(
    parameters: Mapping[str, Any],
    block_name: str,
    required_fields: frozenset[str],
) -> Mapping[str, Any]:
    """Return one complete current-native canonical settings block."""
    block = parameters.get(block_name)
    if block is None:
        raise ValueError(
            "simulation schema version "
            f"{_CURRENT_VERSION} requires parameters.{block_name}"
        )
    if not isinstance(block, Mapping):
        raise TypeError(f"{block_name} must be a mapping")
    missing = sorted(required_fields.difference(block))
    if missing:
        raise ValueError(f"{block_name} requires fields: {', '.join(missing)}")
    return block


def ball_setup_from_json_dict(data: Mapping[str, Any]) -> BallSetup:
    """Import ball geometry from a run or parameter mapping.

    Current version-5 run documents require the complete canonical block. Older
    documents intentionally migrate to Ground/0 so replay retains their
    original fixed-ball geometry, including for drivers whose *new-run* default
    is now Tee.
    """
    require(isinstance(data, Mapping), "simulation JSON must be a mapping", data)
    current_parameters = _current_parameters(data)
    parameters = data.get("parameters", data)
    require(
        isinstance(parameters, Mapping),
        "simulation parameters must be a mapping",
        parameters,
    )
    setup = (
        _require_current_block(
            current_parameters, "ball_setup", _CURRENT_BALL_SETUP_FIELDS
        )
        if current_parameters is not None
        else parameters.get("ball_setup")
    )
    require(
        setup is None or isinstance(setup, Mapping),
        "ball_setup must be a mapping when present",
        setup,
    )
    return BallSetup.from_json_dict(setup)


def manual_delivery_from_json_dict(data: Mapping[str, Any]) -> ManualDeliveryConfig:
    """Import a complete current declaration or default an older run."""
    require(isinstance(data, Mapping), "simulation JSON must be a mapping", data)
    current_parameters = _current_parameters(data)
    parameters = data.get("parameters", data)
    require(
        isinstance(parameters, Mapping),
        "simulation parameters must be a mapping",
        parameters,
    )
    declaration = (
        _require_current_block(
            current_parameters,
            "manual_delivery",
            _CURRENT_MANUAL_DELIVERY_FIELDS,
        )
        if current_parameters is not None
        else parameters.get("manual_delivery")
    )
    require(
        declaration is None or isinstance(declaration, Mapping),
        "manual_delivery must be a mapping when present",
        declaration,
    )
    if declaration is None:
        return ManualDeliveryConfig()
    defaults = ManualDeliveryConfig()
    return ManualDeliveryConfig(
        attack_angle_deg=declaration.get("attack_angle_deg", defaults.attack_angle_deg),
        club_path_deg=declaration.get("club_path_deg", defaults.club_path_deg),
        forward_shaft_lean_deg=declaration.get(
            "forward_shaft_lean_deg", defaults.forward_shaft_lean_deg
        ),
        shaft_axis_datum=declaration.get("shaft_axis_datum", defaults.shaft_axis_datum),
    )


def series_rows(
    run: SimulationRun, spatial_target: SpatialTarget | None = None
) -> list[tuple[Any, ...]]:
    """Flatten the swing and flight series into phase-tagged rows."""
    rows: list[tuple[Any, ...]] = []
    contact_columns = _contact_columns(run)
    target_columns = (
        target_csv_values(spatial_target) if spatial_target is not None else ()
    )
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
                *contact_columns,
                *target_columns,
            )
        )
    t0 = run.impact_time_s
    for t, pos, vel in zip(
        run.flight_times, run.flight_positions, run.flight_velocities, strict=True
    ):
        assert t0 is not None
        rows.append(
            (
                "flight",
                t0 + float(t),
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                float(np.linalg.norm(vel)),
                *contact_columns,
                *target_columns,
            )
        )
    return rows


def torque_series_rows(run: SimulationRun) -> list[tuple[float, str, float]]:
    """Return long-form applied torque rows keyed by stable joint ID."""
    return [
        (float(time_s), joint_id, float(run.swing_applied_torques_nm[row, column]))
        for row, time_s in enumerate(run.swing_times)
        for column, joint_id in enumerate(run.swing_joint_ids)
    ]


def screw_series_rows(run: SimulationRun) -> list[tuple[Any, ...]]:
    """Return the typed club screw decomposition for every swing sample."""
    rows: list[tuple[Any, ...]] = []
    for time_s, point, twist in zip(
        run.swing_times, run.swing_positions, run.swing_twists, strict=True
    ):
        motion = analyze_twist(twist, point)
        axis_point = motion.axis_point_m
        rows.append(
            (
                float(time_s),
                motion.kind.value,
                motion.angular_rate_rad_s,
                motion.pitch_m_rad,
                motion.axial_speed_m_s,
                motion.radius_m,
                *motion.axis_direction.tolist(),
                *(axis_point.tolist() if axis_point is not None else (None,) * 3),
                *motion.orbital_velocity_m_s.tolist(),
                *motion.axial_velocity_m_s.tolist(),
                motion.reconstruction_residual_m_s,
            )
        )
    return rows


def _contact_columns(
    run: SimulationRun,
) -> tuple[int, int, float | None, float, float, float]:
    """Return numeric contact metadata repeated on each flat CSV row."""
    outcome = run.impact_outcome
    return (
        int(outcome.mode.value == "fixed_ball_contact"),
        int(outcome.is_hit),
        run.impact_time_s,
        outcome.candidate_time_s,
        outcome.closest_approach_m,
        outcome.contact_margin_m,
    )


def write_csv(
    run: SimulationRun,
    path: str | Path,
    *,
    spatial_target: SpatialTarget | None = None,
) -> None:
    """Write the run's time series as CSV.

    Args:
        run: The simulation run to export.
        spatial_target: Canonical target to persist. When omitted, the explicit
            default landing target is written so every native v5 document is
            complete.
        path: Destination file path.
    """
    require(isinstance(run, SimulationRun), "run must be a SimulationRun", run)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        target_header = TARGET_CSV_COLUMNS if spatial_target is not None else ()
        writer.writerow((*CSV_COLUMNS, *target_header))
        writer.writerows(series_rows(run, spatial_target))


def write_torque_csv(run: SimulationRun, path: str | Path) -> None:
    """Write a separate long-form applied joint-torque history CSV."""
    require(isinstance(run, SimulationRun), "run must be a SimulationRun", run)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TORQUE_CSV_COLUMNS)
        writer.writerows(torque_series_rows(run))


def write_screw_csv(run: SimulationRun, path: str | Path) -> None:
    """Write a dedicated SI/app-frame club screw-motion time series."""
    require(isinstance(run, SimulationRun), "run must be a SimulationRun", run)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SCREW_CSV_COLUMNS)
        writer.writerows(screw_series_rows(run))


def run_to_json_dict(
    run: SimulationRun, *, spatial_target: SpatialTarget | None = None
) -> dict[str, Any]:
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

    document: dict[str, Any] = {
        "format": "rate_of_closure.simulation_run/5",
        "model_limitations": {
            "contact_tracking": {
                "basis": "tracked_reference_point",
                "description": (
                    "Forced alignment and sampled fixed-ball contact track the "
                    "clubhead reference point, not swept face-mesh contact."
                ),
            },
            "impact_velocity": {
                "basis": "clubhead_reference_translation",
                "description": (
                    "The current rigid impact and ball-flight pipeline consumes "
                    "reference-point translation. Shaft-induced contact-point "
                    "velocity is analyzed separately and does not alter flight."
                ),
            },
        },
        "parameters": {
            "source_kind": config.source_kind,
            "club": config.club.name,
            "ball_setup": config.ball_setup.to_json_dict(),
            "flight_model": config.flight_model,
            "contact_mode": config.contact_mode.value,
            "swing_run_mode": config.swing_run_config.mode.value,
            "prescribed_profile_id": config.swing_run_config.prescribed_profile_id,
            "locked_joint_ids": list(
                config.swing_run_config.joint_locks.locked_joint_ids
            ),
            "impact_time_s": run.impact_time_s,
            "swing_duration_s": config.swing_duration_s,
            "plane_tilts_deg": {
                "yaw": config.plane.yaw_deg,
                "side_tilt": config.plane.side_tilt_deg,
                "forward_tilt": config.plane.forward_tilt_deg,
            },
            "manual_delivery": {
                "attack_angle_deg": config.manual_attack_angle_deg,
                "club_path_deg": config.manual_club_path_deg,
                "forward_shaft_lean_deg": config.manual_forward_shaft_lean_deg,
                "shaft_axis_datum": config.manual_shaft_axis_datum.value,
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
        "impact_outcome": run.impact_outcome.to_dict(),
        "club_assembly_usage": run.club_assembly_usage.to_json_dict(),
        "delivery": _delivery_dict(run),
        "launch": _launch_dict(run),
        "series": {
            "columns": list(CSV_COLUMNS),
            "rows": [list(row) for row in series_rows(run)],
            "swing_joints_app_m": run.swing_joints.tolist(),
            "swing_applied_joint_torques": {
                "unit": "N*m",
                "joint_ids": list(run.swing_joint_ids),
                "values": run.swing_applied_torques_nm.tolist(),
            },
            "club_screw_motion": {
                "frame": "app/world",
                "units": "SI",
                "columns": list(SCREW_CSV_COLUMNS),
                "rows": [list(row) for row in screw_series_rows(run)],
            },
        },
    }
    document.update(
        target_document_fields(
            spatial_target if spatial_target is not None else default_spatial_target()
        )
    )
    return document


def _delivery_dict(run: SimulationRun) -> dict[str, float] | None:
    """Serialize delivery fields only when contact occurred."""
    delivery = run.delivery
    if delivery is None:
        return None
    return {
        "clubhead_speed_mps": float(np.linalg.norm(delivery.clubhead_velocity)),
        "spin_loft_deg": delivery.spin_loft_deg,
        "face_to_path_deg": delivery.face_to_path_deg,
        "spin_axis_tilt_deg": delivery.spin_axis_tilt_deg,
    }


def _launch_dict(run: SimulationRun) -> dict[str, float | None] | None:
    """Serialize finite launch fields only when contact occurred."""
    if run.launch is None:
        return None
    return {
        key: float(value) if math.isfinite(value) else None
        for key, value in run.launch.items()
    }


def write_json(
    run: SimulationRun,
    path: str | Path,
    *,
    spatial_target: SpatialTarget | None = None,
) -> None:
    """Write the run's summary + series as a JSON document.

    Args:
        run: The simulation run to export.
        path: Destination file path.
    """
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(
            run_to_json_dict(run, spatial_target=spatial_target), handle, indent=2
        )
