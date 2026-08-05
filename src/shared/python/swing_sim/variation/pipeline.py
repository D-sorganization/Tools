"""Pipeline-slice evaluation for the variation engine (#4120 V3).

One pure function per study run: registry-keyed variables in, output
columns out. Mirrors :mod:`shared.python.swing_sim.solver.objective`
(delivery -> impact -> flight; swing-source derivation via the same
private helpers, credited there to UpstreamDrift's movement_optimizer
scaffolding) extended with lateral / apex / landing outputs and
club-parameter overrides (mass / MOI / COR into the impact solve).

Sign conventions (launch-monitor style, matching :mod:`..solver.goals`):
``launch_azimuth_deg`` and ``lateral_m`` are positive **right** of the
target line; ``spin_axis_deg`` positive = fade/slice side.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from ..flight import (
    FlightResult,
    LaunchConditions,
    derive_launch_conditions,
    simulate,
    to_flight_frame,
)
from ..impact import (
    DeliveryParameters,
    ImpactModelType,
    ImpactSolverAPI,
    derive_delivery,
    face_basis,
)
from ..impact.constants import GOLF_BALL_MASS_KG, GOLF_BALL_RADIUS_M
from ..impact.types import ImpactParameters
from ..impact_interval import (
    ClubRigidBody,
    ImpactIntervalConfig,
    ImpactIntervalInitialState,
    KelvinVoigtContactLaw,
    solve_impact_interval,
)
from ..solver.objective import (
    EvaluationConfig,
    _derive_from_swing,
    _spin_axis_tilt_deg,
)
from ..solver.tuning import MPH_TO_MPS
from .spec import CATEGORY_SWING

_MAX_ANGLE_DEG = 89.0

DELIVERY_OUTPUTS: tuple[str, ...] = (
    "club_path_deg",
    "face_angle_deg",
    "attack_angle_deg",
    "dynamic_loft_deg",
)
LAUNCH_OUTPUTS: tuple[str, ...] = (
    "ball_speed_mph",
    "launch_angle_deg",
    "launch_azimuth_deg",
    "spin_rpm",
    "spin_axis_deg",
)
FLIGHT_OUTPUTS: tuple[str, ...] = (
    "carry_m",
    "lateral_m",
    "apex_m",
    "landing_angle_deg",
    "flight_time_s",
)
IMPACT_INTERVAL_OUTPUTS: tuple[str, ...] = (
    "contact_duration_us",
    "peak_normal_force_n",
    "integrated_normal_impulse_n_s",
    "face_angle_change_deg",
    "dynamic_loft_change_deg",
    "twist_change_deg",
    "energy_residual_j",
)


def outputs_for_mode(mode: str) -> tuple[str, ...]:
    """Canonical output-column names for a pipeline mode."""
    if mode == "launch":
        return LAUNCH_OUTPUTS + FLIGHT_OUTPUTS
    if mode == "impact_interval":
        return (
            DELIVERY_OUTPUTS + IMPACT_INTERVAL_OUTPUTS + LAUNCH_OUTPUTS + FLIGHT_OUTPUTS
        )
    return DELIVERY_OUTPUTS + LAUNCH_OUTPUTS + FLIGHT_OUTPUTS


def _clamp_angle(value: float) -> float:
    """Clamp an angle [deg] into the delivery contract's open range."""
    return min(max(value, -_MAX_ANGLE_DEG), _MAX_ANGLE_DEG)


def _solver_named(variables: Mapping[str, float]) -> dict[str, float]:
    """Registry-keyed mapping -> the solver's short variable names."""
    named: dict[str, float] = {}
    for key, value in variables.items():
        category, name = key.rsplit(".", 1)
        if category == CATEGORY_SWING:
            named[f"swing_{name}"] = value
        else:
            named[name] = value
    return named


def _flight_outputs(result: FlightResult) -> dict[str, float]:
    """FlightResult -> output columns (lateral converted to + = right)."""
    return {
        "carry_m": float(result.carry_distance),
        "lateral_m": -float(result.lateral_deviation),
        "apex_m": float(result.max_height),
        "landing_angle_deg": float(result.landing_angle),
        "flight_time_s": float(result.flight_time),
    }


def evaluate_run(
    variables: Mapping[str, float],
    mode: str,
    config: EvaluationConfig,
) -> dict[str, float]:
    """Run one sampled variable set through its pipeline slice.

    Pure function (no shared state): safe to call from worker threads.

    Args:
        variables: Full registry-keyed variable mapping for one run.
        mode: ``"delivery"`` / ``"swing"`` / ``"launch"``.
        config: Evaluation knobs (flight model, swing grid) — the same
            :class:`~shared.python.swing_sim.solver.objective.EvaluationConfig`
            the solver uses.

    Returns:
        Mapping of output name (:func:`outputs_for_mode`) -> value.
    """
    if mode == "launch":
        return _evaluate_launch(variables, config)
    if mode == "impact_interval":
        return _evaluate_impact_interval(variables, config)
    named = _solver_named(variables)
    if mode == "swing":
        speed, path_deg, aoa_deg = _derive_from_swing(named, config)
    else:
        speed = named["clubhead_speed_mps"]
        path_deg = _clamp_angle(named["club_path_deg"])
        aoa_deg = _clamp_angle(named["attack_angle_deg"])

    params = DeliveryParameters(
        clubhead_speed_mps=max(speed, 1e-3),
        club_path_deg=path_deg,
        face_angle_deg=_clamp_angle(named["face_angle_deg"]),
        attack_angle_deg=aoa_deg,
        dynamic_loft_deg=_clamp_angle(named["dynamic_loft_deg"]),
        lie_deg=_clamp_angle(named["lie_deg"]),
        impact_offset_toe_mm=named["impact_offset_toe_mm"],
        impact_offset_high_mm=named["impact_offset_high_mm"],
    )
    derived = derive_delivery(params)
    cor = min(max(named["cor"], 0.0), 1.0)
    api = ImpactSolverAPI(
        model_type=ImpactModelType.RIGID_BODY,
        params=ImpactParameters(cor=cor),
    )
    post = api.solve_with_gear_effect(
        timestamp=0.0,
        clubhead_velocity=derived.clubhead_velocity,
        clubhead_orientation=derived.face_normal,
        impact_offset=derived.impact_offset,
        clubhead_mass=max(named["head_mass_kg"], 1e-6),
        clubhead_moi=max(named["head_moi_kg_m2"], 0.0),
        record=False,
    )
    launch = derive_launch_conditions(
        to_flight_frame(post.ball_velocity),
        to_flight_frame(post.ball_angular_velocity),
    )
    outputs: dict[str, float] = {
        "club_path_deg": params.club_path_deg,
        "face_angle_deg": params.face_angle_deg,
        "attack_angle_deg": params.attack_angle_deg,
        "dynamic_loft_deg": params.dynamic_loft_deg,
        "ball_speed_mph": launch.ball_speed / MPH_TO_MPS,
        "launch_angle_deg": math.degrees(launch.launch_angle),
        # Flight-frame azimuth is + toward +y (left); report + = right.
        "launch_azimuth_deg": -math.degrees(launch.azimuth_angle),
        "spin_rpm": launch.spin_rate,
        "spin_axis_deg": _spin_axis_tilt_deg(post.ball_angular_velocity),
    }
    result = simulate(
        launch,
        model_name=config.flight_model,
        max_time=config.flight_max_time_s,
        dt=config.flight_dt_s,
    )
    outputs.update(_flight_outputs(result))
    return outputs


def _evaluate_impact_interval(
    variables: Mapping[str, float], config: EvaluationConfig
) -> dict[str, float]:
    """Delivery -> six-DOF interval -> flight variation slice."""
    named = _solver_named(variables)
    params = DeliveryParameters(
        clubhead_speed_mps=max(named["clubhead_speed_mps"], 1.0e-3),
        club_path_deg=_clamp_angle(named["club_path_deg"]),
        face_angle_deg=_clamp_angle(named["face_angle_deg"]),
        attack_angle_deg=_clamp_angle(named["attack_angle_deg"]),
        dynamic_loft_deg=_clamp_angle(named["dynamic_loft_deg"]),
        lie_deg=_clamp_angle(named["lie_deg"]),
        impact_offset_toe_mm=named["impact_offset_toe_mm"],
        impact_offset_high_mm=named["impact_offset_high_mm"],
    )
    derived = derive_delivery(params)
    toe_axis, up_axis = face_basis(derived.face_normal)
    orientation = np.column_stack((derived.face_normal, up_axis, toe_axis))
    toe_m, high_m = derived.impact_offset
    body = ClubRigidBody(
        mass_kg=max(named["head_mass_kg"], 1.0e-6),
        inertia_body_kg_m2=max(named["head_moi_kg_m2"], 1.0e-9) * np.eye(3),
        cg_to_contact_body_m=np.array([named["cg_depth_m"], high_m, toe_m]),
        cg_to_attachment_body_m=np.array([0.0, 0.90, 0.0]),
        face_normal_body=np.array([1.0, 0.0, 0.0]),
    )
    contact_world = orientation @ body.cg_to_contact_body_m
    ball_position = GOLF_BALL_RADIUS_M * derived.face_normal
    initial = ImpactIntervalInitialState(
        club_position_m=-contact_world,
        club_orientation=orientation,
        club_velocity_mps=derived.clubhead_velocity,
        club_angular_velocity_rad_s=derived.clubhead_angular_velocity,
        ball_position_m=ball_position,
        ball_velocity_mps=np.zeros(3),
        ball_angular_velocity_rad_s=np.zeros(3),
    )
    reduced_mass = GOLF_BALL_MASS_KG * body.mass_kg / (GOLF_BALL_MASS_KG + body.mass_kg)
    cor = min(max(named["cor"], 1.0e-6), 1.0)
    law = KelvinVoigtContactLaw.from_restitution(
        stiffness_n_per_m=max(named["contact_stiffness_n_per_m"], 1.0),
        restitution=cor,
        effective_mass_kg=reduced_mass,
    )
    interval = solve_impact_interval(
        initial,
        body,
        ImpactIntervalConfig(
            contact_law=law,
            time_step_s=1.0e-7,
            maximum_time_s=2.0e-3,
            friction_coefficient=max(named["friction_coefficient"], 0.0),
        ),
    )
    post = interval.to_post_impact_state()
    launch = derive_launch_conditions(
        to_flight_frame(post.ball_velocity),
        to_flight_frame(post.ball_angular_velocity),
    )
    outputs: dict[str, float] = {
        "club_path_deg": params.club_path_deg,
        "face_angle_deg": params.face_angle_deg,
        "attack_angle_deg": params.attack_angle_deg,
        "dynamic_loft_deg": params.dynamic_loft_deg,
        "contact_duration_us": interval.contact_duration_s * 1.0e6,
        "peak_normal_force_n": float(np.max(interval.normal_force_n)),
        "integrated_normal_impulse_n_s": interval.audit.integrated_normal_impulse_n_s,
        "face_angle_change_deg": float(
            interval.face_angle_deg[-1] - interval.face_angle_deg[0]
        ),
        "dynamic_loft_change_deg": float(
            interval.dynamic_loft_deg[-1] - interval.dynamic_loft_deg[0]
        ),
        "twist_change_deg": math.degrees(
            float(interval.twist_angle_rad[-1] - interval.twist_angle_rad[0])
        ),
        "energy_residual_j": interval.audit.energy_residual_j,
        "ball_speed_mph": launch.ball_speed / MPH_TO_MPS,
        "launch_angle_deg": math.degrees(launch.launch_angle),
        "launch_azimuth_deg": -math.degrees(launch.azimuth_angle),
        "spin_rpm": launch.spin_rate,
        "spin_axis_deg": _spin_axis_tilt_deg(post.ball_angular_velocity),
    }
    flight = simulate(
        launch,
        model_name=config.flight_model,
        max_time=config.flight_max_time_s,
        dt=config.flight_dt_s,
    )
    outputs.update(_flight_outputs(flight))
    return outputs


def _evaluate_launch(
    variables: Mapping[str, float], config: EvaluationConfig
) -> dict[str, float]:
    """Launch mode: direct launch conditions -> ball flight only.

    Sign mapping: registry azimuth/spin-axis are + right / + fade; the
    flight frame's azimuth and legacy spin-axis decomposition are + left,
    so both are negated on the way in.
    """
    named = _solver_named(variables)
    launch = LaunchConditions.from_imperial(
        ball_speed_mph=named["ball_speed_mph"],
        launch_angle_deg=_clamp_angle(named["launch_angle_deg"]),
        spin_rate_rpm=max(named["spin_rpm"], 0.0),
        azimuth_angle_deg=-named["launch_azimuth_deg"],
        spin_axis_angle_deg=-named["spin_axis_deg"],
    )
    outputs: dict[str, float] = {
        "ball_speed_mph": named["ball_speed_mph"],
        "launch_angle_deg": named["launch_angle_deg"],
        "launch_azimuth_deg": named["launch_azimuth_deg"],
        "spin_rpm": named["spin_rpm"],
        "spin_axis_deg": named["spin_axis_deg"],
    }
    result = simulate(
        launch,
        model_name=config.flight_model,
        max_time=config.flight_max_time_s,
        dt=config.flight_dt_s,
    )
    outputs.update(_flight_outputs(result))
    return outputs


__all__ = [
    "DELIVERY_OUTPUTS",
    "FLIGHT_OUTPUTS",
    "LAUNCH_OUTPUTS",
    "IMPACT_INTERVAL_OUTPUTS",
    "evaluate_run",
    "outputs_for_mode",
]
