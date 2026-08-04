"""Residual-vector builder for the impact-parameter solver (#4103, #4109).

Scaffolding modeled on UpstreamDrift's
``movement_optimizer/trajectory/optimizer_cost.py`` (pure, free functions
computing objective terms so the driver stays orchestration-only), with
golf-impact semantics: a candidate variable vector runs the
delivery -> impact pipeline (and ball flight when carry goals are present)
and is scored as weighted residuals against an
:class:`~shared.python.swing_sim.solver.goals.ImpactGoal`.

Rust seam (documented, do NOT remove)
-------------------------------------
:func:`evaluate_candidate` is deliberately a single **pure function**
``(variables, partition, goal[, config]) -> residuals`` with plain-float
inputs and a plain ``numpy`` output: a later Rust port of the inner
delivery -> impact evaluation can replace it wholesale behind a facade
(swap the function, keep the signature), exactly like
``swing_sim._rust_facade`` swaps the pendulum integrator. Nothing in
:mod:`.solve` may reach around this function into the impact internals.

Units and frames
----------------
Variables and goal quantities are the launch-monitor units documented in
:mod:`.goals` (deg / m/s / mph / RPM / m). Internally the delivery and
impact stages work in the AffineDrift app frame (x target, y up, z right);
the launch derivation and flight run in the flight frame (x forward,
y left, z up) via :mod:`shared.python.swing_sim.flight.frames`. Achieved
``launch_azimuth_deg`` is negated from the flight-frame azimuth so that
positive means right of the target line, matching the face/path sign
convention. ``spin_axis_deg`` uses the delivery D-plane convention:
positive = fade/slice side.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

import numpy as np

from shared.python.contracts import require

from ..flight import derive_launch_conditions, simulate, to_flight_frame
from ..impact import (
    DeliveryParameters,
    ImpactModelType,
    ImpactSolverAPI,
    derive_delivery,
)
from ..swing_source import DoublePendulumSwing
from ..types import PendulumParameters, PendulumState, PlaneOrientation
from .goals import ImpactGoal, VariablePartition
from .tuning import (
    DEFAULT_FLIGHT_DT_S,
    DEFAULT_FLIGHT_MAX_TIME_S,
    DEFAULT_FLIGHT_MODEL,
    DEFAULT_SWING_DT_S,
    DEFAULT_SWING_DURATION_S,
    MIN_CLUBHEAD_SPEED_MPS,
    MPH_TO_MPS,
    SCALE_ANGLE_DEG,
    SCALE_BALL_SPEED_MPH,
    SCALE_CARRY_M,
    SCALE_SPIN_AXIS_DEG,
    SCALE_SPIN_RPM,
    SWING_TIME_SEARCH_SAMPLES,
)

_RESIDUAL_SCALES: Mapping[str, float] = {
    "club_path_deg": SCALE_ANGLE_DEG,
    "face_angle_deg": SCALE_ANGLE_DEG,
    "attack_angle_deg": SCALE_ANGLE_DEG,
    "dynamic_loft_deg": SCALE_ANGLE_DEG,
    "ball_speed_mph": SCALE_BALL_SPEED_MPH,
    "launch_angle_deg": SCALE_ANGLE_DEG,
    "launch_azimuth_deg": SCALE_ANGLE_DEG,
    "spin_rpm": SCALE_SPIN_RPM,
    "spin_axis_deg": SCALE_SPIN_AXIS_DEG,
    "carry_m": SCALE_CARRY_M,
}

_MAX_DELIVERY_ANGLE_DEG = 89.0
_EPS = 1e-12


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    """Knobs for candidate evaluation (kept off the hot signature).

    Attributes:
        flight_model: Registry flight-model name for carry goals.
        flight_max_time_s: Maximum simulated flight time [s].
        flight_dt_s: Flight sampling interval [s].
        swing_duration_s: Integrated swing duration [s] per pendulum
            candidate (swing-source mode).
        swing_dt_s: Pendulum RK4 step [s] (swing-source mode).
    """

    flight_model: str = DEFAULT_FLIGHT_MODEL
    flight_max_time_s: float = DEFAULT_FLIGHT_MAX_TIME_S
    flight_dt_s: float = DEFAULT_FLIGHT_DT_S
    swing_duration_s: float = DEFAULT_SWING_DURATION_S
    swing_dt_s: float = DEFAULT_SWING_DT_S

    def __post_init__(self) -> None:
        for name in (
            "flight_max_time_s",
            "flight_dt_s",
            "swing_duration_s",
            "swing_dt_s",
        ):
            value = getattr(self, name)
            require(
                math.isfinite(value) and value > 0.0,
                f"{name} must be finite and > 0",
                value,
            )


def _clamp_angle(value: float) -> float:
    """Clamp an angle [deg] into the delivery contract's open range."""
    return min(max(value, -_MAX_DELIVERY_ANGLE_DEG), _MAX_DELIVERY_ANGLE_DEG)


def _derive_from_swing(
    variables: Mapping[str, float], config: EvaluationConfig
) -> tuple[float, float, float]:
    """Swing-source mode: pendulum swing -> (speed [m/s], path, AoA [deg]).

    Builds a :class:`DoublePendulumSwing` on the candidate's tilted plane
    (pure-Python backend: candidates run in worker threads and must not
    depend on wheel availability), locates the peak-clubhead-speed instant
    on a coarse grid, applies the candidate's impact-time offset, and
    converts the sampled world-frame twist (x fwd / y left / z up) to the
    app frame to read off speed, club path, and attack angle.
    """
    plane = PlaneOrientation(
        yaw_deg=variables["swing_yaw_deg"],
        side_tilt_deg=variables["swing_side_tilt_deg"],
        forward_tilt_deg=variables["swing_forward_tilt_deg"],
    )
    defaults = PendulumParameters.golf_default()
    parameters = dataclasses.replace(
        defaults,
        d1=max(variables["swing_damping_shoulder"], 0.0),
        d2=max(variables["swing_damping_wrist"], 0.0),
    )
    # Start at theta1 = -pi/2 (arm horizontal on the backswing side) so
    # the gravity-driven downswing travels toward +x (the target line);
    # the package default of +pi/2 swings away from the target.
    initial_state = PendulumState(
        theta1=-math.pi / 2.0, theta2=0.0, omega1=0.0, omega2=0.0
    )
    swing = DoublePendulumSwing(
        parameters=parameters,
        plane=plane,
        initial_state=initial_state,
        duration=config.swing_duration_s,
        dt=config.swing_dt_s,
        backend="python",
    )
    times = np.linspace(0.0, swing.duration, SWING_TIME_SEARCH_SAMPLES)
    speeds = [float(np.linalg.norm(swing.sample(float(t)).twist[3:])) for t in times]
    t_peak = float(times[int(np.argmax(speeds))])
    t_impact = min(
        max(t_peak + variables["swing_impact_time_offset_s"], 0.0),
        swing.duration,
    )
    v_world = swing.sample(t_impact).twist[3:]
    # Swing world frame == flight frame (x fwd, y left, z up) -> app frame.
    v_app = np.array([v_world[0], v_world[2], -v_world[1]])
    speed = max(float(np.linalg.norm(v_app)), MIN_CLUBHEAD_SPEED_MPS)
    aoa_deg = math.degrees(math.asin(float(np.clip(v_app[1] / speed, -1.0, 1.0))))
    path_deg = math.degrees(math.atan2(float(v_app[2]), float(v_app[0])))
    return speed, _clamp_angle(path_deg), _clamp_angle(aoa_deg)


def _spin_axis_tilt_deg(spin_vector: np.ndarray) -> float:
    """Signed D-plane spin-axis tilt [deg] from an app-frame spin vector.

    Same convention as ``delivery.derive_delivery``: pure backspin is the
    +z (right) axis; positive tilt = fade/slice side. Zero spin reports 0.
    """
    magnitude = float(np.linalg.norm(spin_vector))
    if magnitude < _EPS:
        return 0.0
    axis = spin_vector / magnitude
    horizontal = math.hypot(float(axis[0]), float(axis[2]))
    return math.degrees(math.atan2(-float(axis[1]), horizontal))


def achieved_quantities(
    variables: Mapping[str, float],
    partition: VariablePartition,
    goal: ImpactGoal,
    config: EvaluationConfig | None = None,
) -> dict[str, float]:
    """Run delivery -> impact (-> flight) and report achieved quantities.

    Pure function of its inputs (no recorder state, no globals mutated).
    Returns the delivery-level quantities always, launch-level quantities
    when the goal needs them, and ``carry_m`` when the goal needs flight.

    Args:
        variables: Full variable mapping (from
            :meth:`VariablePartition.assemble`), launch-monitor units.
        partition: The variable partition (selects swing-source mode).
        goal: The goal (selects how deep the pipeline must run).
        config: Optional evaluation knobs.

    Returns:
        Mapping of goal-quantity name -> achieved value (documented units).
    """
    cfg = config or EvaluationConfig()

    if partition.use_swing_source:
        speed, path_deg, aoa_deg = _derive_from_swing(variables, cfg)
    else:
        speed = variables["clubhead_speed_mps"]
        path_deg = variables["club_path_deg"]
        aoa_deg = variables["attack_angle_deg"]

    params = DeliveryParameters(
        clubhead_speed_mps=speed,
        club_path_deg=path_deg,
        face_angle_deg=_clamp_angle(variables["face_angle_deg"]),
        attack_angle_deg=aoa_deg,
        dynamic_loft_deg=_clamp_angle(variables["dynamic_loft_deg"]),
        lie_deg=_clamp_angle(variables["lie_deg"]),
        impact_offset_toe_mm=variables["impact_offset_toe_mm"],
        impact_offset_high_mm=variables["impact_offset_high_mm"],
    )

    achieved: dict[str, float] = {
        "club_path_deg": params.club_path_deg,
        "face_angle_deg": params.face_angle_deg,
        "attack_angle_deg": params.attack_angle_deg,
        "dynamic_loft_deg": params.dynamic_loft_deg,
    }
    if not goal.needs_launch:
        return achieved

    derived = derive_delivery(params)
    api = ImpactSolverAPI(model_type=ImpactModelType.RIGID_BODY)
    post = api.solve_with_gear_effect(
        timestamp=0.0,
        clubhead_velocity=derived.clubhead_velocity,
        clubhead_orientation=derived.face_normal,
        impact_offset=derived.impact_offset,
        record=False,
    )

    launch = derive_launch_conditions(
        to_flight_frame(post.ball_velocity),
        to_flight_frame(post.ball_angular_velocity),
    )
    achieved["ball_speed_mph"] = launch.ball_speed / MPH_TO_MPS
    achieved["launch_angle_deg"] = math.degrees(launch.launch_angle)
    # Flight azimuth is + toward +y (left); goals use + = right of target.
    achieved["launch_azimuth_deg"] = -math.degrees(launch.azimuth_angle)
    achieved["spin_rpm"] = launch.spin_rate
    achieved["spin_axis_deg"] = _spin_axis_tilt_deg(post.ball_angular_velocity)

    if goal.needs_flight:
        result = simulate(
            launch,
            model_name=cfg.flight_model,
            max_time=cfg.flight_max_time_s,
            dt=cfg.flight_dt_s,
        )
        achieved["carry_m"] = result.carry_distance
    return achieved


def evaluate_candidate(
    variables: Mapping[str, float],
    partition: VariablePartition,
    goal: ImpactGoal,
    config: EvaluationConfig | None = None,
) -> np.ndarray:
    """Weighted residual vector for a full candidate variable mapping.

    This is the Rust-portable seam (see module docstring): pure function,
    plain-float inputs, ``(n_goals,)`` float output ordered like
    :meth:`ImpactGoal.items`. Each residual is
    ``weight * (achieved - target) / scale`` with the per-quantity scales
    from :mod:`.tuning`.
    """
    achieved = achieved_quantities(variables, partition, goal, config)
    return np.array(
        [
            term.weight * (achieved[name] - term.target) / _RESIDUAL_SCALES[name]
            for name, term in goal.items()
        ]
    )


def residuals(
    x: np.ndarray,
    partition: VariablePartition,
    goal: ImpactGoal,
    config: EvaluationConfig | None = None,
) -> np.ndarray:
    """Residual vector for a free-variable vector (driver entry point)."""
    return evaluate_candidate(partition.assemble(x), partition, goal, config)


__all__ = [
    "EvaluationConfig",
    "achieved_quantities",
    "evaluate_candidate",
    "residuals",
]
