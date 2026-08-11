"""Swing kinetics façade and run adapter for the double-pendulum model.

Per-sample joint torques, reaction forces, and powers use SI units and joint
order proximal-to-distal. ``theta1`` is the arm absolute angle from in-plane
down; ``theta2`` is the club angle relative to the arm. Positive torque is
counter-clockwise about the swing-plane normal and maps to increasing theta.
Reaction forces are the force the proximal side exerts on the distal side,
expressed in the app frame (x target, y up, z right). Gravity is included.

Ideal joint-lock reactions remain separate from commanded applied torque, so
a constraint is never presented as actuator input. Pure dynamics live in
``_kinetics_dynamics``; the immutable output contract lives in
``_kinetics_series``. Re-exports here preserve the established public surface.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np

from rate_of_closure._contracts import require
from rate_of_closure.simulation._kinetics_dynamics import (
    inverse_dynamics,
    reaction_forces,
    simulate_forced,
    zero_torque_counterfactual,
)
from rate_of_closure.simulation._kinetics_series import (
    CLUBHEAD_MASS_KG,
    KINETIC_JOINT_NAMES,
    KineticsSeries,
)
from rate_of_closure.simulation.session import SimulationRun
from rate_of_closure.simulation.sources import (
    APP_FROM_SWING,
    AppFrameSwing,
    make_source,
)
from shared.python.swing_sim import reference
from shared.python.swing_sim.swing_source import DoublePendulumSwing

__all__ = [
    "CLUBHEAD_MASS_KG",
    "KINETIC_JOINT_NAMES",
    "KineticsSeries",
    "compute_kinetics",
    "inverse_dynamics",
    "kinetics_for_run",
    "simulate_forced",
    "zero_torque_counterfactual",
]

# Compatibility seam retained for focused force tests and existing consumers.
_reaction_forces = reaction_forces


def _to_app(plane_r: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Map in-plane vectors (local x, local up) into the app frame."""
    world = vectors[:, :1] * plane_r[:, 0] + vectors[:, 1:2] * plane_r[:, 2]
    return cast(np.ndarray, np.asarray(world @ APP_FROM_SWING.T))


def compute_kinetics(
    run: SimulationRun,
    clubhead_mass_kg: float = CLUBHEAD_MASS_KG,
    *,
    analysis_time_s: float | None = None,
) -> KineticsSeries | None:
    """Return swing kinetics for a supported run.

    Only the ``double_pendulum`` source exposes the joint states required by
    the equations of motion. Manual and triple-pendulum runs return ``None``.
    The deterministic source is rebuilt from the run configuration, and its
    public ``state_at`` accessor is sampled on the run's stored time grid.

    ``analysis_time_s`` can mark a completed miss for presentation without
    truncating arrays. The canonical :func:`kinetics_for_run` contract still
    returns ``None`` when impact did not occur.
    """
    require(isinstance(run, SimulationRun), "run must be a SimulationRun", run)
    require(
        math.isfinite(clubhead_mass_kg) and clubhead_mass_kg > 0.0,
        "clubhead_mass_kg must be finite and > 0",
        clubhead_mass_kg,
    )
    reference_time_s = (
        run.impact_time_s if analysis_time_s is None else float(analysis_time_s)
    )
    if run.config.source_kind != "double_pendulum" or reference_time_s is None:
        return None
    require(
        math.isfinite(reference_time_s)
        and 0.0 <= reference_time_s <= float(run.swing_times[-1]),
        "analysis_time_s must lie within the sampled swing",
        reference_time_s,
    )

    source = make_source(
        "double_pendulum",
        run.config.scenario,
        plane=run.config.plane,
        duration=run.config.swing_duration_s,
        run_config=run.config.swing_run_config,
        torque_library=run.config.torque_library,
    )
    assert isinstance(source, AppFrameSwing)
    pendulum = source.inner
    assert isinstance(pendulum, DoublePendulumSwing)

    times = np.asarray(run.swing_times, dtype=float)
    dt = float(times[1] - times[0])
    duration = float(pendulum.duration)
    states = np.array(
        [
            (state.theta1, state.theta2, state.omega1, state.omega2)
            for state in (
                pendulum.state_at(min(float(time), duration)) for time in times
            )
        ]
    )

    parameters = pendulum.parameters
    plane_r = reference.plane_rotation(
        pendulum.plane.yaw_rad,
        pendulum.plane.side_tilt_rad,
        pendulum.plane.forward_tilt_rad,
    )
    g_inplane = reference.in_plane_gravity(plane_r, reference_g())

    torques = inverse_dynamics(parameters, states, g_inplane, dt)
    if run.config.swing_run_config.joint_locks.has_locks:
        applied = np.asarray(run.swing_applied_torques_nm, dtype=float)
        constraint_reaction = torques["applied"] - applied
    else:
        applied = torques["applied"]
        constraint_reaction = np.zeros_like(applied)

    theta = states[:, :2]
    omega = states[:, 2:]
    shoulder, wrist, clubhead = reaction_forces(
        parameters,
        theta,
        omega,
        torques["alpha"],
        g_inplane,
        clubhead_mass_kg,
    )
    ztcf = zero_torque_counterfactual(
        parameters,
        states,
        g_inplane,
        locked=run.config.swing_run_config.joint_locks.mask,
    )
    ztcf_shoulder, ztcf_wrist, ztcf_clubhead = reaction_forces(
        parameters,
        theta,
        omega,
        ztcf["acceleration"],
        g_inplane,
        clubhead_mass_kg,
    )
    net = torques["inertial"]
    power = net * omega

    offset = (
        run.swing_positions[0]
        - APP_FROM_SWING @ pendulum.sample(float(times[0])).pose[:3, 3]
    )
    wrist_local = np.stack(
        [
            parameters.l1 * np.sin(theta[:, 0]),
            -parameters.l1 * np.cos(theta[:, 0]),
        ],
        axis=1,
    )
    return KineticsSeries(
        t=times,
        joint_names=KINETIC_JOINT_NAMES,
        torque_applied_nm=applied,
        torque_constraint_reaction_nm=constraint_reaction,
        torque_gravity_nm=torques["gravity"],
        torque_damping_nm=torques["damping"],
        torque_inertial_nm=net,
        ztcf_acceleration_rad_s2=ztcf["acceleration"],
        ztcf_inertial_torque_nm=ztcf["inertial_torque"],
        power_w=power,
        shoulder_force_n=_to_app(plane_r, shoulder),
        wrist_force_n=_to_app(plane_r, wrist),
        clubhead_force_n=_to_app(plane_r, clubhead),
        ztcf_shoulder_force_n=_to_app(plane_r, ztcf_shoulder),
        ztcf_wrist_force_n=_to_app(plane_r, ztcf_wrist),
        ztcf_clubhead_force_n=_to_app(plane_r, ztcf_clubhead),
        pivot_position_m=np.asarray(offset, dtype=float),
        wrist_positions_m=_to_app(plane_r, wrist_local) + offset,
        clubhead_positions_m=np.asarray(run.swing_positions, dtype=float),
        plane_x_app=np.asarray(APP_FROM_SWING @ plane_r[:, 0]),
        plane_up_app=np.asarray(APP_FROM_SWING @ plane_r[:, 2]),
        impact_time_s=reference_time_s,
    )


def reference_g() -> float:
    """Return gravity magnitude used by the swing sources [m/s²]."""
    from shared.python.swing_sim.types import DEFAULT_GRAVITY_M_S2

    return float(DEFAULT_GRAVITY_M_S2)


# Single-slot cache: immutable runs are queried repeatedly by plots and overlays.
_CACHE: dict[str, object] = {"id": None, "series": None}


def kinetics_for_run(run: SimulationRun) -> KineticsSeries | None:
    """Return cached :func:`compute_kinetics`, keyed by run identity."""
    require(isinstance(run, SimulationRun), "run must be a SimulationRun", run)
    if _CACHE["id"] == id(run):
        return _CACHE["series"]  # type: ignore[return-value]
    series = compute_kinetics(run)
    _CACHE["id"] = id(run)
    _CACHE["series"] = series
    return series
