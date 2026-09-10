"""Whole free-system spatial momentum and off-center friction response."""

import json
from collections.abc import Callable
from dataclasses import replace

import numpy as np
from scipy.linalg import expm

from shared.python.golf_club._shaft_moving_chain import _kinetics
from shared.python.swing_sim.impact._friction_contact_response import initial_response
from shared.python.swing_sim.impact._friction_contact_trajectory import (
    FrictionTrajectoryProblem,
    FrictionTrajectoryState,
    integrate_friction_contact,
)

from .test_friction_contact_trajectory import _controls, _friction_case
from .test_normal_shaft_contact import _spatial_case
from .test_shaft_inertia import _hat


def _free_spatial_case() -> tuple:
    problem, initial = _friction_case(0.3)
    contact, shaft_state, ball = _spatial_case()
    chain = contact.chain
    elastic = replace(chain.shaft.elastic, loads=())
    chain = replace(chain, shaft=replace(chain.shaft, elastic=elastic), grips=())
    contact = replace(contact, chain=chain)
    normal = replace(problem.normal, contact=contact, anchor_history=lambda time_s: ())
    face_rotation = np.asarray(shaft_state.poses)[1, :3, :3]
    history = replace(
        initial.tangential,
        normal=face_rotation[:, 2],
        elastic_deflection_m=3e-5 * face_rotation[:, 0],
    )
    state = replace(initial.mechanical, shaft=shaft_state, ball=ball)
    return replace(problem, normal=normal), replace(
        initial, mechanical=state, tangential=history
    )


def _spatial_momentum(
    problem: FrictionTrajectoryProblem, state: FrictionTrajectoryState
) -> np.ndarray:
    """Noether momentum from canonical mass and independent world moment arms.

    Mass interpolation is shared; this independently checks force/moment
    balance, not the quadrature's physical accuracy or mesh convergence.
    """
    mechanical = state.mechanical
    shaft, ball = mechanical.shaft, mechanical.ball
    mass = _kinetics(problem.normal.contact.chain, shaft).mass
    body_momenta = (mass @ np.asarray(shaft.twists).ravel()).reshape(-1, 6)
    poses = np.concatenate((np.asarray(shaft.poses), [ball.pose]))
    ball_twist = np.asarray(ball.twist)
    momenta = np.vstack(
        (body_momenta, np.r_[0.046 * ball_twist[:3], 8e-6 * ball_twist[3:]])
    )
    rotations = poses[:, :3, :3]
    linear = np.einsum("nij,nj->ni", rotations, momenta[:, :3])
    angular = np.einsum("nij,nj->ni", rotations, momenta[:, 3:])
    angular += np.cross(poses[:, :3, 3], linear)
    return np.r_[linear.sum(axis=0), angular.sum(axis=0)]


def _perturb(
    state: FrictionTrajectoryState, rates: np.ndarray, time_s: float
) -> FrictionTrajectoryState:
    mechanical = state.mechanical
    shaft, ball = mechanical.shaft, mechanical.ball
    poses = np.concatenate((np.asarray(shaft.poses), [ball.pose]))
    twists = mechanical.twists
    moved = np.array(
        [
            pose @ expm(_hat(time_s * twist))
            for pose, twist in zip(poses, twists, strict=True)
        ]
    )
    velocities = twists + time_s * rates
    result = replace(
        mechanical,
        shaft=replace(shaft, poses=moved[:-1], twists=velocities[:-1]),
        ball=replace(ball, pose=moved[-1], twist=velocities[-1]),
    )
    return replace(state, mechanical=result)


def test_off_center_internal_contact_preserves_instantaneous_total_momentum() -> None:
    problem, initial = _free_spatial_case()
    response = initial_response(problem, initial, 0.0)
    assert np.linalg.norm(response.tangential_force_n) > 0
    assert np.linalg.norm(response.bodies.contact.face_offset_m) > 0.01
    assert np.linalg.norm(response.bodies.ball.twist_rate[3:]) > 0
    for step in (2e-7, 1e-7):
        before = _spatial_momentum(problem, _perturb(initial, response.rates, -step))
        after = _spatial_momentum(problem, _perturb(initial, response.rates, step))
        np.testing.assert_allclose((after - before) / (2 * step), 0, atol=1e-6, rtol=0)


def test_time_discrete_total_momentum_defect_decreases_with_refinement(
    record_property: Callable[[str, object], None],
) -> None:
    problem, initial = _free_spatial_case()
    expected = _spatial_momentum(problem, initial)
    errors = []
    for steps in (4, 8, 16):
        last = integrate_friction_contact(problem, initial, _controls(steps)).samples[
            -1
        ]
        # Explicit momentum scales: 1 kg m/s and 1 kg m^2/s.
        errors.append(np.linalg.norm(_spatial_momentum(problem, last.state) - expected))
    assert errors[-1] < 1e-4
    assert errors[2] < errors[1] < errors[0]
    record_property("scaled_momentum_defects", json.dumps(errors))
