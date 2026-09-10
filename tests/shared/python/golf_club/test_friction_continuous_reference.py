"""Independent continuous sticking history with high-order adaptive integration.

Canonical body/shaft mechanics and Lie chart differential are shared. The
history ODE and time method do not use the endpoint return map or root solve.
This checks smooth time convergence, not the physical coefficients or sliding.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from shared.python.golf_club._rkmk_step import material_chart_rates
from shared.python.swing_sim.impact._friction_contact_response import friction_response
from shared.python.swing_sim.impact._friction_contact_trajectory import (
    FrictionTrajectoryProblem,
    FrictionTrajectoryState,
    integrate_friction_contact,
)
from shared.python.swing_sim.impact._friction_transport import ContactTransport
from shared.python.swing_sim.impact._normal_contact_trajectory import _shift

from .test_friction_contact_trajectory import _controls
from .test_friction_momentum import _free_spatial_case


@dataclass(frozen=True)
class _StickReference:
    problem: FrictionTrajectoryProblem
    initial: FrictionTrajectoryState

    def unpack(self, time_s: float, vector: np.ndarray) -> tuple:
        shape = self.initial.mechanical.twists.shape
        size = int(np.prod(shape))
        coordinates, velocity = vector[:size], vector[size : 2 * size]
        mechanical = _shift(
            self.initial.mechanical, coordinates.reshape(shape), velocity.reshape(shape)
        )
        model = self.problem.normal.contact_at(time_s)
        contact = model.kinematics(mechanical.shaft, mechanical.ball)
        normal = np.asarray(contact.normal)
        raw_history = vector[-3:]
        assert abs(raw_history @ normal) < 1e-12  # m; monitor constraint drift
        history = np.cross(normal, np.cross(raw_history, normal))
        state = replace(
            self.initial,
            mechanical=mechanical,
            tangential=replace(
                self.initial.tangential, normal=normal, elastic_deflection_m=history
            ),
        )
        return coordinates.reshape(shape), state, model, contact

    def derivative(self, time_s: float, vector: np.ndarray) -> np.ndarray:
        coordinates, state, model, contact = self.unpack(time_s, vector)
        response = friction_response(model, state, contact, "ball")
        law = self.problem.tangential_law
        effort = np.linalg.norm(response.tangential_force_n)
        assert response.normal.force_n > 0
        assert effort < 0.5 * law.friction_coefficient * response.normal.force_n
        normal = np.asarray(contact.normal)
        # Continuous constitutive spin, derived independently of finite Q.
        spin = (
            np.asarray(contact.face.pose)[:3, :3] @ np.asarray(contact.face.twist)[3:]
        )
        if self.problem.transport is ContactTransport.MEAN_NORMAL_SPIN:
            ball_spin = (
                np.asarray(contact.ball.pose)[:3, :3]
                @ np.asarray(contact.ball.twist)[3:]
            )
            spin = spin + 0.5 * normal * (normal @ (ball_spin - spin))
        slip = np.cross(normal, np.cross(contact.relative_velocity_mps, normal))
        history_rate = np.cross(spin, state.tangential.elastic_deflection_m) + slip
        chart_rates = material_chart_rates(coordinates, state.mechanical.twists)
        return np.r_[chart_rates.ravel(), response.rates.ravel(), history_rate]

    def integrate(
        self, end_s: float, rtol: float, atol: float
    ) -> FrictionTrajectoryState:
        twists = self.initial.mechanical.twists
        initial = np.r_[
            np.zeros(twists.size),
            twists.ravel(),
            self.initial.tangential.elastic_deflection_m,
        ]
        result = solve_ivp(
            self.derivative,
            (0, end_s),
            initial,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=end_s / 8,
        )
        assert result.success and result.t[-1] == end_s
        return self.unpack(end_s, result.y[:, -1])[1]


def _scaled_output(state: FrictionTrajectoryState) -> np.ndarray:
    mechanical = state.mechanical
    poses = np.concatenate((mechanical.shaft.poses, [mechanical.ball.pose]))
    # 1 m translations, unit rotation matrices, 1 m/s linear velocities,
    # 100 rad/s angular velocities and 1 mm elastic history are explicit scales.
    velocity_scales = np.array([1, 1, 1, 100, 100, 100])
    return np.r_[
        poses[:, :3, 3].ravel(),
        poses[:, :3, :3].ravel(),
        (mechanical.twists / velocity_scales).ravel(),
        np.asarray(state.tangential.elastic_deflection_m) / 0.001,
    ]


@pytest.mark.parametrize("transport", list(ContactTransport))
def test_nonlinear_sticking_refines_against_independent_continuous_history(
    transport: ContactTransport,
    record_property: Callable[[str, object], None],
) -> None:
    problem, initial = _free_spatial_case()
    problem = replace(problem, transport=transport)
    reference = _StickReference(problem, initial)
    end = _controls().bounds_s[1]
    expected = _scaled_output(reference.integrate(end, 1e-11, 1e-13))
    tighter = _scaled_output(reference.integrate(end, 1e-12, 1e-14))
    np.testing.assert_allclose(expected, tighter, atol=1e-10, rtol=0)
    errors = []
    for steps in (4, 8, 16):
        final = integrate_friction_contact(problem, initial, _controls(steps)).samples[
            -1
        ]
        errors.append(float(np.linalg.norm(_scaled_output(final.state) - tighter)))
        assert final.plastic_dissipation_j == 0
    assert errors[-1] < 1e-3
    ratios = np.asarray(errors[:-1]) / errors[1:]
    np.testing.assert_array_less(1.7, ratios)
    np.testing.assert_array_less(ratios, 2.3)
    record_property("continuous_sticking_errors", json.dumps(errors))
