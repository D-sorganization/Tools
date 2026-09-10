"""Continuous planar sliding oracle, independent of the discrete return map.

Canonical mechanical response and chart differential are shared. Constant-sign
planar slip and a positive outward plastic rate keep active history on its
moving Coulomb boundary. Zero force removes that history during unloading.
Only initially compressed sliding and its release are covered; new contact,
reversal and nonplanar friction remain outside this reference.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp

from shared.python.golf_club._rkmk_step import material_chart_rates
from shared.python.swing_sim.impact._friction_contact_response import (
    FrictionContactResponse,
)
from shared.python.swing_sim.impact._friction_trajectory_contracts import (
    FrictionTrajectoryProblem,
    FrictionTrajectoryState,
)
from shared.python.swing_sim.impact._normal_contact_trajectory import (
    NormalContactTrajectoryState,
    _shift,
)
from shared.python.swing_sim.impact._normal_contact_work import (
    NormalContactWork,
    normal_contact_work,
)
from shared.python.swing_sim.impact._normal_shaft_contact import NormalShaftContact
from shared.python.swing_sim.impact._spatial_contact_kinematics import (
    ContactBodyState,
    PlaneSphereKinematics,
)

from .test_friction_contact_trajectory import _friction_case

PlanarGeometry = tuple[
    np.ndarray, NormalContactTrajectoryState, NormalShaftContact, PlaneSphereKinematics
]


def _origin_rates(
    body: ContactBodyState, rate: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(body.pose)[:3, :3]
    twist = np.asarray(body.twist)
    return rotation @ twist[:3], rotation @ (rate[:3] + np.cross(twist[3:], twist[:3]))


def gap_acceleration(
    contact: PlaneSphereKinematics, face_rate: np.ndarray, ball_rate: np.ndarray
) -> float:
    """Differentiate the fixed material plane's signed center distance twice.

    The plane offset contributes a constant n_material dot p_material and
    therefore disappears from both derivatives. Body linear twist derivatives
    require the omega cross v term before becoming world accelerations.
    """
    face, ball = contact.face, contact.ball
    face_pose, ball_pose = np.asarray(face.pose), np.asarray(ball.pose)
    normal = np.asarray(contact.normal)
    spin = face_pose[:3, :3] @ np.asarray(face.twist)[3:]
    alpha = face_pose[:3, :3] @ face_rate[3:]
    normal_rate = np.cross(spin, normal)
    normal_acceleration = np.cross(alpha, normal) + np.cross(spin, normal_rate)
    face_velocity, face_acceleration = _origin_rates(face, face_rate)
    ball_velocity, ball_acceleration = _origin_rates(ball, ball_rate)
    return float(
        normal_acceleration @ (ball_pose[:3, 3] - face_pose[:3, 3])
        + 2 * normal_rate @ (ball_velocity - face_velocity)
        + normal @ (ball_acceleration - face_acceleration)
    )


def _slip(contact: PlaneSphereKinematics) -> np.ndarray:
    normal = np.asarray(contact.normal)
    return np.asarray(np.cross(normal, np.cross(contact.relative_velocity_mps, normal)))


def _assert_planar(contact: PlaneSphereKinematics) -> None:
    assert abs(contact.normal[1]) < 1e-12
    assert abs(contact.relative_velocity_mps[1]) < 1e-12
    for body in (contact.face, contact.ball):
        spin = np.asarray(body.pose)[:3, :3] @ np.asarray(body.twist)[3:]
        assert np.max(np.abs(spin[[0, 2]])) < 1e-10
    assert _slip(contact)[0] > 1.0  # m/s; constant-sign branch, clear of reversal


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    assert np.all(np.isfinite(result))
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SlidingReferenceTrace:
    """Owned trajectory and detected raw-force/gap zero crossings.

    Event detection uses sign changes between accepted steps. These records
    are not a proof that arbitrary repeated or grazing events are complete.
    """

    times_s: np.ndarray
    vectors: np.ndarray
    event_times: tuple[np.ndarray, ...]
    event_vectors: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        times, vectors = _readonly(self.times_s), _readonly(self.vectors)
        assert times.ndim == 1 and vectors.ndim == 2
        assert vectors.shape[1] == times.size and np.all(np.diff(times) > 0)
        event_times = tuple(_readonly(value) for value in self.event_times)
        event_vectors = tuple(
            _readonly(value.reshape(-1, vectors.shape[0]))
            for value in self.event_vectors
        )
        assert len(event_times) == len(event_vectors) == 2
        for time, vector in zip(event_times, event_vectors, strict=True):
            assert time.ndim == 1 and vector.shape[0] == time.size
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "vectors", vectors)
        object.__setattr__(self, "event_times", event_times)
        object.__setattr__(self, "event_vectors", event_vectors)


@dataclass(frozen=True)
class SlidingReference:
    problem: FrictionTrajectoryProblem
    initial: FrictionTrajectoryState

    def geometry(self, time_s: float, vector: np.ndarray) -> PlanarGeometry:
        initial_mechanical = self.initial.mechanical
        shape = initial_mechanical.twists.shape
        size = int(np.prod(shape))
        coordinates = vector[:size].reshape(shape)
        mechanical = _shift(
            initial_mechanical, coordinates, vector[size : 2 * size].reshape(shape)
        )
        normal_problem = self.problem.normal
        model = normal_problem.contact_at(time_s)
        contact = model.kinematics(mechanical.shaft, mechanical.ball)
        _assert_planar(contact)
        return coordinates, mechanical, model, contact

    def snapshot(
        self, time_s: float, vector: np.ndarray
    ) -> tuple[np.ndarray, FrictionTrajectoryState, FrictionContactResponse]:
        geometry = self.geometry(time_s, vector)
        _, _, model, contact = geometry
        normal = normal_contact_work(model.law, -contact.gap_m, -contact.gap_rate_mps)
        law = self.problem.tangential_law
        slip = _slip(contact)
        traction = (
            -law.friction_coefficient * normal.force_n * slip / np.linalg.norm(slip)
        )
        return self.respond(geometry, normal, traction)

    def respond(
        self, geometry: PlanarGeometry, normal: NormalContactWork, traction: np.ndarray
    ) -> tuple[np.ndarray, FrictionTrajectoryState, FrictionContactResponse]:
        """Share force application without sharing an oracle's phase law."""
        coordinates, mechanical, model, contact = geometry
        law = self.problem.tangential_law
        history = replace(
            self.initial.tangential,
            normal=contact.normal,
            elastic_deflection_m=-traction / law.stiffness_n_per_m,
        )
        state = FrictionTrajectoryState(mechanical, history)
        force = normal.force_n * np.asarray(contact.normal) + traction
        bodies = model._body_response(mechanical.shaft, contact, force, "ball")
        response = FrictionContactResponse(bodies, normal, tuple(traction))
        return coordinates, state, response

    def plastic_power(self, response: FrictionContactResponse) -> float:
        if response.normal.force_n == 0:
            return 0.0
        bodies = response.bodies
        normal_problem = self.problem.normal
        model = normal_problem.contact
        normal_law, tangent_law = model.law, self.problem.tangential_law
        contact = bodies.contact
        acceleration = gap_acceleration(
            contact,
            np.asarray(bodies.shaft.twist_rates)[model.face_node],
            np.asarray(bodies.ball.twist_rate),
        )
        normal_rate = (
            -normal_law.stiffness_n_per_m * contact.gap_rate_mps
            - normal_law.damping_n_s_per_m * acceleration
        )
        cap_rate = (
            tangent_law.friction_coefficient
            * normal_rate
            / tangent_law.stiffness_n_per_m
        )
        plastic_rate = float(np.linalg.norm(_slip(contact)) - cap_rate)
        assert plastic_rate > 0  # moving boundary must be loaded outward
        return tangent_law.friction_coefficient * response.normal.force_n * plastic_rate

    def derivative(self, time_s: float, vector: np.ndarray) -> np.ndarray:
        coordinates, state, response = self.snapshot(time_s, vector)
        chart_rate = material_chart_rates(coordinates, state.mechanical.twists)
        result = np.r_[
            chart_rate.ravel(),
            response.rates.ravel(),
            response.work_powers_w,
            response.normal.force_n,
            response.tangential_force_n,
            self.plastic_power(response),
        ]
        return np.asarray(result)

    def integrate(
        self, end_s: float, rtol: float, atol: float
    ) -> tuple[FrictionTrajectoryState, FrictionContactResponse, np.ndarray]:
        trace = self.trace(end_s, rtol, atol)
        _, state, response = self.snapshot(end_s, trace.vectors[:, -1])
        return state, response, trace.vectors[-10:, -1]

    def force_cutoff_event(self, time_s: float, vector: np.ndarray) -> float:
        _, _, model, contact = self.geometry(time_s, vector)
        return float(
            model.law.unclipped_normal_force(-contact.gap_m, -contact.gap_rate_mps)
        )

    def separation_event(self, time_s: float, vector: np.ndarray) -> float:
        _, _, _, contact = self.geometry(time_s, vector)
        return contact.gap_m

    def trace(self, end_s: float, rtol: float, atol: float) -> SlidingReferenceTrace:
        mechanical = self.initial.mechanical
        twists = mechanical.twists
        # Ten integrals: five work ports, normal/vector tangent impulse, plastic work.
        initial = np.r_[np.zeros(twists.size), twists.ravel(), np.zeros(10)]
        _, state, response = self.snapshot(0, initial)
        assert response.normal.force_n > 0, (
            "reference requires initially compressed sliding"
        )
        assert state.tangential == self.initial.tangential, (
            "initial history must be saturated"
        )
        result = solve_ivp(
            self.derivative,
            (0, end_s),
            initial,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=end_s / 8,
            events=(self.force_cutoff_event, self.separation_event),
        )
        assert result.success and result.t[-1] == end_s
        if any(len(events) > 1 for events in result.t_events):
            raise ValueError(
                "reference excludes repeated contact or cutoff transitions"
            )
        return SlidingReferenceTrace(
            result.t, result.y, tuple(result.t_events), tuple(result.y_events)
        )


def sliding_case(ball_velocity_mps: float = -0.4) -> SlidingReference:
    problem, initial = _friction_case(0.02)
    mechanical = initial.mechanical
    ball = replace(mechanical.ball, twist=(5, 0, ball_velocity_mps, 0, 0, 0))
    initial = replace(initial, mechanical=replace(mechanical, ball=ball))
    provisional = SlidingReference(problem, initial)
    twists = initial.mechanical.twists
    _, saturated, _ = provisional.snapshot(
        0, np.r_[np.zeros(twists.size), twists.ravel()]
    )
    return SlidingReference(problem, saturated)


def sliding_observables(
    state: FrictionTrajectoryState,
    response: FrictionContactResponse,
    integrals: np.ndarray,
) -> np.ndarray:
    """Separate SI outputs; sharing this readout does not share time integration."""
    ball = state.mechanical.ball
    rotation, twist = np.asarray(ball.pose)[:3, :3], np.asarray(ball.twist)
    result = np.r_[
        response.normal.force_n,
        response.tangential_force_n[0],
        (rotation @ twist[:3])[0],
        (rotation @ twist[3:])[1],
        integrals[5],
        integrals[6],
        integrals[9],
        integrals[:5],
    ]
    return np.asarray(result)
