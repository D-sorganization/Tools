"""Continuous planar free/elastic/sliding reference with explicit entry events.

Only positive slip, one first touch and one elastic-to-sliding transition are
qualified. Event integration extends each smooth phase to locate its boundary;
it never applies the production endpoint solver or discrete return map.
"""

from dataclasses import dataclass, replace
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp

from shared.python.golf_club._rkmk_step import material_chart_rates
from shared.python.swing_sim.impact._friction_contact_response import (
    FrictionContactResponse,
)
from shared.python.swing_sim.impact._friction_trajectory_contracts import (
    FrictionTrajectoryState,
)
from shared.python.swing_sim.impact._normal_contact_work import (
    NormalContactWork,
    normal_contact_work,
)
from shared.python.swing_sim.impact._normal_shaft_contact import NormalShaftContact
from shared.python.swing_sim.impact._spatial_contact_kinematics import (
    PlaneSphereKinematics,
)

from ._friction_sliding_reference import (
    SlidingReference,
    SlidingReferenceTrace,
    _readonly,
    _slip,
    sliding_case,
)


def _incoming_normal(
    model: NormalShaftContact, contact: PlaneSphereKinematics
) -> NormalContactWork:
    """Use the dashpot right-hand limit only at the resolved entry boundary."""
    compression, rate = -contact.gap_m, -contact.gap_rate_mps
    if compression > 0:
        return normal_contact_work(model.law, compression, rate)
    assert abs(compression) < 1e-10 and rate > 0, "unresolved contact entry"
    force = model.law.unclipped_normal_force(0, rate)
    assert 0 < force < model.law.maximum_force_n
    power = force * rate
    return NormalContactWork(force, 0, 0, power, 0, power)


@dataclass(frozen=True)
class EntryTrace:
    entry_times_s: tuple[float, float]
    entry_vectors: tuple[np.ndarray, np.ndarray]
    release: SlidingReferenceTrace

    def __post_init__(self) -> None:
        touch, sliding = self.entry_times_s
        assert np.isfinite(touch) and 0 < touch < sliding
        vectors = tuple(_readonly(value) for value in self.entry_vectors)
        assert len(vectors) == 2 and vectors[0].shape == vectors[1].shape
        object.__setattr__(self, "entry_vectors", vectors)


@dataclass(frozen=True)
class _EntryEvent:
    reference: "EntryReference"
    active: bool
    terminal = True
    direction = -1

    def __call__(self, time_s: float, vector: np.ndarray) -> float:
        if not self.active:
            sliding = self.reference.sliding
            return sliding.separation_event(time_s, vector)
        _, _, response = self.reference.elastic_snapshot(time_s, vector)
        sliding = self.reference.sliding
        problem = sliding.problem
        law = problem.tangential_law
        return float(
            law.friction_coefficient * response.normal.force_n
            - law.stiffness_n_per_m * vector[-1]
        )


@dataclass(frozen=True)
class EntryReference:
    sliding: SlidingReference
    tolerances: tuple[float, float] = (1e-11, 1e-13)

    def initial_vector(self) -> np.ndarray:
        initial = self.sliding.initial
        mechanical = initial.mechanical
        twists = mechanical.twists
        assert initial.tangential.elastic_energy_j == 0
        vector = np.r_[np.zeros(twists.size), twists.ravel(), np.zeros(11)]
        assert self.sliding.separation_event(0, vector) > 0
        return np.asarray(vector)

    def free_snapshot(
        self, time_s: float, vector: np.ndarray
    ) -> tuple[np.ndarray, FrictionTrajectoryState, FrictionContactResponse]:
        geometry = self.sliding.geometry(time_s, vector)
        return self.sliding.respond(
            geometry, NormalContactWork(0, 0, 0, 0, 0, 0), np.zeros(3)
        )

    def elastic_snapshot(
        self, time_s: float, vector: np.ndarray
    ) -> tuple[np.ndarray, FrictionTrajectoryState, FrictionContactResponse]:
        geometry = self.sliding.geometry(time_s, vector)
        _, _, model, contact = geometry
        normal = _incoming_normal(model, contact)
        problem = self.sliding.problem
        law = problem.tangential_law
        slip = _slip(contact)
        traction = -law.stiffness_n_per_m * vector[-1] * slip / np.linalg.norm(slip)
        return self.sliding.respond(geometry, normal, traction)

    def derivative(self, time_s: float, vector: np.ndarray, active: bool) -> np.ndarray:
        snapshot = self.elastic_snapshot if active else self.free_snapshot
        coordinates, state, response = snapshot(time_s, vector)
        chart_rate = material_chart_rates(coordinates, state.mechanical.twists)
        contact = response.bodies.contact
        history_rate = float(np.linalg.norm(_slip(contact))) if active else 0.0
        return np.asarray(
            np.r_[
                chart_rate.ravel(),
                response.rates.ravel(),
                response.work_powers_w,
                response.normal.force_n,
                response.tangential_force_n,
                0,  # plastic work vanishes in both entry phases
                history_rate,
            ]
        )

    def segment(
        self, bounds: tuple[float, float], vector: np.ndarray, active: bool
    ) -> tuple[float, np.ndarray]:
        event = _EntryEvent(self, active)
        result = solve_ivp(
            partial(self.derivative, active=active),
            bounds,
            vector,
            method="DOP853",
            rtol=self.tolerances[0],
            atol=self.tolerances[1],
            max_step=(bounds[1] - bounds[0]) / 16,
            events=event,
        )
        assert result.success and len(result.t_events[0]) == 1
        time = float(result.t[-1])
        final = np.asarray(result.y[:, -1])
        tolerance = 1e-8 if active else 1e-11  # force [N] versus gap [m]
        assert abs(event(time, final)) < tolerance, "entry event is unresolved"
        assert bounds[0] < time < bounds[1]
        return time, final

    def trace(self, end_s: float) -> EntryTrace:
        vector = self.initial_vector()
        touch, touched = self.segment((0, end_s), vector, False)
        sliding, loaded = self.segment((touch, end_s), touched, True)
        result = solve_ivp(
            self.sliding.derivative,
            (sliding, end_s),
            loaded[:-1],
            method="DOP853",
            rtol=self.tolerances[0],
            atol=self.tolerances[1],
            max_step=(end_s - sliding) / 16,
            events=(self.sliding.force_cutoff_event, self.sliding.separation_event),
        )
        assert result.success and result.t[-1] == end_s
        assert [len(times) for times in result.t_events] == [1, 1]
        trace = SlidingReferenceTrace(
            result.t, result.y, tuple(result.t_events), tuple(result.y_events)
        )
        return EntryTrace((touch, sliding), (touched, loaded), trace)


def entry_case() -> EntryReference:
    sliding = sliding_case()
    initial = sliding.initial
    mechanical = initial.mechanical
    ball = mechanical.ball
    pose = np.asarray(ball.pose).copy()
    pose[2, 3] += 0.0011  # shift -1 mm compression to +0.1 mm clearance
    initial = replace(
        initial,
        mechanical=replace(mechanical, ball=replace(ball, pose=pose)),
        tangential=replace(initial.tangential, elastic_deflection_m=(0, 0, 0)),
    )
    return EntryReference(replace(sliding, initial=initial))
