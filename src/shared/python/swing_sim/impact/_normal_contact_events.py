"""Private adaptive spatial normal contact, retaining canonical force/work laws."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp

from ...golf_club._grip_contracts import finite_array
from ...golf_club._rkmk_step import material_chart_rates
from ._normal_contact_trajectory import (
    ContactWorkIntegrals,
    NormalContactTrajectoryProblem,
    NormalContactTrajectorySample,
    NormalContactTrajectoryState,
    _Ledger,
    _powers,
    _rates,
    _shift,
)
from ._normal_event_contracts import (
    AdaptiveContactControls,
    AdaptiveNormalContactTrajectory,
    NormalContactEvent,
    NormalContactRootSample,
)
from ._normal_event_contracts import (
    ContactAbsoluteTolerances as ContactAbsoluteTolerances,
)
from ._normal_shaft_contact import NormalShaftContact, NormalShaftContactResponse
from ._spatial_contact_kinematics import PlaneSphereKinematics


@dataclass
class _Budget:
    problem: NormalContactTrajectoryProblem
    maximum: int
    count: int = 0

    def evaluate(
        self, state: NormalContactTrajectoryState, time_s: float
    ) -> NormalShaftContactResponse:
        if self.count >= self.maximum:
            raise ValueError("adaptive contact evaluation budget exhausted")
        self.count += 1
        response: NormalShaftContactResponse = self.problem.evaluate(state, time_s)
        return response


@dataclass(frozen=True)
class _Chart:
    initial: NormalContactTrajectoryState
    budget: _Budget
    ledger: _Ledger
    contact: NormalShaftContact

    @property
    def rows(self) -> int:
        return int(self.initial.twists.shape[0])

    def unpack(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        size = self.rows * 6
        vector = finite_array(vector, (2 * size + 5,), "adaptive chart state")
        return (
            vector[:size].reshape(-1, 6),
            vector[size : 2 * size].reshape(-1, 6),
            vector[-5:],
        )

    def state(self, vector: np.ndarray) -> NormalContactTrajectoryState:
        coordinates, velocities, _ = self.unpack(vector)
        return _shift(self.initial, coordinates, velocities)

    def geometry(self, vector: np.ndarray) -> PlaneSphereKinematics:
        state = self.state(vector)
        return self.contact.kinematics(state.shaft, state.ball)

    def unclipped_force(self, geometry: PlaneSphereKinematics) -> float:
        law = self.contact.law
        return float(
            law.unclipped_normal_force(-geometry.gap_m, -geometry.gap_rate_mps)
        )

    def touch_limit(self, geometry: PlaneSphereKinematics) -> float:
        law = self.contact.law
        force = max(0.0, -law.damping_n_s_per_m * geometry.gap_rate_mps)
        if force > law.maximum_force_n:
            raise ValueError(
                "normal contact force ceiling invalidates first-touch limit"
            )
        return float(force)

    def root_sample(self, time_s: float, vector: np.ndarray) -> NormalContactRootSample:
        state = self.state(vector)
        return NormalContactRootSample(
            time_s, state, self.budget.evaluate(state, time_s)
        )

    def derivative(self, time_s: float, vector: np.ndarray) -> np.ndarray:
        coordinates, velocities, _ = self.unpack(vector)
        state = _shift(self.initial, coordinates, velocities)
        response = self.budget.evaluate(state, float(time_s))
        result: np.ndarray = np.r_[
            material_chart_rates(coordinates, velocities).ravel(),
            _rates(response).ravel(),
            _powers(response),
        ]
        return result

    def sample(
        self, time_s: float, vector: np.ndarray
    ) -> NormalContactTrajectorySample:
        state = self.state(vector)
        work = ContactWorkIntegrals(
            tuple(np.asarray(self.ledger.work.values) + vector[-5:])
        )
        response = self.budget.evaluate(state, time_s)
        return replace(self.ledger, work=work).sample(time_s, state, response)


@dataclass(frozen=True)
class _Event:
    chart: _Chart
    force: bool
    direction: float
    kind: str
    terminal: bool = False

    def __call__(self, time_s: float, vector: np.ndarray) -> float:
        geometry = self.chart.geometry(vector)
        if self.force:
            return self.chart.unclipped_force(geometry)
        return float(geometry.gap_m)

    def record(self, time_s: float, vector: np.ndarray) -> NormalContactEvent | None:
        geometry = self.chart.geometry(vector)
        if self.force and geometry.gap_m >= 0:
            return None  # A zero of the extended force in clearance is not contact.
        # Grazing roots have no transversal event identity; do not invent one.
        if not self.force and geometry.gap_rate_mps * self.direction <= 0:
            return None
        force_limit = (
            self.chart.touch_limit(geometry) if self.kind == "first_touch" else 0.0
        )
        sample = self.chart.root_sample(time_s, vector)
        return NormalContactEvent(self.kind, sample, force_limit)


def _advance(
    chart: _Chart,
    cell: tuple[float, float, float],
    controls: AdaptiveContactControls,
) -> tuple[NormalContactTrajectorySample, tuple[NormalContactEvent, ...]]:
    events = tuple(
        _Event(chart, force, direction, kind)
        for force, direction, kind in (
            (False, -1, "first_touch"),
            (True, -1, "force_release"),
            (False, 1, "geometric_separation"),
            (True, 1, "force_reactivation"),
        )
    )
    initial = np.r_[np.zeros(chart.rows * 6), chart.initial.twists.ravel(), np.zeros(5)]
    solution = solve_ivp(
        chart.derivative,
        (cell[0], cell[2]),
        initial,
        method="DOP853",
        max_step=controls.maximum_step_s,
        rtol=controls.relative_tolerance,
        atol=controls.absolute_tolerances.vector(chart.rows),
        events=events,
    )
    if not solution.success or solution.status != 0 or solution.t[-1] != cell[2]:
        raise ValueError("adaptive contact did not complete the requested chart")
    found = []
    for event, times, vectors in zip(
        events, solution.t_events, solution.y_events, strict=True
    ):
        for time_s, vector in zip(times, vectors, strict=True):
            item = event.record(float(time_s), vector)
            if item is not None:
                found.append(item)
    return chart.sample(cell[2], solution.y[:, -1]), tuple(found)


def integrate_adaptive_normal_contact(
    problem: NormalContactTrajectoryProblem,
    initial: NormalContactTrajectoryState,
    controls: AdaptiveContactControls,
) -> AdaptiveNormalContactTrajectory:
    """Integrate local body charts/work and locate transversal scalar roots.

    DOP853 controls local error and uses dense root interpolation. The maximum
    accepted step is explicit; repeated roots inside one step can be missed.
    Reset each chart at requested endpoints, retain its proper-pose domains,
    and count all force-response evaluations (including root/output samples).
    No corrected energy, event completeness, peak maximum, tangential history,
    material qualification or acoustic energy is inferred from success.
    """
    if not isinstance(problem, NormalContactTrajectoryProblem):
        raise TypeError("problem must be NormalContactTrajectoryProblem")
    if not isinstance(initial, NormalContactTrajectoryState):
        raise TypeError("initial must be NormalContactTrajectoryState")
    if type(controls) is not AdaptiveContactControls:
        raise TypeError("adaptive contact requires AdaptiveContactControls")
    cells = controls.time_cells()
    budget = _Budget(problem, controls.max_evaluations)
    try:
        # Subnormal time increments used internally by SciPy are permissible;
        # every physical state, response and work endpoint is validated finite.
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            response = budget.evaluate(initial, controls.bounds_s[0])
            ledger = _Ledger(response.energy.total_energy_j)
            samples = [ledger.sample(controls.bounds_s[0], initial, response)]
            found: dict[tuple[str, float], NormalContactEvent] = {}
            for cell in cells:
                chart = _Chart(samples[-1].state, budget, ledger, problem.contact)
                final, events = _advance(chart, cell, controls)
                samples.append(final)
                ledger = replace(ledger, work=final.work)
                for item in events:
                    found[(item.kind, item.sample.time_s)] = item
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        raise ValueError("adaptive contact numerical evaluation failed") from error
    ordered = tuple(sorted(found.values(), key=lambda item: item.sample.time_s))
    return AdaptiveNormalContactTrajectory(
        controls, tuple(samples), ordered, budget.count
    )


__all__ = ()
