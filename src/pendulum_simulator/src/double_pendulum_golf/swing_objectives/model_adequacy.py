"""Measuring what the two-link model can and cannot be asked to do.

Epic #4775 started from an observation — the speed-optimal downswing brings the
hands to a standstill at impact — and this module turns it into a measurement.

The mechanism, in one paragraph
-------------------------------
Hub torque does not only turn the arms. Through the off-diagonal mass-matrix
term ``M12`` it also drives the wrist *open*, so a hard-driving hub makes the
club lag further rather than release. In a free rollout at full drive the wrist
cock grows from 100 deg to 184 deg and the club never releases at all. The only
way this model can bring the club through to ``phi = 0`` at impact is to cut, and
then reverse, the hub torque — which necessarily decelerates the arms. Releasing
the club and stopping the hands are therefore the *same act* here, not two
independent choices the optimizer happens to combine.

That coupling is structural, and :func:`hand_speed_frontier` measures its price:
sweep a floor on hand speed at impact, and watch clubhead speed fall and then
the collocation dynamics defect exceed threshold.

Experiment settings and modeled braking
---------------------------------------
Historical parameter sweep notes:
* The original frontier exploration evaluated downswings at a duration of 0.36 s
  with +/- 250 N*m hub torque limits.
* The revised objective comparison uses a downswing duration of 0.28 s.
* In the corrected clubhead mass sweep (0.238 kg equivalent tip mass), approximately
  24% modeled braking remains: even with the corrected club coupling, the optimizer
  still reverses hub torque for roughly a quarter of the downswing to achieve club
  release at impact in this planar fixed-hub geometry.

Dynamics-defect predicate meaning
---------------------------------
The ``reachable`` attribute evaluates the predicate ``result.max_defect < _REACHABLE_DEFECT``
(where ``_REACHABLE_DEFECT = 1e-6``). This predicate strictly checks that the maximum
discrete collocation defect along the trajectory satisfies the Hermite-Simpson
integration tolerance.

Importantly:
* A dynamics-defect threshold is **not** verification of every problem constraint
  (e.g., path inequality constraints, boundary state tolerances, or actuator rate bounds).
* It is **not** a certificate of global reachability or global infeasibility of the
  underlying continuous optimal control problem. When the numerical NLP solver
  terminates with defects exceeding the threshold, it records solver failure under
  the specific grid transcription and initial guess, not a mathematical proof that no
  feasible trajectory exists.
Legacy results remain defined by this defect predicate; any metric or solver behavior
changes must be proposed and tested separately.

Why the obvious fixes do not work
---------------------------------
* **Distributed club inertia** — ruled out analytically in
  :mod:`double_pendulum_golf.swing_objectives.impact_optimality`. For a real
  driver the energy-optimal hand speed is *negative*; no physically realistic
  club moves it forward.
* **Hill-type actuation limits** — see
  :mod:`double_pendulum_golf.swing_objectives.actuation`. They stop the hub
  torque from reversing, which is right, but the resulting candidates carry
  order-one dynamics defects (~1.0), preventing their presentation as feasible
  or measured swings.

What the literature says is missing is a *moving hub*: the torso keeps rotating
through impact, and the hands pull inward along a shortening radius
(`Miura 2001 <https://doi.org/10.1007/BF02844309>`_, "parametric acceleration").
Neither is expressible with a fixed pivot and a constant ``L1``. The repository
already carries a three-segment model in
:mod:`double_pendulum_golf.physics_triple`, which is where that work belongs.

Closes #4779.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from double_pendulum_golf.swing_objectives.downswing import (
    DownswingConfig,
    DownswingOptimizer,
)
from double_pendulum_golf.swing_objectives.reference_kinematics import (
    TOUR_DRIVER_BANDS,
    ObservableBand,
)

__all__ = [
    "FrontierPoint",
    "HandSpeedFrontier",
    "hand_speed_frontier",
    "swing_observables",
]

FloatArray = npt.NDArray[np.float64]

#: Collocation dynamics defect threshold for the reachable predicate.
#: Checks whether discrete Hermite-Simpson integration defects satisfy 1e-6.
#: Does not verify all constraints or certify global reachability/infeasibility.
_REACHABLE_DEFECT = 1e-6

_RAD_TO_DEG = 180.0 / np.pi


def swing_observables(
    states: FloatArray,
    clubhead_speed: FloatArray,
    duration_s: float,
    arm_length_m: float,
) -> dict[str, float]:
    """Reduce a swing to the observables the reference bands are defined on.

    Args:
        states: ``(N, 4)`` trajectory of ``[theta1, phi, omega1, phidot]``.
        clubhead_speed: ``(N,)`` clubhead speed in m/s.
        duration_s: Downswing duration in s.
        arm_length_m: Hub-to-hands distance in m.

    Returns:
        Observable key to value, ready for
        :func:`~double_pendulum_golf.swing_objectives.reference_kinematics.score_against_reference`.

    Pre: ``states`` has at least two rows.
    Post: every returned value is finite.
    """
    if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] != 4:
        raise ValueError("states must be (N, 4) with at least two rows")

    arm_rate = float(states[-1, 2])
    club_rate = float(states[-1, 2] + states[-1, 3])
    cock = states[:, 1]
    target = 0.5 * cock[0]
    released = np.flatnonzero(cock <= target)
    release_index = int(released[0]) if released.size else len(cock) - 1

    observables = {
        "clubhead_speed_ms": float(clubhead_speed[-1]),
        "hand_speed_ms": abs(arm_rate) * arm_length_m,
        "downswing_time_s": float(duration_s),
        "club_arm_rate_ratio": (
            abs(club_rate / arm_rate) if abs(arm_rate) > 1e-9 else float("inf")
        ),
        "wrist_cock_impact_deg": float(cock[-1] * _RAD_TO_DEG),
        "release_fraction": float(release_index / (len(cock) - 1)),
    }
    if not np.isfinite(observables["club_arm_rate_ratio"]):
        # An arm that has stopped dead has no meaningful ratio; report the
        # largest band edge exceeded rather than an infinity a score cannot use.
        observables["club_arm_rate_ratio"] = 1.0e6
    return observables


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    """One point on the hand-speed / clubhead-speed trade-off.

    Attributes:
        hand_speed_floor_ms: The floor that was imposed.
        reachable: Whether the solution satisfied the dynamics defect tolerance
            predicate (max_defect < _REACHABLE_DEFECT). This is a collocation
            defect predicate, not verification of every constraint or a
            certificate of global reachability/infeasibility.
        max_defect: Largest collocation defect at the returned point.
        clubhead_speed_ms: Clubhead speed achieved at impact.
        hand_speed_ms: Hand speed achieved at impact.
        club_arm_rate_ratio: Club-to-arm angular rate ratio at impact.
        braking_fraction: Fraction of the downswing where hub torque opposes arm
            motion — the model actively braking the arms.
        peak_braking_torque_nm: Largest torque applied against the arm's motion.
    """

    hand_speed_floor_ms: float
    reachable: bool
    max_defect: float
    clubhead_speed_ms: float
    hand_speed_ms: float
    club_arm_rate_ratio: float
    braking_fraction: float
    peak_braking_torque_nm: float


@dataclass(frozen=True, slots=True)
class HandSpeedFrontier:
    """The measured price of asking this model to keep the hands moving.

    Attributes:
        points: One entry per requested floor, in the order requested.
        bands: Reference bands the frontier is read against.
    """

    points: tuple[FrontierPoint, ...]
    bands: tuple[ObservableBand, ...]

    @property
    def reachable_points(self) -> tuple[FrontierPoint, ...]:
        """Only the floors the model could actually satisfy."""
        return tuple(point for point in self.points if point.reachable)

    @property
    def max_reachable_hand_speed_ms(self) -> float:
        """Highest hand speed the model can deliver at all. Zero if none."""
        reachable = self.reachable_points
        return max((p.hand_speed_ms for p in reachable), default=0.0)

    @property
    def reaches_measured_hand_speed(self) -> bool:
        """Whether the model can reach the *measured* golfer hand-speed band."""
        band = next(b for b in self.bands if b.key == "hand_speed_ms")
        return any(band.contains(p.hand_speed_ms) for p in self.reachable_points)


def _analyse(config: DownswingConfig, floor: float) -> FrontierPoint:
    """Solve at one hand-speed floor and reduce the answer to a frontier point."""
    result = DownswingOptimizer(replace(config, min_hand_speed_ms=floor)).solve(
        "clubhead_speed"
    )
    states, torques = result.states, result.torques
    arm_rate = float(states[-1, 2])
    club_rate = float(states[-1, 2] + states[-1, 3])

    opposing = np.sign(torques[:, 0]) * np.sign(states[:, 2]) < 0
    braking_torque = np.abs(torques[opposing, 0])
    return FrontierPoint(
        hand_speed_floor_ms=floor,
        reachable=bool(result.max_defect < _REACHABLE_DEFECT),
        max_defect=float(result.max_defect),
        clubhead_speed_ms=float(result.signals.clubhead_speed[-1]),
        hand_speed_ms=abs(arm_rate) * config.params.L1,
        club_arm_rate_ratio=(abs(club_rate / arm_rate) if abs(arm_rate) > 1e-9 else 1.0e6),
        braking_fraction=float(np.mean(opposing)),
        peak_braking_torque_nm=float(braking_torque.max()) if braking_torque.size else 0.0,
    )


def hand_speed_frontier(
    config: DownswingConfig,
    floors_ms: Sequence[float],
    bands: tuple[ObservableBand, ...] = TOUR_DRIVER_BANDS,
) -> HandSpeedFrontier:
    """Measure clubhead speed against an imposed floor on hand speed at impact.

    Each floor is solved as its own clubhead-speed maximisation. Rising floors
    buy realism and cost speed, until the collocation dynamics defect exceeds
    the tolerance threshold (_REACHABLE_DEFECT) — which is the numerical price
    under this collocation transcription that this module exists to record.

    Args:
        config: Base downswing configuration; its ``min_hand_speed_ms`` is
            replaced per point.
        floors_ms: Hand-speed floors to impose, in m/s.
        bands: Reference bands used to judge whether the model reaches measured
            golfer behaviour.

    Returns:
        The frontier.

    Pre: ``floors_ms`` is non-empty and every entry is finite and non-negative.
    Post: one point per requested floor, in order.
    """
    floors = [float(value) for value in floors_ms]
    if not floors:
        raise ValueError("floors_ms must contain at least one floor")
    if any(not np.isfinite(v) or v < 0.0 for v in floors):
        raise ValueError(f"floors_ms must be finite and non-negative, got {floors}")

    return HandSpeedFrontier(
        points=tuple(_analyse(config, floor) for floor in floors), bands=bands
    )
