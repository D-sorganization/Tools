"""Ranking the swing objectives by how closely they match real golfers.

This is the question epic #4766 set out to answer — *which objective is a good
golfer actually optimizing?* — asked properly: solve each objective under
identical conditions, reduce each resulting swing to the observables the
literature reports, and score it against measured bands.

The measured answer, and why it is not the answer anyone wanted
---------------------------------------------------------------
Run over the shipped two-link model, the five objectives land within **0.6% of
each other** on total deviation from measured golfer kinematics, while all of
them sit an order of magnitude outside the bands. The spread between objectives
is far smaller than the gap between any of them and a real swing.

So the objective is **not** what makes these swings unrealistic. The dominant
term is the model's structural inability to keep the hands moving through impact
(see :mod:`double_pendulum_golf.swing_objectives.model_adequacy`), and until that
is addressed, ranking objectives on realism is measuring noise on top of a much
larger bias. :func:`objective_realism_ranking` reports the spread alongside the
ranking precisely so a caller can see when the ranking is not meaningful, and
:attr:`ObjectiveRealismRanking.is_discriminating` answers that directly.

This is a negative result about the *model*, not about the objectives. Under a
model that can reach the measured bands, the same ranking becomes informative.

Closes #4780.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from double_pendulum_golf.swing_objectives.downswing import (
    DownswingConfig,
    DownswingOptimizer,
)
from double_pendulum_golf.swing_objectives.model_adequacy import swing_observables
from double_pendulum_golf.swing_objectives.objectives import SWING_OBJECTIVES
from double_pendulum_golf.swing_objectives.reference_kinematics import (
    TOUR_DRIVER_BANDS,
    ObservableBand,
    score_against_reference,
)

__all__ = [
    "ObjectiveRealism",
    "ObjectiveRealismRanking",
    "objective_realism_ranking",
]

#: Relative spread below which the ranking is reported as non-discriminating:
#: the objectives differ by less than this fraction of their mean deviation.
_DISCRIMINATION_THRESHOLD = 0.05

_REACHABLE_DEFECT = 1e-6


@dataclass(frozen=True, slots=True)
class ObjectiveRealism:
    """How one objective's optimum compares with measured golfer kinematics.

    Attributes:
        key: Objective identifier.
        feasible: Whether the solve produced a dynamically feasible trajectory.
        total_deviation: Summed distance outside the measured bands, in
            half-widths. Zero means every observable is inside its band.
        inside_count: How many observables fall inside their band.
        worst_observable: The observable furthest outside its band.
        clubhead_speed_ms: Clubhead speed at impact.
        observables: Every scored observable, for reporting.
    """

    key: str
    feasible: bool
    total_deviation: float
    inside_count: int
    worst_observable: str
    clubhead_speed_ms: float
    observables: dict[str, float]


@dataclass(frozen=True, slots=True)
class ObjectiveRealismRanking:
    """The objectives ordered by distance from measured golfer kinematics.

    Attributes:
        entries: Objectives, most golf-like first.
        band_count: How many observables were available to score against.
    """

    entries: tuple[ObjectiveRealism, ...]
    band_count: int

    @property
    def best(self) -> ObjectiveRealism:
        """The objective whose optimum sits closest to measured behaviour."""
        return self.entries[0]

    @property
    def deviation_spread(self) -> float:
        """Difference in total deviation between the best and worst objective."""
        if len(self.entries) < 2:
            return 0.0
        return float(self.entries[-1].total_deviation - self.entries[0].total_deviation)

    @property
    def mean_deviation(self) -> float:
        """Average distance from measured behaviour across all objectives."""
        return float(np.mean([entry.total_deviation for entry in self.entries]))

    @property
    def is_discriminating(self) -> bool:
        """Whether the ranking separates the objectives by a meaningful margin.

        False means every objective sits essentially the same distance from a
        real swing, so the ordering is noise on top of a much larger model bias
        and must not be reported as "this is what golfers optimize".
        """
        if self.mean_deviation <= 0.0:
            return True
        return bool(self.deviation_spread / self.mean_deviation > _DISCRIMINATION_THRESHOLD)

    @property
    def reaches_measured_behaviour(self) -> bool:
        """Whether any objective lands fully inside the measured bands."""
        return any(entry.inside_count == self.band_count for entry in self.entries)


def objective_realism_ranking(
    config: DownswingConfig,
    objective_keys: Sequence[str] | None = None,
    bands: tuple[ObservableBand, ...] = TOUR_DRIVER_BANDS,
) -> ObjectiveRealismRanking:
    """Rank objectives by how closely their optima match measured golfers.

    Args:
        config: Conditions held identical across every objective.
        objective_keys: Objectives to rank. Defaults to all five.
        bands: Measured reference bands to score against.

    Returns:
        The ranking, most golf-like first.

    Raises:
        KeyError: If an objective key is not registered.

    Pre: ``config`` validated at construction.
    Post: entries are sorted by ascending total deviation.
    """
    keys = tuple(objective_keys) if objective_keys is not None else tuple(SWING_OBJECTIVES)
    if not keys:
        raise ValueError("objective_keys must name at least one objective")

    optimizer = DownswingOptimizer(config)
    entries = []
    for key in keys:
        result = optimizer.solve(key)
        observables = swing_observables(
            result.states,
            result.signals.clubhead_speed,
            config.duration_s,
            config.params.L1,
        )
        score = score_against_reference(observables, bands)
        entries.append(
            ObjectiveRealism(
                key=key,
                feasible=bool(result.max_defect < _REACHABLE_DEFECT),
                total_deviation=score.total_deviation,
                inside_count=score.inside_count,
                worst_observable=score.worst[0],
                clubhead_speed_ms=observables["clubhead_speed_ms"],
                observables=observables,
            )
        )

    entries.sort(key=lambda entry: entry.total_deviation)
    return ObjectiveRealismRanking(entries=tuple(entries), band_count=len(bands))
