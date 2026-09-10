"""Explicit prescribed spatial force/couple histories on existing shaft nodes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from ._grip_contracts import finite_array
from ._shaft_chain import IndexedPointLoad
from ._shaft_moving_contracts import InertialMovingChain
from ._validation import require_finite_float, require_identifier


def append_point_loads(
    chain: InertialMovingChain, loads: tuple[IndexedPointLoad, ...]
) -> InertialMovingChain:
    """Retain baseline loads once; return a separately owned loaded chain."""
    if not isinstance(chain, InertialMovingChain):
        raise TypeError("chain must be InertialMovingChain")
    if not isinstance(loads, tuple) or any(
        not isinstance(load, IndexedPointLoad) for load in loads
    ):
        raise TypeError("history must return a tuple of IndexedPointLoad")
    shaft = chain.shaft
    if any(load.node >= shaft.node_count for load in loads):
        raise ValueError("history load node is outside the shaft")
    elastic = shaft.elastic
    return replace(
        chain,
        shaft=replace(shaft, elastic=replace(elastic, loads=(*elastic.loads, *loads))),
    )


@dataclass(frozen=True)
class PrescribedPointLoads:
    """Additional loads in an explicit inertial observer over a closed interval.

    The deterministic callback supplies forces [N] and free couples [N m] in
    observer axes, with offsets [m] in each selected node's material axes.
    Samples are complete additional-load sets, never accumulated stage loads.
    Existing loads remain. No interpolation, missing-data substitution, force
    potential, feedback law or body-following load is inferred. Discontinuities
    require explicit event/time refinement; a coverage declaration is not data
    completeness, callback consistency or physical qualification evidence.
    """

    observer_id: str
    bounds_s: tuple[float, float]
    sample: Callable[[float], tuple[IndexedPointLoad, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observer_id", require_identifier(self.observer_id, "observer")
        )
        bounds = finite_array(self.bounds_s, (2,), "load history bounds")
        if not 0 <= bounds[0] < bounds[1]:
            raise ValueError("load history bounds must be nonnegative and increasing")
        object.__setattr__(self, "bounds_s", tuple(float(value) for value in bounds))
        if not callable(self.sample):
            raise TypeError("load history sample must be callable")

    def apply(self, chain: InertialMovingChain, time_s: float) -> InertialMovingChain:
        """Validate time/frame before sampling and reuse canonical load assembly."""
        if not isinstance(chain, InertialMovingChain):
            raise TypeError("chain must be InertialMovingChain")
        time_s = require_finite_float(time_s, "load history time")
        if not self.bounds_s[0] <= time_s <= self.bounds_s[1]:
            raise ValueError("time is outside declared load history coverage")
        if chain.shaft.frame.frame_id != self.observer_id:
            raise ValueError("load history and shaft observers must agree")
        return append_point_loads(chain, self.sample(time_s))


__all__ = ()
