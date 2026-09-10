"""Bounded prescribed-anchor problems for inertial shaft time integration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from ._grip_contracts import _node_index, finite_array
from ._grip_moving_kinematics import MaterialPointMotion
from ._shaft_load_history import PrescribedPointLoads
from ._shaft_moving_contracts import InertialMovingChain, MovingChainControls

AnchorHistory = Callable[[float], tuple[MaterialPointMotion, ...]]


@dataclass(frozen=True)
class MovingTrajectoryProblem:
    """Fixed material laws with explicit anchor and optional additional loads.

    The callback returns one MaterialPointMotion per existing grip in order,
    including its pose, body twist and body-twist derivative in the declared
    inertial observer. It must be deterministic and kinematically consistent;
    sampling cannot establish either property. No interpolation, differentiation
    or missing-data substitution is performed. Additional force/couple histories
    use canonical point-load power; baseline loads remain. Material and grip
    coefficients are fixed; varying coefficients require extra storage/work terms.
    """

    chain: InertialMovingChain
    controls: MovingChainControls
    anchor_history: AnchorHistory
    additional_load_history: PrescribedPointLoads | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.chain, InertialMovingChain):
            raise TypeError("chain must be InertialMovingChain")
        if not isinstance(self.controls, MovingChainControls):
            raise TypeError("controls must be MovingChainControls")
        if not callable(self.anchor_history):
            raise TypeError("anchor history must be callable")
        if self.additional_load_history is not None and not isinstance(
            self.additional_load_history, PrescribedPointLoads
        ):
            raise TypeError("additional_load_history must be PrescribedPointLoads")

    def chain_at(self, time_s: float) -> InertialMovingChain:
        """Reuse fixed constitutive laws; validate every supplied anchor port."""
        anchors = tuple(self.anchor_history(time_s))
        if len(anchors) != len(self.chain.grips):
            raise ValueError("history anchor count does not match the grip count")
        ports = tuple(
            replace(port, anchor=anchor)
            for port, anchor in zip(self.chain.grips, anchors, strict=True)
        )
        chain = replace(self.chain, grips=ports)
        if self.additional_load_history is not None:
            return self.additional_load_history.apply(chain, time_s)
        return chain


def _positive_count(value: object, name: str) -> int:
    count = _node_index(value)
    if count == 0:
        raise ValueError(f"{name} must be positive")
    return int(count)


@dataclass(frozen=True)
class MovingTrajectoryControls:
    """Forward SI time interval, fixed step count and acceleration-call budget.

    The method needs exactly 2*steps+1 acceleration evaluations. The budget is
    checked before any history call, and cannot authorize a partial trajectory.
    These controls do not claim stability, truncation error or mesh convergence.
    """

    bounds_s: tuple[float, float]
    steps: int
    max_evaluations: int

    def __post_init__(self) -> None:
        bounds = finite_array(self.bounds_s, (2,), "time bounds")
        if not 0 <= bounds[0] < bounds[1]:
            raise ValueError("time bounds must be nonnegative and strictly increasing")
        object.__setattr__(self, "bounds_s", tuple(float(x) for x in bounds))
        for name in ("steps", "max_evaluations"):
            object.__setattr__(self, name, _positive_count(getattr(self, name), name))

    def time_cells(self) -> tuple[tuple[float, float, float], ...]:
        """Preflight the whole endpoint/midpoint grid before evaluating physics."""
        if 2 * self.steps + 1 > self.max_evaluations:
            raise ValueError(
                "trajectory acceleration evaluation budget is insufficient"
            )
        times = np.linspace(*self.bounds_s, self.steps + 1)
        cells = []
        for start, end in zip(times[:-1], times[1:], strict=True):
            middle = start + (end - start) / 2
            if not np.isfinite(middle) or not start < middle < end:
                raise ValueError("trajectory time grid is unresolved")
            cells.append((float(start), float(middle), float(end)))
        return tuple(cells)


__all__ = ()
