"""Owned coupled mechanical/history state and bounded endpoint-solve controls."""

from dataclasses import dataclass
from enum import Enum

from ...golf_club._shaft_trajectory_contracts import (
    MovingTrajectoryControls,
    _positive_count,
)
from ...golf_club._validation import require_finite_float
from ._friction_transport import ContactTransport
from ._normal_contact_trajectory import (
    NormalContactTrajectoryProblem,
    NormalContactTrajectoryState,
)
from ._tangential_contact_work import TangentialContactLaw, TangentialContactState


class FrictionTermination(Enum):
    """Initial data versus the numerical criterion that ended an actual solve."""

    INITIAL = "initial"
    RESIDUAL = "roundoff-residual"
    BACKEND = "backend-iterate"


@dataclass(frozen=True)
class FrictionConvergence:
    """Recorded termination and freshly checked dimensionless endpoint residual.

    INITIAL marks supplied data, not a solved equation. Neither termination
    criterion certifies a time-discretization error or physical validity.
    """

    reason: FrictionTermination
    scaled_residual: float

    def __post_init__(self) -> None:
        if not isinstance(self.reason, FrictionTermination):
            raise TypeError("convergence reason must be FrictionTermination")
        value = require_finite_float(self.scaled_residual, "endpoint residual")
        if value < 0:
            raise ValueError("endpoint residual must be nonnegative")
        if self.reason is FrictionTermination.INITIAL and value != 0:
            raise ValueError("initial data cannot claim a solved residual")
        object.__setattr__(self, "scaled_residual", value)


@dataclass(frozen=True)
class FrictionTrajectoryProblem:
    """Reuse normal/shaft/grip/load laws with an explicit tangential convention."""

    normal: NormalContactTrajectoryProblem
    tangential_law: TangentialContactLaw
    transport: ContactTransport

    def __post_init__(self) -> None:
        for name, expected in (
            ("normal", NormalContactTrajectoryProblem),
            ("tangential_law", TangentialContactLaw),
            ("transport", ContactTransport),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")


@dataclass(frozen=True)
class FrictionTrajectoryState:
    """Accepted shaft/ball state and elastic history in the same observer."""

    mechanical: NormalContactTrajectoryState
    tangential: TangentialContactState

    def __post_init__(self) -> None:
        if not isinstance(self.mechanical, NormalContactTrajectoryState):
            raise TypeError("mechanical must be NormalContactTrajectoryState")
        if not isinstance(self.tangential, TangentialContactState):
            raise TypeError("tangential must be TangentialContactState")
        ball = self.mechanical.ball
        if ball.observer_id != self.tangential.observer_id:
            raise ValueError("mechanics and tangential history observers must agree")


@dataclass(frozen=True)
class FrictionTrajectoryControls(MovingTrajectoryControls):
    """Fixed SI grid, actual global/per-step calls, and dimensionless residual.

    Velocity residuals use the existing mechanical length/time scales. Grid
    preflight reuses the existing endpoint/midpoint representability contract.
    Actual nonlinear evaluations count toward both budgets, including the final
    independently re-evaluated endpoint. Failed steps return no partial result.
    """

    max_step_evaluations: int
    scaled_residual_tolerance: float

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "max_step_evaluations",
            _positive_count(self.max_step_evaluations, "step evaluation budget"),
        )
        tolerance = require_finite_float(
            self.scaled_residual_tolerance, "solver tolerance", positive=True
        )
        if tolerance >= 1:
            raise ValueError("solver tolerance must be below one")
        object.__setattr__(self, "scaled_residual_tolerance", tolerance)


__all__ = ()
