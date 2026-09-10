"""Private SI controls and bounded evidence for adaptive normal contact."""

from dataclasses import dataclass, fields

import numpy as np

from ...golf_club._shaft_trajectory_contracts import MovingTrajectoryControls
from ...golf_club._validation import require_finite_float
from ._normal_contact_trajectory import (
    NormalContactTrajectorySample,
    NormalContactTrajectoryState,
)
from ._normal_shaft_contact import NormalShaftContactResponse


@dataclass(frozen=True)
class ContactAbsoluteTolerances:
    """Local absolute errors in metres, radians, m/s, rad/s and joules.

    These are numerical request parameters, not measurement uncertainties or
    bounds on global error. Each physical channel retains its own units.
    """

    position_m: float
    rotation_rad: float
    linear_velocity_mps: float
    angular_velocity_radps: float
    work_j: float

    def __post_init__(self) -> None:
        for item in fields(self):
            object.__setattr__(
                self,
                item.name,
                require_finite_float(
                    getattr(self, item.name), item.name, positive=True
                ),
            )

    def vector(self, nodes: int) -> np.ndarray:
        return np.r_[
            np.tile([self.position_m] * 3 + [self.rotation_rad] * 3, nodes),
            np.tile(
                [self.linear_velocity_mps] * 3 + [self.angular_velocity_radps] * 3,
                nodes,
            ),
            [self.work_j] * 5,
        ]


@dataclass(frozen=True)
class AdaptiveContactControls(MovingTrajectoryControls):
    """Fixed chart/output grid; adaptive substeps and a hard response budget.

    ``steps`` counts chart cells. Sign-change event finding can miss repeated
    crossings in one accepted step. Independent step/event refinement remains
    necessary; no stability, global error or complete-root guarantee is given.
    """

    maximum_step_s: float
    relative_tolerance: float
    absolute_tolerances: ContactAbsoluteTolerances

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("maximum_step_s", "relative_tolerance"):
            object.__setattr__(
                self,
                name,
                require_finite_float(getattr(self, name), name, positive=True),
            )
        if not 100 * np.finfo(float).eps <= self.relative_tolerance < 1:
            raise ValueError("relative tolerance must be representable and below one")
        if not isinstance(self.absolute_tolerances, ContactAbsoluteTolerances):
            raise TypeError("absolute tolerances must have explicit physical units")


@dataclass(frozen=True)
class NormalContactRootSample:
    """Root state/response; dense interpolated work is not a loss certificate.

    High-order dense work interpolation need not preserve nonnegativity near
    a loss onset. Work is reported only at validated integration endpoints.
    """

    time_s: float
    state: NormalContactTrajectoryState
    response: NormalShaftContactResponse


@dataclass(frozen=True)
class NormalContactEvent:
    """Transversal root sample; first-touch force is a one-sided model limit."""

    kind: str
    sample: NormalContactRootSample
    incoming_force_limit_n: float


@dataclass(frozen=True)
class AdaptiveNormalContactTrajectory:
    """Complete chart endpoints and sampled roots, without physical approval."""

    controls: AdaptiveContactControls
    samples: tuple[NormalContactTrajectorySample, ...]
    events: tuple[NormalContactEvent, ...]
    evaluation_count: int

    @property
    def evidence_status(self) -> str:
        return "event-sampled-contact-unqualified"


__all__ = ()
