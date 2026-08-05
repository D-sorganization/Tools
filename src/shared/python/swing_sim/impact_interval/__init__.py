"""Six-DOF impact-interval dynamics façade (Tools #4130).

Downstream consumers import only from this module. Internal numerical and
rotation helpers remain replaceable so future Rust, shaft-flex, and contact
patch models can preserve the public scientific record contract.
"""

from __future__ import annotations

from .contact import KelvinVoigtContactLaw
from .solver import solve_impact_interval
from .types import (
    BoundaryKind,
    ClubRigidBody,
    ImpactIntervalAudit,
    ImpactIntervalConfig,
    ImpactIntervalInitialState,
    ImpactIntervalResult,
    ImpactIntervalSample,
)

__all__ = [
    "BoundaryKind",
    "ClubRigidBody",
    "ImpactIntervalAudit",
    "ImpactIntervalConfig",
    "ImpactIntervalInitialState",
    "ImpactIntervalResult",
    "ImpactIntervalSample",
    "KelvinVoigtContactLaw",
    "solve_impact_interval",
]
