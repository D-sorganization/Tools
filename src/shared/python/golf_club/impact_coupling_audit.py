"""Additive energy/event audit for the v1 lumped contact model (IA-T2, #5071).

This is a translating fixed-anchor inertial frame, not a rotating golfer frame.
Clipped unloading dissipates remaining contact spring energy separately from
viscous losses. No physiological percentage or universal rigid-link bound is
inferred. Import this submodule directly to avoid expanding the eager facade.
"""

from ._coupled_impact_solver import _solve_coupled_impact
from ._coupled_impact_state import (
    CoupledImpactAudit as CoupledImpactAudit,
)
from ._coupled_impact_state import (
    CoupledImpactEnergyLedger as CoupledImpactEnergyLedger,
)
from ._coupled_impact_state import (
    CoupledImpactInitialState as CoupledImpactInitialState,
)
from .impact_coupling import CoupledImpactConfig

__all__ = [
    "CoupledImpactInitialState",
    "CoupledImpactEnergyLedger",
    "CoupledImpactAudit",
    "audit_coupled_impact",
]


def audit_coupled_impact(
    config: CoupledImpactConfig,
    *,
    initial_state: CoupledImpactInitialState | None = None,
) -> CoupledImpactAudit:
    """Return finite terminal state and a complete passive energy/work ledger.

    Preconditions: validated config and finite initial state at first touch;
    dt_s resolves the mass-normalized stiffness/damping rate bound. Maximum
    accepted steps are dt_s; an adaptive high-order method resolves event roots.
    Postconditions: geometric separation reached before max_time_s; no silent
    timeout result. Raises TypeError/ValueError for invalid inputs and RuntimeError
    for incomplete integration. First force release and geometric clearance are
    reported separately. Peak force is sampled at accepted steps plus the initial
    right-hand limit; refine dt_s when peak accuracy matters.
    """
    if not isinstance(config, CoupledImpactConfig):
        raise TypeError("config must be CoupledImpactConfig")
    return _solve_coupled_impact(config, initial_state=initial_state)
