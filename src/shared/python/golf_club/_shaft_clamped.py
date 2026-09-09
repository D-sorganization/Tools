"""Shared domain and balance checks for clamped loaded-chain diagnostics."""

from __future__ import annotations

import numpy as np

from ._grip_contracts import finite_array
from ._shaft_equilibrium import EquilibriumControls, _check_strains
from ._shaft_loaded_dynamics import LoadedChainDynamics, linearized_chain_dynamics
from ._shaft_rotating_chain import RotatingSectionChain


def balanced_clamped_dynamics(
    chain: RotatingSectionChain, poses: object, controls: EquilibriumControls
) -> LoadedChainDynamics:
    """Recompute clamped-root balance and strain domain, retaining every row.

    Root reaction remains visible. No candidate record, nonlinear stability,
    future frame motion, damping or frequency bandwidth is accepted by inference.
    """
    if not isinstance(chain, RotatingSectionChain) or not isinstance(
        controls, EquilibriumControls
    ):
        raise TypeError("expected RotatingSectionChain and EquilibriumControls")
    current = finite_array(poses, (chain.node_count, 4, 4), "chain poses")
    operators = linearized_chain_dynamics(chain, current)
    _check_strains(chain, current, controls.strain_limits)
    free = operators.residual[6:].reshape(-1, 6)
    if np.any(np.abs(free[:, :3]) > controls.force_tolerance_n) or np.any(
        np.abs(free[:, 3:]) > controls.moment_tolerance_nm
    ):
        raise ValueError("free nodes must satisfy the declared balance tolerances")
    return operators


__all__ = ()
