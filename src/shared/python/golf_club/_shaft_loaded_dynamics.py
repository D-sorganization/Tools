"""Consistent full-node dynamic snapshots about relatively resting loaded shapes.

This private assembly reuses the geometric elastic and inertial kernels. No
straight-shaft coefficients, damping, constraints or stability are inferred.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grip_contracts import finite_array
from ._shaft_equilibrium import moving_residual_jacobian
from ._shaft_frame_inertia import rotating_section_inertia
from ._shaft_rotating_chain import RotatingSectionChain

_NODE_DOF = 6


@dataclass(frozen=True)
class LoadedChainDynamics:
    """Fresh SI material-coordinate arrays with all nodal forces and moments.

    For perturbations about zero relative velocity, the frozen affine balance
    is r + M*delta_acceleration + G*delta_velocity + K*delta_pose = 0.
    Stiffness differentiates the moving material residual, not fixed-chart
    virtual work; these agree on free rows/columns at an exact clamped root.
    G represents gyroscopic transport, not damping. Nonzero r is retained.
    This record does not qualify mass rank, equilibrium, material domain,
    autonomous evolution, stability, modal bandwidth or experimental validity.
    """

    residual: np.ndarray
    mass: np.ndarray
    gyroscopic: np.ndarray
    stiffness: np.ndarray


def linearized_chain_dynamics(
    chain: RotatingSectionChain, poses: object
) -> LoadedChainDynamics:
    """Assemble finite independent arrays at explicit proper observer-frame poses.

    The input chain supplies reference-weighted mass, section laws, loads and
    prescribed frame motion. No boundary is eliminated; singular quadrature
    and nonsymmetric stiffness remain visible for subsequent qualification.
    At time-varying frame motion this is only an instantaneous snapshot.
    """
    if not isinstance(chain, RotatingSectionChain):
        raise TypeError("chain must be RotatingSectionChain")
    current = finite_array(poses, (chain.node_count, 4, 4), "chain poses")
    response = chain.linearize(current)
    size = _NODE_DOF * chain.node_count
    mass, gyroscopic = np.zeros((size, size)), np.zeros((size, size))
    for index, inertia in enumerate(chain.inertias):
        pair = current[index : index + 2]
        kinetics = inertia.evaluate(pair, np.zeros(2 * _NODE_DOF))
        transport = rotating_section_inertia(inertia, pair, chain.frame)
        rows = slice(_NODE_DOF * index, _NODE_DOF * (index + 2))
        mass[rows, rows] += kinetics.mass
        gyroscopic[rows, rows] += transport.gyroscopic
    return LoadedChainDynamics(
        response.residual,
        finite_array(mass, (size, size), "loaded chain mass"),
        finite_array(gyroscopic, (size, size), "loaded chain gyroscopic matrix"),
        moving_residual_jacobian(response),
    )


__all__ = ()
