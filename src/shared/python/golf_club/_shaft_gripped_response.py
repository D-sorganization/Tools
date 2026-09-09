"""Frozen point response of an all-node shaft with finite stationary grips."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grip_contracts import finite_array
from ._grip_stationary import stationary_grip_operators
from ._rotating_body_contracts import RotatingFrameState
from ._shaft_equilibrium import EquilibriumControls
from ._shaft_gripped_chain import GrippedSectionChain
from ._shaft_gripped_dynamics import balanced_gripped_dynamics
from ._shaft_harmonic_response import (
    TipHarmonicControls,
    _point_response,
    _scaled_pencil,
)


@dataclass(frozen=True)
class FrozenGrippedCompliance:
    """Fresh SI complex arrays using exp(+i*omega*t), with every node retained.

    Input columns are point forces then torques, in tip material axes. Output
    rows are point translations then infinitesimal rotations. Each transfer is
    the incremental grip-on-shaft wrench in that attachment's node material
    axes, including its preload derivative; equilibrium wrenches are separate.
    Relative stationary anchors can belong to a driven observer. This algebraic
    particular solution proves neither stable time evolution nor qualified
    bandwidth, hand impedance, contact response or acoustic radiation.
    """

    angular_frequency_rad_s: float
    displacement_compliance: np.ndarray
    velocity_mobility: np.ndarray
    nodal_displacement: np.ndarray
    grip_wrench_transfers: tuple[np.ndarray, ...]
    grip_wrenches: tuple[np.ndarray, ...]
    frame: RotatingFrameState
    scaled_rcond: float
    coefficient_resolution: float
    relative_residuals: np.ndarray

    @property
    def stability_status(self) -> str:
        return "unqualified"


def _grip_transfers(
    chain: GrippedSectionChain,
    poses: np.ndarray,
    motion: np.ndarray,
    frequency: float,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    transfers, wrenches = [], []
    for item in chain.grips:
        port = stationary_grip_operators(
            item.grip, item.root_motion(poses[item.node]), item.anchor
        )
        pencil = (
            port.stiffness - frequency**2 * port.mass + 1j * frequency * port.damping
        )
        rows = slice(6 * item.node, 6 * (item.node + 1))
        transfers.append(-pencil @ motion[rows])
        wrenches.append(np.array(port.response.root_wrench))
    return tuple(transfers), tuple(wrenches)


def gripped_tip_compliance(
    chain: GrippedSectionChain,
    poses: object,
    equilibrium: EquilibriumControls,
    controls: TipHarmonicControls,
) -> FrozenGrippedCompliance:
    """Recheck all-node balance/domain and solve without a clamp or regularizer.

    G and C remain separate coefficients for cancellation diagnostics. The
    original pencil and scaled solve must both resolve; no epsilon damping,
    pseudoinverse or inferred zero-frequency stability is supplied.
    """
    if not isinstance(controls, TipHarmonicControls):
        raise TypeError("expected TipHarmonicControls")
    operators = balanced_gripped_dynamics(chain, poses, equilibrium)
    current = finite_array(poses, (chain.node_count, 4, 4), "gripped poses")
    frequency = controls.angular_frequency_rad_s
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            terms = (
                operators.stiffness,
                -(frequency**2) * operators.mass,
                1j * frequency * operators.gyroscopic,
                1j * frequency * operators.damping,
            )
            dynamic, scale, coefficient_scale = _scaled_pencil(
                terms, controls.length_m, 0
            )
            point = _point_response(dynamic, scale, coefficient_scale, controls)
            transfers, wrenches = _grip_transfers(
                chain, current, point.displacement, frequency
            )
            if not all(np.all(np.isfinite(item)) for item in (*transfers, *wrenches)):
                raise ValueError("gripped response is nonfinite after SI mapping")
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        raise ValueError(
            "gripped harmonic response numerical evaluation failed"
        ) from error
    return FrozenGrippedCompliance(
        frequency,
        point.compliance,
        point.mobility,
        point.displacement,
        transfers,
        wrenches,
        operators.frame,
        point.rcond,
        point.resolution,
        point.residuals,
    )


__all__ = ()
