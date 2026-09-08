"""All-node frozen modes for finite stationary grip-supported shaft chains."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ._grip_contracts import finite_array
from ._grip_finite_response import finite_grip_response
from ._shaft_damped_spectrum import DampedPencil, frozen_damped_spectrum
from ._shaft_equilibrium import EquilibriumControls
from ._shaft_gripped_chain import GrippedSectionChain
from ._shaft_gripped_dynamics import balanced_gripped_dynamics
from ._shaft_spectrum import FrozenSpectrum, SpectrumScales, _validate_result


def _grip_wrenches(
    chain: GrippedSectionChain, poses: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Keep each equilibrium support reaction in its own node material axes."""
    return tuple(
        np.array(
            finite_grip_response(
                item.grip, item.root_motion(poses[item.node]), item.anchor
            ).root_wrench
        )
        for item in chain.grips
    )


def gripped_chain_spectrum(
    chain: GrippedSectionChain,
    poses: object,
    controls: EquilibriumControls,
    scales: SpectrumScales,
) -> FrozenSpectrum:
    """Recheck every node/strain and retain finite-support reactions and frame.

    Require resolved positive mass for this ODE route; singular descriptor
    systems need a separate formulation. No root is clamped, no damping is
    added, and a driven-frame snapshot is not an autonomous stability proof.
    """
    if not isinstance(scales, SpectrumScales):
        raise TypeError("expected SpectrumScales")
    operators = balanced_gripped_dynamics(chain, poses, controls)
    current = finite_array(poses, (chain.node_count, 4, 4), "gripped poses")
    coordinate_scale = np.tile([scales.length_m] * 3 + [1.0] * 3, chain.node_count)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            congruence = coordinate_scale[:, None] * coordinate_scale[None, :]
            pencil = DampedPencil(
                *(
                    matrix * congruence
                    for matrix in (
                        operators.mass,
                        operators.gyroscopic,
                        operators.damping,
                        operators.stiffness,
                    )
                )
            )
            result = frozen_damped_spectrum(pencil, scales)
            wrenches = _grip_wrenches(chain, current)
            result = replace(
                result,
                displacement_modes=result.displacement_modes
                * coordinate_scale[:, None],
                velocity_modes=result.velocity_modes * coordinate_scale[:, None],
                frame=operators.frame,
                grip_wrenches=wrenches,
            )
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        raise ValueError("gripped spectrum numerical evaluation failed") from error
    _validate_result(result, scales.residual_tolerance)
    return result


__all__ = ()
