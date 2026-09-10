"""All-node equilibrium candidates for finite grip-supported shaft chains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grip_finite_response import FiniteGripResponse, finite_grip_response
from ._shaft_equilibrium import EquilibriumControls, _iterate_chain
from ._shaft_gripped_chain import GrippedSectionChain


@dataclass(frozen=True)
class GrippedEquilibriumCandidate:
    """Fresh balanced poses and separate physical support ports; no hidden clamp.

    Elastic energy includes both shaft and grip potential storage, excluding
    prescribed load/frame work. A converged force balance proves neither
    uniqueness, stability, physical calibration nor an autonomous trajectory.
    """

    poses: np.ndarray
    elastic_energy_j: float
    grip_responses: tuple[FiniteGripResponse, ...]
    force_residual_n: float
    moment_residual_nm: float
    iterations: int

    @property
    def stability_status(self) -> str:
        return "unqualified"


def solve_gripped_chain(
    chain: GrippedSectionChain, seed: object, controls: EquilibriumControls
) -> GrippedEquilibriumCandidate:
    """Reuse bounded Newton/backtracking with every nodal pose free to move.

    Material strain and principal-chart limits remain enforced at every trial.
    Nonconvergence or singular steps raise RuntimeError, never an unfinished
    candidate. The existing clamped solver retains its separate contract.
    """
    if not isinstance(chain, GrippedSectionChain) or not isinstance(
        controls, EquilibriumControls
    ):
        raise TypeError("expected GrippedSectionChain and EquilibriumControls")
    poses, response, iterations = _iterate_chain(
        chain, seed, controls, first_free_node=0
    )
    ports = tuple(
        finite_grip_response(item.grip, item.root_motion(poses[item.node]), item.anchor)
        for item in chain.grips
    )
    residual = response.residual.reshape(-1, 6)
    return GrippedEquilibriumCandidate(
        poses.copy(),
        response.elastic_energy_j,
        ports,
        float(np.max(np.abs(residual[:, :3]))),
        float(np.max(np.abs(residual[:, 3:]))),
        iterations,
    )


__all__ = ()
