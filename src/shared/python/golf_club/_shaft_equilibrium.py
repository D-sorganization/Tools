"""Private clamped-chain root finding with explicit material-domain checks.

Convergence is force balance only. No stability, impact, frequency
response or experimental qualification is implied by the returned candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grip_contracts import Vector6, finite_array, vector6
from ._shaft_chain import ChainLinearization, SectionChain, _material_chart_connection
from ._shaft_rotating_chain import RotatingSectionChain
from ._shaft_se3 import exp_twist
from ._validation import require_finite_float

_NODE_DOF = 6
_BACKTRACK_FACTOR = 0.5
_ARMIJO_FRACTION = 1e-4


@dataclass(frozen=True)
class EquilibriumControls:
    """Explicit physical bounds and numerical controls; no fitted defaults.

    Positive componentwise absolute strain limits use dimensionless extension/
    shear and curvature [1/m]. They bound this section law, not a general
    material failure criterion. Numerical step caps are local m and rad;
    residual tolerances are N and N m. Iteration counts are positive integers.
    """

    strain_limits: Vector6
    force_tolerance_n: float
    moment_tolerance_nm: float
    translation_step_m: float
    rotation_step_rad: float
    max_iterations: int
    max_backtracks: int

    def __post_init__(self) -> None:
        limits = finite_array(self.strain_limits, (6,), "strain limits")
        if np.any(limits <= 0):
            raise ValueError("strain limits must be positive")
        object.__setattr__(self, "strain_limits", vector6(limits, "strain limits"))
        for name in (
            "force_tolerance_n",
            "moment_tolerance_nm",
            "translation_step_m",
            "rotation_step_rad",
        ):
            value = require_finite_float(getattr(self, name), name, positive=True)
            object.__setattr__(self, name, value)
        exp_twist([0, 0, 0, self.rotation_step_rad, 0, 0])
        for name in ("max_iterations", "max_backtracks"):
            count: object = getattr(self, name)
            if isinstance(count, (bool, np.bool_)) or not isinstance(
                count, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer")
            if count <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, int(count))


@dataclass(frozen=True)
class EquilibriumCandidate:
    """Fresh converged state with node zero clamped at the supplied seed pose.

    Support wrench is support-on-shaft, resolved in the root material frame.
    Residual maxima exclude the constrained node; elastic energy excludes all
    applied-load work. This record must never be consumed as a stable dynamic
    operating point without a separate stability and model-domain assessment.
    """

    poses: np.ndarray
    support_wrench: np.ndarray
    elastic_energy_j: float
    force_residual_n: float
    moment_residual_nm: float
    iterations: int

    @property
    def stability_status(self) -> str:
        """Root finding cannot classify physical or dynamic stability."""
        return "unqualified"


def moving_residual_jacobian(response: ChainLinearization) -> np.ndarray:
    """Convert fixed-chart curvature to the derivative of material residual.

    At chart origin, r_chart(q)=Jr(q)^T r_body(H Exp(q)); hence
    D r_body[a] = K_chart a + ad(a)^T r_body/2, blockwise for every node.
    This term is retained away from equilibrium; no symmetry is imposed.
    """
    if not isinstance(response, ChainLinearization):
        raise TypeError("response must be a ChainLinearization")
    size = np.size(response.residual)
    connection = _material_chart_connection(response.residual)
    tangent = finite_array(response.tangent, (size, size), "chain tangent")
    return finite_array(tangent + connection, (size, size), "moving residual Jacobian")


def _check_strains(
    chain: SectionChain | RotatingSectionChain, poses: np.ndarray, limits: Vector6
) -> None:
    for index, section in enumerate(chain.sections):
        strain = section.strain(poses[index], poses[index + 1])
        if np.any(np.abs(strain) > limits):
            raise ValueError(f"section {index} exceeds declared strain limits")


def _scales(controls: EquilibriumControls, nodes: int) -> tuple[np.ndarray, np.ndarray]:
    force = [controls.force_tolerance_n] * 3 + [controls.moment_tolerance_nm] * 3
    step = [controls.translation_step_m] * 3 + [controls.rotation_step_rad] * 3
    return np.tile(force, nodes), np.tile(step, nodes)


def _newton_step(
    response: ChainLinearization, scales: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    force_scale, step_scale = scales
    matrix = moving_residual_jacobian(response)[_NODE_DOF:, _NODE_DOF:]
    scaled = matrix * step_scale[None, :] / force_scale[:, None]
    residual = response.residual[_NODE_DOF:] / force_scale
    try:
        scaled_step = np.linalg.solve(scaled, -residual)
    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            "equilibrium Jacobian is singular; no converged state"
        ) from error
    direction = finite_array(scaled_step * step_scale, residual.shape, "Newton step")
    # Cap each node's vector norm, rather than each component independently.
    blocks = (direction / step_scale).reshape(-1, _NODE_DOF)
    largest = max(
        1.0,
        float(np.max(np.linalg.norm(blocks[:, :3], axis=1))),
        float(np.max(np.linalg.norm(blocks[:, 3:], axis=1))),
    )
    return direction / largest


def _trial_step(poses: np.ndarray, direction: np.ndarray) -> np.ndarray:
    moved = poses.copy()
    for index, delta in enumerate(direction.reshape(-1, _NODE_DOF), start=1):
        moved[index] = poses[index] @ exp_twist(delta)
    return moved


def _backtrack(
    chain: SectionChain | RotatingSectionChain,
    state: tuple[np.ndarray, ChainLinearization],
    direction: np.ndarray,
    controls: EquilibriumControls,
) -> tuple[np.ndarray, ChainLinearization]:
    poses, response = state
    force_scale, _ = _scales(controls, chain.node_count - 1)
    norm = np.linalg.norm(response.residual[_NODE_DOF:] / force_scale)
    fraction = 1.0
    for _ in range(controls.max_backtracks):
        try:
            trial = _trial_step(poses, fraction * direction)
            _check_strains(chain, trial, controls.strain_limits)
            trial_response = chain.linearize(trial)
        except ValueError:
            # A trial may leave the declared material or numerical chart domain.
            fraction *= _BACKTRACK_FACTOR
            continue
        trial_norm = np.linalg.norm(trial_response.residual[_NODE_DOF:] / force_scale)
        if trial_norm <= (1 - _ARMIJO_FRACTION * fraction) * norm:
            return trial, trial_response
        fraction *= _BACKTRACK_FACTOR
    raise RuntimeError("no admissible decreasing equilibrium step; no converged state")


def _candidate(
    poses: np.ndarray, response: ChainLinearization, iterations: int
) -> EquilibriumCandidate:
    free = response.residual[_NODE_DOF:].reshape(-1, _NODE_DOF)
    return EquilibriumCandidate(
        poses.copy(),
        response.residual[:_NODE_DOF].copy(),
        response.elastic_energy_j,
        float(np.max(np.abs(free[:, :3]))),
        float(np.max(np.abs(free[:, 3:]))),
        iterations,
    )


def solve_clamped_chain(
    chain: SectionChain | RotatingSectionChain,
    seed: object,
    controls: EquilibriumControls,
) -> EquilibriumCandidate:
    """Find force balance while preserving the exact root pose and input arrays.

    Malformed or out-of-domain initial states raise TypeError/ValueError.
    Nonconvergence, singular operators or exhausted admissible steps raise
    RuntimeError. Never return an unfinished state or a stability certificate.
    """
    if not isinstance(chain, (SectionChain, RotatingSectionChain)) or not isinstance(
        controls, EquilibriumControls
    ):
        raise TypeError(
            "expected SectionChain or RotatingSectionChain and EquilibriumControls"
        )
    poses = finite_array(seed, (chain.node_count, 4, 4), "equilibrium seed")
    response = chain.linearize(poses)
    _check_strains(chain, poses, controls.strain_limits)
    scales = _scales(controls, chain.node_count - 1)
    for iteration in range(controls.max_iterations + 1):
        candidate = _candidate(poses, response, iteration)
        if (
            candidate.force_residual_n <= controls.force_tolerance_n
            and candidate.moment_residual_nm <= controls.moment_tolerance_nm
        ):
            return candidate
        if iteration == controls.max_iterations:
            break
        direction = _newton_step(response, scales)
        poses, response = _backtrack(chain, (poses, response), direction, controls)
    raise RuntimeError("equilibrium iteration budget exhausted; no converged state")


__all__ = ()
