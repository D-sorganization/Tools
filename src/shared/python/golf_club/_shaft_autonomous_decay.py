"""Numerical Lyapunov envelopes for constant homogeneous regular linear ODEs.

This private companion does not qualify a frozen model as an autonomous swing,
contact trajectory, physical energy, calibrated uncertainty or acoustic result.
The supplied, already length-scaled pencil is evaluated without plant repair.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

from ._grip_contracts import finite_array
from ._shaft_damped_spectrum import DampedPencil, _validated_damped_generator
from ._shaft_spectrum import SpectrumScales

_SCOPE = "constant_homogeneous_regular_linear_ode"
_Matrix = tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class DecayControls:
    """Numerical tolerances and an assumed dimensionless generator error bound.

    Tolerances lie strictly in (0, 1). The nonnegative error bound is an assumed
    upper bound on ||delta A||_2 in the declared scaled state coordinates. It is
    not estimated here. Zero assesses the computed generator only. Qualification
    uses floating-point arithmetic, never interval-verified certification.
    """

    residual_tolerance: float
    definiteness_rcond_floor: float
    generator_error_bound: float

    def __post_init__(self) -> None:
        for name in (
            "residual_tolerance",
            "definiteness_rcond_floor",
            "generator_error_bound",
        ):
            value = float(finite_array(getattr(self, name), (), name))
            if name == "generator_error_bound":
                if value < 0:
                    raise ValueError("generator_error_bound must be nonnegative")
            elif not 0 < value < 1:
                raise ValueError(f"{name} must lie strictly between zero and one")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class DecayEvidence:
    """Copied candidate P and recomputed Q=-(A.T P+P A), in scaled coordinates.

    Residual is ||Q-I||_F/||I||_F. The reported symmetry defect is measured before
    explicitly replacing the free candidate by (P+P.T)/2; A is never changed.
    Signed rconds are lambda_min/lambda_max_abs. Negative or unresolved margins
    do not establish instability. The robust margin is lambda_min(Q)-2||P||_2 d.
    """

    storage_matrix: _Matrix
    dissipation_matrix: _Matrix
    relative_residual: float
    candidate_symmetry_defect: float
    storage_rcond: float
    dissipation_rcond: float
    robust_dissipation_margin: float


@dataclass(frozen=True)
class DecayEnvelope:
    """||x(t)||_2 <= norm_prefactor exp(-decay_rate_s_inv*t) ||x(0)||_2.

    State x=(y, dy/dtau), tau=t/T. This coordinate-dependent Euclidean norm
    mixes scaled displacement and scaled velocity; it is not club energy.
    """

    norm_prefactor: float
    decay_rate_s_inv: float


@dataclass(frozen=True)
class DecayAssessment:
    """Numerical support or absence of an established autonomous decay bound."""

    status: str
    reason: str
    evidence: DecayEvidence | None = None
    envelope: DecayEnvelope | None = None
    scope: str = _SCOPE


def _matrix_tuple(matrix: np.ndarray) -> _Matrix:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _candidate_storage(generator: np.ndarray) -> tuple[np.ndarray, float]:
    """Choose a real symmetric candidate; near-singular solver warnings refuse."""
    identity = np.eye(len(generator))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        raw = solve_continuous_lyapunov(generator.T, -identity)
    candidate = finite_array(raw, generator.shape, "Lyapunov candidate")
    norm = float(np.linalg.norm(candidate))
    if norm == 0:
        raise ValueError("zero Lyapunov candidate")
    defect = float(np.linalg.norm(candidate - candidate.T) / norm)
    return candidate / 2 + candidate.T / 2, defect


def _signed_rcond(eigenvalues: np.ndarray) -> float:
    magnitude = float(np.max(np.abs(eigenvalues)))
    return float(eigenvalues[0] / magnitude) if magnitude > 0 else 0.0


def _evaluate_candidate(
    generator: np.ndarray, controls: DecayControls
) -> DecayEvidence:
    storage, symmetry = _candidate_storage(generator)
    product = storage @ generator
    dissipation = -(product + product.T)
    storage_eigenvalues = np.linalg.eigvalsh(storage)
    dissipation_eigenvalues = np.linalg.eigvalsh(dissipation)
    identity = np.eye(len(generator))
    residual = float(np.linalg.norm(dissipation - identity) / np.linalg.norm(identity))
    margin = float(
        dissipation_eigenvalues[0]
        - 2 * np.max(np.abs(storage_eigenvalues)) * controls.generator_error_bound
    )
    diagnostics = finite_array(
        [residual, symmetry, margin], (3,), "Lyapunov diagnostics"
    )
    return DecayEvidence(
        _matrix_tuple(storage),
        _matrix_tuple(dissipation),
        float(diagnostics[0]),
        float(diagnostics[1]),
        _signed_rcond(storage_eigenvalues),
        _signed_rcond(dissipation_eigenvalues),
        float(diagnostics[2]),
    )


def _qualify_evidence(evidence: DecayEvidence, controls: DecayControls) -> str:
    if max(evidence.relative_residual, evidence.candidate_symmetry_defect) > (
        controls.residual_tolerance
    ):
        return "candidate symmetry or Lyapunov residual exceeds tolerance"
    if min(evidence.storage_rcond, evidence.dissipation_rcond) <= (
        controls.definiteness_rcond_floor
    ):
        return "positive storage or dissipation is not numerically resolved"
    dissipation_scale = float(np.linalg.norm(evidence.dissipation_matrix, ord=2))
    if evidence.robust_dissipation_margin <= (
        controls.definiteness_rcond_floor * dissipation_scale
    ):
        return "robust dissipation margin is not positive and numerically resolved"
    return ""


def _decay_envelope(evidence: DecayEvidence, time_s: float) -> DecayEnvelope:
    eigenvalues = np.linalg.eigvalsh(evidence.storage_matrix)
    low, high = eigenvalues[0], eigenvalues[-1]
    values = finite_array(
        [np.sqrt(high / low), (evidence.robust_dissipation_margin / high) / 2 / time_s],
        (2,),
        "decay envelope",
    )
    if values[0] < 1 or values[1] <= 0:
        raise ValueError("decay envelope is not numerically representable")
    return DecayEnvelope(float(values[0]), float(values[1]))


def assess_autonomous_decay(
    pencil: DampedPencil, scales: SpectrumScales, controls: DecayControls
) -> DecayAssessment:
    """Assess a constant homogeneous ODE under explicitly declared assumptions.

    Invalid plant/control contracts raise. A failed, inaccurate or unresolved
    Lyapunov candidate returns not_established, never an instability verdict.
    P solves A.T P+P A=-I; Q is recomputed from the unchanged generator. Positive
    resolved P,Q yield the bound via d(x.T P x)/dtau=-x.T Q x. The error margin
    follows ||delta A.T P+P delta A||_2 <= 2||P||_2 ||delta A||_2. Floating-point
    formation and eigenvalue errors are not independently interval bounded.
    """
    if not isinstance(controls, DecayControls):
        raise TypeError("controls must be DecayControls")
    generator, _ = _validated_damped_generator(pencil, scales)
    evidence = None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            evidence = _evaluate_candidate(generator, controls)
            reason = _qualify_evidence(evidence, controls)
            if reason:
                return DecayAssessment("not_established", reason, evidence)
            envelope = _decay_envelope(evidence, scales.time_s)
    except (
        np.linalg.LinAlgError,
        FloatingPointError,
        OverflowError,
        RuntimeWarning,
        TypeError,
        ValueError,
    ) as error:
        return DecayAssessment("not_established", str(error), evidence)
    return DecayAssessment(
        "numerically_supported",
        "resolved Lyapunov envelope for the stated autonomous ODE and error bound",
        evidence,
        envelope,
    )


__all__ = ()
