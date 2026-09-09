"""Conditional response envelopes retaining a constant residual in a linear ODE.

This private companion reuses homogeneous Lyapunov evidence. A frozen shaft
snapshot is not promoted to a constant operating model by calling this kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ._grip_contracts import finite_array
from ._shaft_autonomous_decay import (
    DecayAssessment,
    DecayControls,
    DecayEnvelope,
    assess_autonomous_decay,
)
from ._shaft_damped_spectrum import DampedPencil
from ._shaft_spectrum import SpectrumScales


def _nonnegative(value: object, name: str) -> float:
    result = float(finite_array(value, (), name))
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass(frozen=True)
class ForcedResponseControls:
    """Separate assumed operator and additive-input uncertainty bounds.

    ``decay.generator_error_bound`` concerns dimensionless A in tau=t/T.
    ``input_error_bound_s_inv`` bounds the additional physical-time input norm
    in x=(y,dy/dtau), uniformly over the assessed interval. Neither is estimated
    from solver residuals, calibration data or a nonlinear trajectory here.
    """

    decay: DecayControls
    input_error_bound_s_inv: float

    def __post_init__(self) -> None:
        if not isinstance(self.decay, DecayControls):
            raise TypeError("decay must be DecayControls")
        object.__setattr__(
            self,
            "input_error_bound_s_inv",
            _nonnegative(self.input_error_bound_s_inv, "input error bound"),
        )


@dataclass(frozen=True)
class ResponseBound:
    """Separated nonnegative terms in the scaled-state Euclidean norm."""

    initial_state_term: float
    input_term: float

    @property
    def total(self) -> float:
        return _nonnegative(self.initial_state_term + self.input_term, "total bound")


def _convolution_time(rate: float, time: float) -> float:
    """Integrate exp(-rate*s), retaining its t limit when rate*t underflows."""
    product = rate * time
    if product < 1:
        ratio = -math.expm1(-product) / product if product > 0 else 1.0
        return time * ratio
    return -math.expm1(-product) / rate


@dataclass(frozen=True)
class ForcedEnvelope:
    """K exp(-c t)||x0|| + K U (1-exp(-c t))/c for ||u(t)|| <= U.

    Physical time is in seconds. The norm depends on the stated length/time
    scaling and is not energy, acoustic amplitude or an impact-quality score.
    The bound is conditional numerical evidence, not interval certification.
    """

    decay: DecayEnvelope
    input_norm_bound_s_inv: float

    def __post_init__(self) -> None:
        if not isinstance(self.decay, DecayEnvelope):
            raise TypeError("decay must be DecayEnvelope")
        prefactor = _nonnegative(self.decay.norm_prefactor, "norm prefactor")
        rate = _nonnegative(self.decay.decay_rate_s_inv, "decay rate")
        if prefactor < 1 or rate == 0:
            raise ValueError("norm prefactor must be at least one and rate positive")
        object.__setattr__(self, "decay", DecayEnvelope(prefactor, rate))
        object.__setattr__(
            self,
            "input_norm_bound_s_inv",
            _nonnegative(self.input_norm_bound_s_inv, "input norm bound"),
        )

    def bound(self, time_s: object, initial_norm: object) -> ResponseBound:
        """Require finite nonnegative time/norm; refuse unrepresentable output."""
        time = _nonnegative(time_s, "time")
        initial = _nonnegative(initial_norm, "initial norm")
        prefactor = self.decay.norm_prefactor
        rate = self.decay.decay_rate_s_inv
        initial_term = prefactor * math.exp(-rate * time) * initial
        input_term = prefactor * (
            self.input_norm_bound_s_inv * _convolution_time(rate, time)
        )
        if initial > 0 and initial_term == 0:
            raise ValueError("positive initial-state term is unresolved by underflow")
        if time > 0 and self.input_norm_bound_s_inv > 0 and input_term == 0:
            raise ValueError("positive input term is unresolved by underflow")
        result = ResponseBound(
            _nonnegative(initial_term, "initial-state term"),
            _nonnegative(input_term, "input term"),
        )
        _ = result.total
        return result


@dataclass(frozen=True)
class AffineResponseAssessment:
    """Copied nominal residual/input and unchanged homogeneous evidence.

    The residual is the LEFT-side load in r+M ydd+(G+C)yd+K y=0, with
    work-conjugate scaling: S.T r and S.T operator S. ``input_per_s`` is
    [0,-T M^-1 r] in those scaled coordinates.
    This calculation neither finds equilibrium nor changes its tolerance.
    """

    status: str
    reason: str
    homogeneous: DecayAssessment
    residual: tuple[float, ...]
    input_per_s: tuple[float, ...]
    envelope: ForcedEnvelope | None
    scope: str = "constant_affine_regular_linear_ode"


def _nominal_input(
    mass: np.ndarray, load: np.ndarray, time_s: float
) -> tuple[float, ...]:
    """Convert the left-side residual to physical-time scaled-state forcing."""
    forcing = -time_s * np.linalg.solve(mass, load)
    forcing = finite_array(forcing, load.shape, "nominal input")
    if np.any(load != 0) and not np.any(forcing != 0):
        raise ValueError("nonzero residual input is unresolved by underflow")
    return (0.0,) * len(load) + tuple(float(item) for item in forcing)


def _evaluate_affine_assessment(
    result: AffineResponseAssessment,
    mass: np.ndarray,
    scales: SpectrumScales,
    controls: ForcedResponseControls,
) -> AffineResponseAssessment:
    """Retain available evidence even when input formation is unresolved."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            input_values = _nominal_input(
                mass, np.asarray(result.residual), scales.time_s
            )
            result = replace(result, input_per_s=input_values)
            magnitude = _nonnegative(
                math.hypot(*input_values) + controls.input_error_bound_s_inv,
                "total input norm bound",
            )
        homogeneous = result.homogeneous
        if homogeneous.envelope is None:
            return result
        envelope = ForcedEnvelope(homogeneous.envelope, magnitude)
    except (
        np.linalg.LinAlgError,
        FloatingPointError,
        OverflowError,
        ValueError,
    ) as error:
        return replace(result, reason=str(error))
    return replace(
        result,
        status="numerically_supported",
        reason="conditional affine response under the declared operator/input bounds",
        envelope=envelope,
    )


def assess_affine_response(
    pencil: DampedPencil,
    residual: object,
    scales: SpectrumScales,
    controls: ForcedResponseControls,
) -> AffineResponseAssessment:
    """Assess an explicitly assumed constant affine ODE, retaining its load.

    Preconditions: common length-scaled residual/pencil coordinates and the
    existing regular positive-mass plant domain. Plant/input contract errors
    raise. Unresolved decay or numerical evaluation returns not_established.
    Constant frame/anchor/load history is a caller modeling assumption; no
    snapshot, small balance residual or passive grip establishes it here.
    """
    if not isinstance(controls, ForcedResponseControls):
        raise TypeError("controls must be ForcedResponseControls")
    if not isinstance(pencil, DampedPencil) or not isinstance(scales, SpectrumScales):
        raise TypeError("expected DampedPencil and SpectrumScales")
    mass = np.asarray(pencil.mass)
    load = finite_array(residual, (len(mass),), "residual")
    copied_load = tuple(float(item) for item in load)
    homogeneous = assess_autonomous_decay(pencil, scales, controls.decay)
    result = AffineResponseAssessment(
        "not_established", homogeneous.reason, homogeneous, copied_load, (), None
    )
    return _evaluate_affine_assessment(result, mass, scales, controls)


__all__ = ()
