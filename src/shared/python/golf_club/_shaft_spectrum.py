"""Private frozen shaft dynamics spectra; no stability certificates are inferred."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ._grip_contracts import finite_array
from ._rotating_body_contracts import RotatingFrameState
from ._shaft_clamped import balanced_clamped_dynamics
from ._shaft_equilibrium import EquilibriumControls
from ._shaft_rotating_chain import RotatingSectionChain

_NODE_DOF = 6


@dataclass(frozen=True)
class SpectrumScales:
    """Explicit SI coordinate scales and dimensionless numerical tolerances.

    Length scales translations relative to rotations; time scales velocity
    relative to displacement. Both must be finite and positive. The mass
    reciprocal-condition floor and residual tolerance lie strictly in (0, 1).
    These choices affect conditioning, never the underlying physical model.
    """

    length_m: float
    time_s: float
    mass_rcond_floor: float
    residual_tolerance: float

    def __post_init__(self) -> None:
        for name in ("length_m", "time_s", "mass_rcond_floor", "residual_tolerance"):
            value = float(finite_array(getattr(self, name), (), name))
            if value <= 0 or (name.endswith(("floor", "tolerance")) and value >= 1):
                raise ValueError(f"{name} must be positive and tolerances below one")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class FrozenSpectrum:
    """Fresh complex modes and diagnostics of a frozen linear differential system.

    Rates have units 1/s, velocity modes are displacement modes times rate.
    Modes have arbitrary complex normalization, and repeated-mode bases need
    not be unique. Residuals and eigenbasis conditioning refer to the declared
    dimensionless state coordinates. Small backward errors do not establish
    small forward errors near defective or clustered eigenvalues. Root support
    wrench, when present, is support-on-shaft in the root material frame.
    """

    rates_s_inv: np.ndarray
    displacement_modes: np.ndarray
    velocity_modes: np.ndarray
    relative_residuals: np.ndarray
    polynomial_relative_residuals: np.ndarray
    scaled_eigenbasis_rcond: float
    support_wrench: np.ndarray | None = None
    frame: RotatingFrameState | None = None
    grip_wrenches: tuple[np.ndarray, ...] = ()

    @property
    def stability_status(self) -> str:
        """Frozen eigenvalues alone do not qualify swing or nonlinear stability."""
        return "unqualified"


def _scaled_generator(
    mass: object, gyroscopic: object, stiffness: object, scales: SpectrumScales
) -> np.ndarray:
    if not isinstance(scales, SpectrumScales):
        raise TypeError("scales must be SpectrumScales")
    gyro = finite_array(gyroscopic, np.asarray(mass).shape, "gyroscopic matrix")
    _require_skew(gyro, scales.residual_tolerance)
    return _general_generator(mass, gyro, stiffness, scales)


def _require_skew(gyro: np.ndarray, tolerance: float) -> None:
    if np.linalg.norm(gyro + gyro.T) > tolerance * np.linalg.norm(gyro):
        raise ValueError("gyroscopic matrix must be skew symmetric")


def _validated_mass(mass: object, scales: SpectrumScales) -> np.ndarray:
    """Share unchanged mass-domain checks between spectra and acceleration solves."""
    if not isinstance(scales, SpectrumScales):
        raise TypeError("scales must be SpectrumScales")
    shape = np.asarray(mass).shape
    if len(shape) != 2 or shape[0] == 0 or shape[0] != shape[1]:
        raise ValueError("mass must be a nonempty square matrix")
    inertia = finite_array(mass, shape, "mass")
    tolerance = scales.residual_tolerance
    if np.linalg.norm(inertia - inertia.T) > tolerance * np.linalg.norm(inertia):
        raise ValueError("mass must be symmetric within the declared tolerance")
    # No projection or regularization: diagnose the supplied mass, then solve it.
    eigenvalues = np.linalg.eigvalsh(inertia)
    if (
        eigenvalues[0] <= 0
        or eigenvalues[0] / eigenvalues[-1] <= scales.mass_rcond_floor
    ):
        raise ValueError("mass must be positive definite and numerically resolved")
    return inertia


def _general_generator(
    mass: object, velocity: object, stiffness: object, scales: SpectrumScales
) -> np.ndarray:
    """Shared positive-mass kernel; callers qualify the velocity coefficient."""
    inertia = _validated_mass(mass, scales)
    if scales.time_s**2 == 0:
        raise ValueError("time scale squared must be numerically representable")
    transport = finite_array(velocity, inertia.shape, "velocity coefficient")
    tangent = finite_array(stiffness, inertia.shape, "stiffness")
    size = len(inertia)
    generator = np.zeros((2 * size, 2 * size))
    generator[:size, size:] = np.eye(size)
    generator[size:, :size] = -(scales.time_s**2) * np.linalg.solve(inertia, tangent)
    generator[size:, size:] = -scales.time_s * np.linalg.solve(inertia, transport)
    result: np.ndarray = finite_array(generator, generator.shape, "scaled generator")
    return result


def _polynomial_residuals(
    matrices: tuple[object, ...], rates: np.ndarray, modes: np.ndarray
) -> np.ndarray:
    """Normwise residual in the original quadratic pencil, using Frobenius norms."""
    mass, *velocity, stiffness = (np.asarray(matrix) for matrix in matrices)
    defect = (mass @ modes) * rates**2 + stiffness @ modes
    for matrix in velocity:
        defect += (matrix @ modes) * rates
    mode_norm = np.linalg.norm(modes, axis=0)
    if np.any(mode_norm == 0):
        raise ValueError("polynomial residual requires a nonzero displacement mode")
    denominator = (
        np.linalg.norm(mass) * np.abs(rates) ** 2
        + sum(float(np.linalg.norm(matrix)) for matrix in velocity) * np.abs(rates)
        + np.linalg.norm(stiffness)
    ) * mode_norm
    result: np.ndarray = np.zeros_like(denominator)
    np.divide(
        np.linalg.norm(defect, axis=0),
        denominator,
        out=result,
        where=denominator > 0,
    )
    return result


def _validate_result(result: FrozenSpectrum, tolerance: float) -> None:
    if not all(
        np.all(np.isfinite(value))
        for value in (
            result.rates_s_inv,
            result.displacement_modes,
            result.velocity_modes,
            result.relative_residuals,
            result.polynomial_relative_residuals,
            result.scaled_eigenbasis_rcond,
        )
    ) or any(
        np.any(value > tolerance)
        for value in (
            result.relative_residuals,
            result.polynomial_relative_residuals,
        )
    ):
        raise ValueError("frozen spectrum has nonfinite output or unresolved residuals")


def _frozen_spectrum(
    mass: object, gyroscopic: object, stiffness: object, scales: SpectrumScales
) -> FrozenSpectrum:
    """Diagnose already length-scaled M q'' + G q' + K q = 0.

    Require finite real square matrices, resolved positive mass and skew G.
    Retain nonsymmetric K, growing roots and deficient eigenbases. No damping,
    boundary conditions, equilibrium or autonomous evolution are inferred.
    Unresolved eigen residuals or failed numerical operations raise ValueError.
    """
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            generator = _scaled_generator(mass, gyroscopic, stiffness, scales)
            result = _spectrum_from_generator(
                generator, (mass, gyroscopic, stiffness), scales
            )
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        raise ValueError("frozen spectrum numerical evaluation failed") from error
    _validate_result(result, scales.residual_tolerance)
    return result


def _spectrum_from_generator(
    generator: np.ndarray, matrices: tuple[object, ...], scales: SpectrumScales
) -> FrozenSpectrum:
    """Share eigenpair diagnostics, retaining each original pencil coefficient."""
    rates, modes = np.linalg.eig(generator)
    defects = generator @ modes - modes * rates
    denominator = (np.linalg.norm(generator) + np.abs(rates)) * np.linalg.norm(
        modes, axis=0
    )
    residuals = np.divide(
        np.linalg.norm(defects, axis=0),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    singular_values = np.linalg.svd(modes, compute_uv=False)
    rcond = float(singular_values[-1] / singular_values[0])
    size = generator.shape[0] // 2
    return FrozenSpectrum(
        rates / scales.time_s,
        modes[:size].copy(),
        modes[size:].copy() / scales.time_s,
        residuals,
        _polynomial_residuals(matrices, rates / scales.time_s, modes[:size]),
        rcond,
    )


def clamped_chain_spectrum(
    chain: RotatingSectionChain,
    poses: object,
    controls: EquilibriumControls,
    scales: SpectrumScales,
) -> FrozenSpectrum:
    """Recheck material domain and free-node balance, then clamp only node zero.

    Uses full loaded M/G/K with a congruent translation/rotation length scale.
    Returned modes use physical material coordinates. A balanced snapshot can
    still be unstable or nonautonomous; no physical bandwidth is established.
    """
    if not isinstance(scales, SpectrumScales):
        raise TypeError("expected SpectrumScales")
    operators = balanced_clamped_dynamics(chain, poses, controls)
    coordinate_scale = np.tile([scales.length_m] * 3 + [1.0] * 3, chain.node_count - 1)
    congruence = coordinate_scale[:, None] * coordinate_scale[None, :]
    mass, gyroscopic, stiffness = (
        matrix[_NODE_DOF:, _NODE_DOF:] * congruence
        for matrix in (
            operators.mass,
            operators.gyroscopic,
            operators.stiffness,
        )
    )
    result = _frozen_spectrum(mass, gyroscopic, stiffness, scales)
    return replace(
        result,
        displacement_modes=result.displacement_modes * coordinate_scale[:, None],
        velocity_modes=result.velocity_modes * coordinate_scale[:, None],
        support_wrench=operators.residual[:_NODE_DOF].copy(),
    )


__all__ = ()
