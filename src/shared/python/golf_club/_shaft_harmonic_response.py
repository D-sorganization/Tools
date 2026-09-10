"""Private force/torque compliance of frozen loaded-shaft equations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grip_contracts import finite_array
from ._shaft_clamped import balanced_clamped_dynamics
from ._shaft_equilibrium import EquilibriumControls
from ._shaft_loaded_dynamics import LoadedChainDynamics
from ._shaft_rotating_chain import RotatingSectionChain


@dataclass(frozen=True)
class TipHarmonicControls:
    """Explicit material-frame point, SI frequency/length and numerical limits.

    Frequency is nonnegative angular frequency in rad/s; zero means a static
    perturbation. The point offset is measured from the tip node in its material
    axes. The floor applies to dynamic reciprocal condition and coefficient
    resolution; it and the residual tolerance are in (0,1). No damping, human
    parameters or acoustic calibration is supplied.
    """

    angular_frequency_rad_s: float
    point_offset_m: tuple[float, float, float]
    length_m: float
    dynamic_rcond_floor: float
    residual_tolerance: float

    def __post_init__(self) -> None:
        point = finite_array(self.point_offset_m, (3,), "point offset")
        object.__setattr__(self, "point_offset_m", tuple(float(item) for item in point))
        for name in (
            "angular_frequency_rad_s",
            "length_m",
            "dynamic_rcond_floor",
            "residual_tolerance",
        ):
            value = float(finite_array(getattr(self, name), (), name))
            if value < 0 or (name != "angular_frequency_rad_s" and value == 0):
                raise ValueError(
                    "frequency must be nonnegative and other scales positive"
                )
            if name.endswith(("floor", "tolerance")) and value >= 1:
                raise ValueError("numerical tolerances must be below one")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class FrozenTipCompliance:
    """Fresh complex transfer arrays with the exp(+i*omega*t) convention.

    Input columns are material point forces (N), then torques (N m). Output
    rows are point translations (m), then infinitesimal rotations (rad); the
    velocity mobility has corresponding rates. Free displacement retains every
    nonroot node. Support transfer is the harmonic support-on-shaft root wrench;
    support_wrench retains the distinct equilibrium reaction. Arrays are fresh.
    This is an algebraic particular solution, not proof of a stable steady
    response, autonomous swing motion, valid bandwidth or acoustic radiation.
    """

    angular_frequency_rad_s: float
    displacement_compliance: np.ndarray
    velocity_mobility: np.ndarray
    free_displacement: np.ndarray
    support_wrench_transfer: np.ndarray
    support_wrench: np.ndarray
    scaled_rcond: float
    coefficient_resolution: float
    relative_residuals: np.ndarray

    @property
    def stability_status(self) -> str:
        return "unqualified"


def _point_motion(offset: tuple[float, float, float]) -> np.ndarray:
    """Linear-first tip-to-point velocity map; its transpose transfers wrench."""
    result = np.eye(6)
    result[:3, 3:] = np.column_stack([np.cross(axis, offset) for axis in np.eye(3)])
    return result


def _resolved_response(
    dynamic: np.ndarray,
    force: np.ndarray,
    controls: TipHarmonicControls,
    coefficient_scale: float,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    singular = np.linalg.svd(dynamic, compute_uv=False)
    rcond = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.0
    if not np.isfinite(rcond) or rcond <= controls.dynamic_rcond_floor:
        raise ValueError("dynamic matrix is singular or below the condition floor")
    resolution = (
        float(singular[-1] / coefficient_scale) if coefficient_scale > 0 else 0.0
    )
    if not np.isfinite(resolution) or resolution <= controls.dynamic_rcond_floor:
        raise ValueError("dynamic coefficient cancellation is numerically unresolved")
    motion = np.linalg.solve(dynamic, force)
    defect = dynamic @ motion - force
    denominator = np.linalg.norm(dynamic) * np.linalg.norm(
        motion, axis=0
    ) + np.linalg.norm(force, axis=0)
    residuals = np.linalg.norm(defect, axis=0) / denominator
    if (
        not np.all(np.isfinite(motion))
        or not np.all(np.isfinite(residuals))
        or np.any(residuals > controls.residual_tolerance)
    ):
        raise ValueError("harmonic response is nonfinite or has unresolved residuals")
    return motion, rcond, resolution, residuals


def _dynamic_operators(
    operators: LoadedChainDynamics, controls: TipHarmonicControls
) -> tuple[np.ndarray, np.ndarray, float]:
    """Keep cancellation scale relative to the original length-scaled pencil."""
    frequency = controls.angular_frequency_rad_s
    terms = (
        operators.stiffness,
        -(frequency**2) * operators.mass,
        1j * frequency * operators.gyroscopic,
    )
    return _scaled_pencil(terms, controls.length_m, 1)


def _scaled_pencil(
    terms: tuple[np.ndarray, ...], length_m: float, first_free_node: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Scale retained nodes while measuring cancellation against original terms."""
    start = 6 * first_free_node
    scale = np.tile([length_m] * 3 + [1.0] * 3, len(terms[0]) // 6 - first_free_node)
    congruence = scale[:, None] * scale[None, :]
    coefficient_scale = sum(
        (float(np.linalg.norm(term[start:, start:] * congruence)) for term in terms),
        0.0,
    )
    return np.add.reduce(terms), scale, coefficient_scale


@dataclass(frozen=True)
class _PointResponse:
    displacement: np.ndarray
    compliance: np.ndarray
    mobility: np.ndarray
    rcond: float
    resolution: float
    residuals: np.ndarray


def _point_response(
    dynamic: np.ndarray,
    scale: np.ndarray,
    coefficient_scale: float,
    controls: TipHarmonicControls,
) -> _PointResponse:
    """Share the length-scaled point-wrench solve and conjugate SI output map."""
    port = _point_motion(controls.point_offset_m)
    reduced = dynamic * scale[:, None] * scale[None, :]
    force = np.zeros((len(scale), 6))
    force[-6:] = port.T
    motion, rcond, resolution, residuals = _resolved_response(
        reduced, scale[:, None] * force, controls, coefficient_scale
    )
    physical = scale[:, None] * motion
    compliance = port @ physical[-6:]
    mobility = 1j * controls.angular_frequency_rad_s * compliance
    if not all(np.all(np.isfinite(item)) for item in (physical, compliance, mobility)):
        raise ValueError("harmonic response is nonfinite after SI mapping")
    return _PointResponse(physical, compliance, mobility, rcond, resolution, residuals)


def clamped_tip_compliance(
    chain: RotatingSectionChain,
    poses: object,
    equilibrium: EquilibriumControls,
    controls: TipHarmonicControls,
) -> FrozenTipCompliance:
    """Solve full nonsymmetric loaded equations for six point-wrench inputs.

    Clamp node zero only, recheck balance/domain, and retain full inertial,
    gyroscopic and stiffness coupling. Numerical resonances fail closed; no
    added damping, pseudoinverse, mass repair or stiffness projection is used.
    Singular mass need not prevent an algebraic frequency solution; no ODE,
    descriptor-system regularity or stability qualification follows from it.
    """
    if not isinstance(controls, TipHarmonicControls):
        raise TypeError("expected TipHarmonicControls")
    operators = balanced_clamped_dynamics(chain, poses, equilibrium)
    frequency = controls.angular_frequency_rad_s
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            dynamic, scale, coefficient_scale = _dynamic_operators(operators, controls)
            point = _point_response(dynamic[6:, 6:], scale, coefficient_scale, controls)
            support = dynamic[:6, 6:] @ point.displacement
            if not np.all(np.isfinite(support)):
                raise ValueError("harmonic response is nonfinite after SI mapping")
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        raise ValueError("harmonic response numerical evaluation failed") from error
    return FrozenTipCompliance(
        frequency,
        point.compliance,
        point.mobility,
        point.displacement,
        support,
        operators.residual[:6].copy(),
        point.rcond,
        point.resolution,
        point.residuals,
    )


__all__ = ()
