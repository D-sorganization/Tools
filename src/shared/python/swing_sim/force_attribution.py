"""Coordinate-explicit force, impulse, power, and work attribution.

The equation convention is ``M(q) qdd + h(q, v) + g(q) + d(v) = tau``.
The Christoffel/monomial split used here assigns squared-speed monomials to
``squared_speed`` and cross-speed monomials to ``coriolis``.  That split is a
property of the declared generalized coordinates; it is not a set of
independently applied or directly measured forces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .reference import coriolis_vector, damping_vector, gravity_vector, mass_matrix
from .types import PendulumParameters

FloatArray = NDArray[np.float64]
COMPONENT_NAMES = (
    "coriolis",
    "squared_speed",
    "velocity_residual",
    "gravity",
    "damping",
    "applied",
)
ATTRIBUTION_SCHEMA_VERSION = "force-attribution/v1"


def _vector(name: str, value: object, size: int) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


@runtime_checkable
class AttributionProvider(Protocol):
    """LoD boundary for one coordinate-frozen mechanical model."""

    coordinate_names: tuple[str, ...]
    endpoint_name: str

    def mass_matrix(self, q: FloatArray) -> FloatArray: ...

    def mass_matrix_derivatives(self, q: FloatArray) -> FloatArray: ...

    def velocity_bias(self, q: FloatArray, velocity: FloatArray) -> FloatArray: ...

    def gravity(self, q: FloatArray) -> FloatArray: ...

    def damping(self, velocity: FloatArray) -> FloatArray: ...

    def endpoint_jacobian(self, q: FloatArray) -> FloatArray: ...


@dataclass(frozen=True)
class AttributionComponent:
    """One equation term and its equal-and-opposite generalized drive."""

    equation_term: FloatArray
    generalized_drive: FloatArray
    endpoint_force_n: FloatArray
    endpoint_generalized_drive_nm: FloatArray
    mapping_residual_nm: FloatArray
    mapping_rank: int
    mapping_status: str
    tangent_force_n: float | None
    generalized_power_w: float
    endpoint_power_w: float


@dataclass(frozen=True)
class StateAttribution:
    """Exact pointwise attribution under a frozen coordinate convention."""

    q: FloatArray
    velocity: FloatArray
    velocity_bias: FloatArray
    components: dict[str, AttributionComponent]
    total_generalized_drive: FloatArray
    acceleration: FloatArray
    endpoint_velocity_m_s: FloatArray
    coordinate_names: tuple[str, ...]
    endpoint_name: str
    schema_version: str = ATTRIBUTION_SCHEMA_VERSION
    convention: str = "christoffel_first_kind_cross_vs_squared_speed"


@dataclass(frozen=True)
class ComponentHistory:
    """Time history for one attributed source."""

    generalized_drive_nm: FloatArray
    endpoint_force_n: FloatArray
    mapping_residual_nm: FloatArray
    tangent_force_n: FloatArray
    generalized_power_w: FloatArray
    endpoint_power_w: FloatArray


@dataclass(frozen=True)
class ComponentMetrics:
    """Integrated metrics; impulse and work remain separate estimands."""

    signed_generalized_impulse_nm_s: FloatArray
    absolute_generalized_impulse_nm_s: FloatArray
    signed_tangent_impulse_n_s: float | None
    absolute_tangent_impulse_n_s: float | None
    generalized_work_j: float
    endpoint_work_j: float
    tangent_impulse_cancellation: float | None


@dataclass(frozen=True)
class TrajectoryAttribution:
    """Pointwise histories and integrated metrics for an achieved trajectory."""

    time_s: FloatArray
    components: dict[str, ComponentHistory]
    metrics: dict[str, ComponentMetrics]
    total_generalized_drive_nm: FloatArray
    acceleration_rad_s2: FloatArray
    coordinate_names: tuple[str, ...]
    endpoint_name: str
    schema_version: str = ATTRIBUTION_SCHEMA_VERSION


def _christoffel_split(
    mass_derivatives: FloatArray, velocity: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return cross-speed and squared-speed terms from first-kind symbols."""
    size = velocity.size
    expected = (size, size, size)
    if mass_derivatives.shape != expected:
        raise ValueError(
            "mass_matrix_derivatives must have shape "
            f"{expected} ordered as (derivative_coordinate, row, column)"
        )
    coriolis = np.zeros(size, dtype=np.float64)
    squared = np.zeros(size, dtype=np.float64)
    for output in range(size):
        for first in range(size):
            gamma_diagonal = 0.5 * (
                2.0 * mass_derivatives[first, output, first]
                - mass_derivatives[output, first, first]
            )
            squared[output] += gamma_diagonal * velocity[first] ** 2
            for second in range(first + 1, size):
                gamma = 0.5 * (
                    mass_derivatives[second, output, first]
                    + mass_derivatives[first, output, second]
                    - mass_derivatives[output, first, second]
                )
                coriolis[output] += 2.0 * gamma * velocity[first] * velocity[second]
    return coriolis, squared


def _map_endpoint(
    jacobian: FloatArray,
    generalized_drive: FloatArray,
    velocity: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, int, str, float | None, float]:
    if jacobian.ndim != 2 or jacobian.shape[1] != generalized_drive.size:
        raise ValueError(
            "endpoint_jacobian must have shape (task_dimension, coordinates)"
        )
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("endpoint_jacobian must contain only finite values")
    endpoint_force, _, rank, _ = np.linalg.lstsq(
        jacobian.T, generalized_drive, rcond=None
    )
    reconstructed = jacobian.T @ endpoint_force
    residual = generalized_drive - reconstructed
    tolerance = 1e-10 * max(1.0, float(np.linalg.norm(generalized_drive)))
    exact = float(np.linalg.norm(residual)) <= tolerance
    coordinate_count = generalized_drive.size
    if exact and rank == coordinate_count:
        status = "exact_force_only"
    elif rank < coordinate_count:
        status = "rank_deficient_force_only"
    else:
        status = "least_squares_force_only"
    endpoint_velocity = jacobian @ velocity
    speed = float(np.linalg.norm(endpoint_velocity))
    tangent_force = (
        None if speed <= 1e-12 else float(endpoint_force @ endpoint_velocity / speed)
    )
    endpoint_power = float(endpoint_force @ endpoint_velocity)
    return (
        endpoint_force,
        reconstructed,
        residual,
        int(rank),
        status,
        tangent_force,
        endpoint_power,
    )


def attribute_state(
    provider: AttributionProvider,
    q: FloatArray,
    velocity: FloatArray,
    applied_torque_nm: FloatArray,
) -> StateAttribution:
    """Attribute one achieved state and solve its forward acceleration.

    Endpoint forces are force-only virtual-work equivalents.  Rank and the
    unreconstructed generalized residual are always reported, preventing a
    joint couple from being silently relabeled as hand-path force.
    """
    size = len(provider.coordinate_names)
    if size < 1 or len(set(provider.coordinate_names)) != size:
        raise ValueError("coordinate_names must be unique and non-empty")
    q_array = _vector("q", q, size)
    velocity_array = _vector("velocity", velocity, size)
    applied = _vector("applied_torque_nm", applied_torque_nm, size)
    mass = np.asarray(provider.mass_matrix(q_array), dtype=np.float64)
    if mass.shape != (size, size) or not np.all(np.isfinite(mass)):
        raise ValueError(f"mass_matrix must be finite with shape ({size}, {size})")
    derivatives = np.asarray(
        provider.mass_matrix_derivatives(q_array), dtype=np.float64
    )
    coriolis, squared = _christoffel_split(derivatives, velocity_array)
    velocity_bias = _vector(
        "velocity_bias", provider.velocity_bias(q_array, velocity_array), size
    )
    equation_terms = {
        "coriolis": coriolis,
        "squared_speed": squared,
        "velocity_residual": velocity_bias - coriolis - squared,
        "gravity": _vector("gravity", provider.gravity(q_array), size),
        "damping": _vector("damping", provider.damping(velocity_array), size),
        "applied": -applied,
    }
    jacobian = np.asarray(provider.endpoint_jacobian(q_array), dtype=np.float64)
    components: dict[str, AttributionComponent] = {}
    for name in COMPONENT_NAMES:
        equation_term = equation_terms[name]
        drive = -equation_term
        mapped = _map_endpoint(jacobian, drive, velocity_array)
        components[name] = AttributionComponent(
            equation_term=equation_term,
            generalized_drive=drive,
            endpoint_force_n=mapped[0],
            endpoint_generalized_drive_nm=mapped[1],
            mapping_residual_nm=mapped[2],
            mapping_rank=mapped[3],
            mapping_status=mapped[4],
            tangent_force_n=mapped[5],
            generalized_power_w=float(drive @ velocity_array),
            endpoint_power_w=mapped[6],
        )
    total_drive = sum(
        (component.generalized_drive for component in components.values()),
        start=np.zeros(size, dtype=np.float64),
    )
    try:
        acceleration = np.linalg.solve(mass, total_drive)
    except np.linalg.LinAlgError as error:
        raise ValueError("mass_matrix must be nonsingular") from error
    return StateAttribution(
        q=q_array,
        velocity=velocity_array,
        velocity_bias=velocity_bias,
        components=components,
        total_generalized_drive=total_drive,
        acceleration=acceleration,
        endpoint_velocity_m_s=jacobian @ velocity_array,
        coordinate_names=provider.coordinate_names,
        endpoint_name=provider.endpoint_name,
    )


def _validated_history(
    provider: AttributionProvider,
    time_s: FloatArray,
    q: FloatArray,
    velocity: FloatArray,
    applied_torque_nm: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    time = np.asarray(time_s, dtype=np.float64).reshape(-1)
    if time.size < 2:
        raise ValueError("time_s must contain at least two samples")
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
        raise ValueError("time_s must be finite and strictly increasing")
    expected = (time.size, len(provider.coordinate_names))
    histories = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (q, velocity, applied_torque_nm)
    )
    if any(value.shape != expected for value in histories):
        raise ValueError(
            f"q, velocity, and applied_torque_nm must each have shape {expected}"
        )
    if not all(np.all(np.isfinite(value)) for value in histories):
        raise ValueError("trajectory histories must contain only finite values")
    return time, histories[0], histories[1], histories[2]


def attribute_trajectory(
    provider: AttributionProvider,
    time_s: FloatArray,
    q: FloatArray,
    velocity: FloatArray,
    applied_torque_nm: FloatArray,
) -> TrajectoryAttribution:
    """Attribute an achieved trajectory and integrate source-specific metrics."""
    time, q_history, velocity_history, applied_history = _validated_history(
        provider, time_s, q, velocity, applied_torque_nm
    )
    states = tuple(
        attribute_state(provider, q_sample, velocity_sample, applied_sample)
        for q_sample, velocity_sample, applied_sample in zip(
            q_history, velocity_history, applied_history, strict=True
        )
    )
    histories: dict[str, ComponentHistory] = {}
    metrics: dict[str, ComponentMetrics] = {}
    for name in COMPONENT_NAMES:
        state_components = tuple(state.components[name] for state in states)
        tangent = np.array(
            [
                np.nan if item.tangent_force_n is None else item.tangent_force_n
                for item in state_components
            ],
            dtype=np.float64,
        )
        history = ComponentHistory(
            generalized_drive_nm=np.stack(
                [item.generalized_drive for item in state_components]
            ),
            endpoint_force_n=np.stack(
                [item.endpoint_force_n for item in state_components]
            ),
            mapping_residual_nm=np.stack(
                [item.mapping_residual_nm for item in state_components]
            ),
            tangent_force_n=tangent,
            generalized_power_w=np.array(
                [item.generalized_power_w for item in state_components]
            ),
            endpoint_power_w=np.array(
                [item.endpoint_power_w for item in state_components]
            ),
        )
        histories[name] = history
        tangent_valid = bool(np.all(np.isfinite(tangent)))
        signed_tangent = float(np.trapezoid(tangent, time)) if tangent_valid else None
        absolute_tangent = (
            float(np.trapezoid(np.abs(tangent), time)) if tangent_valid else None
        )
        cancellation = None
        if (
            signed_tangent is not None
            and absolute_tangent is not None
            and absolute_tangent > 0.0
        ):
            cancellation = 1.0 - abs(signed_tangent) / absolute_tangent
        metrics[name] = ComponentMetrics(
            signed_generalized_impulse_nm_s=np.trapezoid(
                history.generalized_drive_nm, time, axis=0
            ),
            absolute_generalized_impulse_nm_s=np.trapezoid(
                np.abs(history.generalized_drive_nm), time, axis=0
            ),
            signed_tangent_impulse_n_s=signed_tangent,
            absolute_tangent_impulse_n_s=absolute_tangent,
            generalized_work_j=float(np.trapezoid(history.generalized_power_w, time)),
            endpoint_work_j=float(np.trapezoid(history.endpoint_power_w, time)),
            tangent_impulse_cancellation=cancellation,
        )
    return TrajectoryAttribution(
        time_s=time,
        components=histories,
        metrics=metrics,
        total_generalized_drive_nm=np.stack(
            [state.total_generalized_drive for state in states]
        ),
        acceleration_rad_s2=np.stack([state.acceleration for state in states]),
        coordinate_names=provider.coordinate_names,
        endpoint_name=provider.endpoint_name,
    )


def component_impulse_objective(
    attribution: TrajectoryAttribution,
    component: str,
    *,
    absolute: bool = False,
) -> float:
    """Return a minimizer-compatible objective for component tangent impulse."""
    if component not in attribution.metrics:
        raise ValueError(f"unknown component {component!r}")
    metric = attribution.metrics[component]
    value = (
        metric.absolute_tangent_impulse_n_s
        if absolute
        else metric.signed_tangent_impulse_n_s
    )
    if value is None:
        raise ValueError(
            "tangent impulse is unavailable when endpoint speed reaches zero"
        )
    return -value


@dataclass(frozen=True)
class DoublePendulumAttributionProvider:
    """Exact Tools double-pendulum provider in relative-angle coordinates."""

    parameters: PendulumParameters
    g_inplane: tuple[float, float]
    coordinate_names: tuple[str, ...] = ("shoulder_absolute", "wrist_relative")
    endpoint_name: str = "wrist_hand_path"

    def __post_init__(self) -> None:
        gravity = _vector("g_inplane", self.g_inplane, 2)
        object.__setattr__(self, "g_inplane", (float(gravity[0]), float(gravity[1])))

    def mass_matrix(self, q: FloatArray) -> FloatArray:
        q_array = _vector("q", q, 2)
        return np.asarray(mass_matrix(self.parameters, float(q_array[1])))

    def mass_matrix_derivatives(self, q: FloatArray) -> FloatArray:
        q_array = _vector("q", q, 2)
        coupling = self.parameters.m2 * self.parameters.l1 * self.parameters.lc2
        derivative = -coupling * np.sin(q_array[1])
        result = np.zeros((2, 2, 2), dtype=np.float64)
        result[1] = np.array([[2.0 * derivative, derivative], [derivative, 0.0]])
        return result

    def velocity_bias(self, q: FloatArray, velocity: FloatArray) -> FloatArray:
        q_array = _vector("q", q, 2)
        speed = _vector("velocity", velocity, 2)
        return np.asarray(
            coriolis_vector(
                self.parameters,
                float(q_array[1]),
                float(speed[0]),
                float(speed[1]),
            ),
            dtype=np.float64,
        )

    def gravity(self, q: FloatArray) -> FloatArray:
        q_array = _vector("q", q, 2)
        return np.asarray(
            gravity_vector(
                self.parameters,
                float(q_array[0]),
                float(q_array[1]),
                self.g_inplane,
            ),
            dtype=np.float64,
        )

    def damping(self, velocity: FloatArray) -> FloatArray:
        speed = _vector("velocity", velocity, 2)
        return np.asarray(
            damping_vector(self.parameters, float(speed[0]), float(speed[1])),
            dtype=np.float64,
        )

    def endpoint_jacobian(self, q: FloatArray) -> FloatArray:
        q_array = _vector("q", q, 2)
        theta = float(q_array[0])
        return np.array(
            [
                [self.parameters.l1 * np.cos(theta), 0.0],
                [self.parameters.l1 * np.sin(theta), 0.0],
            ],
            dtype=np.float64,
        )


__all__ = [
    "ATTRIBUTION_SCHEMA_VERSION",
    "AttributionComponent",
    "AttributionProvider",
    "ComponentHistory",
    "ComponentMetrics",
    "DoublePendulumAttributionProvider",
    "StateAttribution",
    "TrajectoryAttribution",
    "attribute_state",
    "attribute_trajectory",
    "component_impulse_objective",
]
