"""Passive restitution plus Coulomb sphere-plane impact resolution."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ._vector_math import add, cross, dot, norm, scale, subtract, unit
from .contract_types import GroundContactState, GroundSurfaceProfile, Vector3
from .impact_types import (
    ImpactEnergyLedger,
    ImpactImpulseResult,
    ImpactRegime,
    ImpactRejectionReason,
    ImpactStateError,
    SphereProperties,
)

_VELOCITY_TOLERANCE_M_S = 1e-12
_IMPULSE_TOLERANCE_N_S = 1e-12
_ENERGY_ABSOLUTE_TOLERANCE_J = 1e-10
_ENERGY_RELATIVE_TOLERANCE = 1e-10
_REGIME_STICKING = ImpactRegime("sticking")
_REGIME_SLIDING = ImpactRegime("sliding")
_REJECTION_GRAZING = ImpactRejectionReason("grazing")
_REJECTION_OUTGOING = ImpactRejectionReason("outgoing")


@dataclass(frozen=True)
class _ImpactSolution:
    state: GroundContactState
    surface: GroundSurfaceProfile
    body: SphereProperties
    contact_arm_m: Vector3
    contact_velocity_before_m_s: Vector3
    restitution: float
    normal_impulse_n_s: float
    tangent_impulse_n_s: Vector3
    regime: ImpactRegime


def _restitution(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("normal restitution must be finite")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("normal restitution must lie within [0, 1]")
    return number


def _contact_velocity(
    state: GroundContactState,
    surface: GroundSurfaceProfile,
    contact_arm_m: Vector3,
) -> Vector3:
    rotational = cross(state.angular_velocity_rad_s, contact_arm_m)
    return subtract(add(state.velocity_m_s, rotational), surface.surface_velocity_m_s)


def _kinetic_energy(
    state: GroundContactState,
    body: SphereProperties,
) -> float:
    translation = 0.5 * body.mass_kg * dot(state.velocity_m_s, state.velocity_m_s)
    rotation = (
        0.5
        * body.inertia_kg_m2
        * dot(
            state.angular_velocity_rad_s,
            state.angular_velocity_rad_s,
        )
    )
    return translation + rotation


def _tangential_impulse(
    contact_tangent_m_s: Vector3,
    normal_impulse_n_s: float,
    surface: GroundSurfaceProfile,
    effective_mass_kg: float,
) -> tuple[Vector3, ImpactRegime]:
    desired = scale(contact_tangent_m_s, -effective_mass_kg)
    static_limit = surface.static_friction * normal_impulse_n_s
    if norm(desired) <= static_limit + _IMPULSE_TOLERANCE_N_S:
        return desired, _REGIME_STICKING
    direction = unit(contact_tangent_m_s, tolerance=_VELOCITY_TOLERANCE_M_S)
    return (
        scale(direction, -surface.kinetic_friction * normal_impulse_n_s),
        _REGIME_SLIDING,
    )


def _energy_ledger(
    solution: _ImpactSolution,
    after: GroundContactState,
    impulse_n_s: Vector3,
) -> ImpactEnergyLedger:
    kinetic_before = _kinetic_energy(solution.state, solution.body)
    kinetic_after = _kinetic_energy(after, solution.body)
    boundary_work = dot(impulse_n_s, solution.surface.surface_velocity_m_s)
    dissipation = kinetic_before + boundary_work - kinetic_after
    tolerance = _ENERGY_ABSOLUTE_TOLERANCE_J + _ENERGY_RELATIVE_TOLERANCE * max(
        kinetic_before,
        kinetic_after,
        abs(boundary_work),
    )
    if dissipation < -tolerance:
        raise ValueError("impact violates passive energy accounting")
    return ImpactEnergyLedger(
        kinetic_before,
        kinetic_after,
        boundary_work,
        max(0.0, dissipation),
    )


def _validate_contact_postconditions(
    solution: _ImpactSolution,
    after_m_s: Vector3,
) -> None:
    normal = solution.surface.normal_unit
    before_m_s = solution.contact_velocity_before_m_s
    before_normal = dot(before_m_s, normal)
    after_normal = dot(after_m_s, normal)
    if not math.isclose(
        after_normal,
        -solution.restitution * before_normal,
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise ValueError("impact restitution postcondition failed")
    tangent_before = subtract(before_m_s, scale(normal, before_normal))
    tangent_after = subtract(after_m_s, scale(normal, after_normal))
    if solution.regime is _REGIME_STICKING and norm(tangent_after) > 1e-9:
        raise ValueError("sticking impact left residual contact slip")
    if (
        solution.regime is _REGIME_SLIDING
        and dot(tangent_before, tangent_after) < -1e-10
    ):
        raise ValueError("sliding impact reversed contact slip")


def _resolved_tangent(
    contact_velocity_m_s: Vector3,
    normal_impulse_n_s: float,
    surface: GroundSurfaceProfile,
    body: SphereProperties,
) -> tuple[Vector3, ImpactRegime]:
    normal_speed = dot(contact_velocity_m_s, surface.normal_unit)
    tangent_velocity = subtract(
        contact_velocity_m_s,
        scale(surface.normal_unit, normal_speed),
    )
    if norm(tangent_velocity) <= _VELOCITY_TOLERANCE_M_S:
        return (0.0, 0.0, 0.0), _REGIME_STICKING
    return _tangential_impulse(
        tangent_velocity,
        normal_impulse_n_s,
        surface,
        body.tangential_effective_mass_kg,
    )


def _updated_state(
    solution: _ImpactSolution,
) -> tuple[GroundContactState, Vector3]:
    total_impulse = add(
        scale(solution.surface.normal_unit, solution.normal_impulse_n_s),
        solution.tangent_impulse_n_s,
    )
    after = replace(
        solution.state,
        velocity_m_s=add(
            solution.state.velocity_m_s,
            scale(total_impulse, 1.0 / solution.body.mass_kg),
        ),
        angular_velocity_rad_s=add(
            solution.state.angular_velocity_rad_s,
            scale(
                cross(solution.contact_arm_m, solution.tangent_impulse_n_s),
                1.0 / solution.body.inertia_kg_m2,
            ),
        ),
    )
    return after, total_impulse


def _friction_utilization(
    tangent_impulse_n_s: Vector3,
    normal_impulse_n_s: float,
    surface: GroundSurfaceProfile,
) -> float:
    static_limit = surface.static_friction * normal_impulse_n_s
    tangent_magnitude = norm(tangent_impulse_n_s)
    if tangent_magnitude > static_limit + _IMPULSE_TOLERANCE_N_S:
        raise ValueError("impact tangential impulse exceeds the friction cone")
    if static_limit <= _IMPULSE_TOLERANCE_N_S:
        return 0.0
    return tangent_magnitude / static_limit


def _validate_records(
    state: GroundContactState,
    surface: GroundSurfaceProfile,
    body: SphereProperties,
) -> None:
    if type(state) is not GroundContactState:
        raise ValueError("impact requires an exact ground contact state")
    if type(surface) is not GroundSurfaceProfile:
        raise ValueError("impact requires an exact ground surface profile")
    if type(body) is not SphereProperties:
        raise ValueError("impact requires exact sphere properties")
    if state.frame is not surface.frame:
        raise ValueError("impact state and surface frames must match")


def _build_solution(
    state: GroundContactState,
    surface: GroundSurfaceProfile,
    body: SphereProperties,
    restitution: float,
) -> _ImpactSolution:
    normal = surface.normal_unit
    contact_arm = scale(normal, -body.radius_m)
    before_contact = _contact_velocity(state, surface, contact_arm)
    incoming_normal = dot(before_contact, normal)
    if abs(incoming_normal) <= _VELOCITY_TOLERANCE_M_S:
        raise ImpactStateError(_REJECTION_GRAZING)
    if incoming_normal > 0.0:
        raise ImpactStateError(_REJECTION_OUTGOING)
    normal_impulse = -(1.0 + restitution) * body.mass_kg * incoming_normal
    tangent_impulse, regime = _resolved_tangent(
        before_contact, normal_impulse, surface, body
    )
    return _ImpactSolution(
        state,
        surface,
        body,
        contact_arm,
        before_contact,
        restitution,
        normal_impulse,
        tangent_impulse,
        regime,
    )


def resolve_sphere_plane_impact(
    state: GroundContactState,
    surface: GroundSurfaceProfile,
    body: SphereProperties,
    *,
    normal_restitution: float | None = None,
) -> ImpactImpulseResult:
    """Resolve an incoming impact and enforce passive contact postconditions."""
    _validate_records(state, surface, body)
    selected = (
        surface.normal_restitution if normal_restitution is None else normal_restitution
    )
    restitution = _restitution(selected)
    solution = _build_solution(state, surface, body, restitution)
    after, total_impulse = _updated_state(solution)
    after_contact = _contact_velocity(after, surface, solution.contact_arm_m)
    _validate_contact_postconditions(solution, after_contact)
    energy = _energy_ledger(solution, after, total_impulse)
    utilization = _friction_utilization(
        solution.tangent_impulse_n_s,
        solution.normal_impulse_n_s,
        surface,
    )
    return ImpactImpulseResult(
        state,
        after,
        solution.regime,
        solution.normal_impulse_n_s,
        solution.tangent_impulse_n_s,
        total_impulse,
        solution.contact_velocity_before_m_s,
        after_contact,
        restitution,
        utilization,
        energy,
    )


__all__ = ["resolve_sphere_plane_impact"]
