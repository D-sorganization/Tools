"""Adapter from measured shaft profiles to rigid assembly mass properties."""

from __future__ import annotations

from .shaft_profile import ShaftProfile
from .types import ComponentMassProperties, ComponentRole

_GAUSS_NODES = (
    -0.906179845938664,
    -0.538469310105683,
    0.0,
    0.538469310105683,
    0.906179845938664,
)
_GAUSS_WEIGHTS = (
    0.236926885056189,
    0.478628670499366,
    0.568888888888889,
    0.478628670499366,
    0.236926885056189,
)


def shaft_component_mass_properties(
    profile: ShaftProfile,
) -> ComponentMassProperties:
    """Integrate one cut shaft into the canonical rigid-component contract.

    The shaft-local z axis runs from the cut butt toward the cut tip. The x and
    y axes are the transverse axes used by the shaft stiffness profile. Annular
    cross-section rotary inertia is inferred from measured linear density and
    diameters; the result does not infer material modulus from geometry.
    """
    if not isinstance(profile, ShaftProfile):
        raise TypeError("profile must be ShaftProfile")
    mass, first_moment, second_moment, transverse_local, polar = _integrals(profile)
    center_z = first_moment / mass
    distributed = second_moment - mass * center_z**2
    transverse = distributed + transverse_local
    return ComponentMassProperties(
        component_id=profile.shaft_id,
        role=ComponentRole.SHAFT,
        frame_id=profile.frame_id,
        mass_kg=mass,
        center_of_mass_m=(0.0, 0.0, center_z),
        inertia_at_com_kg_m2=(
            (transverse, 0.0, 0.0),
            (0.0, transverse, 0.0),
            (0.0, 0.0, polar),
        ),
    )


def _integrals(profile: ShaftProfile) -> tuple[float, float, float, float, float]:
    start = profile.butt_trim_m
    end = profile.raw_length_m - profile.tip_trim_m
    boundaries = [start, end]
    boundaries.extend(
        station.position_m
        for station in profile.stations
        if start < station.position_m < end
    )
    totals = [0.0, 0.0, 0.0, 0.0, 0.0]
    ordered = sorted(boundaries)
    for left, right in zip(ordered, ordered[1:], strict=False):
        midpoint = 0.5 * (left + right)
        half_width = 0.5 * (right - left)
        for node, weight in zip(_GAUSS_NODES, _GAUSS_WEIGHTS, strict=True):
            raw_position = midpoint + half_width * node
            station = profile.station_at(raw_position)
            local_z = raw_position - start
            density = station.linear_density_kg_m
            radial_squared = station.outer_diameter_m**2 + station.inner_diameter_m**2
            scale = half_width * weight
            totals[0] += scale * density
            totals[1] += scale * density * local_z
            totals[2] += scale * density * local_z**2
            totals[3] += scale * density * radial_squared / 16.0
            totals[4] += scale * density * radial_squared / 8.0
    return tuple(totals)  # type: ignore[return-value]


__all__ = ["shaft_component_mass_properties"]
