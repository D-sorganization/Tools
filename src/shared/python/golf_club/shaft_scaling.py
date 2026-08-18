"""Explicit immutable scaling operations for measured shaft profiles."""

from __future__ import annotations

from dataclasses import dataclass

from ._validation import require_finite_float, require_identifier
from .shaft_profile import (
    ShaftProfile,
    ShaftProfileProvenance,
    ShaftStation,
)


@dataclass(frozen=True)
class ShaftProfileScaling:
    """Dimensionless what-if factors applied to measured station properties."""

    mass_scale: float = 1.0
    ei_about_x_scale: float = 1.0
    ei_about_y_scale: float = 1.0
    gj_scale: float = 1.0
    damping_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                require_finite_float(getattr(self, name), name, positive=True),
            )


def scale_shaft_profile(
    profile: ShaftProfile,
    scaling: ShaftProfileScaling,
    *,
    shaft_id: str,
) -> ShaftProfile:
    """Return a derived profile without modifying geometry or the source profile."""
    if not isinstance(profile, ShaftProfile):
        raise TypeError("profile must be ShaftProfile")
    if not isinstance(scaling, ShaftProfileScaling):
        raise TypeError("scaling must be ShaftProfileScaling")
    derived_id = require_identifier(shaft_id, "shaft_id")
    source = profile.provenance
    stations = tuple(
        ShaftStation(
            position_m=station.position_m,
            outer_diameter_m=station.outer_diameter_m,
            inner_diameter_m=station.inner_diameter_m,
            linear_density_kg_m=(station.linear_density_kg_m * scaling.mass_scale),
            ei_about_x_n_m2=(station.ei_about_x_n_m2 * scaling.ei_about_x_scale),
            ei_about_y_n_m2=(station.ei_about_y_n_m2 * scaling.ei_about_y_scale),
            gj_n_m2=station.gj_n_m2 * scaling.gj_scale,
            damping_ratio=station.damping_ratio * scaling.damping_scale,
            spine_angle_rad=station.spine_angle_rad,
        )
        for station in profile.stations
    )
    return ShaftProfile(
        shaft_id=derived_id,
        frame_id=profile.frame_id,
        raw_length_m=profile.raw_length_m,
        cut_length_m=profile.cut_length_m,
        tip_trim_m=profile.tip_trim_m,
        butt_trim_m=profile.butt_trim_m,
        insertion_depth_m=profile.insertion_depth_m,
        stations=stations,
        provenance=ShaftProfileProvenance(
            source_name=f"derived from {source.source_name}",
            measurement_method=(
                f"{source.measurement_method}; deterministic parameter scaling"
            ),
            uncertainty_note=(
                f"{source.uncertainty_note} Scaling is a what-if transformation, "
                "not an additional measurement."
            ),
            source_uri=source.source_uri,
            data_license=source.data_license,
        ),
    )


__all__ = ["ShaftProfileScaling", "scale_shaft_profile"]
