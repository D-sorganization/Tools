"""Cylindrical bath drafting defaults for radial electrode layouts."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin
from typing import Any

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class CylindricalBathDefaults:
    inner_diameter_in: float = 50.0
    depth_in: float = 14.0
    refractory_thickness_in: float = 6.0


@dataclass(frozen=True)
class CylindricalElectrodeDefaults:
    count: int = 3
    diameter_in: float = 2.0
    insertion_into_inner_circle_in: float = 14.0
    extension_past_inner_circle_in: float = 36.0
    center_height_fraction: float = 0.5


@dataclass(frozen=True)
class RadialElectrodePlacement:
    index: int
    angle_degrees: float
    center_x_mm: float
    center_y_mm: float
    inner_tip_x_mm: float
    inner_tip_y_mm: float
    outer_tip_x_mm: float
    outer_tip_y_mm: float


@dataclass(frozen=True)
class CylindricalBathLayout:
    bath: CylindricalBathDefaults
    electrodes: CylindricalElectrodeDefaults
    placements: tuple[RadialElectrodePlacement, ...]

    @property
    def inner_radius_mm(self) -> float:
        return (self.bath.inner_diameter_in * 0.5) * MM_PER_INCH

    @property
    def outer_radius_mm(self) -> float:
        thickness_mm = self.bath.refractory_thickness_in * MM_PER_INCH
        return self.inner_radius_mm + thickness_mm

    @property
    def depth_mm(self) -> float:
        return self.bath.depth_in * MM_PER_INCH

    @property
    def electrode_radius_mm(self) -> float:
        return (self.electrodes.diameter_in * 0.5) * MM_PER_INCH

    @property
    def electrode_length_mm(self) -> float:
        return (
            self.electrodes.insertion_into_inner_circle_in
            + self.electrodes.extension_past_inner_circle_in
        ) * MM_PER_INCH

    @property
    def electrode_center_z_mm(self) -> float:
        return self.depth_mm * self.electrodes.center_height_fraction

    @property
    def electrode_inner_tip_radius_mm(self) -> float:
        return self.inner_radius_mm - (
            self.electrodes.insertion_into_inner_circle_in * MM_PER_INCH
        )

    @property
    def electrode_outer_tip_radius_mm(self) -> float:
        return self.inner_radius_mm + (
            self.electrodes.extension_past_inner_circle_in * MM_PER_INCH
        )

    @property
    def electrode_center_radius_mm(self) -> float:
        return (
            self.electrode_inner_tip_radius_mm + self.electrode_outer_tip_radius_mm
        ) * 0.5

    def to_manifest(self) -> dict[str, Any]:
        return {
            "project": "cylindrical_bath_layout",
            "units": {"source": "inches", "cad": "millimeters"},
            "bath": {
                "shape": "cylindrical",
                "inner_diameter_in": self.bath.inner_diameter_in,
                "depth_in": self.bath.depth_in,
                "refractory_thickness_in": self.bath.refractory_thickness_in,
                "outer_diameter_in": self.bath.inner_diameter_in
                + (2.0 * self.bath.refractory_thickness_in),
            },
            "electrodes": {
                "count": self.electrodes.count,
                "diameter_in": self.electrodes.diameter_in,
                "insertion_into_inner_circle_in": (
                    self.electrodes.insertion_into_inner_circle_in
                ),
                "extension_past_inner_circle_in": (
                    self.electrodes.extension_past_inner_circle_in
                ),
                "modeled_length_in": (
                    self.electrodes.insertion_into_inner_circle_in
                    + self.electrodes.extension_past_inner_circle_in
                ),
                "center_height_fraction": self.electrodes.center_height_fraction,
            },
            "drafting_assumptions": {
                "axis_convention": "Z up",
                "electrodes_centered_vertically": True,
                "electrode_tip_reference": (
                    "Modeled from the inner-wall penetration tip to the external end."
                ),
                "conflicting_length_note_ignored": (
                    "Used explicit 14 in insertion and 36 in extension values."
                ),
            },
            "placements": [
                {
                    "index": item.index,
                    "angle_degrees": item.angle_degrees,
                    "center_mm": [item.center_x_mm, item.center_y_mm, 0.0],
                    "inner_tip_mm": [item.inner_tip_x_mm, item.inner_tip_y_mm, 0.0],
                    "outer_tip_mm": [item.outer_tip_x_mm, item.outer_tip_y_mm, 0.0],
                }
                for item in self.placements
            ],
        }


def _build_default_placements(
    layout: CylindricalBathLayout,
) -> tuple[RadialElectrodePlacement, ...]:
    placements: list[RadialElectrodePlacement] = []
    for index in range(layout.electrodes.count):
        angle_degrees = index * (360.0 / layout.electrodes.count)
        angle_radians = radians(angle_degrees)
        direction_x = cos(angle_radians)
        direction_y = sin(angle_radians)
        placements.append(
            RadialElectrodePlacement(
                index=index + 1,
                angle_degrees=angle_degrees,
                center_x_mm=layout.electrode_center_radius_mm * direction_x,
                center_y_mm=layout.electrode_center_radius_mm * direction_y,
                inner_tip_x_mm=layout.electrode_inner_tip_radius_mm * direction_x,
                inner_tip_y_mm=layout.electrode_inner_tip_radius_mm * direction_y,
                outer_tip_x_mm=layout.electrode_outer_tip_radius_mm * direction_x,
                outer_tip_y_mm=layout.electrode_outer_tip_radius_mm * direction_y,
            )
        )
    return tuple(placements)


def build_default_cylindrical_bath_layout() -> CylindricalBathLayout:
    bath = CylindricalBathDefaults()
    electrodes = CylindricalElectrodeDefaults()
    layout = CylindricalBathLayout(
        bath=bath,
        electrodes=electrodes,
        placements=(),
    )
    return CylindricalBathLayout(
        bath=bath,
        electrodes=electrodes,
        placements=_build_default_placements(layout),
    )


DEFAULT_CYLINDRICAL_BATH_LAYOUT = build_default_cylindrical_bath_layout()
