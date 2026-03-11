"""Validated defaults and geometry helpers for the vessel drafter tool."""

from __future__ import annotations

from dataclasses import dataclass
from math import radians
from typing import Any

from vessel_drafter.contracts import (
    require_fraction,
    require_integer_at_least,
    require_less_or_equal,
    require_nonnegative,
    require_positive,
)
from vessel_drafter.models.vessel_materials import (
    DEFAULT_VESSEL_MATERIALS_BY_NAME,
    MaterialProperties,
)

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class MaterialLayer:
    properties: MaterialProperties
    thickness_in: float

    @property
    def name(self) -> str:
        return self.properties.name

    @property
    def display_name(self) -> str:
        return self.properties.display_name

    @property
    def color_hex(self) -> str:
        return self.properties.color_hex

    @property
    def category(self) -> str:
        return self.properties.category

    @property
    def density_lb_per_ft3(self) -> float:
        return self.properties.density_lb_per_ft3

    @property
    def thermal_conductivity_w_per_mk(self) -> float:
        return self.properties.thermal_conductivity_w_per_mk

    @property
    def thermal_expansion_um_per_m_c(self) -> float:
        return self.properties.thermal_expansion_um_per_m_c

    @property
    def preview_alpha(self) -> float:
        return self.properties.preview_alpha


@dataclass(frozen=True)
class RadialBand:
    name: str
    color_hex: str
    inner_radius_in: float
    outer_radius_in: float

    @property
    def label(self) -> str:
        return self.name

    @property
    def inner_offset_in(self) -> float:
        return self.inner_radius_in


@dataclass(frozen=True)
class VesselElectrodePlacement:
    index: int
    angle_degrees: float
    angle_radians: float
    inner_tip_radius_in: float
    outer_tip_radius_in: float
    diameter_in: float


@dataclass(frozen=True)
class VesselSidePort:
    clock_angle_degrees: float
    diameter_in: float
    height_above_glass_surface_in: float

    def __post_init__(self) -> None:
        require_positive("diameter_in", self.diameter_in)
        require_nonnegative(
            "height_above_glass_surface_in",
            self.height_above_glass_surface_in,
        )

    @property
    def normalized_clock_angle_degrees(self) -> float:
        return self.clock_angle_degrees % 360.0

    @property
    def normalized_clock_angle_radians(self) -> float:
        return radians(self.normalized_clock_angle_degrees)

    @property
    def radius_in(self) -> float:
        return self.diameter_in * 0.5

    def centerline_height_in(self, layout: VesselDrafterLayout) -> float:
        return layout.glass_depth_in + self.height_above_glass_surface_in


@dataclass(frozen=True)
class VesselLidPort:
    clock_angle_degrees: float
    diameter_in: float
    radial_distance_from_center_in: float

    def __post_init__(self) -> None:
        require_positive("diameter_in", self.diameter_in)
        require_nonnegative(
            "radial_distance_from_center_in",
            self.radial_distance_from_center_in,
        )

    @property
    def normalized_clock_angle_degrees(self) -> float:
        return self.clock_angle_degrees % 360.0

    @property
    def normalized_clock_angle_radians(self) -> float:
        return radians(self.normalized_clock_angle_degrees)

    @property
    def radius_in(self) -> float:
        return self.diameter_in * 0.5


@dataclass(frozen=True)
class VesselDrafterLayout:
    inner_diameter_in: float = 50.0
    glass_depth_in: float = 14.0
    plenum_height_in: float = 14.0
    head_depth_in: float = 12.5
    hot_face_thickness_in: float = 6.0
    ifb_thickness_in: float = 4.5
    duraboard_thickness_in: float = 1.0
    steel_thickness_in: float = 0.5
    electrode_count: int = 3
    electrode_diameter_in: float = 2.0
    electrode_insertion_into_inner_circle_in: float = 14.0
    electrode_extension_past_inner_circle_in: float = 36.0
    electrode_centerline_height_fraction: float = 0.5
    side_ports: tuple[VesselSidePort, ...] = ()
    lid_ports: tuple[VesselLidPort, ...] = ()

    def __post_init__(self) -> None:
        require_positive("inner_diameter_in", self.inner_diameter_in)
        require_positive("glass_depth_in", self.glass_depth_in)
        require_positive("plenum_height_in", self.plenum_height_in)
        require_positive("head_depth_in", self.head_depth_in)
        require_positive("hot_face_thickness_in", self.hot_face_thickness_in)
        require_positive("ifb_thickness_in", self.ifb_thickness_in)
        require_positive("duraboard_thickness_in", self.duraboard_thickness_in)
        require_positive("steel_thickness_in", self.steel_thickness_in)
        require_integer_at_least("electrode_count", self.electrode_count, 1)
        require_positive("electrode_diameter_in", self.electrode_diameter_in)
        require_positive(
            "electrode_insertion_into_inner_circle_in",
            self.electrode_insertion_into_inner_circle_in,
        )
        require_positive(
            "electrode_extension_past_inner_circle_in",
            self.electrode_extension_past_inner_circle_in,
        )
        require_fraction(
            "electrode_centerline_height_fraction",
            self.electrode_centerline_height_fraction,
        )
        require_less_or_equal(
            "electrode_insertion_into_inner_circle_in",
            self.electrode_insertion_into_inner_circle_in,
            self.inner_radius_in,
        )
        self._validate_side_ports()
        self._validate_lid_ports()

    @property
    def inner_radius_in(self) -> float:
        return self.inner_diameter_in * 0.5

    @property
    def straight_shell_height_in(self) -> float:
        return self.glass_depth_in + self.plenum_height_in

    @property
    def total_shell_thickness_in(self) -> float:
        return sum(layer.thickness_in for layer in self.layers[1:])

    @property
    def outer_radius_in(self) -> float:
        return self.shell_bands[-1].outer_radius_in

    @property
    def outer_diameter_in(self) -> float:
        return self.outer_radius_in * 2.0

    @property
    def full_height_in(self) -> float:
        return (
            self.outer_head_depth_in
            + self.straight_shell_height_in
            + self.outer_head_depth_in
        )

    @property
    def outer_head_depth_in(self) -> float:
        return self.head_depth_for_offset_in(self.total_shell_thickness_in)

    @property
    def glass_height_mm(self) -> float:
        return self.glass_depth_in * MM_PER_INCH

    @property
    def straight_shell_height_mm(self) -> float:
        return self.straight_shell_height_in * MM_PER_INCH

    @property
    def electrode_centerline_height_in(self) -> float:
        return self.glass_depth_in * self.electrode_centerline_height_fraction

    @property
    def electrode_centerline_height_mm(self) -> float:
        return self.electrode_centerline_height_in * MM_PER_INCH

    @property
    def electrode_radius_in(self) -> float:
        return self.electrode_diameter_in * 0.5

    @property
    def electrode_radius_mm(self) -> float:
        return self.electrode_radius_in * MM_PER_INCH

    @property
    def electrode_length_in(self) -> float:
        return (
            self.electrode_insertion_into_inner_circle_in
            + self.electrode_extension_past_inner_circle_in
        )

    @property
    def electrode_length_mm(self) -> float:
        return self.electrode_length_in * MM_PER_INCH

    @property
    def electrode_inner_tip_radius_in(self) -> float:
        return self.inner_radius_in - self.electrode_insertion_into_inner_circle_in

    @property
    def electrode_outer_tip_radius_in(self) -> float:
        return self.inner_radius_in + self.electrode_extension_past_inner_circle_in

    @property
    def layers(self) -> tuple[MaterialLayer, ...]:
        materials = self.material_properties_by_name
        return (
            MaterialLayer(materials["glass_bath"], self.inner_radius_in),
            MaterialLayer(materials["hot_face_refractory"], self.hot_face_thickness_in),
            MaterialLayer(materials["ifb"], self.ifb_thickness_in),
            MaterialLayer(materials["duraboard"], self.duraboard_thickness_in),
            MaterialLayer(materials["steel_shell"], self.steel_thickness_in),
        )

    @property
    def material_properties_by_name(self) -> dict[str, MaterialProperties]:
        return dict(DEFAULT_VESSEL_MATERIALS_BY_NAME)

    @property
    def shell_bands(self) -> tuple[RadialBand, ...]:
        running_radius = self.inner_radius_in
        bands: list[RadialBand] = []
        for layer in self.layers[1:]:
            outer_radius = running_radius + layer.thickness_in
            bands.append(
                RadialBand(
                    name=layer.name,
                    color_hex=layer.color_hex,
                    inner_radius_in=running_radius,
                    outer_radius_in=outer_radius,
                )
            )
            running_radius = outer_radius
        return tuple(bands)

    @property
    def radial_bands(self) -> tuple[RadialBand, ...]:
        glass = self.layers[0]
        return (
            RadialBand(
                name=glass.name,
                color_hex=glass.color_hex,
                inner_radius_in=0.0,
                outer_radius_in=self.inner_radius_in,
            ),
            *self.shell_bands,
        )

    @property
    def electrode_placements(self) -> tuple[VesselElectrodePlacement, ...]:
        placements: list[VesselElectrodePlacement] = []
        for index in range(self.electrode_count):
            angle_degrees = index * (360.0 / self.electrode_count)
            placements.append(
                VesselElectrodePlacement(
                    index=index + 1,
                    angle_degrees=angle_degrees,
                    angle_radians=radians(angle_degrees),
                    inner_tip_radius_in=self.electrode_inner_tip_radius_in,
                    outer_tip_radius_in=self.electrode_outer_tip_radius_in,
                    diameter_in=self.electrode_diameter_in,
                )
            )
        return tuple(placements)

    def head_depth_for_offset_in(self, offset_in: float) -> float:
        require_nonnegative("offset_in", offset_in)
        return self.head_depth_in + offset_in

    def _validate_side_ports(self) -> None:
        for index, port in enumerate(self.side_ports):
            require_less_or_equal(
                f"side_ports[{index}].height_above_glass_surface_in",
                port.height_above_glass_surface_in + port.radius_in,
                self.plenum_height_in,
            )
            require_less_or_equal(
                f"side_ports[{index}].diameter_in",
                port.radius_in,
                port.height_above_glass_surface_in,
            )

    def _validate_lid_ports(self) -> None:
        for index, port in enumerate(self.lid_ports):
            require_less_or_equal(
                f"lid_ports[{index}].radial_distance_from_center_in",
                port.radial_distance_from_center_in + port.radius_in,
                self.outer_radius_in,
            )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "project": "vessel_drafter_default",
            "units": {"source": "inches", "cad": "millimeters"},
            "vessel": {
                "inner_diameter_in": self.inner_diameter_in,
                "glass_depth_in": self.glass_depth_in,
                "plenum_height_in": self.plenum_height_in,
                "head_depth_in": self.head_depth_in,
                "outer_diameter_in": self.outer_diameter_in,
                "full_height_in": self.full_height_in,
                "outer_head_depth_in": self.outer_head_depth_in,
            },
            "materials": {
                layer.name: {
                    "thickness_in": layer.thickness_in,
                    "display_name": layer.display_name,
                    "color_hex": layer.color_hex,
                    "density_lb_per_ft3": layer.density_lb_per_ft3,
                    "thermal_conductivity_w_per_mk": (
                        layer.thermal_conductivity_w_per_mk
                    ),
                    "thermal_expansion_um_per_m_c": (
                        layer.thermal_expansion_um_per_m_c
                    ),
                }
                for layer in self.layers[1:]
            },
            "glass_bath": {
                "display_name": self.layers[0].display_name,
                "color_hex": self.layers[0].color_hex,
                "height_in": self.glass_depth_in,
                "density_lb_per_ft3": self.layers[0].density_lb_per_ft3,
                "thermal_conductivity_w_per_mk": (
                    self.layers[0].thermal_conductivity_w_per_mk
                ),
                "thermal_expansion_um_per_m_c": (
                    self.layers[0].thermal_expansion_um_per_m_c
                ),
            },
            "electrodes": {
                "count": self.electrode_count,
                "diameter_in": self.electrode_diameter_in,
                "insertion_into_inner_circle_in": (
                    self.electrode_insertion_into_inner_circle_in
                ),
                "extension_past_inner_circle_in": (
                    self.electrode_extension_past_inner_circle_in
                ),
                "modeled_length_in": self.electrode_length_in,
                "centerline_height_in": self.electrode_centerline_height_in,
            },
            "ports": {
                "side": [
                    {
                        "clock_angle_degrees": port.normalized_clock_angle_degrees,
                        "diameter_in": port.diameter_in,
                        "height_above_glass_surface_in": (
                            port.height_above_glass_surface_in
                        ),
                        "centerline_height_in": port.centerline_height_in(self),
                    }
                    for port in self.side_ports
                ],
                "lid": [
                    {
                        "clock_angle_degrees": port.normalized_clock_angle_degrees,
                        "diameter_in": port.diameter_in,
                        "radial_distance_from_center_in": (
                            port.radial_distance_from_center_in
                        ),
                    }
                    for port in self.lid_ports
                ],
            },
            "drafting_assumptions": {
                "axis_convention": "Z up",
                "plenum_only_internal_void": True,
                "dished_heads": (
                    "Offset elliptical head profiles with "
                    "constant-thickness shell layers"
                ),
            },
        }


DEFAULT_VESSEL_DRAFTER_LAYOUT = VesselDrafterLayout()
