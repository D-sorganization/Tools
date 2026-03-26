"""STEP-ready geometry for the vessel drafter tool."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Any

from build123d import (
    Axis,
    BuildLine,
    BuildSketch,
    Color,
    Compound,
    Plane,
    Polyline,
    Solid,
    make_face,
    revolve,
)

from vessel_drafter.models.vessel_drafter import (
    DEFAULT_VESSEL_DRAFTER_LAYOUT,
    MM_PER_INCH,
    VesselDrafterLayout,
    VesselElectrodePlacement,
    VesselLidPort,
    VesselSidePort,
)
from vessel_drafter.projects.vessel_drafter_profiles import (
    ProfilePoint,
    build_cavity_boundary_half,
    build_glass_boundary_half,
    build_shell_band_profiles,
    build_shell_boundary_half,
)


@dataclass(frozen=True)
class VesselComponent:
    label: str
    group_label: str
    display_name: str
    color_hex: str
    category: str
    density_lb_per_ft3: float
    thermal_conductivity_w_per_mk: float
    thermal_expansion_um_per_m_c: float
    preview_alpha: float
    shape: Any


def _color_from_hex(color_hex: str) -> Color:
    red = int(color_hex[1:3], 16) / 255.0
    green = int(color_hex[3:5], 16) / 255.0
    blue = int(color_hex[5:7], 16) / 255.0
    return Color(red, green, blue)


def _apply_style(shape: Any, label: str, color_hex: str) -> Any:
    if not (label is not None):
        raise ValueError("label must be provided")
    shape.label = label
    shape.color = _color_from_hex(color_hex)
    return shape


def _points_to_mm(points: tuple[ProfilePoint, ...]) -> list[tuple[float, float]]:
    return [(point.x_in * MM_PER_INCH, point.z_in * MM_PER_INCH) for point in points]


def _build_revolved_profile(
    half_profile: tuple[ProfilePoint, ...],
    label: str,
    color_hex: str,
) -> Any:
    if not (half_profile is not None):
        raise ValueError("half_profile must be provided")
    with BuildSketch(Plane.XZ) as sketch:
        with BuildLine():
            Polyline(_points_to_mm(half_profile), close=True)
        make_face()
    part = revolve(sketch.sketch.face(), axis=Axis.Z)
    return _apply_style(part, label, color_hex)


def _build_glass_bath(layout: VesselDrafterLayout) -> Any:
    return _build_revolved_profile(
        build_glass_boundary_half(layout),
        label="glass_bath",
        color_hex=layout.layers[0].color_hex,
    )


def _build_shell_band_shapes(layout: VesselDrafterLayout) -> list[Any]:
    shapes: list[Any] = []
    for profile in build_shell_band_profiles(layout):
        inner_half = (
            build_cavity_boundary_half(layout)
            if profile.inner_offset_in == 0.0
            else build_shell_boundary_half(layout, profile.inner_offset_in)
        )
        outer_half = build_shell_boundary_half(layout, profile.outer_offset_in)
        shapes.append(
            _build_revolved_profile(
                outer_half + tuple(reversed(inner_half)),
                label=profile.band.label,
                color_hex=profile.band.color_hex,
            )
        )
    return shapes


def _build_electrode(
    placement: VesselElectrodePlacement,
    layout: VesselDrafterLayout,
) -> Any:
    if not (placement is not None):
        raise ValueError("placement must be provided")
    direction_x = cos(placement.angle_radians)
    direction_y = sin(placement.angle_radians)
    inner_tip_x_mm = placement.inner_tip_radius_in * direction_x * MM_PER_INCH
    inner_tip_y_mm = placement.inner_tip_radius_in * direction_y * MM_PER_INCH
    outer_tip_x_mm = placement.outer_tip_radius_in * direction_x * MM_PER_INCH
    outer_tip_y_mm = placement.outer_tip_radius_in * direction_y * MM_PER_INCH
    plane = Plane(
        origin=(
            inner_tip_x_mm,
            inner_tip_y_mm,
            layout.electrode_centerline_height_mm,
        ),
        x_dir=(0.0, 0.0, 1.0),
        z_dir=(
            outer_tip_x_mm - inner_tip_x_mm,
            outer_tip_y_mm - inner_tip_y_mm,
            0.0,
        ),
    )
    electrode = Solid.make_cylinder(
        radius=layout.electrode_radius_mm,
        height=layout.electrode_length_mm,
        plane=plane,
    )
    return _apply_style(electrode, f"electrode_{placement.index}", "#2B2B2B")


def _build_side_port_cutter(
    port: VesselSidePort,
    layout: VesselDrafterLayout,
) -> Any:
    if not (port is not None):
        raise ValueError("port must be provided")
    direction_x = cos(port.normalized_clock_angle_radians)
    direction_y = sin(port.normalized_clock_angle_radians)
    start_radius_in = layout.inner_radius_in - 1.0
    cutter_length_in = layout.total_shell_thickness_in + 2.0
    plane = Plane(
        origin=(
            direction_x * start_radius_in * MM_PER_INCH,
            direction_y * start_radius_in * MM_PER_INCH,
            port.centerline_height_in(layout) * MM_PER_INCH,
        ),
        x_dir=(0.0, 0.0, 1.0),
        z_dir=(direction_x, direction_y, 0.0),
    )
    return Solid.make_cylinder(
        radius=port.radius_in * MM_PER_INCH,
        height=cutter_length_in * MM_PER_INCH,
        plane=plane,
    )


def _build_lid_port_cutter(
    port: VesselLidPort,
    layout: VesselDrafterLayout,
) -> Any:
    if not (port is not None):
        raise ValueError("port must be provided")
    direction_x = cos(port.normalized_clock_angle_radians)
    direction_y = sin(port.normalized_clock_angle_radians)
    start_z_in = layout.glass_depth_in
    cutter_height_in = layout.plenum_height_in + layout.outer_head_depth_in + 1.0
    plane = Plane(
        origin=(
            direction_x * port.radial_distance_from_center_in * MM_PER_INCH,
            direction_y * port.radial_distance_from_center_in * MM_PER_INCH,
            start_z_in * MM_PER_INCH,
        ),
        x_dir=(1.0, 0.0, 0.0),
        z_dir=(0.0, 0.0, 1.0),
    )
    return Solid.make_cylinder(
        radius=port.radius_in * MM_PER_INCH,
        height=cutter_height_in * MM_PER_INCH,
        plane=plane,
    )


def _port_cutters(layout: VesselDrafterLayout) -> tuple[Solid, ...]:
    side_cutters = tuple(_build_side_port_cutter(port, layout) for port in layout.side_ports)
    lid_cutters = tuple(_build_lid_port_cutter(port, layout) for port in layout.lid_ports)
    return side_cutters + lid_cutters


def _cut_ports(shape: Any, cutters: tuple[Solid, ...]) -> Any:
    if not cutters:
        return shape
    return shape.cut(*cutters)


def build_vessel_drafter_components(
    layout: VesselDrafterLayout = DEFAULT_VESSEL_DRAFTER_LAYOUT,
) -> tuple[VesselComponent, ...]:
    cutters = _port_cutters(layout)
    materials = layout.material_properties_by_name
    components: list[VesselComponent] = [
        VesselComponent(
            label="glass_bath",
            group_label="glass_bath",
            display_name=materials["glass_bath"].display_name,
            color_hex=materials["glass_bath"].color_hex,
            category=materials["glass_bath"].category,
            density_lb_per_ft3=materials["glass_bath"].density_lb_per_ft3,
            thermal_conductivity_w_per_mk=(materials["glass_bath"].thermal_conductivity_w_per_mk),
            thermal_expansion_um_per_m_c=(materials["glass_bath"].thermal_expansion_um_per_m_c),
            preview_alpha=materials["glass_bath"].preview_alpha,
            shape=_apply_style(
                _cut_ports(_build_glass_bath(layout), cutters),
                "glass_bath",
                materials["glass_bath"].color_hex,
            ),
        )
    ]
    for shape, band in zip(
        _build_shell_band_shapes(layout),
        layout.shell_bands,
        strict=True,
    ):
        properties = materials[band.label]
        components.append(
            VesselComponent(
                label=band.label,
                group_label=band.label,
                display_name=properties.display_name,
                color_hex=band.color_hex,
                category=properties.category,
                density_lb_per_ft3=properties.density_lb_per_ft3,
                thermal_conductivity_w_per_mk=properties.thermal_conductivity_w_per_mk,
                thermal_expansion_um_per_m_c=(properties.thermal_expansion_um_per_m_c),
                preview_alpha=properties.preview_alpha,
                shape=_apply_style(_cut_ports(shape, cutters), band.label, band.color_hex),
            )
        )
    electrode_properties = materials["electrodes"]
    for item in layout.electrode_placements:
        components.append(
            VesselComponent(
                label=f"electrode_{item.index}",
                group_label="electrodes",
                display_name=electrode_properties.display_name,
                color_hex=electrode_properties.color_hex,
                category=electrode_properties.category,
                density_lb_per_ft3=electrode_properties.density_lb_per_ft3,
                thermal_conductivity_w_per_mk=(electrode_properties.thermal_conductivity_w_per_mk),
                thermal_expansion_um_per_m_c=(electrode_properties.thermal_expansion_um_per_m_c),
                preview_alpha=electrode_properties.preview_alpha,
                shape=_build_electrode(item, layout),
            )
        )
    return tuple(components)


def build_vessel_drafter_shape(
    layout: VesselDrafterLayout = DEFAULT_VESSEL_DRAFTER_LAYOUT,
) -> Compound:
    return Compound(
        label="vessel_drafter_default",
        children=[component.shape for component in build_vessel_drafter_components(layout)],
    )
