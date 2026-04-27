"""Preview data builders for the vessel drafter GUI."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

from vessel_drafter.models.vessel_drafter import (
    DEFAULT_VESSEL_DRAFTER_LAYOUT,
    RadialBand,
    VesselDrafterLayout,
    VesselElectrodePlacement,
)
from vessel_drafter.projects.vessel_drafter_profiles import (
    ProfilePoint,
    build_band_boundary_loops,
    build_full_boundary_loop,
    build_glass_boundary_half,
    build_plenum_boundary_half,
)


@dataclass(frozen=True)
class CrossSectionBandPolygon:
    label: str
    color_hex: str
    outer_loop: tuple[ProfilePoint, ...]
    inner_loop: tuple[ProfilePoint, ...] | None = None


@dataclass(frozen=True)
class ProjectedSidePort:
    clock_angle_degrees: float
    centerline_height_in: float
    diameter_in: float


@dataclass(frozen=True)
class CrossSectionRadialFeature:
    label: str
    color_hex: str
    start_x_in: float
    end_x_in: float
    centerline_height_in: float
    diameter_in: float


@dataclass(frozen=True)
class CrossSectionPreview:
    inner_radius_in: float
    outer_radius_in: float
    straight_shell_height_in: float
    inner_head_depth_in: float
    outer_head_depth_in: float
    glass_height_in: float
    plenum_height_in: float
    electrode_diameter_in: float
    bands: tuple[RadialBand, ...]
    band_polygons: tuple[CrossSectionBandPolygon, ...]
    plenum_loop: tuple[ProfilePoint, ...]
    projected_side_ports: tuple[ProjectedSidePort, ...]
    axial_electrodes: tuple[CrossSectionRadialFeature, ...]


@dataclass(frozen=True)
class PlanRadialFeature:
    label: str
    color_hex: str
    angle_degrees: float
    angle_radians: float
    inner_tip_radius_in: float
    outer_tip_radius_in: float
    diameter_in: float


@dataclass(frozen=True)
class PlanCircularFeature:
    label: str
    color_hex: str
    center_x_in: float
    center_y_in: float
    diameter_in: float


@dataclass(frozen=True)
class PlanPreview:
    outer_radius_in: float
    bands: tuple[RadialBand, ...]
    electrodes: tuple[VesselElectrodePlacement, ...]
    side_ports: tuple[PlanRadialFeature, ...]
    lid_ports: tuple[PlanCircularFeature, ...]


def build_cross_section_preview(
    layout: VesselDrafterLayout = DEFAULT_VESSEL_DRAFTER_LAYOUT,
) -> CrossSectionPreview:
    band_polygons = [
        CrossSectionBandPolygon(
            label="glass_bath",
            color_hex=layout.layers[0].color_hex,
            outer_loop=build_full_boundary_loop(build_glass_boundary_half(layout)),
        )
    ]
    band_polygons.extend(
        [
            CrossSectionBandPolygon(
                label=profile.band.label,
                color_hex=profile.band.color_hex,
                outer_loop=outer_loop,
                inner_loop=inner_loop,
            )
            for (profile, outer_loop, inner_loop) in build_band_boundary_loops(layout)
        ]
    )

    return CrossSectionPreview(
        inner_radius_in=layout.inner_radius_in,
        outer_radius_in=layout.outer_radius_in,
        straight_shell_height_in=layout.straight_shell_height_in,
        inner_head_depth_in=layout.head_depth_in,
        outer_head_depth_in=layout.outer_head_depth_in,
        glass_height_in=layout.glass_depth_in,
        plenum_height_in=layout.plenum_height_in,
        electrode_diameter_in=layout.electrode_diameter_in,
        bands=layout.radial_bands,
        band_polygons=tuple(band_polygons),
        plenum_loop=build_full_boundary_loop(build_plenum_boundary_half(layout)),
        projected_side_ports=tuple(
            ProjectedSidePort(
                clock_angle_degrees=port.normalized_clock_angle_degrees,
                centerline_height_in=port.centerline_height_in(layout),
                diameter_in=port.diameter_in,
            )
            for port in layout.side_ports
        ),
        axial_electrodes=tuple(_build_axial_electrodes(layout)),
    )


def build_plan_preview(
    layout: VesselDrafterLayout = DEFAULT_VESSEL_DRAFTER_LAYOUT,
) -> PlanPreview:
    return PlanPreview(
        outer_radius_in=layout.outer_radius_in,
        bands=layout.radial_bands,
        electrodes=layout.electrode_placements,
        side_ports=tuple(
            PlanRadialFeature(
                label=f"side_port_{index + 1}",
                color_hex="#6AAED6",
                angle_degrees=port.normalized_clock_angle_degrees,
                angle_radians=port.normalized_clock_angle_radians,
                inner_tip_radius_in=layout.inner_radius_in,
                outer_tip_radius_in=layout.outer_radius_in,
                diameter_in=port.diameter_in,
            )
            for index, port in enumerate(layout.side_ports)
        ),
        lid_ports=tuple(
            PlanCircularFeature(
                label=f"lid_port_{index + 1}",
                color_hex="#CFE8FF",
                center_x_in=(
                    cos(port.normalized_clock_angle_radians)
                    * port.radial_distance_from_center_in
                ),
                center_y_in=(
                    sin(port.normalized_clock_angle_radians)
                    * port.radial_distance_from_center_in
                ),
                diameter_in=port.diameter_in,
            )
            for index, port in enumerate(layout.lid_ports)
        ),
    )


def _build_axial_electrodes(
    layout: VesselDrafterLayout,
) -> tuple[CrossSectionRadialFeature, ...]:
    features: list[CrossSectionRadialFeature] = []
    for placement in layout.electrode_placements:
        projected_y = sin(placement.angle_radians)
        if abs(projected_y) > 1e-9:
            continue
        direction = 1.0 if cos(placement.angle_radians) >= 0.0 else -1.0
        features.append(
            CrossSectionRadialFeature(
                label=f"electrode_{placement.index}",
                color_hex="#2B2B2B",
                start_x_in=placement.inner_tip_radius_in * direction,
                end_x_in=placement.outer_tip_radius_in * direction,
                centerline_height_in=layout.electrode_centerline_height_in,
                diameter_in=placement.diameter_in,
            )
        )
    return tuple(features)
