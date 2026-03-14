"""Shared vessel head profile helpers used by preview and STEP generation."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt

from vessel_drafter.models.vessel_drafter import (
    DEFAULT_VESSEL_DRAFTER_LAYOUT,
    RadialBand,
    VesselDrafterLayout,
)

HEAD_SAMPLE_COUNT = 48


@dataclass(frozen=True)
class ProfilePoint:
    x_in: float
    z_in: float

    def mirrored_x(self) -> ProfilePoint:
        return ProfilePoint(-self.x_in, self.z_in)


@dataclass(frozen=True)
class ShellBandProfile:
    band: RadialBand
    inner_offset_in: float
    outer_offset_in: float


def build_shell_band_profiles(
    layout: VesselDrafterLayout = DEFAULT_VESSEL_DRAFTER_LAYOUT,
) -> tuple[ShellBandProfile, ...]:
    profiles: list[ShellBandProfile] = []
    running_offset_in = 0.0
    for band in layout.shell_bands:
        next_offset_in = running_offset_in + (
            band.outer_radius_in - band.inner_radius_in
        )
        profiles.append(
            ShellBandProfile(
                band=band,
                inner_offset_in=running_offset_in,
                outer_offset_in=next_offset_in,
            )
        )
        running_offset_in = next_offset_in
    return tuple(profiles)


def build_top_head_curve(
    layout: VesselDrafterLayout = DEFAULT_VESSEL_DRAFTER_LAYOUT,
    offset_in: float = 0.0,
    sample_count: int = HEAD_SAMPLE_COUNT,
) -> tuple[ProfilePoint, ...]:
    return tuple(
        _offset_ellipse_point(
            radius_in=layout.inner_radius_in,
            depth_in=layout.head_depth_in,
            offset_in=offset_in,
            springline_z_in=layout.straight_shell_height_in,
            theta_radians=(pi * 0.5) * (step / sample_count),
            top=True,
        )
        for step in range(sample_count + 1)
    )


def build_bottom_head_curve(
    layout: VesselDrafterLayout = DEFAULT_VESSEL_DRAFTER_LAYOUT,
    offset_in: float = 0.0,
    sample_count: int = HEAD_SAMPLE_COUNT,
) -> tuple[ProfilePoint, ...]:
    return tuple(
        _offset_ellipse_point(
            radius_in=layout.inner_radius_in,
            depth_in=layout.head_depth_in,
            offset_in=offset_in,
            springline_z_in=0.0,
            theta_radians=(pi * 0.5) * (1.0 - (step / sample_count)),
            top=False,
        )
        for step in range(sample_count + 1)
    )


def build_shell_boundary_half(
    layout: VesselDrafterLayout,
    offset_in: float,
) -> tuple[ProfilePoint, ...]:
    radius_in = layout.inner_radius_in + offset_in
    return (
        *build_bottom_head_curve(layout, offset_in),
        ProfilePoint(radius_in, layout.straight_shell_height_in),
        *build_top_head_curve(layout, offset_in)[1:],
    )


def build_cavity_boundary_half(
    layout: VesselDrafterLayout = DEFAULT_VESSEL_DRAFTER_LAYOUT,
) -> tuple[ProfilePoint, ...]:
    return (
        ProfilePoint(0.0, 0.0),
        ProfilePoint(layout.inner_radius_in, 0.0),
        ProfilePoint(layout.inner_radius_in, layout.straight_shell_height_in),
        *build_top_head_curve(layout, 0.0)[1:],
    )


def build_glass_boundary_half(
    layout: VesselDrafterLayout = DEFAULT_VESSEL_DRAFTER_LAYOUT,
) -> tuple[ProfilePoint, ...]:
    return (
        ProfilePoint(0.0, 0.0),
        ProfilePoint(layout.inner_radius_in, 0.0),
        ProfilePoint(layout.inner_radius_in, layout.glass_depth_in),
        ProfilePoint(0.0, layout.glass_depth_in),
    )


def build_plenum_boundary_half(
    layout: VesselDrafterLayout = DEFAULT_VESSEL_DRAFTER_LAYOUT,
) -> tuple[ProfilePoint, ...]:
    return (
        ProfilePoint(0.0, layout.glass_depth_in),
        ProfilePoint(layout.inner_radius_in, layout.glass_depth_in),
        ProfilePoint(layout.inner_radius_in, layout.straight_shell_height_in),
        ProfilePoint(0.0, layout.straight_shell_height_in),
    )


def build_full_boundary_loop(
    half_boundary: tuple[ProfilePoint, ...],
) -> tuple[ProfilePoint, ...]:
    return half_boundary + tuple(point.mirrored_x() for point in half_boundary[-2::-1])


def build_band_boundary_loops(
    layout: VesselDrafterLayout = DEFAULT_VESSEL_DRAFTER_LAYOUT,
) -> tuple[
    tuple[ShellBandProfile, tuple[ProfilePoint, ...], tuple[ProfilePoint, ...]], ...
]:
    loops: list[
        tuple[ShellBandProfile, tuple[ProfilePoint, ...], tuple[ProfilePoint, ...]]
    ] = []
    cavity_loop = build_full_boundary_loop(build_cavity_boundary_half(layout))
    for profile in build_shell_band_profiles(layout):
        inner_loop = (
            cavity_loop
            if profile.inner_offset_in == 0.0
            else build_full_boundary_loop(
                build_shell_boundary_half(layout, profile.inner_offset_in)
            )
        )
        outer_loop = build_full_boundary_loop(
            build_shell_boundary_half(layout, profile.outer_offset_in)
        )
        loops.append((profile, outer_loop, inner_loop))
    return tuple(loops)


def _offset_ellipse_point(
    radius_in: float,
    depth_in: float,
    offset_in: float,
    springline_z_in: float,
    theta_radians: float,
    top: bool,
) -> ProfilePoint:
    assert radius_in is not None, "radius_in must be provided"
    base_x_in = radius_in * cos(theta_radians)
    axial_multiplier = 1.0 if top else -1.0
    base_z_in = springline_z_in + (axial_multiplier * depth_in * sin(theta_radians))
    gradient_x = base_x_in / (radius_in * radius_in)
    gradient_z = (base_z_in - springline_z_in) / (depth_in * depth_in)
    magnitude = sqrt((gradient_x * gradient_x) + (gradient_z * gradient_z))
    normal_x = gradient_x / magnitude
    normal_z = gradient_z / magnitude
    return ProfilePoint(
        x_in=base_x_in + (offset_in * normal_x),
        z_in=base_z_in + (offset_in * normal_z),
    )
