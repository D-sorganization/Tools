"""STEP-ready geometry for the default electrode advisor layout."""

from __future__ import annotations

from build123d import Align, Box, BuildPart, Compound, Cylinder, Locations, Mode, Solid

from vessel_drafter.models.electrode_advisor import (
    DEFAULT_ELECTRODE_ADVISOR_LAYOUT,
    ElectrodeAdvisorLayout,
    ElectrodePlacement,
)


def _build_bath_shell(layout: ElectrodeAdvisorLayout) -> Solid:
    shell = layout.drafting.bath_shell_thickness_mm
    with BuildPart() as bath_shell:
        Box(
            layout.bath_width_mm,
            layout.bath_depth_mm,
            layout.bath_height_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        with Locations((0, 0, shell)):
            Box(
                layout.bath_width_mm - (2.0 * shell),
                layout.bath_depth_mm - (2.0 * shell),
                layout.bath_height_mm - shell,
                mode=Mode.SUBTRACT,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
            )
    bath_shell.part.label = "bath_shell"
    return bath_shell.part


def _build_glass_volume(layout: ElectrodeAdvisorLayout) -> Solid:
    shell = layout.drafting.bath_shell_thickness_mm
    clearance = layout.drafting.glass_clearance_mm
    with BuildPart() as glass_volume:
        Box(
            layout.bath_width_mm - (2.0 * (shell + clearance)),
            layout.bath_depth_mm - (2.0 * (shell + clearance)),
            layout.glass_level_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
    glass_volume.part.label = "glass_volume"
    return glass_volume.part


def _build_electrode_assembly(
    placement: ElectrodePlacement,
    layout: ElectrodeAdvisorLayout,
) -> Solid:
    if not (placement is not None):
        raise ValueError("placement must be provided")
    holder_height = layout.drafting.electrode_holder_height_mm
    holder_radius = (
        placement.diameter_mm * 0.5 * layout.drafting.electrode_holder_radius_factor
    )
    body_radius = placement.diameter_mm * 0.5
    tip_band_height = layout.drafting.tip_band_height_mm

    body_center_z = placement.cad_z_mm - (placement.effective_length_mm * 0.5)
    holder_center_z = placement.cad_z_mm + (holder_height * 0.5)
    tip_band_center_z = (
        placement.cad_z_mm - placement.effective_length_mm + (tip_band_height * 0.5)
    )

    with BuildPart() as electrode_assembly:
        with Locations((placement.cad_x_mm, placement.cad_y_mm, holder_center_z)):
            Cylinder(radius=holder_radius, height=holder_height)
        with Locations((placement.cad_x_mm, placement.cad_y_mm, body_center_z)):
            Cylinder(radius=body_radius, height=placement.effective_length_mm)
        with Locations((placement.cad_x_mm, placement.cad_y_mm, tip_band_center_z)):
            Cylinder(radius=body_radius * 1.01, height=tip_band_height)

    electrode_assembly.part.label = f"electrode_{placement.index}"
    return electrode_assembly.part


def build_default_layout_shape(
    layout: ElectrodeAdvisorLayout = DEFAULT_ELECTRODE_ADVISOR_LAYOUT,
) -> Compound:
    children = [_build_bath_shell(layout), _build_glass_volume(layout)]
    children.extend(
        _build_electrode_assembly(item, layout) for item in layout.placements
    )
    return Compound(label="electrode_advisor_default_layout", children=children)
