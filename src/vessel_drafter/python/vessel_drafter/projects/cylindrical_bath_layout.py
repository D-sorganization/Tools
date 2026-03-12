"""STEP-ready geometry for a cylindrical bath with radial electrodes."""

from __future__ import annotations

from build123d import Align, BuildPart, Compound, Cylinder, Mode, Plane, Solid

from vessel_drafter.models.cylindrical_bath import (
    DEFAULT_CYLINDRICAL_BATH_LAYOUT,
    CylindricalBathLayout,
    RadialElectrodePlacement,
)


def _build_inner_bath_volume(layout: CylindricalBathLayout) -> Solid:
    bath_volume = Cylinder(
        radius=layout.inner_radius_mm,
        height=layout.depth_mm,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    bath_volume.label = "inner_bath_volume"
    return bath_volume


def _build_refractory_annulus(layout: CylindricalBathLayout) -> Solid:
    with BuildPart() as refractory_annulus:
        Cylinder(
            radius=layout.outer_radius_mm,
            height=layout.depth_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        Cylinder(
            radius=layout.inner_radius_mm,
            height=layout.depth_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
            mode=Mode.SUBTRACT,
        )

    refractory_annulus.part.label = "refractory_annulus"
    return refractory_annulus.part


def _build_radial_electrode(
    placement: RadialElectrodePlacement,
    layout: CylindricalBathLayout,
) -> Solid:
    plane = Plane(
        origin=(
            placement.inner_tip_x_mm,
            placement.inner_tip_y_mm,
            layout.electrode_center_z_mm,
        ),
        x_dir=(0.0, 0.0, 1.0),
        z_dir=(
            placement.outer_tip_x_mm - placement.inner_tip_x_mm,
            placement.outer_tip_y_mm - placement.inner_tip_y_mm,
            0.0,
        ),
    )
    electrode = Solid.make_cylinder(
        radius=layout.electrode_radius_mm,
        height=layout.electrode_length_mm,
        plane=plane,
    )
    electrode.label = f"electrode_{placement.index}"
    return electrode


def build_cylindrical_bath_layout_shape(
    layout: CylindricalBathLayout = DEFAULT_CYLINDRICAL_BATH_LAYOUT,
) -> Compound:
    children = [_build_inner_bath_volume(layout), _build_refractory_annulus(layout)]
    children.extend(_build_radial_electrode(item, layout) for item in layout.placements)
    return Compound(label="cylindrical_bath_layout", children=children)
