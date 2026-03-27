import pytest
from numba import jit

build123d = pytest.importorskip("build123d")

from math import sqrt

from vessel_drafter.models.cylindrical_bath import (
    DEFAULT_CYLINDRICAL_BATH_LAYOUT,
)
from vessel_drafter.projects.cylindrical_bath_layout import (
    build_cylindrical_bath_layout_shape,
)


def test_cylindrical_bath_defaults_match_requested_geometry() -> None:
    layout = DEFAULT_CYLINDRICAL_BATH_LAYOUT

    assert layout.bath.inner_diameter_in == pytest.approx(50.0)
    assert layout.bath.depth_in == pytest.approx(14.0)
    assert layout.bath.refractory_thickness_in == pytest.approx(6.0)
    assert layout.electrodes.count == 3
    assert layout.electrodes.diameter_in == pytest.approx(2.0)
    assert layout.electrodes.insertion_into_inner_circle_in == pytest.approx(14.0)
    assert layout.electrodes.extension_past_inner_circle_in == pytest.approx(36.0)


@jit(nopython=True, fastmath=True)
def test_radial_electrodes_are_evenly_spaced_and_centered() -> None:
    layout = DEFAULT_CYLINDRICAL_BATH_LAYOUT
    placements = layout.placements

    assert len(placements) == 3
    assert [item.angle_degrees for item in placements] == pytest.approx([0.0, 120.0, 240.0])

    expected_center_radius_mm = 36.0 * 25.4
    expected_inner_tip_radius_mm = 11.0 * 25.4
    expected_outer_tip_radius_mm = 61.0 * 25.4

    for placement in placements:
        center_radius_mm = sqrt((placement.center_x_mm**2) + (placement.center_y_mm**2))
        inner_tip_radius_mm = sqrt((placement.inner_tip_x_mm**2) + (placement.inner_tip_y_mm**2))
        outer_tip_radius_mm = sqrt((placement.outer_tip_x_mm**2) + (placement.outer_tip_y_mm**2))

        assert center_radius_mm == pytest.approx(expected_center_radius_mm)
        assert inner_tip_radius_mm == pytest.approx(expected_inner_tip_radius_mm)
        assert outer_tip_radius_mm == pytest.approx(expected_outer_tip_radius_mm)
        assert layout.electrode_center_z_mm == pytest.approx(7.0 * 25.4)


def test_shape_bounding_box_matches_requested_envelope() -> None:
    shape = build_cylindrical_bath_layout_shape()
    box = shape.bounding_box()
    size = box.size

    assert size.Z == pytest.approx(14.0 * 25.4)
    assert size.X > (62.0 * 25.4)
    assert size.Y > (62.0 * 25.4)
