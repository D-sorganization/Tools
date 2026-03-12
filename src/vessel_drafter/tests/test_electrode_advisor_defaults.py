from math import sqrt

import pytest
from vessel_drafter.models.electrode_advisor import (
    DEFAULT_ELECTRODE_ADVISOR_LAYOUT,
)
from vessel_drafter.projects.electrode_advisor_default_layout import (
    build_default_layout_shape,
)


def test_defaults_match_current_ui_layout() -> None:
    layout = DEFAULT_ELECTRODE_ADVISOR_LAYOUT

    assert layout.bath.shape == "rectangular"
    assert layout.bath.width_m == pytest.approx(3.0)
    assert layout.bath.depth_m == pytest.approx(2.0)
    assert layout.bath.height_m == pytest.approx(2.5)
    assert layout.bath.glass_level_m == pytest.approx(1.5)
    assert layout.electrodes.count == 3
    assert layout.electrodes.current_length_mm == pytest.approx(1500.0)
    assert layout.electrodes.worn_length_mm == pytest.approx(150.0)
    assert layout.electrodes.operating_current_a == pytest.approx(2500.0)


def test_default_positions_are_evenly_spaced() -> None:
    layout = DEFAULT_ELECTRODE_ADVISOR_LAYOUT
    placements = layout.placements
    assert len(placements) == 3

    expected_radius = 720.0
    for placement in placements:
        radial_distance = sqrt((placement.cad_x_mm**2) + (placement.cad_y_mm**2))
        assert radial_distance == pytest.approx(expected_radius, abs=1e-6)
        assert placement.cad_z_mm == pytest.approx(2600.0)
        assert placement.effective_length_mm == pytest.approx(1350.0)


def test_shape_bounding_box_matches_layout_envelope() -> None:
    shape = build_default_layout_shape()
    box = shape.bounding_box()
    size = box.size

    assert size.X == pytest.approx(3000.0)
    assert size.Y == pytest.approx(2000.0)
    assert size.Z == pytest.approx(2700.0)
