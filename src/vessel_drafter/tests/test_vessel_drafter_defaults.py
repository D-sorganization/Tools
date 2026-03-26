import json
from pathlib import Path

import pytest
from vessel_drafter.models.vessel_drafter import (
    DEFAULT_VESSEL_DRAFTER_LAYOUT,
    VesselDrafterLayout,
    VesselLidPort,
    VesselSidePort,
)


def test_defaults_match_requested_vessel_stackup() -> None:
    layout = DEFAULT_VESSEL_DRAFTER_LAYOUT

    assert layout.inner_diameter_in == pytest.approx(50.0)
    assert layout.glass_depth_in == pytest.approx(14.0)
    assert layout.plenum_height_in == pytest.approx(14.0)
    assert layout.head_depth_in == pytest.approx(12.5)
    assert [layer.name for layer in layout.layers] == [
        "glass_bath",
        "hot_face_refractory",
        "ifb",
        "duraboard",
        "steel_shell",
    ]
    assert layout.total_shell_thickness_in == pytest.approx(12.0)
    assert layout.outer_head_depth_in == pytest.approx(24.5)
    assert layout.side_ports == ()
    assert layout.lid_ports == ()
    assert layout.material_properties_by_name["hot_face_refractory"].density_lb_per_ft3 > 0.0
    assert (
        layout.material_properties_by_name["hot_face_refractory"].thermal_expansion_um_per_m_c > 0.0
    )


def test_port_inputs_round_trip_through_the_layout() -> None:
    layout = VesselDrafterLayout(
        side_ports=(
            VesselSidePort(
                clock_angle_degrees=45.0,
                diameter_in=3.5,
                height_above_glass_surface_in=8.0,
            ),
        ),
        lid_ports=(
            VesselLidPort(
                clock_angle_degrees=90.0,
                diameter_in=4.0,
                radial_distance_from_center_in=10.0,
            ),
        ),
    )

    assert layout.side_ports[0].normalized_clock_angle_degrees == pytest.approx(45.0)
    assert layout.side_ports[0].centerline_height_in(layout) == pytest.approx(22.0)
    assert layout.lid_ports[0].normalized_clock_angle_degrees == pytest.approx(90.0)


def test_invalid_dimensions_raise_value_error() -> None:
    with pytest.raises(ValueError):
        VesselDrafterLayout(inner_diameter_in=0.0)

    with pytest.raises(ValueError):
        VesselDrafterLayout(plenum_height_in=0.0)

    with pytest.raises(ValueError):
        VesselDrafterLayout(electrode_insertion_into_inner_circle_in=30.0)

    with pytest.raises(ValueError):
        VesselDrafterLayout(
            side_ports=(
                VesselSidePort(
                    clock_angle_degrees=0.0,
                    diameter_in=2.0,
                    height_above_glass_surface_in=20.0,
                ),
            )
        )

    with pytest.raises(ValueError):
        VesselDrafterLayout(
            lid_ports=(
                VesselLidPort(
                    clock_angle_degrees=0.0,
                    diameter_in=12.0,
                    radial_distance_from_center_in=32.0,
                ),
            )
        )


def test_default_json_file_matches_layout_values() -> None:
    data_path = Path(__file__).parent.parent / "data" / "vessel_drafter_default.json"
    assert data_path.exists(), f"Default JSON not found at {data_path}"
    data = json.loads(data_path.read_text(encoding="utf-8"))

    layout = DEFAULT_VESSEL_DRAFTER_LAYOUT
    assert data["vessel"]["inner_diameter_in"] == pytest.approx(layout.inner_diameter_in)
    assert data["vessel"]["outer_diameter_in"] == pytest.approx(layout.outer_diameter_in)
    assert data["vessel"]["full_height_in"] == pytest.approx(layout.full_height_in)
