import pytest
from vessel_drafter.models.vessel_drafter import (
    DEFAULT_VESSEL_DRAFTER_LAYOUT,
    VesselDrafterLayout,
    VesselLidPort,
    VesselSidePort,
)
from vessel_drafter.preview.vessel_drafter_preview import (
    build_cross_section_preview,
    build_plan_preview,
)


def test_cross_section_preview_matches_expected_envelope() -> None:
    preview = build_cross_section_preview(DEFAULT_VESSEL_DRAFTER_LAYOUT)

    assert preview.inner_radius_in == pytest.approx(25.0)
    assert preview.outer_radius_in == pytest.approx(37.0)
    assert preview.straight_shell_height_in == pytest.approx(28.0)
    assert preview.inner_head_depth_in == pytest.approx(12.5)
    assert preview.outer_head_depth_in == pytest.approx(24.5)
    assert preview.glass_height_in == pytest.approx(14.0)
    assert preview.plenum_height_in == pytest.approx(14.0)
    assert preview.electrode_diameter_in == pytest.approx(2.0)
    assert [band.label for band in preview.bands] == [
        "glass_bath",
        "hot_face_refractory",
        "ifb",
        "duraboard",
        "steel_shell",
    ]
    assert len(preview.band_polygons) == 5


def test_plan_preview_has_three_radial_electrodes() -> None:
    preview = build_plan_preview(DEFAULT_VESSEL_DRAFTER_LAYOUT)

    assert preview.outer_radius_in == pytest.approx(37.0)
    assert [item.angle_degrees for item in preview.electrodes] == pytest.approx(
        [0.0, 120.0, 240.0]
    )
    assert preview.electrodes[0].outer_tip_radius_in == pytest.approx(61.0)
    assert preview.electrodes[0].diameter_in == pytest.approx(2.0)


def test_preview_includes_side_and_lid_ports() -> None:
    layout = VesselDrafterLayout(
        side_ports=(
            VesselSidePort(
                clock_angle_degrees=30.0,
                diameter_in=3.0,
                height_above_glass_surface_in=6.0,
            ),
        ),
        lid_ports=(
            VesselLidPort(
                clock_angle_degrees=225.0,
                diameter_in=4.0,
                radial_distance_from_center_in=8.0,
            ),
        ),
    )

    cross_section = build_cross_section_preview(layout)
    plan = build_plan_preview(layout)

    assert len(cross_section.projected_side_ports) == 1
    assert cross_section.projected_side_ports[0].centerline_height_in == pytest.approx(
        20.0
    )
    assert len(plan.side_ports) == 1
    assert len(plan.lid_ports) == 1
    assert plan.lid_ports[0].diameter_in == pytest.approx(4.0)
