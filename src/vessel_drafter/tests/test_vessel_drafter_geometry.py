import pytest

build123d = pytest.importorskip("build123d")

from vessel_drafter.models.vessel_drafter import (  # noqa: E402
    DEFAULT_VESSEL_DRAFTER_LAYOUT,
    VesselDrafterLayout,
    VesselLidPort,
)
from vessel_drafter.projects.vessel_drafter_layout import (  # noqa: E402
    build_vessel_drafter_shape,
)


def test_shape_bounding_box_matches_vessel_envelope() -> None:
    shape = build_vessel_drafter_shape()
    size = shape.bounding_box().size

    assert size.Z == pytest.approx(DEFAULT_VESSEL_DRAFTER_LAYOUT.full_height_in * 25.4)
    assert size.X > (74.0 * 25.4)
    assert size.Y > (74.0 * 25.4)


def test_lid_ports_reduce_the_exported_volume() -> None:
    baseline = build_vessel_drafter_shape(DEFAULT_VESSEL_DRAFTER_LAYOUT)
    with_port = build_vessel_drafter_shape(
        VesselDrafterLayout(
            lid_ports=(
                VesselLidPort(
                    clock_angle_degrees=0.0,
                    diameter_in=4.0,
                    radial_distance_from_center_in=8.0,
                ),
            )
        )
    )

    assert with_port.volume < baseline.volume
