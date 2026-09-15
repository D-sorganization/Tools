"""Ground-plane geometry: homography, launch fit, HLA sign."""

from __future__ import annotations

import numpy as np
import pytest

from putting_launch_monitor.geometry import GroundPlane, fit_launch, hla_degrees

from .synthetic import MAT_L_MM, MAT_W_MM, SyntheticCamera, putt_positions

pytestmark = pytest.mark.unit


def test_rectangle_calibration_reprojects_exactly() -> None:
    cam = SyntheticCamera()
    corners = cam.mat_corners_px()
    plane = GroundPlane.from_rectangle(corners, MAT_W_MM, MAT_L_MM)
    world = np.array(
        [[0, 0], [MAT_W_MM, 0], [MAT_W_MM, MAT_L_MM], [0, MAT_L_MM]], float
    )
    assert plane.reprojection_error_px(corners, world) < 1e-6
    # An arbitrary interior point round-trips through a genuine perspective.
    p = np.array([300.0, 900.0])
    assert np.allclose(plane.to_world(cam.project(p)[0]), p, atol=1e-6)
    assert np.allclose(plane.to_image(p), cam.project(p)[0], atol=1e-6)
    assert plane.mm_per_px_at(corners.mean(axis=0)) > 0


def test_calibration_contracts() -> None:
    corners = SyntheticCamera().mat_corners_px()
    with pytest.raises(ValueError, match="dimensions"):
        GroundPlane.from_rectangle(corners, 0, 100)
    with pytest.raises(ValueError, match="four corners"):
        GroundPlane.from_rectangle(corners[:3], 10, 10)
    collinear = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], float)
    with pytest.raises(ValueError, match="degenerate"):
        GroundPlane.from_correspondences(collinear, collinear * 2)
    with pytest.raises(ValueError, match="3x3"):
        GroundPlane(np.eye(2))


@pytest.mark.parametrize("hla", [-4.0, 0.0, 3.5, 12.0])
@pytest.mark.parametrize("speed", [0.8, 2.0, 4.5])
def test_launch_fit_recovers_known_putts_through_the_tilted_camera(
    speed: float, hla: float
) -> None:
    cam = SyntheticCamera()
    plane = GroundPlane.from_rectangle(cam.mat_corners_px(), MAT_W_MM, MAT_L_MM)
    t, world = putt_positions((610.0, 250.0), speed, hla, fps=60.0, seconds=0.6)
    px = cam.project(world) + np.random.default_rng(1).normal(0, 0.3, size=(len(t), 2))
    launch = fit_launch(t, plane.to_world(px), window_mm=300.0)
    assert launch.speed_mps == pytest.approx(speed, rel=0.02)
    assert hla_degrees(launch.direction) == pytest.approx(hla, abs=0.5)
    assert launch.r2 > 0.99 and launch.points >= 4
    assert launch.speed_mph == pytest.approx(launch.speed_mps * 2.2369363, rel=1e-6)


def test_launch_fit_uses_the_opening_window_of_a_decelerating_roll() -> None:
    """Launch speed, not average speed: the ball slows on the mat."""
    t, world = putt_positions(
        (600.0, 200.0), 2.0, 0.0, fps=60.0, seconds=1.5, decel_mps2=0.8
    )
    launch = fit_launch(t, world, window_mm=250.0)
    assert launch.speed_mps == pytest.approx(2.0, rel=0.05)
    assert launch.span_mm <= 250.0 + 40  # one frame past the window at most


def test_launch_fit_contracts() -> None:
    t = np.array([0.0, 0.1, 0.2, 0.3])
    p = np.array([[0, 0], [0, 10], [0, 20], [0, 30]], float)
    with pytest.raises(ValueError, match="too few"):
        fit_launch(t[:2], p[:2])
    with pytest.raises(ValueError, match="strictly increase"):
        fit_launch(np.array([0.0, 0.1, 0.1, 0.3]), p)
    with pytest.raises(ValueError, match="window"):
        fit_launch(t, p, window_mm=0)


def test_hla_sign_is_positive_to_the_right_of_target() -> None:
    assert hla_degrees((0.0, 1.0)) == pytest.approx(0.0)
    assert hla_degrees((0.1, 1.0)) > 0  # drifting to the player's right
    assert hla_degrees((-0.1, 1.0)) < 0
    assert hla_degrees((1.0, 0.0)) == pytest.approx(90.0)
    # Against a rotated target line the reading is relative to that line.
    assert hla_degrees(
        (0.0, 1.0), target=(np.sin(np.radians(5)), np.cos(np.radians(5)))
    ) == pytest.approx(-5.0)
    with pytest.raises(ValueError, match="zero-length"):
        hla_degrees((0.0, 0.0))
