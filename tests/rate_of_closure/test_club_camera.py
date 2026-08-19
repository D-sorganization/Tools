from __future__ import annotations

import json
from pathlib import Path

import pytest

from rate_of_closure.club_camera import (
    DEFAULT_CLUB_CAMERA,
    ClubCamera,
    ClubCameraAction,
    apply_club_camera_action,
    apply_club_camera_drag,
    matplotlib_view,
)

FIXTURE = (
    Path(__file__).parents[2]
    / "src/rate_of_closure/web/src/model/__fixtures__/club_camera_golden_v1.json"
)


def test_camera_actions_match_shared_python_owned_golden() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for case in payload["cases"]:
        actual = apply_club_camera_action(
            DEFAULT_CLUB_CAMERA, ClubCameraAction(case["action"])
        )
        expected = case["expected"]
        assert actual.azimuth_deg == pytest.approx(expected["azimuth_deg"])
        assert actual.elevation_deg == pytest.approx(expected["elevation_deg"])
        assert actual.zoom == pytest.approx(expected["zoom"])


def test_camera_clamps_and_home_resets() -> None:
    camera = DEFAULT_CLUB_CAMERA
    for _ in range(100):
        camera = apply_club_camera_action(camera, ClubCameraAction.UP)
        camera = apply_club_camera_action(camera, ClubCameraAction.ZOOM_IN)
    assert camera.elevation_deg == 80
    assert camera.zoom == 4
    assert (
        apply_club_camera_action(camera, ClubCameraAction.HOME) == DEFAULT_CLUB_CAMERA
    )


def test_matplotlib_mapping_matches_existing_default_view() -> None:
    assert matplotlib_view(DEFAULT_CLUB_CAMERA) == pytest.approx((30.0, -60.0))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_camera_rejects_nonfinite_state_and_drag(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ClubCamera(value, 0.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        apply_club_camera_drag(DEFAULT_CLUB_CAMERA, value, 0.0)


def test_camera_stays_canonical_after_long_action_and_drag_sequences() -> None:
    camera = DEFAULT_CLUB_CAMERA
    for _ in range(10_000):
        camera = apply_club_camera_action(camera, ClubCameraAction.RIGHT)
        camera = apply_club_camera_drag(camera, -1.0, 1.0)
    assert -180.0 <= camera.azimuth_deg < 180.0
    assert camera.elevation_deg == 80.0
    assert 0.3 <= camera.zoom <= 4.0
