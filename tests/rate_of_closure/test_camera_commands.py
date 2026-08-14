"""UI-neutral camera-command contract and cross-runtime golden cases (#4284)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rate_of_closure.application.camera_commands import (
    CameraCommandId,
    CameraState,
    FaceOnSide,
    apply_manual_override,
    camera_preset,
    canvas_angles,
    matplotlib_angles,
    moving_subject_camera_state,
    recenter_camera,
    safe_tracking_zoom,
    set_tracking_enabled,
    update_tracking_target,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

FIXTURE_PATH = (
    Path(__file__).parents[2]
    / "src/rate_of_closure/web/src/model/__fixtures__/camera_commands_v1.json"
)


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_command_ids_and_exact_view_orientations_match_shared_fixture() -> None:
    fixture = _fixture()
    assert fixture["schema"] == "rate-of-closure-camera-commands/v1"
    assert [command.value for command in CameraCommandId] == fixture["command_ids"]
    for case in fixture["presets"]:
        assert isinstance(case, dict)
        command = CameraCommandId(str(case["command_id"]))
        side = FaceOnSide(str(case["face_on_side"]))
        preset = camera_preset(command, side)
        np.testing.assert_allclose(
            preset.view_direction, case["view_direction"], atol=1e-12
        )
        np.testing.assert_allclose(preset.screen_up, case["screen_up"], atol=1e-12)
        yaw, pitch = canvas_angles(preset)
        elevation, azimuth = matplotlib_angles(preset)
        assert yaw == pytest.approx(case["canvas_yaw_rad"], abs=1e-12)
        assert pitch == pytest.approx(case["canvas_pitch_rad"], abs=1e-12)
        assert elevation == pytest.approx(case["matplotlib_elevation_deg"], abs=1e-12)
        assert azimuth == pytest.approx(case["matplotlib_azimuth_deg"], abs=1e-12)


def test_tracking_is_bounded_suspendable_and_exactly_recenterable() -> None:
    initial = CameraState(target_m=(0.0, 0.0, 0.0), zoom=2.5)
    enabled = set_tracking_enabled(initial, True, (0.0, 0.0, 0.0))
    advanced = update_tracking_target(enabled, (10.0, 0.0, 0.0), max_step_m=2.0)
    assert advanced.target_m == pytest.approx((2.0, 0.0, 0.0))
    assert advanced.zoom == pytest.approx(2.5)
    suspended = apply_manual_override(advanced)
    assert suspended.tracking_suspended
    assert update_tracking_target(suspended, (20.0, 0.0, 0.0), 2.0) == suspended
    centered = recenter_camera(suspended, (20.0, 1.0, -2.0))
    assert centered.target_m == pytest.approx((20.0, 1.0, -2.0))
    assert not centered.tracking_suspended
    assert centered.zoom == pytest.approx(2.5)


@pytest.mark.parametrize(
    "bad_target", [(float("nan"), 0.0, 0.0), (0.0, float("inf"), 0.0)]
)
def test_contract_rejects_nonfinite_camera_targets(
    bad_target: tuple[float, float, float],
) -> None:
    with pytest.raises(ValueError, match="finite"):
        CameraState(target_m=bad_target)


def test_auto_fit_only_reduces_unsafe_zoom_and_preserves_safe_zoom() -> None:
    assert safe_tracking_zoom(
        1.2, subject_radius_m=0.3, base_half_extent_m=1.0
    ) == pytest.approx(1.2)
    assert safe_tracking_zoom(
        4.0, subject_radius_m=0.3, base_half_extent_m=1.0
    ) == pytest.approx(2.8)
    with pytest.raises(ValueError, match="positive"):
        safe_tracking_zoom(1.0, subject_radius_m=0.0, base_half_extent_m=1.0)


def test_moving_subject_camera_state_is_share_ready_and_user_reversible() -> None:
    state = moving_subject_camera_state()
    assert state.zoom == pytest.approx(2.0)
    assert state.tracking_enabled
    assert state.auto_fit_enabled
    assert not state.tracking_suspended
    assert not set_tracking_enabled(state, False, (1.0, 2.0, 3.0)).tracking_enabled
