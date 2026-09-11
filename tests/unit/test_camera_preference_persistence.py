"""Unit tests for shared camera controls and preference persistence (#4961)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from shared.python.ui.camera_controls import (
    CAMERA_PREFERENCES_FORMAT,
    CAMERA_VIEWPORT_IDS,
    CameraCommandId,
    CameraPreference,
    CameraPreferences,
    CameraState,
    FaceOnSide,
    apply_camera_preference,
    apply_camera_preset,
    apply_manual_override,
    camera_preset,
    canvas_angles,
    default_camera_preferences,
    matplotlib_angles,
    moving_subject_camera_state,
    preference_from_camera_state,
    recenter_camera,
    safe_tracking_zoom,
    set_tracking_enabled,
    update_tracking_target,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

FIXTURE_PREFERENCES = (
    Path(__file__).parents[2]
    / "src/rate_of_closure/web/src/model/__fixtures__/camera_preferences_v1.json"
)
FIXTURE_COMMANDS = (
    Path(__file__).parents[2]
    / "src/rate_of_closure/web/src/model/__fixtures__/camera_commands_v1.json"
)


def test_shared_camera_preferences_golden_round_trip() -> None:
    """Verify serialized JSON document round-trips with exact values."""
    document = json.loads(FIXTURE_PREFERENCES.read_text(encoding="utf-8"))
    preferences = CameraPreferences.from_document(document)

    assert preferences.to_document() == document
    assert preferences.viewports["impact"].zoom == pytest.approx(1.25)
    assert preferences.viewports["swing"].zoom == pytest.approx(2.5)
    assert preferences.viewports["flight"].zoom == pytest.approx(3.5)
    assert document["format"] == CAMERA_PREFERENCES_FORMAT
    assert tuple(preferences.viewports.keys()) == CAMERA_VIEWPORT_IDS


def test_shared_camera_preferences_defaults() -> None:
    """Default preferences initialize moving subjects with tracking and auto-fit."""
    preferences = default_camera_preferences().viewports

    assert preferences["impact"] == CameraPreference()
    for viewport_id in ("swing", "flight"):
        assert preferences[viewport_id].zoom == pytest.approx(2.0)
        assert preferences[viewport_id].tracking_enabled
        assert preferences[viewport_id].auto_fit_enabled


@pytest.mark.parametrize(
    "mutation",
    [
        lambda doc: {**doc, "format": "camera-preferences/v99"},
        lambda doc: {**doc, "extra_field": True},
        lambda doc: {
            **doc,
            "viewports": {**doc["viewports"], "unknown_viewport": {"zoom": 1.0}},
        },
        lambda doc: {
            **doc,
            "viewports": {
                **doc["viewports"],
                "impact": {**doc["viewports"]["impact"], "zoom": 10.0},
            },
        },
        lambda doc: {
            **doc,
            "viewports": {
                **doc["viewports"],
                "impact": {**doc["viewports"]["impact"], "zoom": 0.1},
            },
        },
        lambda doc: {
            **doc,
            "viewports": {
                **doc["viewports"],
                "impact": {**doc["viewports"]["impact"], "preset_id": "invalid.preset"},
            },
        },
    ],
)
def test_shared_camera_preferences_rejects_invalid_document(
    mutation: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    """Preference parser rejects schema mismatches and invalid presets."""
    document = json.loads(FIXTURE_PREFERENCES.read_text(encoding="utf-8"))
    with pytest.raises((TypeError, ValueError)):
        CameraPreferences.from_document(mutation(document))


def test_preference_capture_and_restore_cycle() -> None:
    """Verify live target and manual suspension are never persisted."""
    fallback = CameraPreference(preset_id=CameraCommandId.VIEW_OVERHEAD)
    runtime = CameraState(
        preset_id=None,
        target_m=(12.0, 3.0, -1.0),
        zoom=2.2,
        tracking_enabled=True,
        tracking_suspended=True,
        auto_fit_enabled=True,
    )
    captured = preference_from_camera_state(runtime, fallback=fallback)
    assert captured.preset_id == CameraCommandId.VIEW_OVERHEAD
    assert captured.zoom == pytest.approx(2.2)
    assert captured.tracking_enabled
    assert captured.auto_fit_enabled

    restored = apply_camera_preference(runtime, captured)
    assert restored.target_m == (12.0, 3.0, -1.0)
    assert not restored.tracking_suspended
    assert restored.preset_id == CameraCommandId.VIEW_OVERHEAD


def test_camera_presets_match_golden_fixture() -> None:
    """Verify deterministic preset orientation calculations match fixture."""
    fixture = json.loads(FIXTURE_COMMANDS.read_text(encoding="utf-8"))
    for case in fixture["presets"]:
        command = CameraCommandId(str(case["command_id"]))
        side = FaceOnSide(str(case["face_on_side"]))
        preset = camera_preset(command, side)
        for _, (v, c) in enumerate(
            zip(preset.view_direction, case["view_direction"], strict=True)
        ):
            assert v == pytest.approx(c, abs=1e-12)
        for _, (u, c) in enumerate(
            zip(preset.screen_up, case["screen_up"], strict=True)
        ):
            assert u == pytest.approx(c, abs=1e-12)
        yaw, pitch = canvas_angles(preset)
        elevation, azimuth = matplotlib_angles(preset)
        assert yaw == pytest.approx(case["canvas_yaw_rad"], abs=1e-12)
        assert pitch == pytest.approx(case["canvas_pitch_rad"], abs=1e-12)
        assert elevation == pytest.approx(case["matplotlib_elevation_deg"], abs=1e-12)
        assert azimuth == pytest.approx(case["matplotlib_azimuth_deg"], abs=1e-12)


def test_camera_tracking_and_manual_override_determinism() -> None:
    """Test tracking advancement, manual override suspension, and recentering."""
    initial = CameraState(target_m=(0.0, 0.0, 0.0), zoom=2.0)
    tracking = set_tracking_enabled(initial, True, (0.0, 0.0, 0.0))
    assert tracking.tracking_enabled

    # Step smaller than max step moves exactly to subject
    stepped = update_tracking_target(tracking, (0.5, 0.0, 0.0), max_step_m=1.0)
    assert stepped.target_m == pytest.approx((0.5, 0.0, 0.0))

    # Step larger than max step clamps to max step
    clamped = update_tracking_target(stepped, (10.0, 0.0, 0.0), max_step_m=1.5)
    assert clamped.target_m == pytest.approx((2.0, 0.0, 0.0))

    # Manual override suspends tracking
    suspended = apply_manual_override(clamped)
    assert suspended.tracking_suspended
    # While suspended, updates do not change target
    unchanged = update_tracking_target(suspended, (20.0, 0.0, 0.0), max_step_m=1.5)
    assert unchanged.target_m == clamped.target_m

    # Recentering clears suspension and snaps to subject
    recentered = recenter_camera(suspended, (5.0, 1.0, 0.0))
    assert not recentered.tracking_suspended
    assert recentered.target_m == pytest.approx((5.0, 1.0, 0.0))


def test_safe_tracking_zoom_bounds() -> None:
    """Verify safe tracking zoom clearance calculation and boundary conditions."""
    # Under limit: preserved
    assert safe_tracking_zoom(
        1.5, subject_radius_m=0.2, base_half_extent_m=1.0
    ) == pytest.approx(1.5)

    # Over limit: reduced to maintain clearance
    # limit = 1.0 * (1 - 0.16) / 0.5 = 0.84 / 0.5 = 1.68
    assert safe_tracking_zoom(
        3.0, subject_radius_m=0.5, base_half_extent_m=1.0
    ) == pytest.approx(1.68)

    # Non-positive values raise ValueError
    with pytest.raises(ValueError, match="positive"):
        safe_tracking_zoom(0.0, subject_radius_m=0.2, base_half_extent_m=1.0)
    with pytest.raises(ValueError, match="positive"):
        safe_tracking_zoom(1.0, subject_radius_m=-0.1, base_half_extent_m=1.0)


def test_moving_subject_camera_state_defaults() -> None:
    """Verify default parameters for moving subject camera state."""
    state = moving_subject_camera_state()
    assert state.zoom == pytest.approx(2.0)
    assert state.tracking_enabled
    assert state.auto_fit_enabled
    assert not state.tracking_suspended
    assert state.target_m == (0.0, 0.0, 0.0)


def test_apply_camera_preset_immutability() -> None:
    """Applying a preset should return an updated immutable state."""
    initial = CameraState(
        preset_id=CameraCommandId.VIEW_ISOMETRIC, face_on_side=FaceOnSide.RIGHT
    )
    updated = apply_camera_preset(initial, CameraCommandId.VIEW_FACE_ON)
    assert updated.preset_id == CameraCommandId.VIEW_FACE_ON
    assert updated.face_on_side == FaceOnSide.RIGHT
    assert initial.preset_id == CameraCommandId.VIEW_ISOMETRIC
    assert initial.face_on_side == FaceOnSide.RIGHT
