"""Cross-runtime durable camera-preference contracts (#4218/#4284)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rate_of_closure.application.camera_commands import (
    CameraCommandId,
    CameraState,
)
from rate_of_closure.application.camera_preferences import (
    CAMERA_PREFERENCES_FORMAT,
    CameraPreference,
    CameraPreferences,
    apply_camera_preference,
    default_camera_preferences,
    preference_from_camera_state,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
FIXTURE = (
    Path(__file__).parents[2]
    / "src/rate_of_closure/web/src/model/__fixtures__/camera_preferences_v1.json"
)


def test_shared_golden_round_trips_with_isolated_viewport_values() -> None:
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    preferences = CameraPreferences.from_document(document)

    assert preferences.to_document() == document
    assert preferences.viewports["impact"].zoom == pytest.approx(1.25)
    assert preferences.viewports["swing"].zoom == pytest.approx(2.5)
    assert preferences.viewports["flight"].zoom == pytest.approx(3.5)
    assert document["format"] == CAMERA_PREFERENCES_FORMAT


def test_migration_defaults_match_moving_subject_contract() -> None:
    preferences = default_camera_preferences().viewports

    assert preferences["impact"] == CameraPreference()
    for viewport_id in ("swing", "flight"):
        assert preferences[viewport_id].zoom == pytest.approx(2.0)
        assert preferences[viewport_id].tracking_enabled
        assert preferences[viewport_id].auto_fit_enabled


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: {**value, "format": "camera-preferences/v9"},
        lambda value: {**value, "unexpected": True},
        lambda value: {
            **value,
            "viewports": {**value["viewports"], "flight": {"zoom": 2.0}},
        },
        lambda value: {
            **value,
            "viewports": {
                **value["viewports"],
                "swing": {**value["viewports"]["swing"], "zoom": 9.0},
            },
        },
    ],
)
def test_contract_rejects_future_or_malformed_documents(mutation) -> None:
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    with pytest.raises((TypeError, ValueError)):
        CameraPreferences.from_document(mutation(document))


def test_capture_and_restore_exclude_target_and_manual_suspension() -> None:
    fallback = CameraPreference(preset_id=CameraCommandId.VIEW_OVERHEAD)
    runtime = CameraState(
        preset_id=None,
        target_m=(14.0, 2.0, -3.0),
        zoom=2.5,
        tracking_enabled=True,
        tracking_suspended=True,
        auto_fit_enabled=True,
    )

    preference = preference_from_camera_state(runtime, fallback)
    assert preference.preset_id is CameraCommandId.VIEW_OVERHEAD
    assert "target_m" not in preference.to_document()
    assert "tracking_suspended" not in preference.to_document()

    other_runtime = replace(runtime, target_m=(99.0, 8.0, 7.0))
    restored = apply_camera_preference(other_runtime, preference)
    assert restored.target_m == (99.0, 8.0, 7.0)
    assert not restored.tracking_suspended
