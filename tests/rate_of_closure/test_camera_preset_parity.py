"""Canonical camera preset contract and club-view parity tests (#4284)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.application.camera_presets import (  # noqa: E402
    CAMERA_COMMAND_IDS,
    CameraCommandId,
    CameraPreset,
    CameraState,
    CameraViewId,
    FaceOnSide,
    apply_camera_view,
    auto_fit_camera,
    camera_preset,
    canvas_angles,
    matplotlib_angles,
)
from rate_of_closure.model import ImpactScenario  # noqa: E402
from rate_of_closure.ui.pyqt6.club_view import VIEW_MODES, Club3DView  # noqa: E402
from rate_of_closure.ui.pyqt6.club_view_geometry import (  # noqa: E402
    canonical_axis_visibility,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

FIXTURE_PATH = (
    Path(__file__).parents[2]
    / "src/rate_of_closure/web/src/model/__fixtures__/camera_presets_v1.json"
)


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_exact_presets_and_adapters_match_one_shared_fixture() -> None:
    fixture = _fixture()
    assert fixture["schema"] == "rate-of-closure-camera-presets/v1"
    assert fixture["frame"] == {"x": "downrange", "y": "up", "z": "right"}
    assert list(CAMERA_COMMAND_IDS) == fixture["command_ids"]
    for case in fixture["presets"]:
        assert isinstance(case, dict)
        preset = camera_preset(case["command_id"], case["face_on_side"])
        np.testing.assert_allclose(
            preset.view_direction, case["view_direction"], atol=1e-12
        )
        np.testing.assert_allclose(preset.screen_up, case["screen_up"], atol=1e-12)
        assert canvas_angles(preset) == pytest.approx(
            (case["canvas_yaw_rad"], case["canvas_pitch_rad"]), abs=1e-12
        )
        assert matplotlib_angles(preset) == pytest.approx(
            (case["matplotlib_elevation_deg"], case["matplotlib_azimuth_deg"]),
            abs=1e-12,
        )


def test_contract_rejects_unknown_nonfinite_and_nonorthogonal_values() -> None:
    with pytest.raises(ValueError, match="unknown camera view"):
        camera_preset("camera.view.unknown", "right")
    with pytest.raises(ValueError, match="unknown face-on side"):
        camera_preset(CameraViewId.FACE_ON, "automatic")
    with pytest.raises(ValueError, match="finite"):
        CameraState(target_m=(float("nan"), 0.0, 0.0))
    with pytest.raises(ValueError, match="unit vector"):
        CameraPreset(CameraViewId.FACE_ON, (2.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="unit vector"):
        CameraPreset(
            CameraViewId.FACE_ON,
            (1.0 + 5e-10, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    with pytest.raises(ValueError, match="perpendicular"):
        CameraPreset(CameraViewId.FACE_ON, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0))


def test_preset_and_reset_are_idempotent_and_preserve_target_and_zoom() -> None:
    state = CameraState(target_m=(1.0, 2.0, 3.0), zoom=2.25)
    snapped = apply_camera_view(state, CameraViewId.DOWN_THE_LINE)
    assert apply_camera_view(snapped, CameraViewId.DOWN_THE_LINE) == snapped
    assert snapped.target_m == state.target_m
    assert snapped.zoom == state.zoom
    reset = apply_camera_view(snapped, CameraViewId.ISOMETRIC)
    assert reset.target_m == state.target_m
    assert reset.zoom == state.zoom


def test_only_auto_fit_changes_scale_and_it_bounds_the_declared_subject() -> None:
    state = CameraState(target_m=(1.0, 2.0, 3.0), zoom=3.5)
    fitted = auto_fit_camera(state, subject_radius_m=0.35, base_half_extent_m=0.42)
    assert fitted.target_m == state.target_m
    assert fitted.preset_id == state.preset_id
    assert fitted.zoom != state.zoom
    assert 0.35 * fitted.zoom <= 0.42 * 0.84 + 1e-12
    with pytest.raises(ValueError, match="positive"):
        auto_fit_camera(state, subject_radius_m=0.0, base_half_extent_m=0.42)


def test_canonical_depth_axis_visibility_contract() -> None:
    assert canonical_axis_visibility(CameraViewId.ISOMETRIC) == (True, True, True)
    assert canonical_axis_visibility(CameraViewId.FACE_ON) == (False, True, True)
    assert canonical_axis_visibility(CameraViewId.DOWN_THE_LINE) == (True, False, True)
    assert canonical_axis_visibility(CameraViewId.OVERHEAD) == (True, True, False)
    assert canonical_axis_visibility(None) == (True, True, True)


def test_pyqt_club_view_exposes_accessible_stable_commands(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = Club3DView()
    qtbot.addWidget(view)
    view.set_scenario(ImpactScenario(clubhead_speed_mph=120.0))
    controls = view.camera_controls()
    assert set(controls.command_widgets()) == set(CAMERA_COMMAND_IDS)
    for command_id, widget in controls.command_widgets().items():
        assert widget.property("cameraCommandId") == command_id
        assert widget.accessibleName()
        assert widget.toolTip()
        assert widget.focusPolicy().value != 0
    view.stop()


def test_pyqt_snap_side_reset_and_zoom_preservation(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = Club3DView()
    qtbot.addWidget(view)
    view.set_scenario(ImpactScenario(clubhead_speed_mph=120.0))
    view.set_zoom(2.0)
    view.apply_camera_command(CameraViewId.DOWN_THE_LINE)
    assert (float(view._axes.elev), float(view._axes.azim)) == pytest.approx(
        (0.0, -90.0)
    )
    view.apply_camera_command(CameraViewId.DOWN_THE_LINE)
    assert view.zoom() == pytest.approx(2.0)
    view.set_face_on_side(FaceOnSide.LEFT)
    view.apply_camera_command(CameraViewId.FACE_ON)
    assert (float(view._axes.elev), float(view._axes.azim)) == pytest.approx(
        (0.0, -180.0)
    )
    view.apply_camera_command(CameraCommandId.RESET_VIEW)
    assert view.camera_state().preset_id is CameraViewId.ISOMETRIC
    assert view.zoom() == pytest.approx(2.0)
    view.stop()


@pytest.mark.parametrize(
    ("view_id", "expected"),
    [
        (CameraViewId.ISOMETRIC, (True, True, True)),
        (CameraViewId.FACE_ON, (False, True, True)),
        (CameraViewId.DOWN_THE_LINE, (True, False, True)),
        (CameraViewId.OVERHEAD, (True, True, False)),
    ],
)
def test_pyqt_canonical_views_hide_only_the_collapsed_depth_axis(
    qtbot, view_id: CameraViewId, expected: tuple[bool, bool, bool]
) -> None:  # type: ignore[no-untyped-def]
    view = Club3DView()
    qtbot.addWidget(view)
    view.set_scenario(ImpactScenario(clubhead_speed_mph=120.0))
    view.apply_camera_command(view_id)

    for axis, visible in zip(
        (view._axes.xaxis, view._axes.yaxis, view._axes.zaxis),
        expected,
        strict=True,
    ):
        assert axis.get_visible() is visible
        assert axis.label.get_visible() is visible
        assert axis.pane.get_visible() is visible
        assert axis.gridlines.get_visible() is visible
        assert all(
            label.get_visible() is visible for label in axis.get_majorticklabels()
        )
        if visible:
            assert axis.get_label_text()
            assert len(axis.get_majorticklocs()) > 0
        else:
            assert axis.get_label_text() == ""
            assert len(axis.get_majorticklocs()) == 0
    view.stop()


def test_pyqt_manual_orbit_restores_every_axis_without_resetting_orbit(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = Club3DView()
    qtbot.addWidget(view)
    view.set_scenario(ImpactScenario(clubhead_speed_mph=120.0))
    view.apply_camera_command(CameraViewId.DOWN_THE_LINE)
    assert not view._axes.yaxis.get_visible()
    view._axes.view_init(elev=31.0, azim=47.0)

    view._on_orbit_release(None)

    for axis in (view._axes.xaxis, view._axes.yaxis, view._axes.zaxis):
        assert axis.get_visible()
        assert axis.label.get_visible()
        assert axis.pane.get_visible()
        assert axis.gridlines.get_visible()
        assert axis.get_label_text()
        assert len(axis.get_majorticklocs()) > 0
        assert all(label.get_visible() for label in axis.get_majorticklabels())
    assert (float(view._axes.elev), float(view._axes.azim)) == pytest.approx(
        (31.0, 47.0)
    )
    view.stop()


def test_pyqt_zoom_preserves_a_manual_orbit(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = Club3DView()
    qtbot.addWidget(view)
    view.set_scenario(ImpactScenario(clubhead_speed_mph=120.0))
    view._axes.view_init(elev=55.0, azim=12.0)
    view._on_orbit_release(None)
    assert not any(
        button.isChecked() for button in view.camera_controls().view_buttons().values()
    )
    view.set_zoom(2.0)
    assert (float(view._axes.elev), float(view._axes.azim)) == pytest.approx(
        (55.0, 12.0)
    )
    view.apply_camera_command(CameraViewId.OVERHEAD)
    assert view.camera_controls().view_buttons()[CameraViewId.OVERHEAD].isChecked()
    view.stop()


def test_pyqt_face_side_change_restores_the_exact_face_on_preset(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = Club3DView()
    qtbot.addWidget(view)
    view.set_scenario(ImpactScenario(clubhead_speed_mph=120.0))
    view.apply_camera_command(CameraViewId.FACE_ON)
    view._axes.view_init(elev=33.0, azim=47.0)
    view._on_orbit_release(None)
    assert not any(
        button.isChecked() for button in view.camera_controls().view_buttons().values()
    )

    view.set_face_on_side(FaceOnSide.LEFT)

    assert view.camera_controls().view_buttons()[CameraViewId.FACE_ON].isChecked()
    assert (float(view._axes.elev), float(view._axes.azim)) == pytest.approx(
        (0.0, -180.0)
    )
    view.stop()


@pytest.mark.parametrize("mode", VIEW_MODES)
@pytest.mark.parametrize("phase", [0.0, 0.5, 1.0])
def test_pyqt_auto_fit_bounds_driver_at_start_impact_and_end(
    qtbot, mode: str, phase: float
) -> None:  # type: ignore[no-untyped-def]
    view = Club3DView()
    qtbot.addWidget(view)
    view.set_scenario(ImpactScenario(clubhead_speed_mph=120.0))
    view.set_view_mode(mode)
    view._phase = phase
    view._draw()
    view.set_zoom(4.0)
    before = view.zoom()
    view.apply_camera_command(CameraCommandId.AUTO_FIT)
    assert view.zoom() != before
    assert view.camera_subject_fits(clearance_fraction=0.16)
    view.stop()
