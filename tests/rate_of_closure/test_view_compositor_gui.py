"""Headless persistence and identity tests for the PyQt multi-view host."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtCore import QSettings, Qt  # noqa: E402
from PyQt6.QtWidgets import QCheckBox, QComboBox, QLabel, QScrollArea  # noqa: E402

from rate_of_closure.application.camera_commands import CameraCommandId  # noqa: E402
from rate_of_closure.ui.pyqt6.flight_view import FlightView  # noqa: E402
from rate_of_closure.ui.pyqt6.simulation_view import SimulationView  # noqa: E402
from rate_of_closure.ui.pyqt6.view_compositor import ViewCompositor  # noqa: E402
from rate_of_closure.view_workspace import (  # noqa: E402
    LegendPlacement,
    PlaybackState,
    ViewKind,
    ViewLayout,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_compositor_recovers_unknown_saved_view_and_keeps_real_hosts(
    qtbot, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    settings = QSettings(str(tmp_path / "views.ini"), QSettings.Format.IniFormat)
    settings.setValue(
        "view_compositor/layout_v1",
        json.dumps(
            {
                "format": "rate_of_closure.view_workspace/1",
                "layout": "grid",
                "slots": [
                    {"id": "future", "kind": "future"},
                    {"id": "impact", "kind": "impact"},
                    {"id": "flight", "kind": "flight"},
                ],
                "active_slot_id": "future",
            }
        ),
    )
    views = {
        kind: QLabel(kind.value)
        for kind in (ViewKind.IMPACT, ViewKind.SWING, ViewKind.FLIGHT)
    }

    compositor = ViewCompositor(views, settings)
    qtbot.addWidget(compositor)

    assert compositor.visible_view_ids() == ("impact", "flight")
    assert compositor.workspace().active_slot_id == "impact"
    assert len({id(compositor.view(kind)) for kind in views}) == 3


def test_toggle_normalizes_cardinality_and_preserves_legend(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = QSettings(str(tmp_path / "cardinality.ini"), QSettings.Format.IniFormat)
    settings.setValue(
        "view_compositor/layout_v1",
        json.dumps(
            {
                "format": "rate_of_closure.view_workspace/1",
                "layout": "single",
                "slots": [{"id": "swing", "kind": "swing", "legend": "hidden"}],
                "active_slot_id": "swing",
            }
        ),
    )
    views = {
        kind: QLabel(kind.value)
        for kind in (ViewKind.IMPACT, ViewKind.SWING, ViewKind.FLIGHT)
    }
    compositor = ViewCompositor(views, settings)
    qtbot.addWidget(compositor)

    compositor._checks[ViewKind.FLIGHT].setChecked(True)
    assert compositor.workspace().layout is ViewLayout.SPLIT_HORIZONTAL
    compositor._checks[ViewKind.IMPACT].setChecked(True)
    assert compositor.workspace().layout is ViewLayout.GRID
    assert compositor.workspace().slots[0].legend is LegendPlacement.HIDDEN
    compositor._checks[ViewKind.FLIGHT].setChecked(False)
    assert compositor.workspace().layout is ViewLayout.SPLIT_HORIZONTAL


def test_playback_updates_are_debounced_before_settings_write(
    qtbot, tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    settings = QSettings(str(tmp_path / "playback.ini"), QSettings.Format.IniFormat)
    views = {
        kind: QLabel(kind.value)
        for kind in (ViewKind.IMPACT, ViewKind.SWING, ViewKind.FLIGHT)
    }
    compositor = ViewCompositor(views, settings)
    qtbot.addWidget(compositor)
    writes: list[float] = []
    monkeypatch.setattr(
        compositor,
        "_persist",
        lambda: writes.append(compositor.workspace().playback.time_s),
    )

    for index in range(10):
        compositor.update_playback(
            PlaybackState(time_s=index / 100.0, playing=True, loop=True, rate=0.5)
        )

    assert writes == []
    qtbot.wait(300)
    assert writes == [pytest.approx(0.09)]


def test_grid_remains_navigable_when_real_views_exceed_available_height(qtbot) -> None:  # type: ignore[no-untyped-def]
    views = {
        kind: QLabel(kind.value)
        for kind in (ViewKind.IMPACT, ViewKind.SWING, ViewKind.FLIGHT)
    }
    for view in views.values():
        view.setMinimumSize(360, 280)
    compositor = ViewCompositor(views)
    qtbot.addWidget(compositor)
    compositor.resize(760, 500)
    compositor.show()

    compositor._checks[ViewKind.FLIGHT].setChecked(True)
    compositor._checks[ViewKind.IMPACT].setChecked(True)
    qtbot.wait(50)

    viewport = compositor.findChild(QScrollArea, "viewCompositorScrollArea")
    assert viewport is not None
    assert viewport.widgetResizable()
    assert viewport.verticalScrollBar().maximum() > 0


def test_keyboard_focus_order_can_build_and_reduce_a_multi_view_layout(
    qtbot,
) -> None:  # type: ignore[no-untyped-def]
    views = {
        kind: QLabel(kind.value)
        for kind in (ViewKind.IMPACT, ViewKind.SWING, ViewKind.FLIGHT)
    }
    compositor = ViewCompositor(views)
    qtbot.addWidget(compositor)
    compositor.show()
    qtbot.waitExposed(compositor)
    compositor.activateWindow()

    layout = compositor.findChild(QComboBox, "viewportLayoutCombo")
    impact = compositor.findChild(QCheckBox, "impactViewportToggle")
    flight = compositor.findChild(QCheckBox, "flightViewportToggle")
    assert layout is not None and impact is not None and flight is not None

    layout.setFocus(Qt.FocusReason.TabFocusReason)
    qtbot.waitUntil(layout.hasFocus)
    qtbot.keyClick(layout, Qt.Key.Key_Down)
    assert compositor.workspace().layout is ViewLayout.SPLIT_HORIZONTAL
    assert layout.hasFocus()
    qtbot.keyClick(layout, Qt.Key.Key_Tab)
    assert impact.hasFocus()

    flight.setFocus()
    qtbot.keyClick(flight, Qt.Key.Key_Space)
    assert compositor.workspace().layout is ViewLayout.GRID
    assert flight.hasFocus()
    qtbot.keyClick(flight, Qt.Key.Key_Space)
    assert compositor.workspace().layout is ViewLayout.SPLIT_HORIZONTAL


def test_versioned_workspace_export_import_round_trip_is_atomic(
    qtbot, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    settings = QSettings(str(tmp_path / "roundtrip.ini"), QSettings.Format.IniFormat)
    views = {
        kind: QLabel(kind.value)
        for kind in (ViewKind.IMPACT, ViewKind.SWING, ViewKind.FLIGHT)
    }
    source = ViewCompositor(views)
    qtbot.addWidget(source)
    source._checks[ViewKind.FLIGHT].setChecked(True)
    source._checks[ViewKind.IMPACT].setChecked(True)
    source.update_playback(
        PlaybackState(time_s=0.42, playing=False, loop=True, rate=0.5)
    )

    document = source.export_workspace_document()
    target = ViewCompositor(
        {
            kind: QLabel(f"target-{kind.value}")
            for kind in (ViewKind.IMPACT, ViewKind.SWING, ViewKind.FLIGHT)
        },
        settings,
    )
    qtbot.addWidget(target)
    target.import_workspace_document(document)

    assert document["format"] == "rate_of_closure.view_workspace/2"
    assert target.workspace() == source.workspace()
    settings.sync()
    reloaded = ViewCompositor(
        {
            kind: QLabel(f"reloaded-{kind.value}")
            for kind in (ViewKind.IMPACT, ViewKind.SWING, ViewKind.FLIGHT)
        },
        settings,
    )
    qtbot.addWidget(reloaded)
    assert reloaded.workspace() == source.workspace()
    before = target.workspace()
    with pytest.raises(ValueError, match="unsupported workspace format"):
        target.import_workspace_document({**document, "format": "future/9"})
    assert target.workspace() == before


def test_camera_preferences_survive_layout_hide_show_and_qsettings_reload(
    qtbot, tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    settings = QSettings(str(tmp_path / "camera.ini"), QSettings.Format.IniFormat)
    swing = SimulationView()
    flight = FlightView()
    compositor = ViewCompositor(
        {
            ViewKind.IMPACT: QLabel("impact"),
            ViewKind.SWING: swing,
            ViewKind.FLIGHT: flight,
        },
        settings,
    )
    qtbot.addWidget(compositor)

    swing.set_camera_zoom(2.25)
    swing.apply_camera_command(CameraCommandId.VIEW_DOWN_THE_LINE)
    before_flight = compositor.workspace().camera_preferences.viewports["flight"]
    compositor.show_single_view(ViewKind.FLIGHT)
    compositor.show_single_view(ViewKind.SWING)

    saved = compositor.workspace().camera_preferences.viewports
    assert saved["swing"].zoom == pytest.approx(2.25)
    assert saved["swing"].preset_id is CameraCommandId.VIEW_DOWN_THE_LINE
    assert saved["flight"] == before_flight
    qtbot.wait(300)
    settings.sync()
    writes: list[str] = []
    monkeypatch.setattr(compositor, "_persist", lambda: writes.append("write"))
    for _ in range(20):
        swing._advance_camera_tracking()
    qtbot.wait(300)
    assert writes == []

    restored_swing = SimulationView()
    restored = ViewCompositor(
        {
            ViewKind.IMPACT: QLabel("restored-impact"),
            ViewKind.SWING: restored_swing,
            ViewKind.FLIGHT: FlightView(),
        },
        settings,
    )
    qtbot.addWidget(restored)
    assert restored_swing.camera_state().zoom == pytest.approx(2.25)
    assert restored_swing.camera_state().preset_id is CameraCommandId.VIEW_DOWN_THE_LINE
