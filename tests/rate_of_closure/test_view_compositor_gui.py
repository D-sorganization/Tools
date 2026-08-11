"""Headless persistence and identity tests for the PyQt multi-view host."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtCore import QSettings  # noqa: E402
from PyQt6.QtWidgets import QLabel, QScrollArea  # noqa: E402

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
