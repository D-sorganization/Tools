"""Headless persistence and identity tests for the PyQt multi-view host."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtCore import QSettings  # noqa: E402
from PyQt6.QtWidgets import QLabel  # noqa: E402

from rate_of_closure.ui.pyqt6.view_compositor import ViewCompositor  # noqa: E402
from rate_of_closure.view_workspace import ViewKind  # noqa: E402

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
