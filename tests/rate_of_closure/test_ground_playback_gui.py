"""Qt interaction coverage for strict imported ground playback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtCore import QSettings  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402

from rate_of_closure.ui.pyqt6.ground_playback_tab import GroundPlaybackTab  # noqa: E402
from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

FIXTURE = (
    Path(__file__).parents[2]
    / "src/rate_of_closure/web/src/model/__fixtures__"
    / "ground_reference_pipeline_golden_v1.json"
)


def _result_text() -> str:
    return json.dumps(json.loads(FIXTURE.read_text(encoding="utf-8"))["result"])


def test_tab_imports_strict_result_and_exposes_accessible_controls(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = GroundPlaybackTab()
    qtbot.addWidget(tab)

    assert "does not execute ground physics" in tab.disclosure_label.text()
    assert not tab.play_button.isEnabled()
    tab.import_json_text(_result_text(), source_name="golden.json")

    assert tab.play_button.isEnabled()
    assert tab.phase_label.text() == "Impact"
    assert tab.summary_table.item(0, 0).text() == "Carry"
    assert tab.summary_table.item(1, 0).text() == "Total"
    assert tab.trajectory_table.rowCount() == 8
    assert tab.warnings_table.rowCount() == 6
    assert tab.warnings_table.item(3, 0).text() == "provenance"
    assert tab.play_button.accessibleName() == "Play ground result"


def test_invalid_import_retains_last_good_result(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = GroundPlaybackTab()
    qtbot.addWidget(tab)
    tab.import_json_text(_result_text(), source_name="good.json")
    original = tab.timeline

    with pytest.raises(ValueError):
        tab.import_json_text('{"schema_version":"wrong"}', source_name="bad.json")

    assert tab.timeline is original
    assert tab.status_label.property("state") == "error"
    assert "bad.json" in tab.status_label.text()


def test_play_pause_step_phase_jump_restart_and_loop(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = GroundPlaybackTab()
    qtbot.addWidget(tab)
    tab.import_json_text(_result_text())

    tab.jump_to_phase("roll")
    assert tab.phase_label.text() == "Roll"
    roll_time = tab.current_time_s
    tab.previous_frame()
    assert tab.current_time_s < roll_time
    tab.next_frame()
    assert tab.current_time_s == pytest.approx(roll_time)
    tab.restart()
    assert tab.current_time_s == pytest.approx(tab.timeline.start_time_s)

    tab.loop_checkbox.setChecked(True)
    tab.set_time(tab.timeline.end_time_s)
    tab.play()
    assert tab.current_time_s == pytest.approx(tab.timeline.start_time_s)
    assert tab.playback_timer.isActive()
    QTest.qWait(25)
    tab.pause()
    assert not tab.playback_timer.isActive()


def test_ground_module_is_discoverable_for_existing_qt_navigation(
    qtbot, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    settings = QSettings(str(tmp_path / "navigation.ini"), QSettings.Format.IniFormat)
    settings.setValue(
        "ui/primary-tabs/v1",
        json.dumps(
            {
                "version": 1,
                "order": ["clubhead", "plots"],
                "visible": ["clubhead", "plots"],
                "active": "clubhead",
            }
        ),
    )
    window = RateOfClosureMainWindow(navigation_settings=settings)
    qtbot.addWidget(window)
    try:
        assert "ground_playback" in window.primary_tab_ids()
        assert "ground_playback" in window.visible_primary_tab_ids()
        assert window._tabs.indexOf(window._ground_playback_tab) >= 0
    finally:
        window.close()
