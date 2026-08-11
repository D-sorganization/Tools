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
from rate_of_closure.ui.pyqt6.ground_playback_tables import (  # noqa: E402
    MAX_VISIBLE_GROUND_ROWS,
    evidence_window,
)
from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

FIXTURE = (
    Path(__file__).parents[2]
    / "src/rate_of_closure/web/src/model/__fixtures__"
    / "flight_to_ground_golden_v1.json"
)
REGIONAL_FIXTURE = FIXTURE.with_name("ground_regional_execution_golden_v1.json")


def _result_text() -> str:
    return json.dumps(json.loads(FIXTURE.read_text(encoding="utf-8"))["result"])


def _regional_text(name: str = "representable") -> str:
    payload = json.loads(REGIONAL_FIXTURE.read_text(encoding="utf-8"))
    return json.dumps(payload[name]["result"])


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
    assert tab.trajectory_table.rowCount() == 4
    assert tab.trajectory_table.columnCount() == 13
    assert tab.trajectory_table.horizontalHeaderItem(7).text() == "vx m/s"
    assert tab.trajectory_table.horizontalHeaderItem(12).text() == "ωz rad/s"
    assert tab.trajectory_table.item(0, 7).text() == "24"
    assert tab.trajectory_table.item(0, 12).text() == "-2"
    assert tab.events_table.columnCount() == 18
    assert tab.events_table.horizontalHeaderItem(6).text() == "vx before m/s"
    assert tab.events_table.horizontalHeaderItem(17).text() == "ωz after rad/s"
    assert tab.events_table.item(0, 6).text() == "31"
    assert tab.events_table.item(0, 9).text() == "24"
    evidence = {
        (
            tab.warnings_table.item(row, 0).text(),
            tab.warnings_table.item(row, 1).text(),
        ): tab.warnings_table.item(row, 2).text()
        for row in range(tab.warnings_table.rowCount())
    }
    assert evidence[("identity", "status")] == "complete"
    assert evidence[("provenance", "input SHA-256")] == "a" * 64
    assert evidence[("calibration", "calibration ID")] == ("literature-default-2026-08")
    assert evidence[("calibration", "confidence")] == "0.60"
    assert tab.play_button.accessibleName() == "Play ground result"
    assert tab.regional_import_button.accessibleName() == (
        "Import strict regional ground execution JSON"
    )


def test_tab_imports_validated_regional_envelope_and_reuses_nested_result(
    qtbot,
) -> None:  # type: ignore[no-untyped-def]
    tab = GroundPlaybackTab()
    qtbot.addWidget(tab)
    tab.import_regional_json_text(_regional_text(), source_name="regional.json")

    assert tab.timeline.result.request_id == "surface-run-analytic"
    assert "regional.json" in tab.status_label.text()

    previous = tab.timeline
    with pytest.raises(ValueError, match="playable ground result"):
        tab.import_regional_json_text(
            _regional_text("failed"), source_name="failed.json"
        )
    assert tab.timeline is previous


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


def test_playback_uses_monotonic_time_speed_and_loop_overshoot(qtbot) -> None:  # type: ignore[no-untyped-def]
    now = [100.0]
    tab = GroundPlaybackTab(clock=lambda: now[0])
    qtbot.addWidget(tab)
    tab.import_json_text(_result_text())

    tab.play()
    now[0] += 0.25
    tab._advance()
    assert tab.current_time_s == pytest.approx(tab.timeline.start_time_s + 0.25)

    tab.speed_combo.setCurrentText("2×")
    now[0] += 0.10
    tab._advance()
    assert tab.current_time_s == pytest.approx(tab.timeline.start_time_s + 0.45)

    tab.pause()
    tab.loop_checkbox.setChecked(True)
    tab.set_time(tab.timeline.end_time_s - 0.05)
    tab.play()
    now[0] += 0.10
    tab._advance()
    assert tab.current_time_s == pytest.approx(tab.timeline.start_time_s + 0.15)
    assert tab.playback_timer.isActive()

    tab.speed_combo.setCurrentText("1×")
    tab.loop_checkbox.setChecked(False)
    now[0] += 0.01
    tab._advance()
    assert tab.current_time_s == pytest.approx(tab.timeline.start_time_s + 0.16)
    assert tab.playback_timer.isActive()


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


def test_large_evidence_window_consumes_only_the_visible_budget() -> None:
    consumed = 0

    def rows():
        nonlocal consumed
        for index in range(100_000):
            consumed += 1
            yield (index,)

    window = evidence_window(rows(), total=100_000)

    assert len(window.rows) == MAX_VISIBLE_GROUND_ROWS
    assert consumed == MAX_VISIBLE_GROUND_ROWS
    assert window.disclosure == (
        "Showing first 256 of 100000 validated rows; full result retained."
    )
