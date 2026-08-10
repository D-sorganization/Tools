"""Qt interaction coverage for strict imported ground playback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtCore import QSettings  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402

from rate_of_closure.ui.pyqt6 import (
    ground_playback_comparison as comparison_ui,  # noqa: E402
)
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


def _comparison_text(*, time_offset_s: float = 0.0) -> str:
    payload = json.loads(_result_text())
    payload["request_id"] = "comparison-run"
    payload["provenance"]["input_sha256"] = "b" * 64
    for point in payload["trajectory"]:
        point["time_s"] += time_offset_s
    for event in payload["events"]:
        event["time_s"] += time_offset_s
    payload["termination"]["time_s"] += time_offset_s
    return json.dumps(payload)


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
    assert tab.trajectory_table.columnCount() == 13
    assert tab.trajectory_table.horizontalHeaderItem(7).text() == "vx m/s"
    assert tab.trajectory_table.horizontalHeaderItem(12).text() == "ωz rad/s"
    assert tab.trajectory_table.item(0, 7).text() == "0.976"
    assert tab.trajectory_table.item(0, 12).text() == "-2.8103"
    assert tab.events_table.columnCount() == 18
    assert tab.events_table.horizontalHeaderItem(6).text() == "vx before m/s"
    assert tab.events_table.horizontalHeaderItem(17).text() == "ωz after rad/s"
    assert tab.events_table.item(0, 6).text() == "1"
    assert tab.events_table.item(0, 9).text() == "0.976"
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


def test_comparison_import_is_atomic_accessible_and_toggleable(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = GroundPlaybackTab()
    qtbot.addWidget(tab)
    tab.import_json_text(_result_text(), source_name="primary.json")
    primary = tab.timeline

    tab.import_comparison_json_text(_comparison_text(), source_name="comparison.json")
    comparison = tab.comparison
    assert tab.timeline is primary
    assert comparison.comparison.result.request_id == "comparison-run"
    assert tab.show_comparison_checkbox.isChecked()
    assert tab.show_comparison_checkbox.accessibleName() == "Show comparison overlay"
    assert tab.comparison_table.accessibleName() == "Ground result comparison table"
    assert tab.comparison_table.rowCount() == 14
    assert tab.comparison_table.horizontalHeaderItem(3).text() == "Comparison − primary"
    assert (
        tab.comparison_provenance_table.accessibleName()
        == "Ground comparison identity status and provenance"
    )
    assert tab.comparison_provenance_table.rowCount() == 12
    assert tab.comparison_provenance_table.item(0, 1).text() == "surface-run-analytic"
    assert tab.comparison_provenance_table.item(0, 2).text() == "comparison-run"
    assert (
        tab.comparison_trajectory_table.accessibleName()
        == "Ground comparison trajectory evidence"
    )
    assert (
        tab.comparison_events_table.accessibleName()
        == "Ground comparison event evidence"
    )
    assert tab.comparison_trajectory_table.rowCount() == 8
    assert tab.comparison_events_table.rowCount() == 4
    assert tab.comparison_trajectory_table.item(0, 7).text() == "0.976"
    assert tab.comparison_trajectory_table.item(0, 12).text() == "-2.8103"
    assert tab.comparison_events_table.item(0, 6).text() == "1"
    assert tab.comparison_events_table.item(0, 9).text() == "0.976"
    assert tab.export_comparison_trajectory_button.isEnabled()
    assert tab.export_comparison_events_button.isEnabled()
    assert "comparison.json" in tab.comparison_status_label.text()

    tab.show_comparison_checkbox.setChecked(False)
    assert not tab.view.comparison_visible
    assert tab.comparison_trajectory_table.rowCount() == 8
    assert tab.comparison_events_table.rowCount() == 4
    with pytest.raises(ValueError):
        tab.import_comparison_json_text(
            '{"schema_version":"wrong"}', source_name="bad.json"
        )
    assert tab.timeline is primary
    assert tab.comparison is comparison
    assert tab.comparison_trajectory_table.rowCount() == 8
    assert tab.comparison_events_table.rowCount() == 4
    assert "Last valid comparison remains loaded" in tab.comparison_status_label.text()


def test_comparison_raw_evidence_uses_exact_shifted_result_and_canonical_csv(
    qtbot,
) -> None:  # type: ignore[no-untyped-def]
    tab = GroundPlaybackTab()
    qtbot.addWidget(tab)
    tab.import_json_text(_result_text(), source_name="primary.json")
    tab.import_comparison_json_text(
        _comparison_text(time_offset_s=0.2), source_name="comparison.json"
    )

    assert tab.trajectory_table.accessibleName() == "Ground primary trajectory evidence"
    assert tab.events_table.accessibleName() == "Ground primary event evidence"
    assert tab.trajectory_table.item(0, 1).text() == "1.005"
    assert tab.comparison_trajectory_table.item(0, 1).text() == "1.205"
    assert tab.events_table.item(0, 2).text() == "1.005"
    assert tab.comparison_events_table.item(0, 2).text() == "1.205"

    trajectory_csv = tab.comparison_trajectory_csv()
    event_csv = tab.comparison_event_csv()
    assert trajectory_csv.startswith("sample_index,time_s,phase,frame")
    assert event_csv.startswith("sequence,event_type,time_s,frame")
    assert "\r" not in trajectory_csv
    assert "\r" not in event_csv
    assert trajectory_csv.endswith("\n")
    assert event_csv.endswith("\n")
    assert trajectory_csv.splitlines()[1].split(",")[1] == "1.205"
    assert event_csv.splitlines()[1].split(",")[2] == "1.205"


def test_successful_primary_replacement_clears_stale_comparison(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = GroundPlaybackTab()
    qtbot.addWidget(tab)
    tab.import_json_text(_result_text())
    tab.import_comparison_json_text(_comparison_text())

    tab.import_json_text(_result_text(), source_name="replacement.json")

    assert not tab.has_comparison
    assert not tab.show_comparison_checkbox.isEnabled()
    assert tab.comparison_trajectory_table.rowCount() == 0
    assert tab.comparison_events_table.rowCount() == 0
    assert not tab.export_comparison_trajectory_button.isEnabled()
    assert not tab.export_comparison_events_button.isEnabled()
    assert "cleared" in tab.comparison_status_label.text().lower()


def test_comparison_export_failure_preserves_existing_destination(
    qtbot, tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    tab = GroundPlaybackTab()
    qtbot.addWidget(tab)
    tab.import_json_text(_result_text())
    tab.import_comparison_json_text(_comparison_text())
    destination = tmp_path / "ground-comparison.json"
    destination.write_text("last-good", encoding="utf-8")
    monkeypatch.setattr(
        comparison_ui.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(destination), "JSON files (*.json)"),
    )

    def fail_atomic_write(_path: Path, _text: str) -> None:
        raise OSError("simulated commit failure")

    monkeypatch.setattr(comparison_ui, "write_atomic_text", fail_atomic_write)

    tab._save_comparison("ground-comparison.json", tab.comparison_json)

    assert destination.read_text(encoding="utf-8") == "last-good"
    assert "simulated commit failure" in tab.comparison_status_label.text()


def test_comparison_playback_uses_union_absolute_time_and_honest_hold(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = GroundPlaybackTab()
    qtbot.addWidget(tab)
    tab.import_json_text(_result_text())
    tab.import_comparison_json_text(_comparison_text(time_offset_s=0.2))

    target = tab.timeline.end_time_s + 0.1
    tab.set_time(target)

    assert tab.current_time_s == pytest.approx(target)
    assert "primary held at rest" in tab.time_label.text().lower()
    assert "comparison active" in tab.time_label.text().lower()
    assert tab._end_time_s == pytest.approx(tab.comparison.comparison.end_time_s)


def test_comparison_visibility_preserves_union_time_and_workspace_clamps_primary(
    qtbot,
) -> None:  # type: ignore[no-untyped-def]
    tab = GroundPlaybackTab()
    qtbot.addWidget(tab)
    tab.import_json_text(_result_text())
    tab.import_comparison_json_text(_comparison_text(time_offset_s=0.2))
    comparison_end = tab.comparison.end_time_s
    target = tab.timeline.end_time_s + 0.1
    tab.set_time(target)

    tab.show_comparison_checkbox.setChecked(False)

    assert tab.current_time_s == pytest.approx(target)
    assert tab._end_time_s == pytest.approx(comparison_end)
    assert not tab.view.comparison_visible
    workspace = json.loads(tab.workspace_json())
    assert workspace["playback"]["time_s"] == pytest.approx(tab.timeline.end_time_s)
    assert tab.current_time_s == pytest.approx(target)
    hidden_seek = target + 0.01
    tab.set_time(hidden_seek)
    assert tab.current_time_s == pytest.approx(hidden_seek)
    assert "comparison active" in tab.time_label.text().lower()

    tab.show_comparison_checkbox.setChecked(True)
    assert tab.current_time_s == pytest.approx(hidden_seek)
    assert tab._end_time_s == pytest.approx(comparison_end)
    assert tab.view.comparison_visible


def test_comparison_file_error_discloses_retained_last_good_result(
    qtbot, tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    tab = GroundPlaybackTab()
    qtbot.addWidget(tab)
    tab.import_json_text(_result_text())
    tab.import_comparison_json_text(_comparison_text(), source_name="good.json")
    invalid = tmp_path / "bad.json"
    invalid.write_text('{"schema_version":"wrong"}', encoding="utf-8")
    monkeypatch.setattr(
        comparison_ui.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(invalid), "JSON files (*.json)"),
    )

    tab._choose_comparison_file()

    assert tab.has_comparison
    assert "Last valid comparison remains loaded" in tab.comparison_status_label.text()


def test_workspace_import_restores_playback_and_view_atomically(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = GroundPlaybackTab()
    qtbot.addWidget(tab)
    tab.import_json_text(_result_text(), source_name="good.json")
    tab.set_time(1.205)
    tab.speed_combo.setCurrentText("2×")
    tab.loop_checkbox.setChecked(True)
    tab.view.apply_workspace_view(yaw_deg=-37.5, pitch_deg=18.0, zoom=1.75)
    encoded = tab.workspace_json()

    restored = GroundPlaybackTab()
    qtbot.addWidget(restored)
    restored.import_workspace_json_text(encoded, source_name="session.json")
    view = restored.view.workspace_view()

    assert restored.current_time_s == pytest.approx(1.205)
    assert restored.speed_combo.currentText() == "2×"
    assert restored.loop_checkbox.isChecked()
    assert view.yaw_deg == pytest.approx(-37.5)
    assert view.pitch_deg == pytest.approx(18.0)
    assert view.zoom == pytest.approx(1.75)
    assert not restored.playback_timer.isActive()
    assert "workspace" in restored.status_label.text().lower()

    original = restored.timeline
    with pytest.raises(ValueError):
        restored.import_workspace_json_text(
            encoded.replace('"speed":2', '"speed":3'), source_name="bad.json"
        )
    assert restored.timeline is original
    assert restored.current_time_s == pytest.approx(1.205)
    assert restored.speed_combo.currentText() == "2×"
    assert restored.loop_checkbox.isChecked()
    assert restored.view.workspace_view().zoom == pytest.approx(1.75)


def test_persistence_and_export_controls_are_accessible(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = GroundPlaybackTab()
    qtbot.addWidget(tab)

    assert (
        tab.import_workspace_button.accessibleName()
        == "Import ground playback workspace"
    )
    assert not tab.save_workspace_button.isEnabled()
    tab.import_json_text(_result_text())

    assert tab.save_workspace_button.isEnabled()
    assert tab.export_result_button.isEnabled()
    assert tab.export_trajectory_button.isEnabled()
    assert tab.export_events_button.isEnabled()
    assert tab.result_json() == tab.timeline.result.to_json()
    assert tab.trajectory_csv().startswith("sample_index,time_s,phase,frame")
    assert tab.event_csv().startswith("sequence,event_type,time_s,frame")


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
