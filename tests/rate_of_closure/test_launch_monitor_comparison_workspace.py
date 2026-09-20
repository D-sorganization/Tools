"""Tests for launch-monitor side-by-side comparison workspace and accessibility."""

from __future__ import annotations

from typing import Any

import pytest
from PyQt6.QtCore import Qt

from rate_of_closure.ui.pyqt6.launch_monitor_analytics_tab import (
    LaunchMonitorAnalyticsTab,
)
from rate_of_closure.ui.pyqt6.launch_monitor_comparison_workspace import (
    LaunchMonitorComparisonWorkspace,
    build_comparison_rows,
)
from rate_of_closure.ui.pyqt6.launch_monitor_preview import demo_frame
from shared.python.swing_sim.conventions import (
    ParameterId,
)


def test_build_comparison_rows_contains_all_parameters() -> None:
    rows = build_comparison_rows()
    assert len(rows) == len(ParameterId)
    assert len(rows) == 28
    ids = {r.parameter_id for r in rows}
    assert ids == {p.value for p in ParameterId}


def test_build_comparison_rows_computes_signed_delta_only_when_comparable() -> None:
    tm_vals = {
        "ball_speed": 70.0,
        "launch_angle": 12.5,
        "total_spin": 2500.0,
        "club_path": 2.0,
        "launch_direction": 1.0,
        "curve": 5.0,
    }
    fs_vals = {
        "ball_speed": 68.0,
        "launch_angle": 13.0,
        "total_spin": 2400.0,
        "club_path": 1.5,
        "launch_direction": 1.0,
        "curve": 5.0,
    }
    rows = build_comparison_rows(tm_vals, fs_vals)
    by_id = {r.parameter_id: r for r in rows}

    # Ball speed: directly comparable
    bs = by_id["ball_speed"]
    assert bs.is_comparable
    assert bs.difference == pytest.approx(2.0)
    assert bs.difference_text == "+2.00"
    assert bs.reasons == ()
    assert bs.reasons_text == "—"

    # Launch angle: directly comparable
    la = by_id["launch_angle"]
    assert la.is_comparable
    assert la.difference == pytest.approx(-0.5)
    assert la.difference_text == "-0.50"

    # Total spin: directly comparable
    ts = by_id["total_spin"]
    assert ts.is_comparable
    assert ts.difference == pytest.approx(100.0)
    assert ts.difference_text == "+100.00"

    # Club path: NOT comparable (reference point + event time mismatch)
    # MUST NEVER FABRICATE A DELTA!
    cp = by_id["club_path"]
    assert not cp.is_comparable
    assert cp.difference is None
    assert "Not comparable" in cp.difference_text
    assert "reference_point" in cp.reasons_text
    assert "event_time" in cp.reasons_text

    # Launch direction: NOT comparable (sign rule unspecified on Foresight)
    ld = by_id["launch_direction"]
    assert not ld.is_comparable
    assert ld.difference is None
    assert "sign_rule" in ld.reasons_text

    # Curve: NOT comparable (unavailable on Foresight)
    crv = by_id["curve"]
    assert not crv.is_comparable
    assert crv.difference is None
    assert "availability" in crv.reasons_text


def test_comparison_workspace_initialization_and_filtering(qtbot: Any) -> None:
    workspace = LaunchMonitorComparisonWorkspace()
    qtbot.addWidget(workspace)

    assert workspace.table.rowCount() == 28
    assert workspace.table.columnCount() == 12

    # Group filter: Ball Launch (4 parameters: ball_speed,
    # launch_angle, launch_direction, smash_factor)
    workspace.group_combo.setCurrentText("Ball Launch")
    assert workspace.table.rowCount() == 4
    for row_idx in range(workspace.table.rowCount()):
        assert workspace.table.item(row_idx, 0).text() == "Ball Launch"

    # Group filter: All Groups
    workspace.group_combo.setCurrentText("All Groups")
    assert workspace.table.rowCount() == 28

    # Search filter: "spin"
    workspace.search_edit.setText("spin")
    # Matches spin loft, total spin, spin axis, back spin, side spin
    assert workspace.table.rowCount() >= 4

    # Search filter: clear
    workspace.search_edit.setText("")
    assert workspace.table.rowCount() == 28


def test_comparison_workspace_set_dataset_and_export(qtbot: Any) -> None:
    workspace = LaunchMonitorComparisonWorkspace()
    qtbot.addWidget(workspace)

    frame = demo_frame()
    workspace.set_dataset(frame, "Demo Frame")

    # Verify JSON export
    exported_json = workspace.export_json()
    assert exported_json["schema_version"] == "launch-monitor-comparison/v1"
    assert exported_json["source_name"] == "Demo Frame"
    assert len(exported_json["rows"]) == 28
    assert exported_json["total_count"] == 28

    # Verify CSV export
    exported_csv = workspace.export_csv()
    assert "Signed Difference (TM - FS)" in exported_csv
    assert "Ball Speed" in exported_csv
    assert "Club Speed" in exported_csv


def test_comparison_workspace_accessibility_attributes(qtbot: Any) -> None:
    workspace = LaunchMonitorComparisonWorkspace()
    qtbot.addWidget(workspace)

    # Check accessible names and tooltips on interactive controls
    controls = [
        (workspace.group_combo, "Filter Parameter Group"),
        (workspace.search_edit, "Search Comparison Parameters"),
        (workspace.table, "Launch Monitor Side-by-Side Comparison Table"),
        (workspace.export_json_btn, "Export Comparison as JSON"),
        (workspace.export_csv_btn, "Export Comparison as CSV"),
    ]
    for widget, expected_accessible_name in controls:
        assert widget.accessibleName() == expected_accessible_name
        assert len(widget.toolTip().strip()) > 0

    # Focus policies and keyboard accessibility
    assert workspace.search_edit.focusPolicy() != Qt.FocusPolicy.NoFocus
    assert workspace.group_combo.focusPolicy() != Qt.FocusPolicy.NoFocus
    assert workspace.table.focusPolicy() != Qt.FocusPolicy.NoFocus
    assert workspace.export_json_btn.focusPolicy() != Qt.FocusPolicy.NoFocus
    assert workspace.export_csv_btn.focusPolicy() != Qt.FocusPolicy.NoFocus

    # Test keyboard typing into search input
    workspace.search_edit.clear()
    qtbot.keyClicks(workspace.search_edit, "speed")
    assert workspace.search_edit.text() == "speed"
    assert (
        workspace.table.rowCount() == 3
    )  # club_speed, ball_speed, and smash_factor (in definition)
    workspace.search_edit.clear()
    assert workspace.table.rowCount() == 28

    # Verify cell tooltips are populated
    for row_idx in range(min(5, workspace.table.rowCount())):
        for col_idx in range(workspace.table.columnCount()):
            item = workspace.table.item(row_idx, col_idx)
            assert item is not None
            assert len(item.toolTip().strip()) > 0


def test_analytics_tab_compare_mode_integration(qtbot: Any) -> None:
    tab = LaunchMonitorAnalyticsTab()
    qtbot.addWidget(tab)

    # Verify Compare TrackMan / Foresight mode exists in combo
    combo_values = [
        tab.convention_combo.itemData(i) for i in range(tab.convention_combo.count())
    ]
    assert "compare_trackman_foresight" in combo_values

    # Select Compare TrackMan / Foresight
    index = combo_values.index("compare_trackman_foresight")
    tab.convention_combo.setCurrentIndex(index)

    # Verify convention evidence updates gracefully
    assert "Compare TrackMan / Foresight" in tab.convention_evidence.text()
    assert "Side-by-side" in tab.convention_evidence.text()

    # Verify comparison workspace is present in tab
    assert hasattr(tab, "comparison_workspace")
    assert tab.comparison_workspace is not None
    assert tab.comparison_workspace.table.rowCount() == 28
