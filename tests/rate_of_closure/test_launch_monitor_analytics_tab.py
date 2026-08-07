"""Desktop presentation tests for the Launch Monitor Analytics tab."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.launch_monitor_player_metrics import (  # noqa: E402
    analyze_sessions,
    calculate_strokes_gained_proxy,
)
from rate_of_closure.ui.pyqt6.launch_monitor_analytics_tab import (  # noqa: E402
    LaunchMonitorAnalyticsTab,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_demo_analysis_populates_results_and_traceability(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = LaunchMonitorAnalyticsTab(auto_discover_campaign=False)
    qtbot.addWidget(tab)

    result = tab.run_analysis()

    assert result.dataset.row_count == 120
    assert tab.result_table.rowCount() >= 3
    assert result.dataset.fingerprint_sha256 in tab.details.toPlainText()
    assert tab.export_result_button.isEnabled()


def test_every_interactive_control_has_accessible_help(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = LaunchMonitorAnalyticsTab(auto_discover_campaign=False)
    qtbot.addWidget(tab)
    controls = (
        tab.import_button,
        tab.demo_button,
        tab.export_data_button,
        tab.export_result_button,
        tab.dataset_combo,
        tab.refresh_button,
        tab.save_project_button,
        tab.load_project_button,
        tab.export_plot_button,
        tab.export_plot_data_button,
        tab.convention_combo,
        tab.outcome_combo,
        tab.predictor_list,
        tab.mode_combo,
        tab.method_combo,
        tab.missing_combo,
        tab.group_combo,
        tab.confidence_spin,
        tab.min_samples_spin,
        tab.run_button,
        tab.player_controls.plot_mode_combo,
        tab.player_controls.lateral_combo,
        tab.player_controls.carry_combo,
        tab.player_controls.session_combo,
        tab.player_controls.player_combo,
        tab.player_controls.target_distance_spin,
        tab.player_controls.start_lie_combo,
        tab.player_controls.end_lie_combo,
        tab.data_preview,
    )
    assert all(control.accessibleName() for control in controls)
    assert all(control.toolTip() for control in controls)


def test_campaign_is_loaded_by_default_when_available(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = LaunchMonitorAnalyticsTab()
    qtbot.addWidget(tab)

    if tab.catalog is None:
        pytest.skip("private campaign repository is not present")
    assert tab.dataset_id == "normalized"
    assert len(tab.frame) == 10_169
    assert tab.dataset_combo.count() == len(tab.catalog.datasets) + 1
    assert tab.source_sha256
    assert tab.data_preview.rowCount() == 500


def test_dispersion_plot_uses_signed_yards_and_backing_rows(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = LaunchMonitorAnalyticsTab(auto_discover_campaign=False)
    qtbot.addWidget(tab)
    tab.player_controls.plot_mode_combo.setCurrentText("Directional Dispersion")

    tab.run_analysis()

    assert "lateral_yd" in tab.plot_widget.backing_data
    assert len(tab.plot_widget.backing_data) == 120
    assert tab.player_payload["mode"] == "Directional Dispersion"
    assert "left" in tab.details.toPlainText()
    summary = tab.player_payload["summary"]
    assert "absolute_p50_yd" in summary
    assert "ellipse_major_radius_yd" in summary
    assert tab.plot_widget.figure.axes[0].patches
    assert {
        "absolute_p50_yd",
        "absolute_p80_yd",
        "ellipse_major_radius_yd",
        "ellipse_minor_radius_yd",
        "ellipse_angle_deg",
    }.issubset(tab.plot_widget.backing_data)


def test_relationship_uses_dataset_unit_and_rejects_mixed_units(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = LaunchMonitorAnalyticsTab(auto_discover_campaign=False)
    qtbot.addWidget(tab)
    frame = pd.DataFrame({"input": [1.0, 2.0], "bias": [3.0, 4.0], "unit": ["m", "m"]})

    tab.plot_widget.plot_relationship(frame, "input", "bias")

    assert tab.plot_widget.figure.axes[0].get_ylabel() == "Bias (m)"
    frame.loc[1, "unit"] = "deg"
    with pytest.raises(ValueError, match="mixed units.*filter"):
        tab.plot_widget.plot_relationship(frame, "input", "bias")


def test_strokes_gained_plot_labels_lateral_color_and_clamping(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = LaunchMonitorAnalyticsTab(auto_discover_campaign=False)
    qtbot.addWidget(tab)
    analysis = calculate_strokes_gained_proxy(
        pd.DataFrame({"carry_yd": [10.0, 900.0], "lateral_yd": [1.0, 2.0]}),
        carry_column="carry_yd",
        lateral_column="lateral_yd",
        target_distance_yd=700.0,
        start_lie="tee",
        end_lie="fairway",
    )

    tab.plot_widget.plot_strokes_gained(analysis)

    assert len(tab.plot_widget.figure.axes) == 2
    assert tab.plot_widget.figure.axes[1].get_ylabel().endswith("right +)")
    assert "Benchmark clamped: 2/2 (100.0%)" in {
        text.get_text() for text in tab.plot_widget.figure.axes[0].texts
    }


def test_session_plot_keeps_player_lines_separate(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = LaunchMonitorAnalyticsTab(auto_discover_campaign=False)
    qtbot.addWidget(tab)
    frame = pd.DataFrame(
        {
            "player_id": ["A", "A", "B", "B"],
            "session_id": ["one", "two", "one", "two"],
            "ball_speed_mph": [100.0, 110.0, 200.0, 190.0],
        }
    )
    analysis = analyze_sessions(
        frame,
        metric_column="ball_speed_mph",
        session_column="session_id",
        player_column="player_id",
    )

    tab.plot_widget.plot_sessions(
        analysis,
        "ball_speed_mph",
        player_column="player_id",
        source_frame=frame,
    )

    handles, labels = tab.plot_widget.figure.axes[0].get_legend_handles_labels()
    assert len(handles) == 2
    assert labels == ["A (slope 10/session)", "B (slope -10/session)"]
    assert [list(handle.lines[0].get_xdata()) for handle in handles] == [
        [1, 2],
        [1, 2],
    ]


def test_imported_project_reloads_data_hash_and_full_state(
    qtbot, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "player.csv"
    pd.DataFrame(
        {
            "session_id": ["a", "a", "b"],
            "carry_m": [1.0, 2.0, 3.0],
            "speed_mph": [4.0, 5.0, 6.0],
        }
    ).to_csv(path, index=False)
    original = LaunchMonitorAnalyticsTab(auto_discover_campaign=False)
    qtbot.addWidget(original)
    original.import_path(path)
    original.mode_combo.setCurrentText("regression")
    original.method_combo.setCurrentText("spearman")
    original.missing_combo.setCurrentText("listwise")
    original.confidence_spin.setValue(0.91)
    original.min_samples_spin.setValue(3)
    original.player_controls.plot_mode_combo.setCurrentText("Session Trend")
    project = original._project()

    restored = LaunchMonitorAnalyticsTab(auto_discover_campaign=False)
    qtbot.addWidget(restored)
    restored._load_project(project)

    assert restored.frame.equals(original.frame)
    assert restored.data_path == str(path.resolve())
    assert restored.source_sha256 == project.dataset_sha256
    assert restored.mode_combo.currentText() == "regression"
    assert restored.method_combo.currentText() == "spearman"
    assert restored.missing_combo.currentText() == "listwise"
    assert restored.confidence_spin.value() == pytest.approx(0.91)
    assert restored.min_samples_spin.value() == 3
    assert restored.player_controls.plot_mode_combo.currentText() == "Session Trend"


def test_project_hash_failure_does_not_mutate_current_data(
    qtbot, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "player.csv"
    pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}).to_csv(path, index=False)
    source = LaunchMonitorAnalyticsTab(auto_discover_campaign=False)
    qtbot.addWidget(source)
    source.import_path(path)
    project = source._project()
    path.write_text("x,y\n9,4\n2,5\n3,6\n", encoding="utf-8")
    target = LaunchMonitorAnalyticsTab(auto_discover_campaign=False)
    qtbot.addWidget(target)
    before = target.frame.copy()

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        target._load_project(project)

    assert target.dataset_id == "demo"
    assert target.frame.equals(before)
