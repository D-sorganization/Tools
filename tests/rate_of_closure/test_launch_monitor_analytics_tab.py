"""Desktop presentation tests for the Launch Monitor Analytics tab."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from PyQt6.QtCore import Qt

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.launch_monitor_import import (  # noqa: E402
    read_launch_monitor_frame,
)
from rate_of_closure.ui.pyqt6.launch_monitor_analytics_tab import (  # noqa: E402
    LaunchMonitorAnalyticsTab,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_demo_analysis_populates_results_and_traceability(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = LaunchMonitorAnalyticsTab()
    qtbot.addWidget(tab)

    result = tab.run_analysis()

    assert result.dataset.row_count == 120
    assert tab.result_table.rowCount() >= 3
    assert result.dataset.fingerprint_sha256 in tab.details.toPlainText()
    assert tab.export_result_button.isEnabled()


def test_every_interactive_control_has_accessible_help(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = LaunchMonitorAnalyticsTab()
    qtbot.addWidget(tab)
    controls = (
        tab.import_button,
        tab.demo_button,
        tab.export_data_button,
        tab.export_result_button,
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
    )
    assert all(control.accessibleName() for control in controls)
    assert all(control.toolTip() for control in controls)


def test_linked_scatter_keyboard_selection_does_not_recompute_analysis(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = LaunchMonitorAnalyticsTab()
    qtbot.addWidget(tab)
    result = tab.run_analysis()
    trace = tab.details.toPlainText()

    tab.preview.setFocus()
    qtbot.keyPress(tab.preview, Qt.Key.Key_End)

    assert tab.preview_status.text().startswith("Displayed 120 of 120 finite pairs")
    assert "Retained row index 119 (zero-based)" in tab.preview_status.text()
    assert "shot demo-120" in tab.preview_status.text()
    assert tab.last_result is result
    assert tab.details.toPlainText() == trace
    assert len(tab.preview._axes.lines) == 1

    qtbot.keyPress(tab.preview, Qt.Key.Key_Escape)
    assert tab.preview_panel.selected_raw_index is None
    assert "No retained source row selected" in tab.preview_status.text()


def test_analysis_contract_change_invalidates_stale_results(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = LaunchMonitorAnalyticsTab()
    qtbot.addWidget(tab)
    old_result = tab.run_analysis()

    tab.outcome_combo.setCurrentText("carry_distance")

    assert old_result.request.outcome == "ball_speed"
    assert tab.last_result is None
    assert not tab.export_result_button.isEnabled()
    assert tab.result_table.rowCount() == 0
    assert "contract changed" in tab.details.toPlainText()
    new_result = tab.run_analysis()
    assert new_result.request.outcome == "carry_distance"
    assert tab.export_result_button.isEnabled()


def test_mixed_decimal_text_axis_is_plotted_without_silent_axis_replacement(
    qtbot,
) -> None:  # type: ignore[no-untyped-def]
    tab = LaunchMonitorAnalyticsTab()
    qtbot.addWidget(tab)
    tab.set_frame(
        pd.DataFrame(
            {
                "string_axis": ["1.0", "2.5", "3e0"],
                "outcome": [4.0, 5.0, 6.0],
                "other": [7.0, 8.0, 9.0],
            }
        )
    )
    tab.outcome_combo.setCurrentText("outcome")
    for index in range(tab.predictor_list.count()):
        item = tab.predictor_list.item(index)
        if item is not None:
            item.setSelected(item.text() == "string_axis")

    plan = tab.preview_panel.set_frame(tab.frame, "outcome", ("string_axis",))

    assert plan.x_field == "string_axis"
    assert [point.x for point in plan.points] == [1.0, 2.5, 3.0]


def test_import_rejects_nested_json_and_ragged_csv(tmp_path, qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = LaunchMonitorAnalyticsTab()
    qtbot.addWidget(tab)
    nested = tmp_path / "nested.json"
    nested.write_text('[{"x":{"nested":1},"y":2}]', encoding="utf-8")
    ragged = tmp_path / "ragged.csv"
    ragged.write_text("x,y\n1,2,3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="portable finite scalars"):
        tab.import_path(nested)
    with pytest.raises(ValueError, match="match the header width"):
        tab.import_path(ragged)


def test_import_coercion_and_json_edges_match_browser_policy(tmp_path) -> None:
    csv_path = tmp_path / "portable.csv"
    csv_path.write_text(
        "x,y,label\n\n1,2, alpha \n1.5e2,3,\n0x10,4,hex\n",
        encoding="utf-8",
    )

    assert read_launch_monitor_frame(csv_path).to_dict(orient="records") == [
        {"x": 1, "y": 2, "label": "alpha"},
        {"x": 150, "y": 3, "label": None},
        {"x": "0x10", "y": 4, "label": "hex"},
    ]

    bodies = (
        '[{"x":NaN}]',
        '[{"x":Infinity}]',
        '[{"x":1e20}]',
        '[{"x":' + str(10**1000) + "}]",
    )
    for body in bodies:
        path = tmp_path / "invalid.json"
        path.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError, match="constant|portable finite"):
            read_launch_monitor_frame(path)
    empty_key = tmp_path / "empty-key.json"
    empty_key.write_text('[{"":1,"y":2}]', encoding="utf-8")
    with pytest.raises(ValueError, match="field names must be non-empty"):
        read_launch_monitor_frame(empty_key)
    unsupported = tmp_path / "shots.txt"
    unsupported.write_text("x,y\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="supports CSV and JSON"):
        read_launch_monitor_frame(unsupported)


def test_pointer_selection_uses_rendered_pixel_distance(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = LaunchMonitorAnalyticsTab()
    qtbot.addWidget(tab)
    tab.set_frame(pd.DataFrame({"x": [0.0, 10.0, 5.0], "y": [0.0, 100.0, 50.0]}))
    plan = tab.preview_panel.set_frame(tab.frame, "y", ("x",))
    tab.preview.draw()
    disagreement = None
    for x_value in range(11):
        for y_value in range(0, 101, 5):
            pixel = tab.preview._axes.transData.transform((x_value, y_value))
            rendered = min(
                plan.points,
                key=lambda point: sum(
                    (left - right) ** 2
                    for left, right in zip(
                        tab.preview._axes.transData.transform((point.x, point.y)),
                        pixel,
                        strict=True,
                    )
                ),
            )
            normalized = min(
                plan.points,
                key=lambda point: (
                    ((point.x - x_value) / 10) ** 2 + ((point.y - y_value) / 100) ** 2
                ),
            )
            if rendered.raw_index != normalized.raw_index:
                disagreement = (pixel, rendered.raw_index)
                break
        if disagreement is not None:
            break
    assert disagreement is not None
    (pixel_x, pixel_y), expected = disagreement

    with qtbot.waitSignal(tab.preview.selection_changed) as blocker:
        tab.preview._select_nearest(
            SimpleNamespace(xdata=1.0, ydata=1.0, x=pixel_x, y=pixel_y)
        )

    assert blocker.args == [expected]
