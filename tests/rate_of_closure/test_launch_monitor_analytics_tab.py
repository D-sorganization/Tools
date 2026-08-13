"""Desktop presentation tests for the Launch Monitor Analytics tab."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from PyQt6.QtCore import Qt

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.launch_monitor_import import (  # noqa: E402
    MAX_IMPORT_BYTES,
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


def test_import_rejects_duplicate_keys_invalid_utf8_and_resource_excess(
    tmp_path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('[{"x":1,"x":2,"y":3}]', encoding="utf-8")
    invalid = tmp_path / "invalid.csv"
    invalid.write_bytes(b"x,y\n1,\xff\n")
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"[" + b" " * MAX_IMPORT_BYTES + b"]")

    with pytest.raises(ValueError, match="Duplicate JSON field"):
        read_launch_monitor_frame(duplicate)
    with pytest.raises(ValueError, match="valid UTF-8"):
        read_launch_monitor_frame(invalid)
    with pytest.raises(ValueError, match="exceeds .* bytes"):
        read_launch_monitor_frame(oversized)

    malformed = tmp_path / "malformed.csv"
    malformed.write_text('x,y\n"unterminated,2', encoding="utf-8")
    with pytest.raises(ValueError, match="CSV is malformed"):
        read_launch_monitor_frame(malformed)

    with pytest.raises(ValueError, match="supports CSV and JSON"):
        read_launch_monitor_frame(tmp_path / "unreadable.txt")


def test_successful_dataset_replacement_resets_all_bound_controls(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = LaunchMonitorAnalyticsTab()
    qtbot.addWidget(tab)
    tab.outcome_combo.setCurrentText("carry_distance")
    tab.group_combo.setCurrentText("session_id")

    tab.set_frame(
        pd.DataFrame(
            {
                "x": list(range(12)),
                "y": [value * value + 1 for value in range(12)],
            }
        ),
        "two.csv",
    )

    assert tab.outcome_combo.currentText() == "x"
    assert [item.text() for item in tab.predictor_list.selectedItems()] == ["y"]
    assert tab.group_combo.currentText() == "(none)"
    assert tab.run_analysis().request.group_by is None

    tab.load_demo()
    result = tab.run_analysis()
    assert result.request.outcome == "ball_speed"
    assert result.request.predictors == ("attack_angle", "club_speed")
    assert result.request.group_by == "monitor_vendor"


def test_failed_programmatic_replacement_preserves_current_dataset(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = LaunchMonitorAnalyticsTab()
    qtbot.addWidget(tab)
    original = tab.frame.copy()
    with pytest.raises(ValueError, match="flat scalar"):
        tab.set_frame(
            pd.DataFrame(
                {
                    "x": [1, 2, 3],
                    "y": [4, 5, 6],
                    "nested": [{"bad": 1}, {"bad": 2}, {"bad": 3}],
                }
            )
        )
    pd.testing.assert_frame_equal(tab.frame, original)
    assert tab.source_name == "Built-In Demonstration Data"


def test_pointer_selection_uses_rendered_pixel_distance(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = LaunchMonitorAnalyticsTab()
    qtbot.addWidget(tab)
    tab.set_frame(pd.DataFrame({"x": [0.0, 10.0, 5.0], "y": [0.0, 100.0, 50.0]}))
    plan = tab.preview_panel.set_frame(tab.frame, "y", ("x",))
    tab.preview.draw()
    display_by_raw = {
        point.raw_index: (tab.preview._plot_x[index], tab.preview._plot_y[index])
        for index, point in enumerate(plan.points)
    }
    disagreement = None
    for x_step in range(-10, 11):
        for y_step in range(-10, 11):
            x_value, y_value = x_step / 10, y_step / 10
            pixel = tab.preview._axes.transData.transform((x_value, y_value))
            rendered = min(
                plan.points,
                key=lambda point: sum(
                    (left - right) ** 2
                    for left, right in zip(
                        tab.preview._axes.transData.transform(
                            display_by_raw[point.raw_index]
                        ),
                        pixel,
                        strict=True,
                    )
                ),
            )
            normalized = min(
                plan.points,
                key=lambda point: (
                    (display_by_raw[point.raw_index][0] - x_value) ** 2
                    + (display_by_raw[point.raw_index][1] - y_value) ** 2
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
            SimpleNamespace(xdata=0.0, ydata=0.0, x=pixel_x, y=pixel_y)
        )

    assert blocker.args == [expected]


def test_generation_reset_preserves_external_listener_and_ignores_old_slot(
    qtbot,
) -> None:  # type: ignore[no-untyped-def]
    tab = LaunchMonitorAnalyticsTab()
    qtbot.addWidget(tab)
    external: list[object] = []
    tab.preview.selection_changed.connect(external.append)
    old_slot = tab.preview_panel._selection_slot
    assert old_slot is not None

    tab.preview_panel.reset_dataset()
    old_slot(4)
    assert tab.preview_panel.selected_raw_index is None

    tab.preview.selection_changed.emit(5)
    assert external == [5]
    assert tab.preview_panel.selected_raw_index == 5


def test_disjoint_numeric_axes_render_an_honest_empty_pair_state(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = LaunchMonitorAnalyticsTab()
    qtbot.addWidget(tab)
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, None, None, None],
            "y": [None, None, None, 4.0, 5.0, 6.0],
        }
    )

    plan = tab.preview_panel.set_frame(frame, "y", ("x",))

    assert plan.finite_count == 0
    assert "Displayed 0 of 0 finite pairs" in tab.preview_status.text()
    assert any(
        "No jointly finite pairs" in text.get_text() for text in tab.preview._axes.texts
    )
