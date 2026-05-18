# ruff: noqa: E501
from typing import Any

"""Tests for the diagnostics tracker and viewer."""


import json
from unittest.mock import MagicMock, patch

import pytest

from double_pendulum_golf.gui.diagnostics import (
    DiagnosticsTracker,
    DiagnosticsViewer,
    get_tracker,
)


@pytest.fixture
def temp_tracker(tmp_path) -> Any:
    """Provide an isolated DiagnosticsTracker."""
    tracker_file = tmp_path / "test_diag.jsonl"
    with patch("double_pendulum_golf.gui.diagnostics._DIAG_FILE", tracker_file):
        with patch("double_pendulum_golf.gui.diagnostics._LOG_DIR", tmp_path):
            tracker = DiagnosticsTracker()
            yield tracker
            tracker.clear()


class TestDiagnosticsTracker:
    def test_singleton_get_tracker(self) -> Any:
        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2

    def test_record_event(self, temp_tracker) -> Any:
        temp_tracker.record(
            "test_cat", "test msg", severity="warning", extra={"k": "v"}
        )
        assert len(temp_tracker.events) == 1
        event = temp_tracker.events[0]
        assert event.category == "test_cat"
        assert event.message == "test msg"
        assert event.severity == "warning"
        assert event.extra == {"k": "v"}

    def test_error_count(self, temp_tracker) -> Any:
        temp_tracker.record("test", "msg")  # default is error
        temp_tracker.record("test", "msg2", severity="critical")
        temp_tracker.record("test", "msg3", severity="info")
        assert temp_tracker.error_count == 2

    def test_record_exception(self, temp_tracker) -> Any:
        try:
            1 / 0
        except ZeroDivisionError as exc:
            temp_tracker.record_exception("math", exc, context="oops")

        events = temp_tracker.events
        assert len(events) == 1
        assert "ZeroDivisionError" in events[0].message
        assert "oops" in events[0].message
        assert "ZeroDivisionError" in events[0].details

    def test_clear_events(self, temp_tracker) -> Any:
        temp_tracker.record("test", "msg")
        assert len(temp_tracker.events) == 1
        temp_tracker.clear()
        assert len(temp_tracker.events) == 0
        assert temp_tracker._file.read_text(encoding="utf-8") == ""

    def test_load_history(self, tmp_path) -> Any:
        diag_file = tmp_path / "diagnostics.jsonl"
        fake_event = {
            "timestamp": "2023-01-01T12:00:00Z",
            "severity": "info",
            "category": "old_cat",
            "message": "old_msg",
        }
        diag_file.write_text(json.dumps(fake_event) + "\n", encoding="utf-8")

        with patch("double_pendulum_golf.gui.diagnostics._DIAG_FILE", diag_file):
            with patch("double_pendulum_golf.gui.diagnostics._LOG_DIR", tmp_path):
                tracker = DiagnosticsTracker()
                assert len(tracker.events) == 1
                assert tracker.events[0].category == "old_cat"

    def test_load_history_handles_corrupt_lines(self, tmp_path) -> Any:
        diag_file = tmp_path / "diagnostics.jsonl"
        content = 'not json\n{"timestamp": "1", "severity": "info", "category": "ok", "message": "m"}\n'
        diag_file.write_text(content, encoding="utf-8")

        with patch("double_pendulum_golf.gui.diagnostics._DIAG_FILE", diag_file):
            with patch("double_pendulum_golf.gui.diagnostics._LOG_DIR", tmp_path):
                tracker = DiagnosticsTracker()
                assert len(tracker.events) == 1
                assert tracker.events[0].category == "ok"


class TestDiagnosticsViewer:
    def test_viewer_init_and_populate(self, temp_tracker, qtbot) -> Any:
        temp_tracker.record("test", "msg1", severity="error")
        temp_tracker.record("test", "msg2", severity="info")

        viewer = DiagnosticsViewer(temp_tracker)
        qtbot.addWidget(viewer)

        assert viewer._table.rowCount() == 2

    def test_filter_combobox_changes(self, temp_tracker, qtbot) -> Any:
        temp_tracker.record("c1", "m1", severity="error")
        temp_tracker.record("c2", "m2", severity="info")

        viewer = DiagnosticsViewer(temp_tracker)
        qtbot.addWidget(viewer)

        viewer._filter_combo.setCurrentText("Info")
        assert viewer._table.rowCount() == 1

        viewer._filter_combo.setCurrentText("All")
        assert viewer._table.rowCount() == 2

    def test_row_selection_shows_details(self, temp_tracker, qtbot) -> Any:
        temp_tracker.record("c1", "m1", details="traceback here", extra={"k": "v"})

        viewer = DiagnosticsViewer(temp_tracker)
        qtbot.addWidget(viewer)

        viewer._table.setCurrentCell(0, 0)

        details_text = viewer._details.toPlainText()
        assert "traceback here" in details_text
        assert "k" in details_text

    def test_clear_button(self, temp_tracker, qtbot) -> Any:
        temp_tracker.record("c1", "m1")
        viewer = DiagnosticsViewer(temp_tracker)
        qtbot.addWidget(viewer)

        viewer._on_clear()

        assert viewer._table.rowCount() == 0
        assert len(temp_tracker.events) == 0

    def test_copy_details(self, temp_tracker, qtbot) -> Any:
        temp_tracker.record("c1", "m", details="trace")
        viewer = DiagnosticsViewer(temp_tracker)
        qtbot.addWidget(viewer)

        viewer._table.setCurrentCell(0, 0)

        with patch(
            "double_pendulum_golf.gui.diagnostics.QApplication.clipboard"
        ) as mock_clip:
            mock_cb = MagicMock()
            mock_clip.return_value = mock_cb
            viewer._copy_details()

            mock_cb.setText.assert_called_once()
            assert "trace" in mock_cb.setText.call_args[0][0]


class TestExceptionHook:
    def test_hook_records_event(self, temp_tracker) -> Any:
        # Patch sys.excepthook before calling _install_exception_hook so it wraps our mock
        with patch("sys.excepthook", MagicMock()) as mock_orig_hook:
            temp_tracker._install_exception_hook()

            import sys

            installed_hook = sys.excepthook

            exc = ValueError("dummy error")
            installed_hook(type(exc), exc, exc.__traceback__)

            assert mock_orig_hook.called

            events = temp_tracker.events
            assert len(events) == 1
            assert events[0].category == "uncaught_exception"
            assert "ValueError" in events[0].message


class TestDiagnosticsGaps:
    def test_show_viewer(self, temp_tracker) -> Any:
        with patch(
            "double_pendulum_golf.gui.diagnostics.DiagnosticsViewer.exec"
        ) as mock_exec:
            temp_tracker.show_viewer()
            mock_exec.assert_called_once()

    def test_flush_oserror(self, temp_tracker) -> Any:
        with patch("pathlib.Path.open", side_effect=OSError):
            # should not crash
            temp_tracker.record("cat", "msg")
            assert len(temp_tracker.events) == 1

    def test_load_history_oserror(self, tmp_path) -> Any:
        import double_pendulum_golf.gui.diagnostics as diag

        with patch.object(diag.Path, "read_text", side_effect=OSError):
            tracker = diag.DiagnosticsTracker()
            assert len(tracker.events) == 0

    def test_load_history_empty_line(self, tmp_path) -> Any:
        import double_pendulum_golf.gui.diagnostics as diag

        diag_file = tmp_path / "diagnostics.jsonl"
        content = "\n"  # empty line
        diag_file.write_text(content, encoding="utf-8")

        with patch("double_pendulum_golf.gui.diagnostics._DIAG_FILE", diag_file):
            with patch("double_pendulum_golf.gui.diagnostics._LOG_DIR", tmp_path):
                tracker = diag.DiagnosticsTracker()
                assert len(tracker.events) == 0

    def test_caller_source_empty(self) -> Any:
        import double_pendulum_golf.gui.diagnostics as diag

        # Requesting a depth much larger than the stack will exhaust frames and hit `return ""`
        assert diag.DiagnosticsTracker._caller_source(depth=9999) == ""

    def test_viewer_bad_timestamp(self, temp_tracker, qtbot) -> Any:
        temp_tracker.record("test", "test", severity="info")
        # Ruin the timestamp
        temp_tracker._events[0].timestamp = "bad_time_string_here"

        viewer = DiagnosticsViewer(temp_tracker)
        qtbot.addWidget(viewer)

        # It should fall back to timestamp[:19]
        assert viewer._table.item(0, 0).text() == "bad_time_string_her"
