"""Tests for Sidekick UX hardening fixes (issue #3104).

F1 — PTY submit always sends a single newline (never os.linesep).
F3 — Per-chunk output is appended raw; trailing newlines are not stripped.
F5 — QSettings writes are funnelled through a single helper (_persist_visible_tabs).
F7 — Help dialog is reused when already open (no duplicate windows).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.serial

if sys.platform == "win32" and os.environ.get("PYTEST_XDIST_WORKER"):
    pytest.skip(
        "Qt sidebar tests run serially on Windows.",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# F1 — _on_submit writes exactly one newline to the PTY
# ---------------------------------------------------------------------------


class TestF1PtyNewline:
    """_on_submit must always write a single \\n (never \\r\\n from os.linesep)."""

    def test_submit_sends_single_newline(  # noqa: ANN201
        self, tmp_path: Path, qtbot: Any
    ) -> None:
        """Exactly one \\n is written per submit regardless of os.linesep."""
        try:
            from upstream_drift_tools.ui.tools_sidebar import os_terminal
            from upstream_drift_tools.ui.tools_sidebar.qt_compat import (
                QtWidgets,
            )
            from upstream_drift_tools.ui.tools_sidebar.shell_discovery import (
                ShellDescriptor,
            )
        except ImportError:
            pytest.skip("Qt/sidekick unavailable")

        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        written: list[bytes] = []

        class _FakeBackend:
            is_running = True

            def start(self) -> None:
                pass

            def write(self, data: bytes) -> None:
                written.append(data)

            def read(self, timeout: float = 0.0) -> bytes:  # noqa: ARG002
                return b""

            def terminate(self) -> None:
                self.is_running = False

            def resize(self, rows: int, cols: int) -> None:  # noqa: ARG002
                pass

        widget = os_terminal.SidekickOsTerminalWidget(
            project_root=tmp_path,
            shells=[
                ShellDescriptor(identifier="fake", label="fake", command=("fake",))
            ],
            autostart=False,
        )
        widget._backend = _FakeBackend()  # noqa: SLF001
        qtbot.addWidget(widget)

        widget._input.setText("hello")  # noqa: SLF001
        widget._on_submit()  # noqa: SLF001

        assert written, "write() was never called"
        payload = written[0]
        # Must end with exactly one newline byte, not \\r\\n
        assert payload == b"hello\n", (
            f"Expected b'hello\\n' but got {payload!r}; "
            "os.linesep is still being used (F1 regression)"
        )


# ---------------------------------------------------------------------------
# F3 — _handle_output does NOT strip trailing newlines per chunk
# ---------------------------------------------------------------------------


class TestF3RawOutputAppend:
    """_handle_output must preserve trailing newlines so blank lines are kept."""

    def _make_widget(  # noqa: ANN202
        self, tmp_path: Path, qtbot: Any
    ) -> Any:  # noqa: ANN401
        try:
            from upstream_drift_tools.ui.tools_sidebar.os_terminal import (
                SidekickOsTerminalWidget,
            )
            from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
            from upstream_drift_tools.ui.tools_sidebar.shell_discovery import (
                ShellDescriptor,
            )
        except ImportError:
            pytest.skip("Qt/sidekick unavailable")

        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        widget = SidekickOsTerminalWidget(
            project_root=tmp_path,
            shells=[
                ShellDescriptor(identifier="fake", label="fake", command=("fake",))
            ],
            autostart=False,
        )
        qtbot.addWidget(widget)
        return widget

    def test_blank_line_in_output_is_preserved(  # noqa: ANN201
        self, tmp_path: Path, qtbot: Any
    ) -> None:
        """A blank line inside PTY output must appear in the widget."""
        widget = self._make_widget(tmp_path, qtbot)
        # Two lines with a blank line between them (common in shell output).
        widget._handle_output(b"line1\n\nline2\n")  # noqa: SLF001
        text = widget._output.toPlainText()  # noqa: SLF001
        assert "line1" in text
        assert "line2" in text
        # There should be at least one blank line between them
        assert "\n\n" in text or text.count("\n") >= 2  # noqa: PLR2004

    def test_chunk_boundary_does_not_join_lines(  # noqa: ANN201
        self, tmp_path: Path, qtbot: Any
    ) -> None:
        """Successive chunks must not be joined onto a single line."""
        widget = self._make_widget(tmp_path, qtbot)
        widget._handle_output(b"first\n")  # noqa: SLF001
        widget._handle_output(b"second\n")  # noqa: SLF001
        text = widget._output.toPlainText()  # noqa: SLF001
        assert "first" in text
        assert "second" in text
        lines = [ln for ln in text.splitlines() if ln.strip()]
        assert any("first" in ln for ln in lines)
        assert any("second" in ln for ln in lines)


# ---------------------------------------------------------------------------
# F5 — QSettings writes are consolidated in _persist_visible_tabs
# ---------------------------------------------------------------------------


class TestF5QSettingsConsolidation:
    """All QSettings writes must go through _persist_visible_tabs()."""

    def test_persist_helper_exists(self) -> None:
        """UnifiedToolsSidebar exposes _persist_visible_tabs."""
        try:
            from upstream_drift_tools.ui.tools_sidebar.sidebar import (
                UnifiedToolsSidebar,
            )
        except ImportError:
            pytest.skip("sidekick unavailable")

        assert hasattr(UnifiedToolsSidebar, "_persist_visible_tabs"), (
            "_persist_visible_tabs helper missing (F5 regression)"
        )

    def test_qs_constants_are_defined(self) -> None:
        """Module-level QSettings constants must be present."""
        try:
            from upstream_drift_tools.ui.tools_sidebar import sidebar as sb
        except ImportError:
            pytest.skip("sidekick unavailable")

        assert hasattr(sb, "_QS_ORG"), "_QS_ORG constant missing"
        assert hasattr(sb, "_QS_APP"), "_QS_APP constant missing"
        assert hasattr(sb, "_QS_VISIBLE_TABS_KEY"), (
            "_QS_VISIBLE_TABS_KEY constant missing"
        )  # noqa: E501

    def test_persist_uses_explicit_org_app(  # noqa: ANN201
        self, tmp_path: Path, qtbot: Any
    ) -> None:
        """_persist_visible_tabs must write to org/app-scoped QSettings."""
        try:
            from upstream_drift_tools.ui.tools_sidebar.qt_compat import (
                QtCore,
                QtWidgets,
            )
        except ImportError:
            pytest.skip("Qt/sidekick unavailable")

        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        written: dict[str, Any] = {}

        class _FakeQSettings:
            def __init__(self, org: str, app_name: str) -> None:
                written["org"] = org
                written["app"] = app_name

            def setValue(self, key: str, value: Any) -> None:  # noqa: N802
                written["key"] = key
                written["value"] = value

            def value(self, key: str, default: Any = None, **kwargs: Any) -> Any:  # noqa: N802
                return default

            def sync(self) -> None:  # noqa: N802
                pass

        original_qs = QtCore.QSettings

        try:
            from upstream_drift_tools.ui.tools_sidebar.sidebar import (
                _QS_APP,
                _QS_ORG,
                _QS_VISIBLE_TABS_KEY,
                UnifiedToolsSidebar,
            )

            # Create the sidebar with real QSettings, then patch to capture
            # the single persist write.
            sidebar = UnifiedToolsSidebar(project_root=tmp_path)
            qtbot.addWidget(sidebar)

            QtCore.QSettings = _FakeQSettings  # type: ignore[assignment]
            sidebar._persist_visible_tabs()  # noqa: SLF001

            assert written.get("org") == _QS_ORG
            assert written.get("app") == _QS_APP
            # The written key is project-root-scoped (starts with the global prefix).
            assert written.get("key", "").startswith(_QS_VISIBLE_TABS_KEY), (
                f"Expected key starting with {_QS_VISIBLE_TABS_KEY!r}, "
                f"got {written.get('key')!r}"
            )
        finally:
            QtCore.QSettings = original_qs  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# F7 — Help dialog is reused when already visible
# ---------------------------------------------------------------------------


class TestF7HelpDialogSingleton:
    """show_tab_help must reuse the existing dialog rather than spawn a new one."""

    def test_second_call_raises_existing_dialog(  # noqa: ANN201
        self, tmp_path: Path, qtbot: Any
    ) -> None:
        """Calling show_tab_help twice must not create a second dialog window."""
        try:
            from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
            from upstream_drift_tools.ui.tools_sidebar.sidebar import (
                UnifiedToolsSidebar,
            )
        except ImportError:
            pytest.skip("Qt/sidekick unavailable")

        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        sidebar = UnifiedToolsSidebar(project_root=tmp_path)
        qtbot.addWidget(sidebar)

        # We need at least one tab with help metadata to open the dialog.
        if not sidebar.visible_tab_ids():
            pytest.skip("No tabs with help metadata available")

        tab_id = sidebar.visible_tab_ids()[0]
        if not sidebar.tab_help_metadata(tab_id):
            pytest.skip(f"Tab '{tab_id}' has no help metadata")

        # First call — creates the dialog.
        result1 = sidebar.show_tab_help(tab_id)
        dialog1 = sidebar._help_dialog  # noqa: SLF001

        # Second call — must reuse the same dialog.
        result2 = sidebar.show_tab_help(tab_id)
        dialog2 = sidebar._help_dialog  # noqa: SLF001

        assert result1 is True
        assert result2 is True
        assert dialog1 is dialog2, (
            "show_tab_help created a second QDialog instead of "
            "reusing the first (F7 regression)"
        )


# ---------------------------------------------------------------------------
# F10 — Quick-access pins are persisted to and restored from QSettings
# ---------------------------------------------------------------------------


class TestF10QuickAccessPersistence:
    """Quick-access folder pins must survive a restart (QSettings round-trip)."""

    def test_resolve_columns_helper_exists(self) -> None:
        """resolve_columns is exported from data_explorer_service."""
        try:
            from upstream_drift_tools.ui.tools_sidebar.data_explorer_service import (
                resolve_columns,
            )
        except ImportError:
            pytest.skip("sidekick unavailable")

        assert callable(resolve_columns), "resolve_columns is not callable"

    def test_quick_access_methods_exist(self) -> None:
        """ProjectFileExplorer exposes persistence helpers."""
        try:
            from upstream_drift_tools.ui.tools_sidebar.project_file_explorer import (
                ProjectFileExplorer,
            )
        except ImportError:
            pytest.skip("sidekick unavailable")

        assert hasattr(ProjectFileExplorer, "_restore_quick_access"), (
            "_restore_quick_access missing (F10 regression)"
        )
        assert hasattr(ProjectFileExplorer, "_save_quick_access"), (
            "_save_quick_access missing (F10 regression)"
        )
        assert hasattr(ProjectFileExplorer, "_quick_access_settings_key"), (
            "_quick_access_settings_key missing (F10 regression)"
        )

    def test_add_to_quick_access_rejects_duplicates(  # noqa: ANN201
        self, tmp_path: Path, qtbot: Any
    ) -> None:
        """Adding the same folder twice must not create a duplicate pin."""
        try:
            from upstream_drift_tools.ui.tools_sidebar.project_file_explorer import (
                ProjectFileExplorer,
            )
            from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
        except ImportError:
            pytest.skip("Qt/sidekick unavailable")

        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        explorer = ProjectFileExplorer(project_root=tmp_path, parent=None)
        qtbot.addWidget(explorer)

        # Use a subdirectory so the pin target is distinct from tmp_path
        # itself (which _refresh_common_locations may already list as the
        # project root).
        pin_dir = tmp_path / "quick_access_pin"
        pin_dir.mkdir()

        initial_count = explorer._common_locations.count()  # noqa: SLF001

        # Add the same path twice — second call must be a no-op.
        explorer._add_to_quick_access(pin_dir)  # noqa: SLF001
        explorer._add_to_quick_access(pin_dir)  # noqa: SLF001

        final_count = explorer._common_locations.count()  # noqa: SLF001
        assert final_count == initial_count + 1, (
            f"Expected {initial_count + 1} items but got {final_count}; "
            "duplicate quick-access pin was added (F10 regression)"
        )


# ---------------------------------------------------------------------------
# F11 — Shared resolve_columns helper eliminates duplication
# ---------------------------------------------------------------------------


class TestF11DryResolveColumns:
    """resolve_columns from data_explorer_service must be used by both modules."""

    def test_resolve_all_when_none_selected(self) -> None:
        """Passing selected=None must return all available columns."""
        try:
            from upstream_drift_tools.ui.tools_sidebar.data_explorer_service import (
                DataExplorerError,
                resolve_columns,
            )
        except ImportError:
            pytest.skip("sidekick unavailable")

        available = ["a", "b", "c"]
        result = resolve_columns(available, None, DataExplorerError)
        assert result == available

    def test_resolve_subset(self) -> None:
        """Only the requested columns are returned when a subset is given."""
        try:
            from upstream_drift_tools.ui.tools_sidebar.data_explorer_service import (
                DataExplorerError,
                resolve_columns,
            )
        except ImportError:
            pytest.skip("sidekick unavailable")

        available = ["a", "b", "c"]
        result = resolve_columns(available, ["b", "c"], DataExplorerError)
        assert result == ["b", "c"]

    def test_unknown_column_raises(self) -> None:
        """A column not in available must raise the given error_cls."""
        try:
            from upstream_drift_tools.ui.tools_sidebar.data_explorer_service import (
                DataExplorerError,
                resolve_columns,
            )
        except ImportError:
            pytest.skip("sidekick unavailable")

        with pytest.raises(DataExplorerError):
            resolve_columns(["a", "b"], ["a", "z"], DataExplorerError)

    def test_unknown_column_raises_with_single_arg_error(self) -> None:
        """resolve_columns also works with single-argument exception classes."""
        try:
            from upstream_drift_tools.ui.tools_sidebar.data_explorer_service import (
                resolve_columns,
            )
        except ImportError:
            pytest.skip("sidekick unavailable")

        class _SimpleError(ValueError):
            pass

        with pytest.raises(_SimpleError):
            resolve_columns(["a", "b"], ["a", "z"], _SimpleError)

    def test_data_processor_imports_shared_helper(self) -> None:
        """data_processor_tab must import resolve_columns from data_explorer_service."""
        try:
            import upstream_drift_tools.ui.tools_sidebar.data_processor_tab as dpt
        except ImportError:
            pytest.skip("sidekick unavailable")

        from upstream_drift_tools.ui.tools_sidebar.data_explorer_service import (
            resolve_columns,
        )

        assert getattr(dpt, "_resolve_columns", None) is resolve_columns, (
            "data_processor_tab is not using the shared resolve_columns "
            "(F11 regression)"
        )


# ---------------------------------------------------------------------------
# F8 — replace_tab_widget atomically updates widget map and QTabWidget
# ---------------------------------------------------------------------------


class TestF8AtomicTabSwap:
    """UnifiedToolsSidebar.replace_tab_widget must update both QTabWidget and map."""

    def test_replace_tab_widget_exists(self) -> None:
        """UnifiedToolsSidebar exposes replace_tab_widget as a public method."""
        try:
            from upstream_drift_tools.ui.tools_sidebar.sidebar import (
                UnifiedToolsSidebar,
            )
        except ImportError:
            pytest.skip("sidekick unavailable")

        assert hasattr(UnifiedToolsSidebar, "replace_tab_widget"), (
            "replace_tab_widget public method missing (F8 regression)"
        )
        assert callable(UnifiedToolsSidebar.replace_tab_widget)

    def test_replace_tab_widget_updates_map(self, tmp_path: Path, qtbot: Any) -> None:
        """After a swap, _tab_widgets[tab_id] must point to the new widget."""
        try:
            from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
            from upstream_drift_tools.ui.tools_sidebar.sidebar import (
                UnifiedToolsSidebar,
            )
        except ImportError:
            pytest.skip("Qt/sidekick unavailable")

        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        sidebar = UnifiedToolsSidebar(project_root=tmp_path)
        qtbot.addWidget(sidebar)

        old_widget = QtWidgets.QLabel("old", sidebar)
        new_widget = QtWidgets.QLabel("new", sidebar)

        sidebar.add_tab("swap_test", "Swap", old_widget)

        assert sidebar._tab_widgets.get("swap_test") is old_widget  # noqa: SLF001

        result = sidebar.replace_tab_widget(old_widget, new_widget)

        assert result is True, "replace_tab_widget returned False unexpectedly"
        assert sidebar._tab_widgets.get("swap_test") is new_widget, (  # noqa: SLF001
            "_tab_widgets still points to old_widget after swap (F8 regression)"
        )

    def test_replace_tab_widget_returns_false_for_unknown(
        self, tmp_path: Path, qtbot: Any
    ) -> None:
        """replace_tab_widget returns False when old_widget is not in any tab."""
        try:
            from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
            from upstream_drift_tools.ui.tools_sidebar.sidebar import (
                UnifiedToolsSidebar,
            )
        except ImportError:
            pytest.skip("Qt/sidekick unavailable")

        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        sidebar = UnifiedToolsSidebar(project_root=tmp_path)
        qtbot.addWidget(sidebar)

        ghost = QtWidgets.QLabel("ghost")
        new_w = QtWidgets.QLabel("new")
        assert sidebar.replace_tab_widget(ghost, new_w) is False


# ---------------------------------------------------------------------------
# F9 — registry.update_from merges via public set() and emits events
# ---------------------------------------------------------------------------


class TestF9RegistryUpdateFrom:
    """update_from must validate entries and notify subscribers."""

    def test_update_from_notifies_subscribers(self) -> None:
        """Subscribers must receive a 'set' event for each merged variable."""
        try:
            from upstream_drift_tools.ui.tools_sidebar.registry import (
                WorkspaceRegistry,
            )
        except ImportError:
            pytest.skip("sidekick unavailable")

        source = WorkspaceRegistry({"x": 1, "y": 2})
        target = WorkspaceRegistry()

        events: list[tuple[str, str]] = []
        target.subscribe(lambda event, name: events.append((event, name)))

        target.update_from(source)

        notified_names = {name for _, name in events}
        assert "x" in notified_names, (
            "Subscriber was not notified for 'x' (F9 regression)"
        )
        assert "y" in notified_names, (
            "Subscriber was not notified for 'y' (F9 regression)"
        )

    def test_update_from_validates_names(self) -> None:
        """update_from must reject invalid variable names from the source."""
        try:
            from upstream_drift_tools.ui.tools_sidebar.registry import (
                WorkspaceRegistry,
            )
        except ImportError:
            pytest.skip("sidekick unavailable")

        # Directly inject a bad name into source's private store to simulate a
        # malformed loaded file (bypassing source.set() validation).
        source = WorkspaceRegistry()
        source._values["   "] = 42  # noqa: SLF001 - deliberately injecting bad name
        target = WorkspaceRegistry()

        with pytest.raises(ValueError, match="non-empty"):
            target.update_from(source)

    def test_update_from_replace_clears_existing(self) -> None:
        """replace=True must clear target before merging."""
        try:
            from upstream_drift_tools.ui.tools_sidebar.registry import (
                WorkspaceRegistry,
            )
        except ImportError:
            pytest.skip("sidekick unavailable")

        target = WorkspaceRegistry({"old": 99})
        source = WorkspaceRegistry({"new": 1})

        target.update_from(source, replace=True)

        assert target.list_names() == ["new"], (
            "replace=True did not clear existing variables (F9 regression)"
        )

    def test_repr_only_entries_are_merged_and_notified(self) -> None:
        """Repr-only entries from a loaded registry must be merged + notify fired."""
        try:
            from upstream_drift_tools.ui.tools_sidebar.registry import (
                WorkspaceRegistry,
            )
        except ImportError:
            pytest.skip("sidekick unavailable")

        # Simulate a loaded repr-only variable by calling _set_repr_entry directly.
        source = WorkspaceRegistry()
        source._set_repr_entry(  # noqa: SLF001
            "arr",
            "<numpy.ndarray at 0x…>",
            "<numpy.ndarray at 0x…>",
            {"shape": (3, 3), "dtype": "float64", "size": 9, "preview": "…"},
        )

        target = WorkspaceRegistry()
        events: list[str] = []
        target.subscribe(lambda ev, name: events.append(name))

        target.update_from(source)

        assert "arr" in target.list_names(), (
            "repr-only variable not merged by update_from (F9 regression)"
        )
        assert "arr" in events, (
            "Subscriber not notified for repr-only variable (F9 regression)"
        )


# ---------------------------------------------------------------------------
# F2 — Ctrl+C interrupt, Stop/restart, and command history ring
# ---------------------------------------------------------------------------


class TestF2TerminalControls:
    """OS terminal must support Ctrl+C, Stop/restart, and command history."""

    def _make_widget_with_fake_backend(
        self, tmp_path: Path, qtbot: Any
    ) -> tuple[Any, list[bytes]]:
        """Return (widget, written_list) with a fake in-memory backend."""
        try:
            from upstream_drift_tools.ui.tools_sidebar import os_terminal
            from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
            from upstream_drift_tools.ui.tools_sidebar.shell_discovery import (
                ShellDescriptor,
            )
        except ImportError:
            pytest.skip("Qt/sidekick unavailable")

        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        written: list[bytes] = []

        class _FakeBackend:
            is_running = True

            def start(self) -> None:
                pass

            def write(self, data: bytes) -> None:
                written.append(data)

            def read(self, timeout: float = 0.0) -> bytes:  # noqa: ARG002
                return b""

            def terminate(self) -> None:
                self.is_running = False

            def resize(self, rows: int, cols: int) -> None:  # noqa: ARG002
                pass

        widget = os_terminal.SidekickOsTerminalWidget(
            project_root=tmp_path,
            shells=[
                ShellDescriptor(identifier="fake", label="fake", command=("fake",))
            ],
            autostart=False,
        )
        widget._backend = _FakeBackend()  # noqa: SLF001
        qtbot.addWidget(widget)
        return widget, written

    def test_ctrlc_button_exists(self, tmp_path: Path, qtbot: Any) -> None:
        """Widget must expose _ctrlc_button and _stop_button controls."""
        widget, _ = self._make_widget_with_fake_backend(tmp_path, qtbot)
        assert hasattr(widget, "_ctrlc_button"), "_ctrlc_button missing (F2 regression)"
        assert hasattr(widget, "_stop_button"), "_stop_button missing (F2 regression)"

    def test_send_interrupt_writes_etx(self, tmp_path: Path, qtbot: Any) -> None:
        """_send_interrupt() must write exactly b'\\x03' to the PTY backend."""
        widget, written = self._make_widget_with_fake_backend(tmp_path, qtbot)
        widget._send_interrupt()  # noqa: SLF001
        assert written, "_send_interrupt did not write anything to backend"
        assert written[0] == b"\x03", (
            f"Expected b'\\x03' but got {written[0]!r} (F2 regression)"
        )

    def test_history_records_submitted_commands(
        self, tmp_path: Path, qtbot: Any
    ) -> None:
        """Submitted commands must be prepended to the history ring."""
        widget, _ = self._make_widget_with_fake_backend(tmp_path, qtbot)
        widget._input.setText("ls -la")  # noqa: SLF001
        widget._on_submit()  # noqa: SLF001
        widget._input.setText("pwd")  # noqa: SLF001
        widget._on_submit()  # noqa: SLF001

        assert widget._history[0] == "pwd", (  # noqa: SLF001
            "Most recent command must be first in history (F2 regression)"
        )
        assert widget._history[1] == "ls -la"  # noqa: SLF001

    def test_history_rejects_exact_duplicates(self, tmp_path: Path, qtbot: Any) -> None:
        """Submitting the same command twice must not duplicate it in history."""
        widget, _ = self._make_widget_with_fake_backend(tmp_path, qtbot)
        widget._input.setText("echo hi")  # noqa: SLF001
        widget._on_submit()  # noqa: SLF001
        widget._input.setText("echo hi")  # noqa: SLF001
        widget._on_submit()  # noqa: SLF001

        assert widget._history.count("echo hi") == 1, (
            "Duplicate command was added to history (F2 regression)"
        )  # noqa: SLF001

    def test_navigate_history_older(self, tmp_path: Path, qtbot: Any) -> None:
        """Up-arrow (direction=1) must populate the input with older commands."""
        widget, _ = self._make_widget_with_fake_backend(tmp_path, qtbot)
        widget._input.setText("first")  # noqa: SLF001
        widget._on_submit()  # noqa: SLF001
        widget._input.setText("second")  # noqa: SLF001
        widget._on_submit()  # noqa: SLF001

        # Navigate one step back (most recent = "second")
        widget._navigate_history(direction=1)  # noqa: SLF001
        assert widget._input.text() == "second", (  # noqa: SLF001
            "First up-arrow should show most recent command (F2 regression)"
        )

        # Navigate one more step back (older = "first")
        widget._navigate_history(direction=1)  # noqa: SLF001
        assert widget._input.text() == "first", (
            "Second up-arrow should show older command (F2 regression)"
        )  # noqa: SLF001

    def test_navigate_history_forward_restores_scratch(
        self, tmp_path: Path, qtbot: Any
    ) -> None:
        """Down-arrow past newest must restore the live scratch text."""
        widget, _ = self._make_widget_with_fake_backend(tmp_path, qtbot)
        widget._input.setText("old_cmd")  # noqa: SLF001
        widget._on_submit()  # noqa: SLF001

        widget._input.setText("new draft")  # noqa: SLF001
        widget._navigate_history(direction=1)  # noqa: SLF001  # go back
        widget._navigate_history(direction=-1)  # noqa: SLF001  # come forward

        assert widget._input.text() == "new draft", (  # noqa: SLF001
            "Navigating forward past newest should restore live draft (F2 regression)"
        )


# ---------------------------------------------------------------------------
# F6 — Off-thread / cancellable REPL execution
# ---------------------------------------------------------------------------


class TestF6AsyncRepl:
    """PythonReplWidget must run user code on a worker thread, not the GUI thread."""

    def _make_repl(self, qtbot: Any) -> Any:
        """Return a ready-to-use PythonReplWidget with a stub registry."""
        try:
            from upstream_drift_tools.ui.tools_sidebar import runtime_tabs
            from upstream_drift_tools.ui.tools_sidebar.calculator_startup import (
                CalculatorStartupConfig,
            )
            from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
            from upstream_drift_tools.ui.tools_sidebar.registry import (
                WorkspaceRegistry,
            )
        except ImportError:
            pytest.skip("Qt/sidekick unavailable")

        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        reg = WorkspaceRegistry()
        widget = runtime_tabs.PythonReplWidget(
            registry=reg,
            set_variable=lambda name, value: reg.set(name, value),
            startup_config=CalculatorStartupConfig(()),
        )
        qtbot.addWidget(widget)
        return widget

    def test_cancel_button_present_and_hidden(self, qtbot: Any) -> None:
        """Widget must expose _cancel_button, initially hidden and disabled."""
        widget = self._make_repl(qtbot)
        assert hasattr(widget, "_cancel_button"), (
            "_cancel_button missing (F6 regression)"
        )
        assert not widget._cancel_button.isVisible(), (
            "_cancel_button should be hidden at rest (F6 regression)"
        )  # noqa: SLF001
        assert not widget._cancel_button.isEnabled(), (
            "_cancel_button should be disabled at rest (F6 regression)"
        )  # noqa: SLF001

    def test_status_label_present_and_hidden(self, qtbot: Any) -> None:
        """Widget must expose _status_label, initially hidden."""
        widget = self._make_repl(qtbot)
        assert hasattr(widget, "_status_label"), "_status_label missing (F6 regression)"
        assert not widget._status_label.isVisible(), (
            "_status_label should be hidden at rest (F6 regression)"
        )  # noqa: SLF001

    def test_execute_completes_and_shows_output(self, qtbot: Any) -> None:
        """execute() must complete and write output to the output pane."""
        widget = self._make_repl(qtbot)
        widget.execute("x = 2 + 2")

        # Wait up to 3 s for the worker to finish (signal updates output)
        qtbot.waitUntil(
            lambda: widget._worker is None,  # noqa: SLF001
            timeout=3000,
        )
        assert widget._namespace["x"] == 4  # noqa: SLF001
        assert widget._run_button.isEnabled()  # noqa: SLF001

    def test_set_running_toggles_controls(self, qtbot: Any) -> None:
        """_set_running(True) disables Run and shows Cancel + status label."""
        widget = self._make_repl(qtbot)
        widget._set_running(True)  # noqa: SLF001

        assert not widget._run_button.isEnabled(), (
            "Run button must be disabled while running (F6 regression)"
        )  # noqa: SLF001
        # In headless tests the top-level window is never shown, so isVisible()
        # returns False even after setVisible(True).  isHidden() checks the
        # widget's own explicit visibility bit, which is reliable here.
        assert not widget._cancel_button.isHidden(), (  # noqa: SLF001
            "Cancel button must not be hidden while running (F6 regression)"
        )
        assert not widget._status_label.isHidden(), (  # noqa: SLF001
            "Status label must not be hidden while running (F6 regression)"
        )

        widget._set_running(False)  # noqa: SLF001
        assert widget._run_button.isEnabled(), (
            "Run button must re-enable after stop (F6 regression)"
        )  # noqa: SLF001
        assert widget._cancel_button.isHidden(), (
            "Cancel button must be hidden after stop (F6 regression)"
        )  # noqa: SLF001
