"""Lifecycle contract tests for the unified Sidekick sidebar."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_sidebar_shutdown_delegates_once_to_each_live_runtime() -> None:
    """A host close must not leave terminal or other tab runtimes alive."""
    from sidekick.ui.tools_sidebar.sidebar import UnifiedToolsSidebar

    terminal = SimpleNamespace(shutdown=MagicMock())
    other_runtime = SimpleNamespace(shutdown=MagicMock())
    passive_widget = object()
    sidebar = SimpleNamespace(
        _shutdown_complete=False,
        _tab_widgets={
            "terminal": terminal,
            "terminal-alias": terminal,
            "other": other_runtime,
            "passive": passive_widget,
        },
    )

    UnifiedToolsSidebar.shutdown(sidebar)
    UnifiedToolsSidebar.shutdown(sidebar)

    terminal.shutdown.assert_called_once_with()
    other_runtime.shutdown.assert_called_once_with()
    assert sidebar._shutdown_complete is True


def test_sidebar_shutdown_continues_after_runtime_cleanup_error() -> None:
    """One faulty runtime must not prevent cleanup of remaining tabs."""
    from sidekick.ui.tools_sidebar.sidebar import UnifiedToolsSidebar

    failing_runtime = SimpleNamespace(
        shutdown=MagicMock(side_effect=RuntimeError("cleanup failed"))
    )
    healthy_runtime = SimpleNamespace(shutdown=MagicMock())
    sidebar = SimpleNamespace(
        _shutdown_complete=False,
        _tab_widgets={
            "failing": failing_runtime,
            "healthy": healthy_runtime,
        },
    )

    UnifiedToolsSidebar.shutdown(sidebar)

    failing_runtime.shutdown.assert_called_once_with()
    healthy_runtime.shutdown.assert_called_once_with()


def test_host_window_close_shuts_down_live_runtime(
    tmp_path: Any,
    qtbot: Any,
) -> None:
    """Closing a generic Qt host must invoke the sidebar lifecycle contract."""
    from PyQt6 import QtWidgets
    from sidekick.ui.tools_sidebar.sidebar import UnifiedToolsSidebar

    main_window = QtWidgets.QMainWindow()
    qtbot.addWidget(main_window)
    sidebar = UnifiedToolsSidebar(
        project_root=tmp_path,
        tab_definitions=[],
        parent=main_window,
    )
    runtime = QtWidgets.QWidget()
    runtime.shutdown = MagicMock()
    sidebar.add_tab("runtime", "Runtime", runtime)
    sidebar.install_as_dock(main_window, title="Sidekick")
    main_window.show()

    main_window.close()

    runtime.shutdown.assert_called_once_with()
