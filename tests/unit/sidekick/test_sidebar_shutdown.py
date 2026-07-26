"""Lifecycle contract tests for the unified Sidekick sidebar."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

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
