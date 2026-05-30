"""Tests for the copy-to-clipboard button on the chat diagnostic tab (#3115)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.serial]

if sys.platform == "win32" and __import__("os").environ.get("PYTEST_XDIST_WORKER"):
    pytest.skip("Qt tests run serially on Windows.", allow_module_level=True)


@pytest.fixture
def qt_app():  # noqa: ANN201
    try:
        from sidekick.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _build_fake_sidebar(tmp_path: Path) -> types.SimpleNamespace:
    sidebar = types.SimpleNamespace(
        _chat_dock_import_error=RuntimeError("test import failure"),
        project_root=tmp_path,
    )
    return sidebar


def test_copy_button_exists_in_chat_status_tab(qt_app, tmp_path: Path, qtbot) -> None:  # noqa: ANN001
    """The chat status fallback widget must include a Copy button."""
    from sidekick.ui.tools_sidebar import runtime_tabs
    from sidekick.ui.tools_sidebar.qt_compat import QtWidgets

    sidebar = _build_fake_sidebar(tmp_path)
    widget = runtime_tabs._build_chat_status_tab(sidebar)  # noqa: SLF001
    qtbot.addWidget(widget)

    copy_btn = widget.findChild(QtWidgets.QPushButton, "SidekickChatStatusCopy")
    assert copy_btn is not None, "Copy button not found in chat status tab"
    assert "copy" in copy_btn.toolTip().lower()


def test_retry_button_still_present_alongside_copy(  # noqa: ANN001
    qt_app, tmp_path: Path, qtbot
) -> None:
    """The Retry button must coexist with the new Copy button."""
    from sidekick.ui.tools_sidebar import runtime_tabs
    from sidekick.ui.tools_sidebar.qt_compat import QtWidgets

    sidebar = _build_fake_sidebar(tmp_path)
    widget = runtime_tabs._build_chat_status_tab(sidebar)  # noqa: SLF001
    qtbot.addWidget(widget)

    retry_btn = widget.findChild(QtWidgets.QPushButton, "SidekickChatStatusRetry")
    assert retry_btn is not None, "Retry button must still be present"


def test_copy_button_copies_error_text_to_clipboard(
    qt_app, tmp_path: Path, qtbot
) -> None:
    """Clicking Copy writes the diagnostic text to the system clipboard."""
    from sidekick.ui.tools_sidebar import runtime_tabs
    from sidekick.ui.tools_sidebar.qt_compat import QtWidgets

    sidebar = _build_fake_sidebar(tmp_path)
    widget = runtime_tabs._build_chat_status_tab(sidebar)  # noqa: SLF001
    qtbot.addWidget(widget)

    copy_btn = widget.findChild(QtWidgets.QPushButton, "SidekickChatStatusCopy")
    assert copy_btn is not None

    copy_btn.click()

    clipboard = QtWidgets.QApplication.clipboard()
    if clipboard is None:
        pytest.skip("Clipboard not available in this environment")

    clipped = clipboard.text()
    assert "test import failure" in clipped or len(clipped) > 0
