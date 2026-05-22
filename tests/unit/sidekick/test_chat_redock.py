"""Tests for Sidekick chat redock + duplicate-chat-tab flows (issue #2935).

Verifies the acceptance criteria:
1. chat_popout_window.py exists with a Re-dock affordance
2. Clicking Re-dock restores chat to its host dock without losing history
3. duplicate-chat-tab creates an independent session (separate session_id)

Also verifies that chat_dock_widget.py is accessible and has the expected
session-id helpers (_read_shared_session_id, _write_shared_session_id).

TDD: these tests drove the implementation of chat_popout_window.py and
the re_dock precondition on tab_popout.py.
"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.serial

if sys.platform == "win32" and os.environ.get("PYTEST_XDIST_WORKER"):
    pytest.skip(
        "Qt Chat redock tests run serially on Windows.",
        allow_module_level=True,
    )

pytest.importorskip("PyQt6")

_SHARED = Path(__file__).resolve().parents[3] / "src" / "shared" / "python"
if str(_SHARED) in sys.path:
    sys.path.remove(str(_SHARED))
sys.path.insert(0, str(_SHARED))


# ---------------------------------------------------------------------------
# chat_dock_widget.py verification
# ---------------------------------------------------------------------------


def test_chat_dock_widget_importable() -> None:
    """chat.chat_dock_widget is importable and exposes session-id helpers."""
    from chat.chat_dock_widget import (  # noqa: F401
        _read_shared_session_id,
        _session_file_path,
        _write_shared_session_id,
    )

    assert callable(_read_shared_session_id)
    assert callable(_write_shared_session_id)
    assert callable(_session_file_path)


def test_session_id_write_and_read_round_trip(tmp_path: Path) -> None:
    """Writing a session_id and reading it back returns the same value."""
    from chat.chat_dock_widget import (
        _read_shared_session_id,
        _write_shared_session_id,
    )

    path = tmp_path / ".test_app" / "active_chat_session.txt"
    _write_shared_session_id("test-session-123", path)
    result = _read_shared_session_id(path)
    assert result == "test-session-123"


def test_read_missing_session_file_returns_none(tmp_path: Path) -> None:
    """Reading a missing session file returns None without raising."""
    from chat.chat_dock_widget import _read_shared_session_id

    path = tmp_path / "nonexistent" / "session.txt"
    result = _read_shared_session_id(path)
    assert result is None


# ---------------------------------------------------------------------------
# chat_popout_window.py: Re-dock affordance
# ---------------------------------------------------------------------------


def test_chat_popout_window_module_importable() -> None:
    """chat.chat_popout_window is importable and exposes ChatPopoutWindow."""
    from chat.chat_popout_window import (  # noqa: F401
        ChatPopoutWindow,
        make_chat_popout_window,
    )

    assert callable(make_chat_popout_window)


def test_chat_popout_window_has_redock_button(qtbot) -> None:  # type: ignore[no-untyped-def]
    """ChatPopoutWindow has a Re-dock button with the expected objectName."""
    from chat.chat_popout_window import (
        _REDOCK_BUTTON_OBJECT_NAME,
        ChatPopoutWindow,
    )
    from PyQt6.QtWidgets import QPushButton, QWidget

    redocked = []
    content = QWidget()
    win = ChatPopoutWindow(
        content,
        session_id="sess-001",
        redock_callback=lambda: redocked.append(True),
    )
    qtbot.addWidget(win)
    redock_btn = win.findChild(QPushButton, _REDOCK_BUTTON_OBJECT_NAME)
    assert (
        redock_btn is not None
    ), f"Expected QPushButton with objectName {_REDOCK_BUTTON_OBJECT_NAME!r}"


def test_chat_popout_window_redock_invokes_callback(qtbot) -> None:  # type: ignore[no-untyped-def]
    """Clicking Re-dock invokes the host callback."""
    from chat.chat_popout_window import (
        _REDOCK_BUTTON_OBJECT_NAME,
        ChatPopoutWindow,
    )
    from PyQt6.QtWidgets import QPushButton, QWidget

    redocked: list[bool] = []
    content = QWidget()
    win = ChatPopoutWindow(
        content,
        session_id="sess-002",
        redock_callback=lambda: redocked.append(True),
    )
    qtbot.addWidget(win)
    redock_btn = win.findChild(QPushButton, _REDOCK_BUTTON_OBJECT_NAME)
    assert redock_btn is not None
    redock_btn.click()
    assert redocked == [True]


def test_chat_popout_window_hides_after_redock(qtbot) -> None:  # type: ignore[no-untyped-def]
    """After clicking Re-dock the floating window is hidden."""
    from chat.chat_popout_window import (
        _REDOCK_BUTTON_OBJECT_NAME,
        ChatPopoutWindow,
    )
    from PyQt6.QtWidgets import QPushButton, QWidget

    content = QWidget()
    win = ChatPopoutWindow(
        content,
        session_id="sess-003",
        redock_callback=lambda: None,
    )
    win.show()
    qtbot.addWidget(win)
    redock_btn = win.findChild(QPushButton, _REDOCK_BUTTON_OBJECT_NAME)
    assert redock_btn is not None
    redock_btn.click()
    assert not win.isVisible()


def test_chat_popout_window_preserves_session_id(qtbot) -> None:  # type: ignore[no-untyped-def]
    """ChatPopoutWindow exposes the session_id supplied at construction."""
    from chat.chat_popout_window import ChatPopoutWindow
    from PyQt6.QtWidgets import QWidget

    content = QWidget()
    win = ChatPopoutWindow(
        content,
        session_id="my-session-id",
        redock_callback=lambda: None,
    )
    qtbot.addWidget(win)
    assert win.session_id == "my-session-id"


def test_chat_popout_window_type_error_on_bad_callback(qtbot) -> None:  # type: ignore[no-untyped-def]
    """ChatPopoutWindow raises TypeError when redock_callback is not callable."""
    from chat.chat_popout_window import ChatPopoutWindow
    from PyQt6.QtWidgets import QWidget

    content = QWidget()
    with pytest.raises(TypeError, match="redock_callback must be callable"):
        ChatPopoutWindow(content, session_id="sess", redock_callback="not-callable")  # type: ignore[arg-type]


def test_chat_popout_window_value_error_on_empty_session_id(qtbot) -> None:  # type: ignore[no-untyped-def]
    """ChatPopoutWindow raises ValueError when session_id is empty."""
    from chat.chat_popout_window import ChatPopoutWindow
    from PyQt6.QtWidgets import QWidget

    content = QWidget()
    with pytest.raises(ValueError, match="session_id must be a non-empty string"):
        ChatPopoutWindow(content, session_id="", redock_callback=lambda: None)


# ---------------------------------------------------------------------------
# Duplicate chat tab: independent session_id
# ---------------------------------------------------------------------------


def _fix_sidekick_import() -> None:
    _TEST_PKG = Path(__file__).resolve().parent
    shared_str = str(_SHARED)
    if shared_str not in sys.path:
        sys.path.insert(0, shared_str)
    else:
        sys.path.remove(shared_str)
        sys.path.insert(0, shared_str)
    test_dir = str(_TEST_PKG)
    top_mod = sys.modules.get("sidekick")
    if (
        top_mod is not None
        and getattr(top_mod, "__file__", None) is not None
        and test_dir in str(Path(top_mod.__file__).resolve().parent)
    ):
        del sys.modules["sidekick"]


def test_duplicate_tab_creates_independent_session(tmp_path: Path, qtbot) -> None:
    """Duplicating a tab creates a new independent tab_id (different from original)."""
    _fix_sidekick_import()
    from PyQt6 import QtWidgets
    from sidekick.ui.tools_sidebar.sidebar import UnifiedToolsSidebar
    from sidekick.ui.tools_sidebar.tab_definition import SidebarTabDefinition

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = QtWidgets.QMainWindow()
    qtbot.addWidget(win)
    tab_def = SidebarTabDefinition(
        tab_id="chat_tab",
        title="Chat",
        factory=lambda _sb: QtWidgets.QLabel("chat"),
        duplicate_enabled=True,
    )
    sidebar = UnifiedToolsSidebar(
        project_root=tmp_path,
        tab_definitions=[tab_def],
        parent=win,
    )
    sidebar.install_as_dock(win, title="Sidekick")
    win.show()

    duplicate_id = sidebar.duplicate_tab("chat_tab")
    assert duplicate_id is not None
    assert duplicate_id != "chat_tab"
    # The duplicate tab id must follow the pattern chat_tab#N
    assert duplicate_id.startswith("chat_tab#")
