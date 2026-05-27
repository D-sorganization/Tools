"""Pop-out window wrapper for the shared chat dock widget (issue #2935).

When the :class:`~chat.ChatDockWidget` is "popped out" of its host
:class:`~PyQt6.QtWidgets.QDockWidget`, it is reparented into a
:class:`ChatPopoutWindow` that:

- Displays the chat history in a floating ``QMainWindow``.
- Provides a "Re-dock" button that returns the chat to its original dock
  without losing the message history or the ``session_id``.
- Preserves the ``session_id`` of the original chat session so that
  re-docking creates a seamless continuation.

Design
------
- **DbC**: preconditions validated on construction and on ``redock()``.
- **LOD**: only talks to the host dock through the ``redock_callback``
  supplied at construction time; does not reach into the host widget.
- **DRY**: re-dock button styling reuses the same factory as the
  Sidekick tab pop-out chrome.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QDockWidget,
        QMainWindow,
        QPushButton,
        QToolBar,
        QWidget,
    )

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False
    QDockWidget = None  # type: ignore[assignment,misc]

_REDOCK_BUTTON_OBJECT_NAME = "ChatRedockButton"


class ChatPopoutWindow(QMainWindow):
    """Floating window that hosts a popped-out chat dock widget.

    Args:
        chat_widget: The widget to display in the floating window.
        session_id: Session identifier of the chat session being displayed.
        redock_callback: Zero-argument callable invoked when the user clicks
            the Re-dock button.  The host is responsible for re-embedding
            *chat_widget* back into its original dock position.
        title: Window title (default ``"Chat"``).
        parent: Optional Qt parent.

    Raises:
        TypeError: If *redock_callback* is not callable.
        ValueError: If *session_id* is empty.
        RuntimeError: If PyQt6 is not installed.
    """

    def __init__(
        self,
        chat_widget: QWidget,
        *,
        session_id: str,
        redock_callback: Callable[[], None],
        title: str = "Chat",
        parent: QWidget | None = None,
    ) -> None:
        if not _QT_AVAILABLE:
            raise RuntimeError(
                "PyQt6 is not installed; install it with: pip install PyQt6"
            )
        if not callable(redock_callback):
            raise TypeError("redock_callback must be callable")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")

        super().__init__(parent)
        self._session_id = session_id
        self._redock_callback = redock_callback

        self.setObjectName("ChatPopoutWindow")
        self.setWindowTitle(title)

        # If we were handed a QDockWidget, host its inner content widget
        # rather than the dock itself. Putting a QDockWidget into
        # QMainWindow.setCentralWidget produces an empty window because the
        # dock's title-bar chrome and child reparenting fight the main-window
        # layout. Extracting the dock's inner widget keeps the bubble list,
        # input row, and combos visible in the floating window.
        self._source_dock: QDockWidget | None = None
        content_widget: QWidget = chat_widget
        if QDockWidget is not None and isinstance(chat_widget, QDockWidget):
            inner = chat_widget.widget()
            if inner is None:
                raise ValueError(
                    "chat_widget is a QDockWidget with no inner widget set"
                )
            self._source_dock = chat_widget
            # Detach the inner widget from the dock so we can reparent it.
            chat_widget.setWidget(None)
            content_widget = inner
        self._content_widget = content_widget
        self.setCentralWidget(content_widget)

        # ── Re-dock toolbar ──────────────────────────────────────────────
        toolbar = QToolBar("Chat", self)
        toolbar.setObjectName("ChatPopoutToolbar")
        toolbar.setMovable(False)

        redock_btn = QPushButton("⬇ Re-dock", self)
        redock_btn.setObjectName(_REDOCK_BUTTON_OBJECT_NAME)
        redock_btn.setToolTip("Return chat to its original dock position")
        redock_btn.setFlat(True)
        redock_btn.clicked.connect(self._on_redock)
        toolbar.addWidget(redock_btn)

        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        log.debug(
            "ChatPopoutWindow created (session_id=%r, title=%r)", session_id, title
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        """The chat session ID preserved from the original dock widget."""
        return self._session_id

    @property
    def content_widget(self) -> QWidget:
        """The reparented inner content widget shown in the popout."""
        return self._content_widget

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def redock(self) -> None:
        """Invoke the re-dock callback and hide this floating window.

        Before invoking the callback we return the inner content widget to
        its source ``QDockWidget`` (when one was extracted in ``__init__``)
        so the host's reparent-back logic can be a simple no-op. The host
        callback is still invoked so dock visibility can be restored.

        Raises:
            RuntimeError: If the re-dock callback raises.
        """
        log.debug(
            "ChatPopoutWindow.redock() called for session_id=%r", self._session_id
        )
        if self._source_dock is not None:
            # Take the content back from the QMainWindow's central area
            # before re-attaching it to the source dock, otherwise Qt will
            # warn about the widget having two parents.
            self.takeCentralWidget()
            self._source_dock.setWidget(self._content_widget)
        self._redock_callback()
        self.hide()

    def _on_redock(self) -> None:
        """Slot wired to the Re-dock button ``clicked`` signal."""
        self.redock()


def make_chat_popout_window(
    chat_widget: Any,
    *,
    session_id: str,
    redock_callback: Callable[[], None],
    title: str = "Chat",
    parent: Any = None,
) -> ChatPopoutWindow:
    """Factory for :class:`ChatPopoutWindow`.

    Args:
        chat_widget: Chat widget to display.
        session_id: Session identifier.
        redock_callback: Callback invoked when Re-dock is clicked.
        title: Window title.
        parent: Optional Qt parent.

    Returns:
        A :class:`ChatPopoutWindow` with the Re-dock button wired.

    Raises:
        TypeError: If *redock_callback* is not callable.
        ValueError: If *session_id* is empty.
        RuntimeError: If PyQt6 is not installed.
    """
    return ChatPopoutWindow(
        chat_widget,
        session_id=session_id,
        redock_callback=redock_callback,
        title=title,
        parent=parent,
    )


__all__ = [
    "ChatPopoutWindow",
    "make_chat_popout_window",
    "_REDOCK_BUTTON_OBJECT_NAME",
]
