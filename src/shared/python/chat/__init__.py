"""Shared AI chat widget and contract models.

Provides a portable ChatDockWidget (QDockWidget + QWebSocket) that connects
to any FastAPI-based chat WebSocket endpoint, plus Pydantic contract models
for the chat protocol.

Usage::

    from shared.python.chat import ChatDockWidget

    dock = ChatDockWidget(
        app_context="gasification",
        app_name="integrated_process_simulator",
        parent=main_window,
    )
    main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
"""

from .models import (
    ChatChunkResponse,
    ChatHistoryResponse,
    ChatMessageRequest,
    ChatSessionInfo,
)

try:
    from .chat_dock_widget import ChatDockWidget, ChatMessageBubble

    _PYQT6_AVAILABLE = True
except ImportError:
    _PYQT6_AVAILABLE = False
    ChatDockWidget = None  # type: ignore[assignment, misc]
    ChatMessageBubble = None  # type: ignore[assignment, misc]

__all__ = [
    "ChatDockWidget",
    "ChatMessageBubble",
    "ChatMessageRequest",
    "ChatChunkResponse",
    "ChatSessionInfo",
    "ChatHistoryResponse",
]
