"""Public contract tests for the Tools-owned shared chat package."""

from __future__ import annotations

from pathlib import Path


def test_chat_facade_exports_contract_models() -> None:
    """Consumers can import the documented shared-chat API from ``chat``."""
    import chat

    expected = {
        "ChatDockWidget",
        "ChatMessageBubble",
        "ChatMessageRequest",
        "ChatChunkResponse",
        "ChatSessionInfo",
        "ChatHistoryResponse",
        "ChatModelInfo",
        "ChatModelListResponse",
        "ChatIndexStatusResponse",
        "ChatServiceBase",
        "ChatSession",
        "ChatMessage",
        "create_chat_router",
    }

    assert hasattr(chat, "__all__"), getattr(chat, "__file__", "<missing>")
    assert expected.issubset(set(chat.__all__))
    lazy_exports = {"ChatDockWidget", "ChatMessageBubble", "create_chat_router"}
    for name in expected - lazy_exports:
        assert getattr(chat, name) is not None


def test_chat_qt_loader_uses_package_relative_import() -> None:
    """The lazy Qt widget loader must work from an installed package."""
    source = Path("src/shared/python/chat/chat_dock_widget.py").read_text(
        encoding="utf-8"
    )

    assert "from . import _chat_dock_widget_qt" in source
    assert "from src.shared.python.chat import _chat_dock_widget_qt" not in source


def test_single_canonical_shared_chat_package() -> None:
    """Tools keeps the canonical reusable chat implementation in one package."""
    canonical = Path("src/shared/python/chat").resolve()
    chat_packages = []
    for init_file in Path("src").rglob("__init__.py"):
        if init_file.parent.name != "chat":
            continue
        if "tests" in init_file.parts:
            continue
        chat_packages.append(init_file.parent.resolve())

    assert chat_packages == [canonical]
