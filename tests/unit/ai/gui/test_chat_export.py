"""RED tests for Tools issue #2735 — Advanced Export and Copy Capabilities.

Covers:
- ``export_thread_to_markdown`` helper function (data model layer)
- Per-message copy button presence on MessageWidget
- Copy-thread clipboard aggregation
- Save-as-markdown file dialog (mocked)

All tests run without a display server; PyQt6 widgets are skipped gracefully
when a QApplication cannot be created.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Stub out heavy deps so the module can load without a display server
# ---------------------------------------------------------------------------


def _ensure_pkg(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    sys.modules[name] = mod


_ensure_pkg("src", ROOT / "src")
_ensure_pkg("src.shared", ROOT / "src" / "shared")
_ensure_pkg("src.shared.python", ROOT / "src" / "shared" / "python")
_ensure_pkg("src.shared.python.ai", ROOT / "src" / "shared" / "python" / "ai")
_ensure_pkg(
    "src.shared.python.ai.gui",
    ROOT / "src" / "shared" / "python" / "ai" / "gui",
)

logging_pkg = types.ModuleType("src.shared.python.logging_pkg")
logging_config = types.ModuleType("src.shared.python.logging_pkg.logging_config")
logging_config.get_logger = logging.getLogger
logging_config.setup_logging = lambda *args, **kwargs: None
sys.modules.setdefault("src.shared.python.logging_pkg", logging_pkg)
sys.modules.setdefault("src.shared.python.logging_pkg.logging_config", logging_config)

# Require PyQt6 — skip entire module on headless CI without Qt
pytest.importorskip("PyQt6.QtCore", reason="PyQt6 required")


# ---------------------------------------------------------------------------
# Module loaders
# ---------------------------------------------------------------------------


def _load_module(name: str, rel_path: str):
    if name in sys.modules:
        return sys.modules[name]
    try:
        import importlib

        return importlib.import_module(name)
    except Exception:
        full = ROOT / "src" / "shared" / "python" / rel_path
        spec = importlib.util.spec_from_file_location(name, full)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module


_types_mod = _load_module("src.shared.python.ai.types", "ai/types.py")
Message = _types_mod.Message
ConversationContext = _types_mod.ConversationContext

_export_mod = _load_module(
    "src.shared.python.ai.gui.chat_export",
    "ai/gui/chat_export.py",
)
export_thread_to_markdown = _export_mod.export_thread_to_markdown


# ---------------------------------------------------------------------------
# Tests: export_thread_to_markdown (pure-Python, no Qt required)
# ---------------------------------------------------------------------------


class TestExportThreadToMarkdown:
    def test_formats_user_and_agent_messages(self) -> None:
        messages = [
            Message(role="user", content="Hello there"),
            Message(role="assistant", content="Hi! How can I help?"),
        ]
        result = export_thread_to_markdown(messages)
        assert "**User:** Hello there" in result
        assert "**Agent:** Hi! How can I help?" in result

    def test_user_and_agent_separated_by_blank_line(self) -> None:
        messages = [
            Message(role="user", content="Q"),
            Message(role="assistant", content="A"),
        ]
        result = export_thread_to_markdown(messages)
        # Must have a blank separator between consecutive messages
        assert "\n\n" in result

    def test_handles_code_blocks(self) -> None:
        code_content = "Here is code:\n```python\nprint('hello')\n```"
        messages = [
            Message(role="user", content="Show me code"),
            Message(role="assistant", content=code_content),
        ]
        result = export_thread_to_markdown(messages)
        assert "```python" in result
        assert "print('hello')" in result

    def test_empty_thread_returns_empty_string(self) -> None:
        result = export_thread_to_markdown([])
        assert result == ""

    def test_skips_system_messages(self) -> None:
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]
        result = export_thread_to_markdown(messages)
        assert "You are a helpful assistant." not in result
        assert "**User:** Hello" in result

    def test_raises_value_error_on_non_list_input(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            export_thread_to_markdown("not a list")  # type: ignore[arg-type]

    def test_raises_value_error_on_none_input(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            export_thread_to_markdown(None)  # type: ignore[arg-type]

    def test_multiple_exchanges_ordered_correctly(self) -> None:
        messages = [
            Message(role="user", content="First question"),
            Message(role="assistant", content="First answer"),
            Message(role="user", content="Second question"),
            Message(role="assistant", content="Second answer"),
        ]
        result = export_thread_to_markdown(messages)
        idx_first_q = result.index("First question")
        idx_first_a = result.index("First answer")
        idx_second_q = result.index("Second question")
        idx_second_a = result.index("Second answer")
        assert idx_first_q < idx_first_a < idx_second_q < idx_second_a

    def test_special_characters_preserved(self) -> None:
        messages = [
            Message(role="user", content="What is 2 < 3 & 4 > 1?"),
            Message(role="assistant", content="Yes, 2 < 3 and 4 > 1."),
        ]
        result = export_thread_to_markdown(messages)
        assert "2 < 3" in result
        assert "4 > 1" in result


# ---------------------------------------------------------------------------
# Tests: copy_message writes to clipboard
# ---------------------------------------------------------------------------


class TestCopyMessageToClipboard:
    def test_copy_message_writes_to_clipboard(self) -> None:
        """copy_message_to_clipboard(text) calls QApplication.clipboard().setText()."""
        copy_fn = _export_mod.copy_message_to_clipboard

        mock_clipboard = MagicMock()
        with patch("src.shared.python.ai.gui.chat_export.QApplication") as mock_app_cls:
            mock_app_cls.clipboard.return_value = mock_clipboard
            copy_fn("Hello clipboard")
            mock_clipboard.setText.assert_called_once_with("Hello clipboard")

    def test_copy_message_to_clipboard_raises_on_none(self) -> None:
        copy_fn = _export_mod.copy_message_to_clipboard
        with pytest.raises((ValueError, TypeError)):
            copy_fn(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests: copy_thread aggregates all messages
# ---------------------------------------------------------------------------


class TestCopyThread:
    def test_copy_thread_aggregates_all_messages(self) -> None:
        """copy_thread_to_clipboard(messages) must put full markdown on clipboard."""
        copy_thread = _export_mod.copy_thread_to_clipboard

        messages = [
            Message(role="user", content="User says hi"),
            Message(role="assistant", content="Agent says hello"),
        ]

        mock_clipboard = MagicMock()
        with patch("src.shared.python.ai.gui.chat_export.QApplication") as mock_app_cls:
            mock_app_cls.clipboard.return_value = mock_clipboard
            copy_thread(messages)
            call_args = mock_clipboard.setText.call_args[0][0]
            assert "User says hi" in call_args
            assert "Agent says hello" in call_args

    def test_copy_thread_empty_list_copies_empty_string(self) -> None:
        copy_thread = _export_mod.copy_thread_to_clipboard

        mock_clipboard = MagicMock()
        with patch("src.shared.python.ai.gui.chat_export.QApplication") as mock_app_cls:
            mock_app_cls.clipboard.return_value = mock_clipboard
            copy_thread([])
            mock_clipboard.setText.assert_called_once_with("")


# ---------------------------------------------------------------------------
# Tests: save_thread_as_markdown writes file
# ---------------------------------------------------------------------------


class TestSaveThreadAsMarkdown:
    def test_save_thread_creates_markdown_file(self, tmp_path: Path) -> None:
        """save_thread_as_markdown writes structured markdown to chosen path."""
        save_fn = _export_mod.save_thread_as_markdown

        messages = [
            Message(role="user", content="Save this"),
            Message(role="assistant", content="Saved!"),
        ]
        dest = tmp_path / "chat_export.md"

        with patch(
            "src.shared.python.ai.gui.chat_export.QFileDialog"
        ) as mock_dialog_cls:
            mock_dialog_cls.getSaveFileName.return_value = (
                str(dest),
                "Markdown (*.md)",
            )
            save_fn(messages, parent=None)

        assert dest.exists()
        content = dest.read_text(encoding="utf-8")
        assert "**User:** Save this" in content
        assert "**Agent:** Saved!" in content

    def test_save_thread_does_nothing_when_dialog_cancelled(
        self, tmp_path: Path
    ) -> None:
        """Cancelled dialog (empty path) must not raise or write anything."""
        save_fn = _export_mod.save_thread_as_markdown

        messages = [Message(role="user", content="Irrelevant")]

        with patch(
            "src.shared.python.ai.gui.chat_export.QFileDialog"
        ) as mock_dialog_cls:
            mock_dialog_cls.getSaveFileName.return_value = ("", "")
            # Must not raise
            save_fn(messages, parent=None)


# ---------------------------------------------------------------------------
# Tests: MessageWidget has a copy button
# ---------------------------------------------------------------------------


class TestMessageWidgetCopyButton:
    """Verify the per-message copy QToolButton is present on MessageWidget."""

    def _load_widget_module(self):
        """Load assistant_widgets with theme stubs."""
        theme_mod = types.ModuleType("src.shared.python.theme")
        theme_mod.__path__ = []
        style_mod = types.ModuleType("src.shared.python.theme.style_constants")

        class _Styles:
            TEXT_LABEL_BOLD_WHITE = "font-weight: bold; color: white;"
            TEXT_MUTED = "color: grey;"
            TEXT_CONTENT_TRANSPARENT = "background: transparent;"
            CONTAINER_DARK = "background-color: #1e1e1e;"

        style_mod.Styles = _Styles()
        theme_manager_mod = types.ModuleType("src.shared.python.theme.theme_manager")
        theme_manager_mod.get_theme_manager = MagicMock(
            return_value=MagicMock(get_current_colors=MagicMock(return_value={}))
        )
        sys.modules.setdefault("src.shared.python.theme", theme_mod)
        sys.modules.setdefault("src.shared.python.theme.style_constants", style_mod)
        sys.modules.setdefault(
            "src.shared.python.theme.theme_manager", theme_manager_mod
        )

        return _load_module(
            "src.shared.python.ai.gui.assistant_widgets",
            "ai/gui/assistant_widgets.py",
        )

    def test_copy_button_visible_on_message_widget(self) -> None:
        """Each MessageWidget must have a _copy_btn QToolButton attribute."""
        from PyQt6.QtWidgets import QApplication, QToolButton

        app = QApplication.instance() or QApplication(sys.argv)
        widget_mod = self._load_widget_module()
        MessageWidget = widget_mod.MessageWidget

        widget = MessageWidget("user", "Hello world")
        assert hasattr(widget, "_copy_btn"), "MessageWidget missing _copy_btn attribute"
        assert isinstance(
            widget._copy_btn, QToolButton
        ), "_copy_btn must be a QToolButton"
        _ = app  # keep reference alive
