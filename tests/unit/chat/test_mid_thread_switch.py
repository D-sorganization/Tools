"""Mid-thread provider switch contract tests (issue #2871).

``ChatDockWidget.switch_provider(name, model, thinking_level)`` lets a
user pivot provider/model/thinking-level inside an active session
without losing prior messages. The DbC is:

* Preconditions:  ``name`` and ``model`` are non-empty strings (after
  ``.strip()``); ``thinking_level`` is one of
  ``{"none","low","medium","high"}``.
* Postconditions: ``_message_history`` is referentially unchanged
  (same list object, same contents); the widget's current provider /
  model / thinking-level state is updated.
* Invariant:      no prior message is dropped, reordered, or mutated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication, QComboBox  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


def _load_in_tree_chat_module():  # type: ignore[no-untyped-def]
    """Load ``_chat_dock_widget_qt`` from the in-tree source (see
    test_dock_widget_dropdowns.py for the rationale)."""
    import importlib
    import importlib.util

    tree_root = Path(__file__).resolve().parents[3] / "src" / "shared" / "python"
    src_path = tree_root / "chat" / "_chat_dock_widget_qt.py"

    sys.path.insert(0, str(tree_root))
    for _name in list(sys.modules):
        if _name == "chat" or _name.startswith("chat."):
            del sys.modules[_name]
    importlib.invalidate_caches()
    chat_mod = importlib.import_module("chat")
    chat_mod.__path__ = [str(tree_root / "chat")]
    spec = importlib.util.spec_from_file_location("chat._chat_dock_widget_qt", src_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not build spec for {src_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["chat._chat_dock_widget_qt"] = module
    spec.loader.exec_module(module)
    return module


def _make_widget_with_history():  # type: ignore[no-untyped-def]
    """Make a ChatDockWidget stand-in with a primed message history."""
    module = _load_in_tree_chat_module()
    ChatDockWidget = module.ChatDockWidget

    namespace = dict(ChatDockWidget.__dict__)
    namespace["__init__"] = lambda self: None
    stand_in_cls = type("ChatDockWidgetStandIn", (object,), namespace)
    widget = stand_in_cls()

    widget._ai_provider_combo = QComboBox()
    widget._ai_provider_combo.addItem("Ollama", "ollama")
    widget._ai_provider_combo.addItem("OpenAI", "openai")
    widget._ai_provider_combo.addItem("Anthropic", "anthropic")

    widget._ai_model_combo = QComboBox()
    widget._ai_model_combo.addItem("llama3", "llama3")
    widget._ai_model_combo.addItem("gpt-4-turbo", "gpt-4-turbo")

    widget._ai_thinking_combo = QComboBox()
    for name in ("none", "low", "medium", "high"):
        widget._ai_thinking_combo.addItem(name.title(), name)

    widget._message_history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    widget._current_provider = "ollama"
    widget._current_model = "llama3"
    widget._current_thinking_level = "none"
    return widget


class TestSwitchProviderDbC:
    """Preconditions / postconditions / invariant for ``switch_provider``."""

    @pytest.mark.parametrize("bad_name", ["", "  "])
    def test_empty_provider_name_raises(self, bad_name: str) -> None:
        widget = _make_widget_with_history()
        with pytest.raises(ValueError):
            widget.switch_provider(bad_name, "gpt-4-turbo", "none")

    @pytest.mark.parametrize("bad_model", ["", "  "])
    def test_empty_model_raises(self, bad_model: str) -> None:
        widget = _make_widget_with_history()
        with pytest.raises(ValueError):
            widget.switch_provider("openai", bad_model, "none")

    @pytest.mark.parametrize(
        "bad_thinking",
        ["", "off", "HIGH", "extreme", "  "],
    )
    def test_invalid_thinking_level_raises(self, bad_thinking: str) -> None:
        widget = _make_widget_with_history()
        with pytest.raises(ValueError):
            widget.switch_provider("openai", "gpt-4-turbo", bad_thinking)

    def test_switch_preserves_history_identity(self) -> None:
        widget = _make_widget_with_history()
        history_before = widget._message_history
        snapshot_before = list(history_before)
        widget.switch_provider("openai", "gpt-4-turbo", "low")
        # Same list object (referential identity preserved).
        assert widget._message_history is history_before
        # And same contents.
        assert widget._message_history == snapshot_before

    def test_switch_updates_state(self) -> None:
        widget = _make_widget_with_history()
        widget.switch_provider("anthropic", "gpt-4-turbo", "high")
        assert widget._current_provider == "anthropic"
        assert widget._current_model == "gpt-4-turbo"
        assert widget._current_thinking_level == "high"

    def test_switch_strips_whitespace(self) -> None:
        widget = _make_widget_with_history()
        widget.switch_provider("  anthropic  ", "  gpt-4-turbo ", "  low ")
        assert widget._current_provider == "anthropic"
        assert widget._current_model == "gpt-4-turbo"
        assert widget._current_thinking_level == "low"


class TestSwitchProviderInvariants:
    """History-immutability invariant across repeated switches."""

    def test_multiple_switches_keep_history(self) -> None:
        widget = _make_widget_with_history()
        history_before = list(widget._message_history)
        widget.switch_provider("openai", "gpt-4-turbo", "none")
        widget.switch_provider("anthropic", "gpt-4-turbo", "medium")
        widget.switch_provider("ollama", "llama3", "none")
        assert widget._message_history == history_before

    def test_switch_does_not_clear_history_after_failed_switch(self) -> None:
        widget = _make_widget_with_history()
        history_before = list(widget._message_history)
        with pytest.raises(ValueError):
            widget.switch_provider("openai", "gpt-4-turbo", "extreme")
        # Failed switch must leave state and history untouched.
        assert widget._message_history == history_before
        assert widget._current_provider == "ollama"
