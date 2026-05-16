"""Header dropdown contract tests for ``ChatDockWidget`` (issue #2871).

These tests exercise the new three header dropdowns
(``_ai_provider_combo``, ``_ai_model_combo``, ``_ai_thinking_combo``)
and the DRY helper ``_build_header_combobox`` plus the
``_apply_settings_change`` router that funnels every dropdown's
``currentIndexChanged`` signal.

To avoid spinning up a Qt event loop the tests construct the widget
via ``ChatDockWidget.__new__`` and stub out ``QDockWidget.__init__``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure the in-tree chat package is loaded (not an editable install in a
# parallel worktree). See conftest.py for the rationale.
_TREE_SRC = str(ROOT / "src" / "shared" / "python")
if _TREE_SRC in sys.path:
    sys.path.remove(_TREE_SRC)
sys.path.insert(0, _TREE_SRC)
_test_dir = str(Path(__file__).parent.resolve())
for _name in list(sys.modules):
    if not (_name == "chat" or _name.startswith("chat.")):
        continue
    _mod = sys.modules.get(_name)
    _file = getattr(_mod, "__file__", "") or ""
    if _test_dir in _file:
        continue
    del sys.modules[_name]

PyQt6 = pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication, QComboBox  # noqa: E402

# Headless QApplication for QComboBox instantiation.
_app = QApplication.instance() or QApplication(sys.argv[:1])


def _load_in_tree_chat_module():  # type: ignore[no-untyped-def]
    """Load ``chat._chat_dock_widget_qt`` from the in-tree source.

    Bypasses any editable install of ``ud_tools`` that may resolve the
    ``chat`` package to a different worktree (a real concern in this
    fleet where multiple side-by-side checkouts are common).
    """
    import importlib
    import importlib.util

    tree_root = Path(__file__).resolve().parents[3] / "src" / "shared" / "python"
    src_path = tree_root / "chat" / "_chat_dock_widget_qt.py"

    # Force ``chat`` package to resolve to the in-tree path BEFORE we
    # load the submodule, so relative imports inside the submodule work.
    # Prepend in-tree path then nuke any cached chat entries unconditionally.
    sys.path.insert(0, str(tree_root))
    for _name in list(sys.modules):
        if _name == "chat" or _name.startswith("chat."):
            del sys.modules[_name]
    importlib.invalidate_caches()
    chat_mod = importlib.import_module("chat")
    # Pin chat.__path__ to the in-tree directory so relative imports
    # like ``from ._theme_protocol import ...`` always resolve here.
    chat_mod.__path__ = [str(tree_root / "chat")]
    spec = importlib.util.spec_from_file_location("chat._chat_dock_widget_qt", src_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not build spec for {src_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["chat._chat_dock_widget_qt"] = module
    spec.loader.exec_module(module)
    return module


def _make_widget():  # type: ignore[no-untyped-def]
    """Build a bare ChatDockWidget stand-in via type() so we never invoke
    ``QDockWidget.__init__`` (which Qt refuses to skip cleanly)."""
    module = _load_in_tree_chat_module()
    ChatDockWidget = module.ChatDockWidget

    namespace = dict(ChatDockWidget.__dict__)
    # Drop the heavy ``__init__`` so the stand-in constructs cleanly.
    namespace["__init__"] = lambda self: None
    stand_in_cls = type("ChatDockWidgetStandIn", (object,), namespace)
    widget = stand_in_cls()
    return widget


class TestBuildHeaderCombobox:
    """DRY helper ``_build_header_combobox(label, items)``."""

    def test_returns_qcombobox_with_items(self) -> None:
        widget = _make_widget()
        combo = widget._build_header_combobox(
            label="provider",
            items=[("Ollama", "ollama"), ("OpenAI", "openai")],
        )
        assert isinstance(combo, QComboBox)
        assert combo.count() == 2
        assert combo.itemText(0) == "Ollama"
        assert combo.itemData(0) == "ollama"

    @pytest.mark.parametrize("bad_label", ["", "  "])
    def test_empty_label_raises(self, bad_label: str) -> None:
        widget = _make_widget()
        with pytest.raises(ValueError):
            widget._build_header_combobox(label=bad_label, items=[("a", "a")])

    def test_empty_items_raises(self) -> None:
        widget = _make_widget()
        with pytest.raises(ValueError):
            widget._build_header_combobox(label="provider", items=[])

    def test_tooltip_uses_label(self) -> None:
        widget = _make_widget()
        combo = widget._build_header_combobox(
            label="thinking",
            items=[("Off", "none")],
        )
        # Tooltip is derived from label so users see what the dropdown drives.
        assert "thinking" in combo.toolTip().lower()


class TestApplySettingsChange:
    """``_apply_settings_change(field, value)`` is the single change router."""

    def test_routes_provider_change(self) -> None:
        widget = _make_widget()
        widget._ai_provider_combo = QComboBox()
        widget._ai_provider_combo.addItem("Ollama", "ollama")
        widget._ai_provider_combo.addItem("OpenAI", "openai")
        widget._ai_model_combo = QComboBox()
        widget._ai_thinking_combo = QComboBox()
        widget._refresh_ai_model_combo = MagicMock()
        widget._refresh_ai_thinking_combo = MagicMock()
        widget._persist_ai_settings = MagicMock()

        widget._apply_settings_change("provider", "openai")

        widget._refresh_ai_model_combo.assert_called_once()
        widget._refresh_ai_thinking_combo.assert_called_once()
        widget._persist_ai_settings.assert_called_once()

    def test_routes_model_change(self) -> None:
        widget = _make_widget()
        widget._ai_provider_combo = QComboBox()
        widget._ai_provider_combo.addItem("Ollama", "ollama")
        widget._ai_model_combo = QComboBox()
        widget._ai_model_combo.addItem("llama3", "llama3")
        widget._ai_thinking_combo = QComboBox()
        widget._refresh_ai_thinking_combo = MagicMock()
        widget._persist_ai_settings = MagicMock()

        widget._apply_settings_change("model", "llama3")

        widget._refresh_ai_thinking_combo.assert_called_once()
        widget._persist_ai_settings.assert_called_once()

    def test_routes_thinking_change(self) -> None:
        widget = _make_widget()
        widget._ai_provider_combo = QComboBox()
        widget._ai_model_combo = QComboBox()
        widget._ai_thinking_combo = QComboBox()
        widget._ai_thinking_combo.addItem("Low", "low")
        widget._persist_ai_settings = MagicMock()

        widget._apply_settings_change("thinking", "low")

        widget._persist_ai_settings.assert_called_once()

    @pytest.mark.parametrize("bad_field", ["", "  ", "color", "ProvIdEr"])
    def test_unknown_field_raises_value_error(self, bad_field: str) -> None:
        widget = _make_widget()
        widget._ai_provider_combo = QComboBox()
        widget._ai_model_combo = QComboBox()
        widget._ai_thinking_combo = QComboBox()
        widget._persist_ai_settings = MagicMock()
        with pytest.raises(ValueError):
            widget._apply_settings_change(bad_field, "x")

    @pytest.mark.parametrize("field", ["provider", "model", "thinking"])
    def test_empty_value_raises(self, field: str) -> None:
        widget = _make_widget()
        widget._ai_provider_combo = QComboBox()
        widget._ai_model_combo = QComboBox()
        widget._ai_thinking_combo = QComboBox()
        widget._refresh_ai_model_combo = MagicMock()
        widget._refresh_ai_thinking_combo = MagicMock()
        widget._persist_ai_settings = MagicMock()
        with pytest.raises(ValueError):
            widget._apply_settings_change(field, "  ")


class TestHeaderHasAiDropdownsApi:
    """The header API must expose the three new combos + helpers."""

    def test_required_methods_exist(self) -> None:
        widget = _make_widget()
        # These are the public/protected surface added by #2871.
        assert hasattr(widget, "_build_header_combobox")
        assert hasattr(widget, "_apply_settings_change")
        assert hasattr(widget, "switch_provider")
        assert hasattr(widget, "_refresh_ai_model_combo")
        assert hasattr(widget, "_refresh_ai_thinking_combo")
