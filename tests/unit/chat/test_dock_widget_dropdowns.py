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
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PyQt6 = pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication, QComboBox, QDockWidget  # noqa: E402

# Headless QApplication for QComboBox instantiation.
_app = QApplication.instance() or QApplication(sys.argv[:1])


def _make_widget():  # type: ignore[no-untyped-def]
    """Build a ChatDockWidget without invoking the heavy ``__init__``."""
    from chat._chat_dock_widget_qt import ChatDockWidget

    with patch.object(QDockWidget, "__init__", return_value=None):
        widget = ChatDockWidget.__new__(ChatDockWidget)
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
