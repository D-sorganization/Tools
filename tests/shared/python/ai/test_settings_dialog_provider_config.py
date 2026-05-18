"""Regression tests for provider-specific settings widgets."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

from PyQt6.QtWidgets import QLabel, QPushButton

# Bootstrap the shared AI package the same way the existing AI tests do.
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PACKAGE_STUBS: list[tuple[str, str | None]] = [
    ("src", "src"),
    ("src.shared", "src/shared"),
    ("src.shared.python", "src/shared/python"),
    ("src.shared.python.config", "src/shared/python/config"),
    ("src.shared.python.ai", "src/shared/python/ai"),
    ("src.shared.python.ai.adapters", "src/shared/python/ai/adapters"),
]
for _mod_name, _rel_path in _PACKAGE_STUBS:
    if _mod_name not in sys.modules:
        import types

        _stub = types.ModuleType(_mod_name)
        if _rel_path is not None:
            _stub.__path__ = [str(ROOT / _rel_path)]
        sys.modules[_mod_name] = _stub


_logging_config_stub = sys.modules.setdefault(
    "src.shared.python.logging_pkg.logging_config",
    types.ModuleType("src.shared.python.logging_pkg.logging_config"),
)
_logging_config_stub.get_logger = logging.getLogger  # type: ignore[attr-defined]
_logging_config_stub.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]

_env_stub = sys.modules.setdefault(
    "src.shared.python.config.environment",
    types.ModuleType("src.shared.python.config.environment"),
)
_env_stub.get_env = lambda key, default=None, required=False: default  # type: ignore[attr-defined]  # noqa: E501
_env_stub.get_env_float = lambda key, default=0.0: float(default)  # type: ignore[attr-defined]  # noqa: E501

from src.shared.python.ai.gui.settings_dialog import AIProvider, ProviderConfigWidget


def _label_texts(widget: ProviderConfigWidget) -> set[str]:
    return {
        label.text()
        for label in widget.findChildren(QLabel)
        if isinstance(label.text(), str) and label.text()
    }


def _button_texts(widget: ProviderConfigWidget) -> set[str]:
    return {
        button.text()
        for button in widget.findChildren(QPushButton)
        if isinstance(button.text(), str) and button.text()
    }


def test_ollama_provider_config_shows_ollama_controls(qapp) -> None:
    widget = ProviderConfigWidget(AIProvider.OLLAMA)

    assert "Ollama Host:" in _label_texts(widget)
    assert "🔄 Refresh Available Models" in _button_texts(widget)
    widget.close()


def test_cline_provider_config_uses_cline_specific_controls(qapp) -> None:
    widget = ProviderConfigWidget(AIProvider.CLINE_CLI)

    labels = _label_texts(widget)
    buttons = _button_texts(widget)

    assert "Cline Host:" in labels
    assert "Ollama Host:" not in labels
    assert "Test Connection" in buttons
    assert "🔄 Refresh Available Models" not in buttons
    widget.close()


def test_bitnet_provider_config_uses_bitnet_specific_controls(qapp) -> None:
    widget = ProviderConfigWidget(AIProvider.BITNET)

    labels = _label_texts(widget)
    buttons = _button_texts(widget)

    assert "BitNet Root:" in labels
    assert "Ollama Host:" not in labels
    assert "Test Connection" not in buttons
    assert "🔄 Refresh Available Models" not in buttons
    assert any("main model selector" in text for text in labels)
    widget.close()
