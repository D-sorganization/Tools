# ruff: noqa: E501
"""Regression tests for provider-specific settings widgets."""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QPushButton

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
