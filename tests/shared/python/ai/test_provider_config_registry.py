# ruff: noqa: E501
"""Tests for ProviderConfigRegistry (Tools #2762)."""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QLabel, QWidget  # noqa: E402

from src.shared.python.ai.gui._provider_config_registry import (  # noqa: E402
    ProviderConfigRegistry,
)
from src.shared.python.ai.gui._provider_registry_data import AIProvider  # noqa: E402


def test_default_registrations_cover_all_providers(qapp) -> None:
    for provider in AIProvider:
        assert ProviderConfigRegistry.is_registered(provider.name), (
            f"missing registration for {provider}"
        )


def test_get_widget_returns_distinct_instances(qapp) -> None:
    a = ProviderConfigRegistry.get_widget(AIProvider.OPENAI)
    b = ProviderConfigRegistry.get_widget(AIProvider.OPENAI)
    assert a is not b
    a.close()
    b.close()


def test_get_widget_unknown_raises(qapp) -> None:
    with pytest.raises(KeyError):
        ProviderConfigRegistry.get_widget("definitely-not-real")


def test_register_and_unregister_round_trip(qapp) -> None:
    class _DummyWidget(QWidget):
        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            QLabel("dummy", self)

    ProviderConfigRegistry.register("dummy_provider", _DummyWidget)
    try:
        assert ProviderConfigRegistry.is_registered("dummy_provider")
        widget = ProviderConfigRegistry.get_widget("dummy_provider")
        assert isinstance(widget, _DummyWidget)
        widget.close()
    finally:
        ProviderConfigRegistry.unregister("dummy_provider")
    assert not ProviderConfigRegistry.is_registered("dummy_provider")


def test_register_rejects_non_widget_factory() -> None:
    with pytest.raises(TypeError):
        ProviderConfigRegistry.register("bad", object)  # type: ignore[arg-type]


def test_register_rejects_empty_id() -> None:
    with pytest.raises(ValueError):
        ProviderConfigRegistry.register("", QWidget)


def test_widgets_are_isolated_per_provider(qapp) -> None:
    """An anthropic widget must not share Qt children with an openai widget."""
    aw = ProviderConfigRegistry.get_widget(AIProvider.ANTHROPIC)
    ow = ProviderConfigRegistry.get_widget(AIProvider.OPENAI)
    assert type(aw).__name__ == "AnthropicConfigWidget"
    assert type(ow).__name__ == "OpenAIConfigWidget"
    aw.close()
    ow.close()
