"""Tests for ProviderConfigRegistry (Tools #2762)."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

# Same bootstrap shim used by test_settings_dialog_provider_config.py.
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




_logging_config_stub = sys.modules.setdefault("src.shared.python.logging_pkg.logging_config", types.ModuleType("src.shared.python.logging_pkg.logging_config"))
_logging_config_stub.get_logger = logging.getLogger  # type: ignore[attr-defined]
_logging_config_stub.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]

_env_stub = sys.modules.setdefault("src.shared.python.config.environment", types.ModuleType("src.shared.python.config.environment"))
_env_stub.get_env = lambda key, default=None, required=False: default  # type: ignore[attr-defined]
_env_stub.get_env_float = lambda key, default=0.0: float(default)  # type: ignore[attr-defined]

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
