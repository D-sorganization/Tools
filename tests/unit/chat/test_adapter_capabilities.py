"""Adapter capability contract tests for Tools issue #2871.

Each ``BaseAgentAdapter`` subclass must expose two new capability
methods so that ``ChatDockWidget`` can populate the new
provider/model/thinking dropdowns without poking into adapter
internals (Law of Demeter):

* ``list_models() -> list[str]``  — non-empty list of provider-known
  model identifiers, never network-dependent in unit-test path.
* ``thinking_capabilities() -> ThinkingCapabilities`` — describes the
  reasoning-budget levels (``none``/``low``/``medium``/``high``) the
  active model supports.

These tests run fully offline by mocking each provider client.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from chat_contracts.models import (  # noqa: E402
    ThinkingCapabilities,
    ThinkingLevel,
)

from src.shared.python.ai.adapters.base import BaseAgentAdapter  # noqa: E402

# ---------------------------------------------------------------------------
# ThinkingLevel + ThinkingCapabilities DbC tests
# ---------------------------------------------------------------------------


class TestThinkingLevelContract:
    """``ThinkingLevel`` is a frozen value object with strict DbC."""

    def test_valid_level_constructs(self) -> None:
        level = ThinkingLevel(name="low", budget_tokens=1024, label="Low")
        assert level.name == "low"
        assert level.budget_tokens == 1024
        assert level.label == "Low"

    @pytest.mark.parametrize(
        "bad_name",
        ["", "  ", "extreme", "off", "high ", "LOW"],
    )
    def test_invalid_name_raises_value_error(self, bad_name: str) -> None:
        with pytest.raises(ValueError):
            ThinkingLevel(name=bad_name, budget_tokens=0, label="x")

    @pytest.mark.parametrize("bad_budget", [-1, -100])
    def test_negative_budget_raises_value_error(self, bad_budget: int) -> None:
        with pytest.raises(ValueError):
            ThinkingLevel(name="low", budget_tokens=bad_budget, label="Low")

    def test_zero_budget_allowed_for_none_level(self) -> None:
        # The "none" level has zero budget by definition; allowed.
        level = ThinkingLevel(name="none", budget_tokens=0, label="Off")
        assert level.budget_tokens == 0

    def test_is_frozen(self) -> None:
        level = ThinkingLevel(name="low", budget_tokens=1, label="L")
        with pytest.raises((AttributeError, Exception)):
            level.name = "high"  # type: ignore[misc]


class TestThinkingCapabilitiesContract:
    """``ThinkingCapabilities`` aggregates ``ThinkingLevel`` instances."""

    def _make_levels(self) -> tuple[ThinkingLevel, ...]:
        return (
            ThinkingLevel(name="none", budget_tokens=0, label="Off"),
            ThinkingLevel(name="low", budget_tokens=512, label="Low"),
        )

    def test_valid_capabilities_construct(self) -> None:
        caps = ThinkingCapabilities(
            provider="openai",
            levels=self._make_levels(),
            default_level_name="none",
        )
        assert caps.provider == "openai"
        assert len(caps.levels) == 2

    def test_empty_provider_raises(self) -> None:
        with pytest.raises(ValueError):
            ThinkingCapabilities(
                provider="",
                levels=self._make_levels(),
                default_level_name="none",
            )

    def test_whitespace_provider_raises(self) -> None:
        with pytest.raises(ValueError):
            ThinkingCapabilities(
                provider="   ",
                levels=self._make_levels(),
                default_level_name="none",
            )

    def test_empty_levels_raises(self) -> None:
        with pytest.raises(ValueError):
            ThinkingCapabilities(
                provider="openai",
                levels=(),
                default_level_name="none",
            )

    def test_default_level_not_in_levels_raises(self) -> None:
        with pytest.raises(ValueError):
            ThinkingCapabilities(
                provider="openai",
                levels=self._make_levels(),
                default_level_name="high",
            )

    def test_level_names_returns_ordered_tuple(self) -> None:
        caps = ThinkingCapabilities(
            provider="openai",
            levels=self._make_levels(),
            default_level_name="none",
        )
        assert caps.level_names() == ("none", "low")

    def test_find_level_returns_matching(self) -> None:
        caps = ThinkingCapabilities(
            provider="openai",
            levels=self._make_levels(),
            default_level_name="none",
        )
        match = caps.find_level("low")
        assert match is not None
        assert match.name == "low"

    def test_find_level_missing_returns_none(self) -> None:
        caps = ThinkingCapabilities(
            provider="openai",
            levels=self._make_levels(),
            default_level_name="none",
        )
        assert caps.find_level("high") is None


# ---------------------------------------------------------------------------
# Adapter factory helpers (mocked clients)
# ---------------------------------------------------------------------------


def _make_anthropic_adapter() -> BaseAgentAdapter:
    from src.shared.python.ai.adapters.anthropic_adapter import AnthropicAdapter

    adapter = AnthropicAdapter(api_key="test-key", model="claude-3-5-sonnet-20240620")
    client_mock = MagicMock()
    client_mock.models.list.side_effect = RuntimeError("network disabled in test")
    adapter._client = client_mock
    return adapter


def _make_openai_adapter() -> BaseAgentAdapter:
    from src.shared.python.ai.adapters.openai_adapter import OpenAIAdapter

    adapter = OpenAIAdapter(api_key="test-key", model="gpt-4-turbo")
    client_mock = MagicMock()
    client_mock.models.list.side_effect = RuntimeError("network disabled in test")
    adapter._client = client_mock
    return adapter


def _make_ollama_adapter() -> BaseAgentAdapter:
    from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter

    adapter = OllamaAdapter()
    client_mock = MagicMock()
    client_mock.get.side_effect = RuntimeError("network disabled in test")
    adapter._client = client_mock
    return adapter


def _make_gemini_adapter() -> BaseAgentAdapter:
    """Build a Gemini adapter without invoking the SDK's process-global state."""
    from src.shared.python.ai.adapters import gemini_adapter as gemini_mod

    adapter = gemini_mod.GeminiAdapter.__new__(gemini_mod.GeminiAdapter)
    adapter._api_key = "test-key"
    adapter._model_name = "gemini-pro"
    adapter._client = None
    adapter._model = MagicMock()
    return adapter


def _make_cline_adapter() -> BaseAgentAdapter:
    from src.shared.python.ai.adapters.cline_adapter import ClineAdapter

    adapter = ClineAdapter()
    client_mock = MagicMock()
    client_mock.get.side_effect = RuntimeError("network disabled in test")
    adapter._client = client_mock
    return adapter


def _make_bitnet_adapter() -> BaseAgentAdapter:
    from src.shared.python.ai.adapters.bitnet_adapter import BitnetAdapter

    return BitnetAdapter(
        model="test.gguf", bitnet_root=str(Path(tempfile.gettempdir()))
    )


def _make_rust_adapter() -> BaseAgentAdapter:
    ai_backend_mock = MagicMock()
    engine_mock = MagicMock()
    engine_mock.generate_response.return_value = "rust response"
    config_mock = MagicMock()
    config_mock.model = "gpt-4"
    ai_backend_mock.AIConfig.return_value = config_mock
    ai_backend_mock.AIEngine.return_value = engine_mock
    ai_backend_mock.MemoryManager.return_value = MagicMock()
    ai_backend_mock.RagPipeline.return_value = MagicMock()

    with patch.dict(sys.modules, {"ai_backend": ai_backend_mock}):
        from src.shared.python.ai.adapters.rust_adapter import RustAgentAdapter

        return RustAgentAdapter(
            api_key="k",
            base_url="http://localhost",
            model="gpt-4",
        )


_ADAPTER_FACTORIES = [
    ("anthropic", _make_anthropic_adapter),
    ("openai", _make_openai_adapter),
    ("ollama", _make_ollama_adapter),
    ("gemini", _make_gemini_adapter),
    ("cline", _make_cline_adapter),
    ("bitnet", _make_bitnet_adapter),
    ("rust", _make_rust_adapter),
]


# ---------------------------------------------------------------------------
# Per-adapter capability contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider_name", "factory"),
    _ADAPTER_FACTORIES,
    ids=[name for name, _ in _ADAPTER_FACTORIES],
)
class TestAdapterCapabilityContract:
    """Every concrete adapter must implement the #2871 capability protocol."""

    def test_list_models_returns_non_empty_list_of_strings(
        self, provider_name: str, factory
    ) -> None:
        adapter = factory()
        models = adapter.list_models()
        assert isinstance(models, list), (
            f"{provider_name}: list_models() must return a list"
        )
        assert models, f"{provider_name}: list_models() must not be empty"
        for entry in models:
            assert isinstance(entry, str) and entry.strip(), (
                f"{provider_name}: every model id must be a non-empty string"
            )

    def test_list_models_is_offline_safe(self, provider_name: str, factory) -> None:
        """``list_models()`` must fall back to a static catalogue when the
        provider client raises (precondition: ``_client`` is a mock that
        raises on any network call)."""
        adapter = factory()
        # Should not raise even though every patched client raises.
        models = adapter.list_models()
        assert models

    def test_thinking_capabilities_returns_dataclass(
        self, provider_name: str, factory
    ) -> None:
        adapter = factory()
        caps = adapter.thinking_capabilities()
        assert isinstance(caps, ThinkingCapabilities)
        assert caps.provider
        # Must always include at least the "none" level.
        names = caps.level_names()
        assert "none" in names, (
            f"{provider_name}: thinking_capabilities must include 'none'"
        )
        assert caps.default_level_name in names

    def test_thinking_capabilities_default_resolvable(
        self, provider_name: str, factory
    ) -> None:
        adapter = factory()
        caps = adapter.thinking_capabilities()
        assert caps.find_level(caps.default_level_name) is not None
