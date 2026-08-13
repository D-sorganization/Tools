"""Tests for AdapterFactory caching behaviour.

Verifies that:
- Calling create() twice with identical arguments returns the *same* instance.
- Calling create() with different configurations returns *different* instances.
- clear_cache() causes subsequent create() calls to construct fresh instances.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# The adapter sub-modules can now be safely imported natively
# without artificial module pollution.
import src.shared.python.ai.adapters.bitnet_adapter  # noqa: F401
import src.shared.python.ai.adapters.cline_adapter  # noqa: F401
import src.shared.python.ai.adapters.ollama_adapter  # noqa: F401

# Now safe to import the factory (bypasses the broken ai/__init__.py).
from src.shared.python.ai.adapters.factory import AdapterFactory  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_adapter() -> MagicMock:
    """Return a fresh MagicMock that quacks like a BaseAgentAdapter."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_factory_cache() -> None:
    """Ensure a clean cache state before and after every test."""
    AdapterFactory.clear_cache()
    yield  # type: ignore[misc]
    AdapterFactory.clear_cache()


# ---------------------------------------------------------------------------
# Cache hit — same instance returned for identical configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "kwargs"),
    [
        ("ollama", {"host": "http://localhost:11434", "model": "llama3"}),
        ("cline", {"host": "http://localhost:7777"}),
        ("bitnet", {"model": "Llama3.2-1B.gguf", "host": "/opt/bitnet"}),
    ],
)
def test_create_same_config_returns_same_instance(provider: str, kwargs: dict) -> None:
    """create() called twice with identical args returns the same object."""
    adapter_mock = _make_mock_adapter()

    adapter_class_paths = {
        "ollama": "src.shared.python.ai.adapters.ollama_adapter.OllamaAdapter",
        "cline": "src.shared.python.ai.adapters.cline_adapter.ClineAdapter",
        "bitnet": "src.shared.python.ai.adapters.bitnet_adapter.BitnetAdapter",
    }

    with patch(adapter_class_paths[provider], return_value=adapter_mock):
        first = AdapterFactory.create(provider, **kwargs)
        second = AdapterFactory.create(provider, **kwargs)

    assert first is second, (
        f"Expected the same adapter instance on second create({provider!r}, {kwargs}), "
        f"but got a different object."
    )


# ---------------------------------------------------------------------------
# Cache miss — different configs → different instances
# ---------------------------------------------------------------------------


def test_create_different_configs_returns_different_instances() -> None:
    """create() with different model overrides returns distinct instances."""
    mock_a = _make_mock_adapter()
    mock_b = _make_mock_adapter()

    with patch(
        "src.shared.python.ai.adapters.ollama_adapter.OllamaAdapter",
        side_effect=[mock_a, mock_b],
    ):
        adapter_a = AdapterFactory.create("ollama", model="llama3")
        adapter_b = AdapterFactory.create("ollama", model="mistral")

    assert adapter_a is not adapter_b, (
        "Different model configurations must produce distinct adapter instances."
    )


def test_create_different_hosts_returns_different_instances() -> None:
    """create() with different host overrides returns distinct instances."""
    mock_a = _make_mock_adapter()
    mock_b = _make_mock_adapter()

    with patch(
        "src.shared.python.ai.adapters.ollama_adapter.OllamaAdapter",
        side_effect=[mock_a, mock_b],
    ):
        adapter_a = AdapterFactory.create("ollama", host="http://host-a:11434")
        adapter_b = AdapterFactory.create("ollama", host="http://host-b:11434")

    assert adapter_a is not adapter_b, (
        "Different host configurations must produce distinct adapter instances."
    )


# ---------------------------------------------------------------------------
# clear_cache() causes fresh construction
# ---------------------------------------------------------------------------


def test_clear_cache_causes_fresh_construction() -> None:
    """After clear_cache(), create() constructs a new adapter instance."""
    mock_first = _make_mock_adapter()
    mock_second = _make_mock_adapter()

    with patch(
        "src.shared.python.ai.adapters.ollama_adapter.OllamaAdapter",
        side_effect=[mock_first, mock_second],
    ):
        first = AdapterFactory.create("ollama", model="llama3")
        AdapterFactory.clear_cache()
        second = AdapterFactory.create("ollama", model="llama3")

    assert first is not second, (
        "After clear_cache(), create() must construct a fresh adapter instance."
    )


def test_clear_cache_empties_internal_dict() -> None:
    """clear_cache() leaves the internal _cache dict empty."""
    adapter_mock = _make_mock_adapter()

    with patch(
        "src.shared.python.ai.adapters.ollama_adapter.OllamaAdapter",
        return_value=adapter_mock,
    ):
        AdapterFactory.create("ollama")

    assert len(AdapterFactory._cache) == 1, (
        "Cache should have one entry after create()."
    )
    AdapterFactory.clear_cache()
    assert len(AdapterFactory._cache) == 0, "Cache should be empty after clear_cache()."


# ---------------------------------------------------------------------------
# Constructor called exactly once per unique key
# ---------------------------------------------------------------------------


def test_constructor_called_once_for_repeated_create() -> None:
    """The underlying adapter constructor is only called once for cache hits."""
    with patch(
        "src.shared.python.ai.adapters.ollama_adapter.OllamaAdapter",
    ) as mock_cls:
        mock_cls.return_value = _make_mock_adapter()
        AdapterFactory.create("ollama", model="llama3")
        AdapterFactory.create("ollama", model="llama3")
        AdapterFactory.create("ollama", model="llama3")

    assert mock_cls.call_count == 1, (
        f"OllamaAdapter constructor should be called once, got {mock_cls.call_count}."
    )
