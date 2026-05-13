"""Regression tests for the shared chat BitNet adapter source contract."""

from __future__ import annotations

from pathlib import Path


def test_bitnet_adapter_uses_current_capability_contract() -> None:
    """BitNet must construct ProviderCapabilities with current field names."""
    source = Path("src/shared/python/ai/adapters/bitnet_adapter.py").read_text(
        encoding="utf-8"
    )

    assert "ProviderCapability.STREAMING" in source
    assert "ProviderCapability.SYSTEM_MESSAGE" in source
    assert "max_tokens=2048" in source
    assert 'provider_name="bitnet"' in source
    assert "supports_streaming" not in source
    assert "context_window" not in source


def test_bitnet_adapter_uses_current_agent_response_contract() -> None:
    """BitNet responses use AgentResponse metadata, not removed fields."""
    source = Path("src/shared/python/ai/adapters/bitnet_adapter.py").read_text(
        encoding="utf-8"
    )

    assert 'metadata={"stdout": result.stdout}' in source
    assert "raw_response=" not in source
    assert 'role="assistant"' not in source


def test_factory_registers_bitnet_provider() -> None:
    """Factory provider lists and creation path include BitNet."""
    source = Path("src/shared/python/ai/adapters/factory.py").read_text(
        encoding="utf-8"
    )

    assert '"bitnet"' in source
    assert 'if provider == "bitnet":' in source
    assert "BitnetAdapter(model=model, bitnet_root=host)" in source
