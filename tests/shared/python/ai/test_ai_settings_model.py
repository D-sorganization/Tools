# ruff: noqa: E501
"""Tests for the headless AISettings dataclass round-trip (Tools #2762)."""

from __future__ import annotations

from src.shared.python.ai._settings_model import AISettings  # noqa: E402
from src.shared.python.ai.access_policy import ChatAccessMode  # noqa: E402
from src.shared.python.ai.gui._provider_registry_data import AIProvider  # noqa: E402


def test_default_settings_values() -> None:
    s = AISettings()
    assert s.provider is AIProvider.OLLAMA
    assert s.response_style == "standard"
    assert s.streaming_enabled is True
    assert s.access_mode == ChatAccessMode.NO_REPO_ACCESS


def test_to_dict_round_trip_preserves_fields() -> None:
    original = AISettings(
        provider=AIProvider.ANTHROPIC,
        model="claude-3-5-sonnet-20241022",
        response_style="concise",
        chat_mode="agent",
        streaming_enabled=False,
        rag_enabled=False,
        auto_index_on_open=True,
        access_mode=ChatAccessMode.AGENT_TOOLS,
    )
    payload = original.to_dict()

    # JSON-serialisable
    import json

    json.dumps(payload)

    restored = AISettings.from_dict(payload)
    assert restored.provider is AIProvider.ANTHROPIC
    assert restored.model == "claude-3-5-sonnet-20241022"
    assert restored.response_style == "concise"
    assert restored.chat_mode == "agent"
    assert restored.streaming_enabled is False
    assert restored.rag_enabled is False
    assert restored.auto_index_on_open is True
    assert restored.access_mode == ChatAccessMode.AGENT_TOOLS


def test_from_dict_normalises_unknown_response_style() -> None:
    s = AISettings.from_dict({"response_style": "verbose"})
    assert s.response_style == "standard"


def test_from_dict_normalises_unknown_chat_mode() -> None:
    s = AISettings.from_dict({"chat_mode": "rogue"})
    assert s.chat_mode == "ask"


def test_from_dict_unknown_provider_falls_back_to_ollama() -> None:
    s = AISettings.from_dict({"provider": "NEVER_SHIPPED"})
    assert s.provider is AIProvider.OLLAMA
