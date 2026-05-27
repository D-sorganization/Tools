"""TDD coverage for the Tools-side Ollama latency tuning.

Mirrors UpstreamDrift's ``tests/api/test_chat_speed_fixes.py`` for the
Ollama adapter so a future divergence between the two copies of the
adapter trips a CI signal in the affected repo first.

The tests do not require a live Ollama server; httpx is fully stubbed.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

# Evict any stub ``src.shared.python.contracts`` namespace package that a
# sibling test (``test_cli_agent_adapters.py``) registers via
# ``sys.modules.setdefault`` — the stub shadows the real ``contracts.py``
# module that ``ollama_adapter`` imports ``precondition`` from. Tools chat
# fleet drift; remove once the cli-agent test stops poisoning sys.modules.
_stub = sys.modules.get("src.shared.python.contracts")
if _stub is not None and not hasattr(_stub, "precondition"):
    sys.modules.pop("src.shared.python.contracts", None)
    sys.modules.pop("src.shared.python.ai.adapters.ollama_adapter", None)


# ── Tool-declaration wire format ────────────────────────────────────────


def test_tool_declarations_to_ollama_handles_none() -> None:
    from src.shared.python.ai.adapters.ollama_adapter import (
        _tool_declarations_to_ollama,
    )

    assert _tool_declarations_to_ollama(None) == []


def test_tool_declarations_to_ollama_handles_empty() -> None:
    from src.shared.python.ai.adapters.ollama_adapter import (
        _tool_declarations_to_ollama,
    )

    assert _tool_declarations_to_ollama([]) == []


def test_tool_declarations_to_ollama_emits_openai_function_shape() -> None:
    from src.shared.python.ai.adapters.base import ToolDeclaration
    from src.shared.python.ai.adapters.ollama_adapter import (
        _tool_declarations_to_ollama,
    )

    td = ToolDeclaration(
        name="get_weather",
        description="Look up weather",
        parameters={"location": {"type": "string"}},
        required=["location"],
    )

    result = _tool_declarations_to_ollama([td])

    assert result == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]


# ── Streaming POST body ────────────────────────────────────────────────


class _StreamResponse:
    def __init__(self) -> None:
        self.raise_for_status = MagicMock()

    def __enter__(self) -> _StreamResponse:
        return self

    def __exit__(self, *_a: Any) -> None:
        pass

    def iter_lines(self) -> Any:
        import json as _json

        yield _json.dumps({"message": {"content": "ok"}, "done": True})


def _capture_post_body() -> tuple[Any, list[dict[str, Any]]]:
    captured: list[dict[str, Any]] = []
    fake = MagicMock()

    def _stream(method: str, url: str, **kw: Any) -> Any:
        captured.append({"method": method, "url": url, **kw})
        return _StreamResponse()

    fake.stream = _stream
    return fake, captured


def _drain(gen: Any) -> None:
    for _ in gen:
        pass


def test_stream_post_sets_keep_alive() -> None:
    from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter
    from src.shared.python.ai.types import ConversationContext

    fake, captured = _capture_post_body()
    adapter = OllamaAdapter(host="http://localhost:11434", model="llama3.1:8b")
    with patch.object(adapter, "_get_client", return_value=fake):
        _drain(adapter.stream_response("hi", ConversationContext(), []))

    assert captured, "no POST was issued"
    assert captured[0]["json"].get("keep_alive") == "30m"


def test_stream_post_sets_num_ctx_in_options() -> None:
    from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter
    from src.shared.python.ai.types import ConversationContext

    fake, captured = _capture_post_body()
    adapter = OllamaAdapter(host="http://localhost:11434", model="llama3.1:8b")
    with patch.object(adapter, "_get_client", return_value=fake):
        _drain(adapter.stream_response("hi", ConversationContext(), []))

    body = captured[0]["json"]
    assert isinstance(body.get("options"), dict)
    assert body["options"].get("num_ctx") == 4096


def test_stream_post_includes_native_tools_when_supplied() -> None:
    from src.shared.python.ai.adapters.base import ToolDeclaration
    from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter
    from src.shared.python.ai.types import ConversationContext

    fake, captured = _capture_post_body()
    adapter = OllamaAdapter(host="http://localhost:11434", model="llama3.1:8b")
    tools = [
        ToolDeclaration(
            name="weather",
            description="d",
            parameters={"x": {"type": "string"}},
            required=["x"],
        )
    ]
    with patch.object(adapter, "_get_client", return_value=fake):
        _drain(adapter.stream_response("hi", ConversationContext(), tools))

    body = captured[0]["json"]
    assert "tools" in body
    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["function"]["name"] == "weather"


def test_stream_post_omits_tools_when_empty() -> None:
    from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter
    from src.shared.python.ai.types import ConversationContext

    fake, captured = _capture_post_body()
    adapter = OllamaAdapter(host="http://localhost:11434", model="llama3.1:8b")
    with patch.object(adapter, "_get_client", return_value=fake):
        _drain(adapter.stream_response("hi", ConversationContext(), []))

    assert "tools" not in captured[0]["json"]
