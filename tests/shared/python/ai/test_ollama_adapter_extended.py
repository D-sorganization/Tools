"""Extended unit tests for OllamaAdapter.

Covers the previously untested happy-path, error-path, and capability
contracts. The existing test_ollama_adapter.py only tests _format_messages
behavior introduced in PR #2750; this file covers the remaining surface:

  - send_message happy path
  - send_message → AIConnectionError on ConnectError
  - send_message → AITimeoutError on TimeoutException
  - send_message → AIProviderError on HTTP error
  - validate_connection success and failure cases
  - list_available_models happy path and connection failure
  - capabilities report STREAMING + SYSTEM_MESSAGE; FUNCTION_CALLING for llama3
  - stream_response yields ordered AgentChunk instances
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: mirror the approach in test_ollama_adapter.py
# ---------------------------------------------------------------------------

ROOT = __import__("pathlib").Path(__file__).resolve().parents[4]
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
_logging_config_stub.get_logger = MagicMock()  # type: ignore[attr-defined]

_env_stub = sys.modules.get("src.shared.python.config.environment")
if not isinstance(_env_stub, types.ModuleType):
    _env_stub = types.ModuleType("src.shared.python.config.environment")
    sys.modules["src.shared.python.config.environment"] = _env_stub
_env_stub.get_env = lambda key, default=None, required=False: default
_env_stub.get_env_float = lambda key, default=0.0: float(default)

from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter  # noqa: E402
from src.shared.python.ai.exceptions import (  # noqa: E402
    AIConnectionError,
    AIProviderError,
    AITimeoutError,
)
from src.shared.python.ai.types import (  # noqa: E402
    ConversationContext,
    ExpertiseLevel,
    ProviderCapability,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def adapter() -> OllamaAdapter:
    return OllamaAdapter(host="http://localhost:11434", model="llama3.1:8b")


@pytest.fixture()
def context() -> ConversationContext:
    return ConversationContext(
        session_id="test-session",
        user_expertise=ExpertiseLevel.INTERMEDIATE,
    )


def _make_mock_response(body: dict[str, Any], status_code: int = 200) -> MagicMock:
    """Return a fake httpx response with .json() and .raise_for_status()."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    if status_code >= 400:
        resp.raise_for_status.side_effect = _httpx_status_error(status_code)
    else:
        resp.raise_for_status.return_value = None
    return resp


def _httpx_status_error(status_code: int) -> Exception:
    import httpx

    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("error", request=request, response=response)


# ---------------------------------------------------------------------------
# send_message
# ---------------------------------------------------------------------------


class TestSendMessageHappyPath:
    def test_returns_agent_response_with_content(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """send_message returns an AgentResponse containing the model reply."""
        fake_body = {
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": "Hello, world!"},
            "done": True,
        }
        mock_client = MagicMock()
        mock_client.post.return_value = _make_mock_response(fake_body)
        adapter._client = mock_client

        response = adapter.send_message("hi", context, [])

        assert response.content == "Hello, world!"
        assert response.finish_reason == "stop"
        assert response.tool_calls == []

    def test_finish_reason_is_length_when_not_done(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """finish_reason is 'length' when the model did not finish naturally."""
        fake_body = {
            "message": {"content": "partial..."},
            "done": False,
        }
        mock_client = MagicMock()
        mock_client.post.return_value = _make_mock_response(fake_body)
        adapter._client = mock_client

        response = adapter.send_message("hi", context, [])

        assert response.finish_reason == "length"

    def test_usage_fields_are_populated_from_eval_counts(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """prompt_eval_count and eval_count map to usage dict."""
        fake_body = {
            "message": {"content": "ok"},
            "done": True,
            "prompt_eval_count": 42,
            "eval_count": 17,
        }
        mock_client = MagicMock()
        mock_client.post.return_value = _make_mock_response(fake_body)
        adapter._client = mock_client

        response = adapter.send_message("hi", context, [])

        assert response.usage["input_tokens"] == 42
        assert response.usage["output_tokens"] == 17

    def test_tool_calls_are_parsed_from_response(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """Tool calls in the model response are extracted."""
        fake_body = {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc_0",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "London"},
                        },
                    }
                ],
            },
            "done": True,
        }
        mock_client = MagicMock()
        mock_client.post.return_value = _make_mock_response(fake_body)
        adapter._client = mock_client

        response = adapter.send_message("hi", context, [])

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments == {"city": "London"}


class TestSendMessageErrors:
    def test_connect_error_raises_ai_connection_error(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """httpx.ConnectError is translated to AIConnectionError."""
        import httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("refused")
        adapter._client = mock_client

        with pytest.raises(AIConnectionError, match="ollama"):
            adapter.send_message("hi", context, [])

    def test_timeout_raises_ai_timeout_error(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """httpx.TimeoutException is translated to AITimeoutError."""
        import httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.TimeoutException("timed out")
        adapter._client = mock_client

        with pytest.raises(AITimeoutError):
            adapter.send_message("hi", context, [])

    def test_http_error_raises_ai_provider_error(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """Non-connect HTTP errors wrap as AIProviderError."""
        import httpx

        mock_client = MagicMock()
        request = httpx.Request("POST", "http://localhost/api/chat")
        response = httpx.Response(500, request=request)
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "server error", request=request, response=response
        )
        adapter._client = mock_client

        with pytest.raises(AIProviderError):
            adapter.send_message("hi", context, [])


# ---------------------------------------------------------------------------
# validate_connection
# ---------------------------------------------------------------------------


class TestValidateConnection:
    def test_success_when_model_is_available(self, adapter: OllamaAdapter) -> None:
        """Returns (True, message) when model appears in /api/tags."""
        fake_body = {"models": [{"name": "llama3.1:8b"}]}
        mock_client = MagicMock()
        mock_client.get.return_value = _make_mock_response(fake_body)
        adapter._client = mock_client

        ok, msg = adapter.validate_connection()

        assert ok is True
        assert "llama3.1" in msg

    def test_failure_when_no_models_installed(self, adapter: OllamaAdapter) -> None:
        """Returns (False, message) when model list is empty."""
        fake_body: dict[str, Any] = {"models": []}
        mock_client = MagicMock()
        mock_client.get.return_value = _make_mock_response(fake_body)
        adapter._client = mock_client

        ok, msg = adapter.validate_connection()

        assert ok is False
        assert "No models" in msg or "ollama pull" in msg

    def test_failure_when_model_not_in_list(self, adapter: OllamaAdapter) -> None:
        """Returns (False, message) when model is not among available models."""
        fake_body = {"models": [{"name": "mistral"}, {"name": "codellama"}]}
        mock_client = MagicMock()
        mock_client.get.return_value = _make_mock_response(fake_body)
        adapter._client = mock_client

        ok, msg = adapter.validate_connection()

        assert ok is False
        assert "llama3.1" in msg or "not found" in msg

    def test_failure_on_connect_error(self, adapter: OllamaAdapter) -> None:
        """Returns (False, message) when Ollama server is unreachable."""
        import httpx

        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ConnectError("refused")
        adapter._client = mock_client

        ok, msg = adapter.validate_connection()

        assert ok is False
        assert "connect" in msg.lower() or "running" in msg.lower()


# ---------------------------------------------------------------------------
# list_available_models
# ---------------------------------------------------------------------------


class TestListAvailableModels:
    def test_returns_model_names(self, adapter: OllamaAdapter) -> None:
        """Returns a list of model name strings."""
        fake_body = {
            "models": [
                {"name": "llama3.1:8b"},
                {"name": "mistral:latest"},
            ]
        }
        mock_client = MagicMock()
        mock_client.get.return_value = _make_mock_response(fake_body)
        adapter._client = mock_client

        names = adapter.list_available_models()

        assert names == ["llama3.1:8b", "mistral:latest"]

    def test_raises_ai_connection_error_on_failure(
        self, adapter: OllamaAdapter
    ) -> None:
        """Network failure wraps as AIConnectionError."""
        import httpx

        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ConnectError("refused")
        adapter._client = mock_client

        with pytest.raises(AIConnectionError):
            adapter.list_available_models()


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


class TestCapabilities:
    @pytest.mark.parametrize(
        "model",
        ["llama3.1:8b", "llama3:latest", "mistral:7b"],
    )
    def test_llama3_and_mistral_support_function_calling(self, model: str) -> None:
        """Models matching 'llama3' or 'mistral' advertise FUNCTION_CALLING."""
        adapter = OllamaAdapter(model=model)
        caps = adapter.capabilities
        assert ProviderCapability.FUNCTION_CALLING in caps.supported

    def test_generic_model_does_not_advertise_function_calling(self) -> None:
        """Unknown models do not claim function calling support."""
        adapter = OllamaAdapter(model="phi2")
        caps = adapter.capabilities
        assert ProviderCapability.FUNCTION_CALLING not in caps.supported

    def test_streaming_and_system_message_always_supported(self) -> None:
        """STREAMING and SYSTEM_MESSAGE are always in capabilities."""
        adapter = OllamaAdapter(model="phi2")
        caps = adapter.capabilities
        assert ProviderCapability.STREAMING in caps.supported
        assert ProviderCapability.SYSTEM_MESSAGE in caps.supported

    def test_capabilities_provider_name_is_ollama(self) -> None:
        """provider_name is always 'ollama'."""
        adapter = OllamaAdapter(model="llama3.1:8b")
        assert adapter.capabilities.provider_name == "ollama"


# ---------------------------------------------------------------------------
# stream_response
# ---------------------------------------------------------------------------


class TestStreamResponse:
    def test_yields_agent_chunks_in_order(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """stream_response yields AgentChunk for each SSE line."""
        lines = [
            json.dumps({"message": {"content": "hel"}, "done": False}),
            json.dumps({"message": {"content": "lo"}, "done": False}),
            json.dumps({"message": {"content": "!"}, "done": True}),
        ]
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = iter(lines)

        mock_client = MagicMock()
        enter = MagicMock(return_value=mock_response)
        mock_client.stream.return_value.__enter__ = enter
        mock_client.stream.return_value.__exit__ = MagicMock(return_value=False)
        adapter._client = mock_client

        chunks = list(adapter.stream_response("hi", context, []))

        assert [c.content for c in chunks] == ["hel", "lo", "!"]

    def test_final_chunk_has_is_final_true(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """The chunk with done=True has is_final=True."""
        lines = [
            json.dumps({"message": {"content": "part1"}, "done": False}),
            json.dumps({"message": {"content": "part2"}, "done": True}),
        ]
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = iter(lines)

        mock_client = MagicMock()
        enter = MagicMock(return_value=mock_response)
        mock_client.stream.return_value.__enter__ = enter
        mock_client.stream.return_value.__exit__ = MagicMock(return_value=False)
        adapter._client = mock_client

        chunks = list(adapter.stream_response("hi", context, []))

        assert chunks[0].is_final is False
        assert chunks[1].is_final is True

    def test_malformed_json_raises_ai_provider_error(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """A non-JSON line in the stream raises AIProviderError."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = iter(["not-json"])

        mock_client = MagicMock()
        enter = MagicMock(return_value=mock_response)
        mock_client.stream.return_value.__enter__ = enter
        mock_client.stream.return_value.__exit__ = MagicMock(return_value=False)
        adapter._client = mock_client

        with pytest.raises(AIProviderError):
            list(adapter.stream_response("hi", context, []))
