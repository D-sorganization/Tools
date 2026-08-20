"""Unit tests for OllamaAdapter.

Covers: initialization, send_message (success + error paths), stream_response,
validate_connection (connected / model-missing / unreachable), list_available_models,
_parse_response, _format_messages (response style), and capabilities.

Bootstrap pattern mirrors test_bitnet_adapter.py.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

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
# Helpers
# ---------------------------------------------------------------------------


def _make_context(response_style: str = "standard") -> ConversationContext:
    ctx = ConversationContext(
        session_id="test-session",
        user_expertise=ExpertiseLevel.INTERMEDIATE,
    )
    ctx.response_style = response_style
    return ctx


def _make_ollama_response(
    content: str = "hello",
    model: str = "llama3.1:8b",
    done: bool = True,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> dict:
    return {
        "model": model,
        "message": {"role": "assistant", "content": content},
        "done": done,
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def adapter() -> OllamaAdapter:
    return OllamaAdapter(host="http://localhost:11434", model="test-model")


@pytest.fixture()
def llama3_adapter() -> OllamaAdapter:
    """Adapter with a llama3-family model (supports function calling)."""
    return OllamaAdapter(host="http://localhost:11434", model="llama3.1:8b")


@pytest.fixture()
def context() -> ConversationContext:
    return ConversationContext(
        session_id="test-session",
        user_expertise=ExpertiseLevel.INTERMEDIATE,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for OllamaAdapter.__init__."""

    @pytest.mark.unit
    def test_host_stored_without_trailing_slash(self) -> None:
        """Trailing slashes are stripped from the host URL."""
        a = OllamaAdapter(host="http://localhost:11434/")
        assert not a._host.endswith("/")

    @pytest.mark.unit
    def test_custom_timeout_stored(self) -> None:
        """Explicit timeout is stored verbatim."""
        a = OllamaAdapter(timeout=30.0)
        assert a._timeout == 30.0

    @pytest.mark.unit
    def test_client_is_none_before_first_use(self) -> None:
        """HTTP client is lazily initialised — None at construction time."""
        a = OllamaAdapter()
        assert a._client is None


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


class TestCapabilities:
    """Tests for OllamaAdapter.capabilities property."""

    @pytest.mark.unit
    def test_provider_name_is_ollama(self, adapter: OllamaAdapter) -> None:
        assert adapter.capabilities.provider_name == "ollama"

    @pytest.mark.unit
    def test_streaming_capability_always_present(self, adapter: OllamaAdapter) -> None:
        assert ProviderCapability.STREAMING in adapter.capabilities.supported

    @pytest.mark.unit
    def test_llama3_model_has_function_calling(
        self, llama3_adapter: OllamaAdapter
    ) -> None:
        """llama3-family models advertise FUNCTION_CALLING capability."""
        caps = llama3_adapter.capabilities
        assert ProviderCapability.FUNCTION_CALLING in caps.supported

    @pytest.mark.unit
    def test_non_llama3_model_no_function_calling(self, adapter: OllamaAdapter) -> None:
        """Non-llama3 models do not advertise FUNCTION_CALLING."""
        assert ProviderCapability.FUNCTION_CALLING not in adapter.capabilities.supported


# ---------------------------------------------------------------------------
# _format_messages (response style)
# ---------------------------------------------------------------------------


class TestFormatMessages:
    """Tests for OllamaAdapter._format_messages."""

    @pytest.mark.unit
    def test_concise_style_in_system_prompt(self, adapter: OllamaAdapter) -> None:
        """ResponseStyle=concise injects concise instructions into system prompt."""
        ctx = _make_context(response_style="concise")
        with patch("httpx.Client"):
            messages = adapter._format_messages(ctx, "Hello", [])
        system_msg = next(m for m in messages if m["role"] == "system")
        assert "Reply concisely" in system_msg["content"]
        assert "Prefer code, tables" in system_msg["content"]

    @pytest.mark.unit
    def test_standard_style_in_system_prompt(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """Default (standard) ResponseStyle injects standard instructions."""
        messages = adapter._format_messages(context, "Hello", [])
        system_msg = next(m for m in messages if m["role"] == "system")
        assert "Reply at a standard level of detail" in system_msg["content"]

    @pytest.mark.unit
    def test_detailed_style_in_system_prompt(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """ResponseStyle=detailed injects detailed instructions."""
        context.response_style = "detailed"
        messages = adapter._format_messages(context, "Hello", [])
        system_msg = next(m for m in messages if m["role"] == "system")
        assert "Reply in detail" in system_msg["content"]
        assert "Walk through reasoning" in system_msg["content"]

    @pytest.mark.unit
    def test_current_message_appended_as_user_role(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """The current message is the last entry with role='user'."""
        messages = adapter._format_messages(context, "probe-message", [])
        assert messages[-1] == {"role": "user", "content": "probe-message"}


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    """Tests for OllamaAdapter._parse_response."""

    @pytest.mark.unit
    def test_parses_content_from_message_field(self, adapter: OllamaAdapter) -> None:
        data = _make_ollama_response(content="test response")
        result = adapter._parse_response(data)
        assert result.content == "test response"

    @pytest.mark.unit
    def test_usage_tokens_populated(self, adapter: OllamaAdapter) -> None:
        data = _make_ollama_response(prompt_tokens=15, completion_tokens=8)
        result = adapter._parse_response(data)
        assert result.usage.get("input_tokens") == 15
        assert result.usage.get("output_tokens") == 8

    @pytest.mark.unit
    def test_finish_reason_stop_when_done(self, adapter: OllamaAdapter) -> None:
        data = _make_ollama_response(done=True)
        result = adapter._parse_response(data)
        assert result.finish_reason == "stop"

    @pytest.mark.unit
    def test_finish_reason_length_when_not_done(self, adapter: OllamaAdapter) -> None:
        data = _make_ollama_response(done=False)
        result = adapter._parse_response(data)
        assert result.finish_reason == "length"

    @pytest.mark.unit
    def test_model_name_in_metadata(self, adapter: OllamaAdapter) -> None:
        data = _make_ollama_response(model="custom-model")
        result = adapter._parse_response(data)
        assert result.metadata.get("model") == "custom-model"


# ---------------------------------------------------------------------------
# send_message — success
# ---------------------------------------------------------------------------


class TestSendMessageSuccess:
    """Happy-path tests for OllamaAdapter.send_message."""

    @pytest.mark.unit
    def test_returns_agent_response_with_content(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """send_message returns an AgentResponse with non-empty content."""
        from src.shared.python.ai.types import AgentResponse

        mock_response = MagicMock()
        mock_response.json.return_value = _make_ollama_response(content="great answer")
        mock_response.raise_for_status.return_value = None

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        adapter._client = mock_client

        result = adapter.send_message("hello", context, [])
        assert isinstance(result, AgentResponse)
        assert result.content == "great answer"

    @pytest.mark.unit
    def test_request_sent_to_chat_endpoint(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """send_message POSTs to /api/chat with stream=False."""
        mock_response = MagicMock()
        mock_response.json.return_value = _make_ollama_response()
        mock_response.raise_for_status.return_value = None

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        adapter._client = mock_client

        adapter.send_message("test", context, [])

        call_kwargs = mock_client.post.call_args
        assert "/api/chat" in call_kwargs[0][0]
        assert call_kwargs[1]["json"]["stream"] is False

    @pytest.mark.unit
    def test_model_name_in_request_body(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """The adapter's model name is sent in the request JSON."""
        mock_response = MagicMock()
        mock_response.json.return_value = _make_ollama_response()
        mock_response.raise_for_status.return_value = None

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        adapter._client = mock_client

        adapter.send_message("test", context, [])

        body = mock_client.post.call_args[1]["json"]
        assert body["model"] == adapter._model


# ---------------------------------------------------------------------------
# send_message — error paths
# ---------------------------------------------------------------------------


class TestSendMessageErrors:
    """Error-path tests for OllamaAdapter.send_message."""

    @pytest.mark.unit
    def test_connect_error_raises_ai_connection_error(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """httpx.ConnectError maps to AIConnectionError."""
        import httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("connection refused")
        adapter._client = mock_client

        with pytest.raises(AIConnectionError) as exc_info:
            adapter.send_message("hello", context, [])

        assert exc_info.value.provider == "ollama"

    @pytest.mark.unit
    def test_timeout_raises_ai_timeout_error(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """httpx.TimeoutException maps to AITimeoutError."""
        import httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.TimeoutException("timed out")
        adapter._client = mock_client

        with pytest.raises(AITimeoutError) as exc_info:
            adapter.send_message("hello", context, [])

        assert exc_info.value.provider == "ollama"

    @pytest.mark.unit
    def test_other_http_error_raises_ai_provider_error(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """Generic HTTP errors map to AIProviderError."""
        import httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "500 server error",
            request=MagicMock(),
            response=MagicMock(),
        )
        adapter._client = mock_client

        with pytest.raises(AIProviderError) as exc_info:
            adapter.send_message("hello", context, [])

        assert exc_info.value.provider == "ollama"


# ---------------------------------------------------------------------------
# stream_response
# ---------------------------------------------------------------------------


class TestStreamResponse:
    """Tests for OllamaAdapter.stream_response."""

    @pytest.mark.unit
    def test_yields_chunks_from_ndjson_lines(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """stream_response yields one AgentChunk per NDJSON line."""
        lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            json.dumps({"message": {"content": " world"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True}),
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = iter(lines)
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        adapter._client = mock_client

        chunks = list(adapter.stream_response("hi", context, []))

        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"
        assert chunks[2].is_final is True

    @pytest.mark.unit
    def test_stream_skips_empty_lines(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """Empty lines in the NDJSON stream are ignored."""
        lines = [
            "",
            json.dumps({"message": {"content": "Hi"}, "done": False}),
            "",
            json.dumps({"message": {"content": ""}, "done": True}),
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = iter(lines)
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        adapter._client = mock_client

        chunks = list(adapter.stream_response("hi", context, []))

        assert len(chunks) == 2

    @pytest.mark.unit
    def test_json_decode_error_raises_ai_provider_error(
        self, adapter: OllamaAdapter, context: ConversationContext
    ) -> None:
        """Malformed JSON lines raise AIProviderError."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = iter(["not-valid-json"])
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        adapter._client = mock_client

        with pytest.raises(AIProviderError) as exc_info:
            list(adapter.stream_response("hi", context, []))

        assert exc_info.value.provider == "ollama"


# ---------------------------------------------------------------------------
# validate_connection
# ---------------------------------------------------------------------------


class TestValidateConnection:
    """Tests for OllamaAdapter.validate_connection."""

    @pytest.mark.unit
    def test_returns_true_when_model_available(self, adapter: OllamaAdapter) -> None:
        """validate_connection returns (True, msg) when model is in the list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "test-model:latest"}, {"name": "llama3.1:8b"}]
        }

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        adapter._client = mock_client

        ok, msg = adapter.validate_connection()

        assert ok is True
        assert "test-model" in msg

    @pytest.mark.unit
    def test_falls_back_when_model_absent(self, adapter: OllamaAdapter) -> None:
        """validate_connection returns True and falls back to first available

        model when configured model is absent.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "mistral:latest"}, {"name": "phi3:mini"}]
        }

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        adapter._client = mock_client

        ok, msg = adapter.validate_connection()

        assert ok is True
        assert adapter._model == "mistral:latest"
        assert "using 'mistral:latest'" in msg.lower()

    @pytest.mark.unit
    def test_returns_false_when_no_models_installed(
        self, adapter: OllamaAdapter
    ) -> None:
        """validate_connection returns (False, hint) when no models are installed."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        adapter._client = mock_client

        ok, msg = adapter.validate_connection()

        assert ok is False
        assert "pull" in msg.lower() or "install" in msg.lower()

    @pytest.mark.unit
    def test_returns_false_on_connect_error(self, adapter: OllamaAdapter) -> None:
        """validate_connection returns (False, msg) on connection failure."""
        import httpx

        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ConnectError("refused")
        adapter._client = mock_client

        ok, msg = adapter.validate_connection()

        assert ok is False
        assert msg


# ---------------------------------------------------------------------------
# list_available_models
# ---------------------------------------------------------------------------


class TestListAvailableModels:
    """Tests for OllamaAdapter.list_available_models."""

    @pytest.mark.unit
    def test_returns_model_names(self, adapter: OllamaAdapter) -> None:
        """list_available_models returns a list of model name strings."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "models": [{"name": "llama3.1:8b"}, {"name": "mistral:latest"}]
        }

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        adapter._client = mock_client

        models = adapter.list_available_models()

        assert models == ["llama3.1:8b", "mistral:latest"]

    @pytest.mark.unit
    def test_raises_ai_connection_error_on_failure(
        self, adapter: OllamaAdapter
    ) -> None:
        """list_available_models raises AIConnectionError when Ollama is unreachable."""
        import httpx

        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ConnectError("refused")
        adapter._client = mock_client

        with pytest.raises(AIConnectionError) as exc_info:
            adapter.list_available_models()

        assert exc_info.value.provider == "ollama"


class TestEmptyCurrentMessage:
    """An empty `current_message` must not become a trailing user turn.

    `chat_service` passes `current_message=""` when the turn the user just
    sent is already the tail of `context.messages`. Appending a blank user
    message there corrupts the request: providers either reject it or answer
    the empty turn instead of the real one.
    """

    @pytest.mark.unit
    def test_blank_current_message_is_not_appended(
        self, adapter: OllamaAdapter
    ) -> None:
        from src.shared.python.ai.types import Message

        ctx = ConversationContext()
        ctx.messages = [Message(role="user", content="hello")]

        messages = adapter._format_messages(ctx, "", [])

        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[-1]["content"] == "hello"

    @pytest.mark.unit
    def test_whitespace_only_current_message_is_not_appended(
        self, adapter: OllamaAdapter
    ) -> None:
        from src.shared.python.ai.types import Message

        ctx = ConversationContext()
        ctx.messages = [Message(role="user", content="hello")]

        messages = adapter._format_messages(ctx, "   \n\t ", [])

        assert [m["role"] for m in messages] == ["system", "user"]

    @pytest.mark.unit
    def test_real_current_message_is_still_appended(
        self, adapter: OllamaAdapter
    ) -> None:
        from src.shared.python.ai.types import Message

        ctx = ConversationContext()
        ctx.messages = [Message(role="user", content="hello")]

        messages = adapter._format_messages(ctx, "and then?", [])

        assert [m["role"] for m in messages] == ["system", "user", "user"]
        assert messages[-1]["content"] == "and then?"


class TestTypedTransportErrors:
    """httpx exposes typed transport errors; classification must not need text.

    `BaseAgentAdapter._classify_error` scans the exception message, so
    `ConnectError("broken")` -- which contains none of its keywords -- was
    reported as a generic provider error, and Ollama's own
    "Is Ollama running?" hint never fired.
    """

    @pytest.mark.unit
    def test_connect_error_without_keywords_is_a_connection_error(
        self, adapter: OllamaAdapter
    ) -> None:
        import httpx

        with (
            patch.object(adapter, "_get_client") as mock_get_client,
            pytest.raises(AIConnectionError) as excinfo,
        ):
            mock_get_client.return_value.post.side_effect = httpx.ConnectError("broken")
            adapter.send_message("hi", ConversationContext(), [])

        assert "ollama serve" in str(excinfo.value)

    @pytest.mark.unit
    def test_timeout_without_keywords_is_a_timeout_error(
        self, adapter: OllamaAdapter
    ) -> None:
        import httpx

        with (
            patch.object(adapter, "_get_client") as mock_get_client,
            pytest.raises(AITimeoutError),
        ):
            mock_get_client.return_value.post.side_effect = httpx.TimeoutException("x")
            adapter.send_message("hi", ConversationContext(), [])
