"""Parametrized adapter-contract tests (issue #2763).

Every adapter that implements BaseAgentAdapter must produce:
1. Normalized token-count keys: ``input_tokens``, ``output_tokens``,
   ``total_tokens`` — all int, all present.
2. At least one ``AgentChunk(is_final=True)`` in ``stream_response``.
3. ``send_message`` raises (ValueError/AIProviderError) when called with an
   empty or blank message rather than returning garbage.

The fixtures use mocked providers so these tests run without live API keys
or installed binaries.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Now import adapter modules.
# ---------------------------------------------------------------------------
from src.shared.python.ai.adapters.base import BaseAgentAdapter  # noqa: E402
from src.shared.python.ai.types import (  # noqa: E402
    AgentChunk,
    AgentResponse,
    ConversationContext,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CANONICAL_USAGE_KEYS = frozenset({"input_tokens", "output_tokens", "total_tokens"})


def _assert_canonical_usage(usage: dict[str, int], adapter_name: str) -> None:
    """Assert that *usage* contains exactly the canonical token-count keys."""
    assert _CANONICAL_USAGE_KEYS == set(usage.keys()), (
        f"{adapter_name}: expected usage keys {_CANONICAL_USAGE_KEYS!r}, "
        f"got {set(usage.keys())!r}"
    )
    for key in _CANONICAL_USAGE_KEYS:
        assert isinstance(
            usage[key], int
        ), f"{adapter_name}: usage['{key}'] must be int, got {type(usage[key])!r}"


def _assert_stream_terminates(chunks: Iterator[AgentChunk], adapter_name: str) -> None:
    """Consume *chunks* and assert at least one has ``is_final=True``."""
    chunk_list = list(chunks)
    finals = [c for c in chunk_list if c.is_final]
    assert (
        finals
    ), f"{adapter_name}: stream_response did not emit any chunk with is_final=True"


# ---------------------------------------------------------------------------
# Adapter mock factories
# ---------------------------------------------------------------------------


def _make_anthropic_adapter() -> BaseAgentAdapter:
    from src.shared.python.ai.adapters.anthropic_adapter import AnthropicAdapter

    adapter = AnthropicAdapter(api_key="test-key", model="claude-3-sonnet-20240229")

    usage_mock = MagicMock()
    usage_mock.input_tokens = 10
    usage_mock.output_tokens = 20

    content_block = MagicMock()
    content_block.type = "text"
    content_block.text = "hello"

    response_mock = MagicMock()
    response_mock.content = [content_block]
    response_mock.usage = usage_mock
    response_mock.stop_reason = "end_turn"
    response_mock.model = "claude-3-sonnet-20240229"
    response_mock.id = "msg_test"

    client_mock = MagicMock()
    client_mock.messages.create.return_value = response_mock
    adapter._client = client_mock
    return adapter


def _make_openai_adapter() -> BaseAgentAdapter:
    from src.shared.python.ai.adapters.openai_adapter import OpenAIAdapter

    adapter = OpenAIAdapter(api_key="test-key")

    usage_mock = MagicMock()
    usage_mock.prompt_tokens = 5
    usage_mock.completion_tokens = 15
    usage_mock.total_tokens = 20

    msg_mock = MagicMock()
    msg_mock.content = "hi"
    msg_mock.tool_calls = []

    choice_mock = MagicMock()
    choice_mock.message = msg_mock
    choice_mock.finish_reason = "stop"

    response_mock = MagicMock()
    response_mock.choices = [choice_mock]
    response_mock.usage = usage_mock
    response_mock.model = "gpt-4"
    response_mock.id = "resp_test"

    client_mock = MagicMock()
    client_mock.chat.completions.create.return_value = response_mock
    adapter._client = client_mock
    return adapter


def _make_ollama_adapter() -> BaseAgentAdapter:
    from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter

    adapter = OllamaAdapter()

    data = {
        "message": {"content": "response text"},
        "done": True,
        "prompt_eval_count": 8,
        "eval_count": 12,
        "model": "llama3.1:8b",
    }

    resp_mock = MagicMock()
    resp_mock.json.return_value = data
    resp_mock.raise_for_status.return_value = None

    client_mock = MagicMock()
    client_mock.post.return_value = resp_mock
    adapter._client = client_mock
    return adapter


def _make_cline_adapter() -> BaseAgentAdapter:
    from src.shared.python.ai.adapters.cline_adapter import ClineAdapter

    adapter = ClineAdapter()

    data: dict[str, Any] = {
        "choices": [
            {
                "message": {"content": "cline reply", "tool_calls": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10},
        "model": "cline",
    }

    resp_mock = MagicMock()
    resp_mock.json.return_value = data
    resp_mock.raise_for_status.return_value = None

    client_mock = MagicMock()
    client_mock.post.return_value = resp_mock
    adapter._client = client_mock
    return adapter


def _make_bitnet_adapter() -> BaseAgentAdapter:
    from src.shared.python.ai.adapters.bitnet_adapter import BitnetAdapter

    return BitnetAdapter(model="test.gguf", bitnet_root="/tmp")


def _make_rust_adapter() -> BaseAgentAdapter:
    """Build a RustAgentAdapter with the ai_backend wheel mocked out."""
    ai_backend_mock = MagicMock()

    engine_mock = MagicMock()
    engine_mock.generate_response.return_value = "rust response"

    config_mock = MagicMock()
    config_mock.model = "gpt-4"

    ai_backend_mock.AIConfig.return_value = config_mock
    ai_backend_mock.AIEngine.return_value = engine_mock
    memory_mock = MagicMock()
    ai_backend_mock.MemoryManager.return_value = memory_mock
    rag_mock = MagicMock()
    ai_backend_mock.RagPipeline.return_value = rag_mock

    with patch.dict(sys.modules, {"ai_backend": ai_backend_mock}):
        from src.shared.python.ai.adapters.rust_adapter import RustAgentAdapter

        adapter = RustAgentAdapter(
            api_key="k",
            base_url="http://localhost",
            model="gpt-4",
        )
    return adapter


def _make_gemini_adapter() -> BaseAgentAdapter:
    genai_mock = MagicMock()
    model_instance = MagicMock()
    chat_mock = MagicMock()

    response_mock = MagicMock()
    response_mock.text = "gemini reply"

    chat_mock.send_message.return_value = response_mock
    model_instance.start_chat.return_value = chat_mock
    genai_mock.configure = MagicMock()
    genai_mock.GenerativeModel.return_value = model_instance

    genai_types = MagicMock()
    genai_types.GenerateContentResponse = MagicMock()

    modules = {
        "google": MagicMock(),
        "google.generativeai": genai_mock,
        "google.generativeai.types": genai_types,
    }

    with patch.dict(sys.modules, modules):
        with (
            patch(
                "src.shared.python.ai.adapters.gemini_adapter.HAS_GEMINI",
                True,
            ),
            patch(
                "src.shared.python.ai.adapters.gemini_adapter.HAS_GEMINI_CLIENT",
                False,
            ),
            patch(
                "src.shared.python.ai.adapters.gemini_adapter.GenerativeModel",
                genai_mock.GenerativeModel,
            ),
            patch(
                "src.shared.python.ai.adapters.gemini_adapter.genai",
                genai_mock,
            ),
        ):
            from src.shared.python.ai.adapters.gemini_adapter import GeminiAdapter

            adapter = GeminiAdapter(api_key="test-key")
            adapter._model = model_instance
    return adapter


@pytest.fixture(autouse=True)
def qapp():  # type: ignore[no-untyped-def]
    """Ensure a QApplication is initialized for QEventLoop signal delivery."""
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        yield None
        return
    app = QApplication.instance() or QApplication([])
    yield app


# ---------------------------------------------------------------------------
# Parametrize
# ---------------------------------------------------------------------------

_ADAPTER_FACTORIES = [
    ("anthropic", _make_anthropic_adapter),
    ("openai", _make_openai_adapter),
    ("ollama", _make_ollama_adapter),
    ("cline", _make_cline_adapter),
    ("bitnet", _make_bitnet_adapter),
    ("rust", _make_rust_adapter),
    ("gemini", _make_gemini_adapter),
]


# ---------------------------------------------------------------------------
# Test: token-count normalization
# ---------------------------------------------------------------------------


@pytest.mark.parity
@pytest.mark.parametrize("name,factory", _ADAPTER_FACTORIES)
def test_send_message_returns_normalized_usage(name: str, factory: Any) -> None:
    """All adapters must return the canonical token-count keys."""
    if name == "bitnet":
        result_mock = MagicMock()
        result_mock.stdout = "BitNet answer"
        result_mock.returncode = 0
        with patch("subprocess.run", return_value=result_mock):
            adapter = factory()
            ctx = ConversationContext()
            resp: AgentResponse = adapter.send_message("hello", ctx, [])
    elif name == "rust":
        adapter = factory()
        ctx = ConversationContext()
        resp = adapter.send_message("hello", ctx, [])
    elif name == "gemini":
        adapter = factory()
        ctx = ConversationContext()
        resp = adapter.send_message("hello", ctx, [])
    else:
        adapter = factory()
        ctx = ConversationContext()
        resp = adapter.send_message("hello", ctx, [])

    _assert_canonical_usage(resp.usage, name)
    assert resp.usage["total_tokens"] == (
        resp.usage["input_tokens"] + resp.usage["output_tokens"]
    ), f"{name}: total_tokens must equal input + output"


# ---------------------------------------------------------------------------
# Test: streaming finality
# ---------------------------------------------------------------------------


@pytest.mark.parity
@pytest.mark.parametrize("name,factory", _ADAPTER_FACTORIES)
def test_stream_response_always_emits_final_chunk(name: str, factory: Any) -> None:
    """Every adapter stream must terminate with at least one is_final=True chunk."""
    if name == "bitnet":
        lines_mock = MagicMock()
        lines_mock.__iter__ = MagicMock(return_value=iter(["line1", "line2"]))
        popen_mock = MagicMock()
        popen_mock.stdout = lines_mock
        popen_mock.wait.return_value = 0
        with patch("subprocess.Popen", return_value=popen_mock):
            adapter = factory()
            ctx = ConversationContext()
            _assert_stream_terminates(adapter.stream_response("hello", ctx, []), name)
        return

    if name == "rust":
        adapter = factory()
        ctx = ConversationContext()
        adapter.engine.stream_response.return_value = ["hello ", "world"]
        _assert_stream_terminates(adapter.stream_response("hello", ctx, []), name)
        return

    if name == "gemini":
        chunk1 = MagicMock()
        chunk1.text = "hello "
        chunk2 = MagicMock()
        chunk2.text = "world"
        model_instance = MagicMock()
        chat_mock = MagicMock()
        chat_mock.send_message.return_value = iter([chunk1, chunk2])
        model_instance.start_chat.return_value = chat_mock
        adapter = factory()
        adapter._model = model_instance
        ctx = ConversationContext()
        _assert_stream_terminates(adapter.stream_response("hello", ctx, []), name)
        return

    if name == "ollama":
        adapter = factory()
        ctx = ConversationContext()
        done_data = [
            '{"message": {"content": "word1"}, "done": false}',
            '{"message": {"content": "word2"}, "done": true}',
        ]
        resp_mock = MagicMock()
        resp_mock.raise_for_status.return_value = None
        resp_mock.iter_lines.return_value = iter(done_data)
        cm_mock = MagicMock()
        cm_mock.__enter__ = MagicMock(return_value=resp_mock)
        cm_mock.__exit__ = MagicMock(return_value=False)
        adapter._client.stream.return_value = cm_mock
        _assert_stream_terminates(adapter.stream_response("hello", ctx, []), name)
        return

    if name == "anthropic":
        adapter = factory()
        ctx = ConversationContext()

        delta1 = MagicMock()
        delta1.text = "hi"
        event1 = MagicMock()
        event1.type = "content_block_delta"
        event1.delta = delta1

        event2 = MagicMock()
        event2.type = "message_stop"

        stream_inner = MagicMock()
        stream_inner.__iter__ = MagicMock(return_value=iter([event1, event2]))
        stream_cm = MagicMock()
        stream_cm.__enter__ = MagicMock(return_value=stream_inner)
        stream_cm.__exit__ = MagicMock(return_value=False)
        adapter._client.messages.stream.return_value = stream_cm
        _assert_stream_terminates(adapter.stream_response("hello", ctx, []), name)
        return

    if name == "openai":
        adapter = factory()
        ctx = ConversationContext()

        chunk1 = MagicMock()
        chunk1.choices = [
            MagicMock(
                delta=MagicMock(content="word", tool_calls=None),
                finish_reason=None,
            )
        ]
        chunk2 = MagicMock()
        chunk2.choices = [
            MagicMock(
                delta=MagicMock(content="", tool_calls=None),
                finish_reason="stop",
            )
        ]
        adapter._client.chat.completions.create.return_value = iter([chunk1, chunk2])
        _assert_stream_terminates(adapter.stream_response("hello", ctx, []), name)
        return

    if name == "cline":
        adapter = factory()
        ctx = ConversationContext()

        lines = [
            'data: {"choices": [{"delta": {"content": "hi"}}]}',
            "data: [DONE]",
        ]
        resp_mock = MagicMock()
        resp_mock.raise_for_status.return_value = None
        resp_mock.iter_lines.return_value = iter(lines)
        cm_mock = MagicMock()
        cm_mock.__enter__ = MagicMock(return_value=resp_mock)
        cm_mock.__exit__ = MagicMock(return_value=False)
        adapter._client.stream.return_value = cm_mock
        _assert_stream_terminates(adapter.stream_response("hello", ctx, []), name)
        return


# ---------------------------------------------------------------------------
# Test: empty-message precondition
# ---------------------------------------------------------------------------


@pytest.mark.parity
@pytest.mark.parametrize("name,factory", _ADAPTER_FACTORIES)
def test_empty_message_raises(name: str, factory: Any) -> None:
    """Sending an empty or blank message must raise rather than silently succeed."""
    from src.shared.python.ai.exceptions import AIProviderError

    if name == "bitnet":
        result_mock = MagicMock()
        result_mock.stdout = ""
        result_mock.returncode = 0
        with patch("subprocess.run", return_value=result_mock):
            adapter = factory()
    else:
        adapter = factory()

    ctx = ConversationContext()

    with pytest.raises((ValueError, AIProviderError)):
        adapter.send_message("", ctx, [])


# ---------------------------------------------------------------------------
# Unit tests: _normalize_token_counts
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNormalizeTokenCounts:
    """Unit tests for BaseAgentAdapter._normalize_token_counts (issue #2763)."""

    @staticmethod
    def _norm(raw: dict[str, int]) -> dict[str, int]:
        return BaseAgentAdapter._normalize_token_counts(raw)

    def test_empty_returns_zeros(self) -> None:
        result = self._norm({})
        assert result == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def test_anthropic_style(self) -> None:
        result = self._norm({"input_tokens": 10, "output_tokens": 20})
        assert result == {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}

    def test_openai_style(self) -> None:
        result = self._norm(
            {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20}
        )
        assert result == {"input_tokens": 5, "output_tokens": 15, "total_tokens": 20}

    def test_openai_style_total_computed_when_missing(self) -> None:
        result = self._norm({"prompt_tokens": 5, "completion_tokens": 10})
        assert result == {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15}

    def test_canonical_keys_always_present(self) -> None:
        result = self._norm({"input_tokens": 1, "output_tokens": 2})
        assert set(result.keys()) == _CANONICAL_USAGE_KEYS

    def test_total_tokens_explicit_wins_over_computed(self) -> None:
        # If provider sends total_tokens=100 but sum would be 30, trust provider.
        result = self._norm(
            {"input_tokens": 10, "output_tokens": 20, "total_tokens": 100}
        )
        assert result["total_tokens"] == 100
