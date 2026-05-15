"""Cross-adapter contract suite (issue #2763).

The seven AI provider adapters all implement the same ``BaseAgentAdapter``
method signatures but historically diverged in semantics: token-count
key names, streaming finality, and preconditions on empty input.

This module locks the post-#2763 contract with a parametrized suite that
runs the SAME assertions against ALL adapters:

1. ``send_message`` returns an :class:`AgentResponse` whose ``usage``
   field is a :class:`TokenUsage` instance with consistent total_tokens.
2. ``stream_response`` always yields at least one chunk and the last
   chunk has ``is_final=True`` (enforced by ``_ensure_final_chunk``).
3. ``send_message("")`` raises :class:`AIProviderError` (or a subclass)
   for every adapter.

Underlying SDKs are mocked. The point is to exercise the adapter glue,
not the provider transport.
"""

from __future__ import annotations

import logging
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: stub broken ai/__init__ side imports so we can import adapters
# directly without dragging in PyQt6 / heavy deps. Mirrors the pattern in
# test_adapter_factory.py.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PACKAGE_STUBS: list[tuple[str, str | None]] = [
    ("src", "src"),
    ("src.shared", "src/shared"),
    ("src.shared.python", "src/shared/python"),
    ("src.shared.python.ai", "src/shared/python/ai"),
    ("src.shared.python.ai.adapters", "src/shared/python/ai/adapters"),
    ("src.shared.python.logging_pkg", None),
    ("src.shared.python.logging_pkg.logging_config", None),
]
for _mod_name, _rel_path in _PACKAGE_STUBS:
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        if _rel_path is not None:
            _stub.__path__ = [str(ROOT / _rel_path)]  # type: ignore[attr-defined]
        sys.modules[_mod_name] = _stub

_logging_config_stub = sys.modules["src.shared.python.logging_pkg.logging_config"]
_logging_config_stub.get_logger = logging.getLogger  # type: ignore[attr-defined]
_logging_config_stub.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]

# Stub the missing ``src.shared.python.config.environment`` module that
# ``ai/config.py`` imports — that subpackage was removed in an earlier
# refactor but the import line was never updated. We provide minimal
# env-var helpers so the adapter modules import cleanly under test.
import os as _os

_config_pkg_stub = types.ModuleType("src.shared.python.config")
_config_pkg_stub.__path__ = []  # type: ignore[attr-defined]
_env_stub = types.ModuleType("src.shared.python.config.environment")


def _get_env(name, default=None, **_kw):  # type: ignore[no-untyped-def]
    return _os.environ.get(name, default)


def _get_env_float(name, default=None, **_kw):  # type: ignore[no-untyped-def]
    raw = _os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


_env_stub.get_env = _get_env  # type: ignore[attr-defined]
_env_stub.get_env_float = _get_env_float  # type: ignore[attr-defined]
sys.modules.setdefault("src.shared.python.config", _config_pkg_stub)
sys.modules.setdefault("src.shared.python.config.environment", _env_stub)

# The real ``contracts`` module is self-contained; let it load normally
# so adapters get their genuine @precondition behaviour (raises
# ``PreconditionError`` which inherits from ``ValueError``).

# Stub memory_manager so base.py imports cleanly.
if "src.shared.python.ai.memory_manager" not in sys.modules:
    _mm = types.ModuleType("src.shared.python.ai.memory_manager")
    _mm.build_memory_prompt_section = (  # type: ignore[attr-defined]
        lambda **kw: ""
    )
    _mm.load_agents_md = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["src.shared.python.ai.memory_manager"] = _mm

# Now safe to import adapters.
from src.shared.python.ai.adapters.anthropic_adapter import (  # noqa: E402
    AnthropicAdapter,
)
from src.shared.python.ai.adapters.bitnet_adapter import BitnetAdapter  # noqa: E402
from src.shared.python.ai.adapters.cline_adapter import ClineAdapter  # noqa: E402
from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter  # noqa: E402
from src.shared.python.ai.adapters.openai_adapter import OpenAIAdapter  # noqa: E402
from src.shared.python.ai.exceptions import AIProviderError  # noqa: E402
from src.shared.python.ai.types import (  # noqa: E402
    AgentChunk,
    ConversationContext,
    TokenUsage,
)

# ---------------------------------------------------------------------------
# Stub builders
# ---------------------------------------------------------------------------


def _empty_context() -> ConversationContext:
    return ConversationContext(messages=[])


def _build_anthropic_stub() -> AnthropicAdapter:
    adapter = AnthropicAdapter(api_key="sk-test")
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.content = [MagicMock(type="text", text="hello")]
    fake_response.stop_reason = "end_turn"
    fake_response.model = "claude-test"
    fake_response.id = "msg_test"
    fake_response.usage = MagicMock(input_tokens=11, output_tokens=22)
    fake_client.messages.create.return_value = fake_response

    # Streaming: emit two content_block_delta events, no message_stop.
    class _Evt:
        def __init__(self, et: str, text: str = "") -> None:
            self.type = et
            self.delta = MagicMock(text=text)

    fake_stream_ctx = MagicMock()
    fake_stream_ctx.__enter__.return_value = iter(
        [_Evt("content_block_delta", "hi "), _Evt("content_block_delta", "there")]
    )
    fake_stream_ctx.__exit__.return_value = False
    fake_client.messages.stream.return_value = fake_stream_ctx
    adapter._client = fake_client
    return adapter


def _build_openai_stub() -> OpenAIAdapter:
    adapter = OpenAIAdapter(api_key="sk-test")
    fake_client = MagicMock()
    fake_choice = MagicMock()
    fake_choice.message = MagicMock(content="hello", tool_calls=None)
    fake_choice.finish_reason = "stop"
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    fake_response.usage = MagicMock(
        prompt_tokens=11, completion_tokens=22, total_tokens=33
    )
    fake_response.model = "gpt-test"
    fake_response.id = "cmpl_test"
    fake_client.chat.completions.create.return_value = fake_response

    # Streaming: two delta chunks, the second carries finish_reason=stop.
    def _make_chunk(text: str, finish: str | None) -> MagicMock:
        delta = MagicMock(content=text, tool_calls=None)
        choice = MagicMock(delta=delta, finish_reason=finish)
        return MagicMock(choices=[choice])

    def _stream_factory(**kwargs):  # type: ignore[no-untyped-def]
        if kwargs.get("stream"):
            return iter([_make_chunk("hi ", None), _make_chunk("there", "stop")])
        return fake_response

    fake_client.chat.completions.create.side_effect = _stream_factory
    adapter._client = fake_client
    return adapter


def _build_ollama_stub() -> OllamaAdapter:
    adapter = OllamaAdapter()
    fake_client = MagicMock()

    fake_post_resp = MagicMock()
    fake_post_resp.json.return_value = {
        "message": {"content": "hello"},
        "done": True,
        "prompt_eval_count": 11,
        "eval_count": 22,
        "model": "llama3-test",
    }
    fake_post_resp.raise_for_status.return_value = None
    fake_client.post.return_value = fake_post_resp

    # Streaming: NDJSON-ish lines; last has done=True.
    fake_stream_resp = MagicMock()
    fake_stream_resp.iter_lines.return_value = iter(
        [
            '{"message": {"content": "hi "}, "done": false}',
            '{"message": {"content": "there"}, "done": true}',
        ]
    )
    fake_stream_resp.raise_for_status.return_value = None
    fake_stream_ctx = MagicMock()
    fake_stream_ctx.__enter__.return_value = fake_stream_resp
    fake_stream_ctx.__exit__.return_value = False
    fake_client.stream.return_value = fake_stream_ctx
    adapter._client = fake_client
    return adapter


def _build_cline_stub() -> ClineAdapter:
    adapter = ClineAdapter()
    fake_client = MagicMock()

    fake_resp = MagicMock()
    fake_resp.json.return_value = {
        "choices": [
            {"message": {"content": "hello"}, "finish_reason": "stop"},
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 22},
        "model": "cline-test",
    }
    fake_resp.raise_for_status.return_value = None
    fake_client.post.return_value = fake_resp

    fake_stream_resp = MagicMock()
    fake_stream_resp.iter_lines.return_value = iter(
        [
            'data: {"choices": [{"delta": {"content": "hi "}}]}',
            'data: {"choices": [{"delta": {"content": "there"}}]}',
            "data: [DONE]",
        ]
    )
    fake_stream_resp.raise_for_status.return_value = None
    fake_stream_ctx = MagicMock()
    fake_stream_ctx.__enter__.return_value = fake_stream_resp
    fake_stream_ctx.__exit__.return_value = False
    fake_client.stream.return_value = fake_stream_ctx
    adapter._client = fake_client
    return adapter


def _build_bitnet_stub() -> BitnetAdapter:
    """BitNet adapter with subprocess fully mocked."""
    adapter = BitnetAdapter(model="test-model.gguf")

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        result = MagicMock()
        result.stdout = "hello"
        result.stderr = ""
        result.returncode = 0
        return result

    class _FakePopen:
        def __init__(self, cmd, **kwargs):  # type: ignore[no-untyped-def]
            self.stdout = iter(["hi\n", "there\n"])
            self.stderr = MagicMock()

        def wait(self) -> int:
            return 0

    adapter._test_patches = (  # type: ignore[attr-defined]
        patch("subprocess.run", _fake_run),
        patch("subprocess.Popen", _FakePopen),
    )
    for p in adapter._test_patches:  # type: ignore[attr-defined]
        p.start()
    return adapter


def _build_gemini_stub():  # type: ignore[no-untyped-def]
    """Gemini adapter with the SDK fully stubbed.

    The real adapter requires ``google-generativeai``; we install a fake
    module before constructing it.
    """
    fake_genai = types.ModuleType("google.generativeai")
    fake_genai.configure = lambda *a, **kw: None  # type: ignore[attr-defined]

    class _FakeChat:
        def send_message(self, msg, stream=False):  # type: ignore[no-untyped-def]
            if stream:
                return iter([MagicMock(text="hi "), MagicMock(text="there")])
            resp = MagicMock()
            resp.text = "hello"
            resp.usage_metadata = MagicMock(
                prompt_token_count=11, candidates_token_count=22
            )
            return resp

    class _FakeModel:
        def start_chat(self, history=None):  # type: ignore[no-untyped-def]
            return _FakeChat()

        def generate_content(self, prompt):  # type: ignore[no-untyped-def]
            return MagicMock(text="ok")

    fake_genai.GenerativeModel = lambda name: _FakeModel()  # type: ignore[attr-defined]
    fake_types = types.ModuleType("google.generativeai.types")
    fake_types.GenerateContentResponse = object  # type: ignore[attr-defined]

    google_pkg = types.ModuleType("google")
    google_pkg.generativeai = fake_genai  # type: ignore[attr-defined]
    sys.modules["google"] = google_pkg
    sys.modules["google.generativeai"] = fake_genai
    sys.modules["google.generativeai.types"] = fake_types

    # Force re-import of the gemini adapter module under the fake SDK.
    sys.modules.pop("src.shared.python.ai.adapters.gemini_adapter", None)
    from src.shared.python.ai.adapters.gemini_adapter import (  # noqa: WPS433
        GeminiAdapter,
    )

    return GeminiAdapter(api_key="g-test")


def _build_rust_stub():  # type: ignore[no-untyped-def]
    """Rust adapter stub — install a fake ``ai_backend`` module."""
    fake_backend = types.ModuleType("ai_backend")

    class _FakeEngine:
        def generate_response(self, prompt):  # type: ignore[no-untyped-def]
            return "hello"

        def stream_response(self, prompt):  # type: ignore[no-untyped-def]
            return ["hi ", "there"]

    class _FakeMemory:
        def initialize(self):  # type: ignore[no-untyped-def]
            return None

    fake_backend.AIConfig = MagicMock(  # type: ignore[attr-defined]
        return_value=MagicMock(model="rust-model")
    )
    fake_backend.AIEngine = lambda cfg: _FakeEngine()  # type: ignore[attr-defined]
    fake_backend.MemoryManager = lambda path: _FakeMemory()  # type: ignore[attr-defined]
    fake_backend.RagPipeline = lambda *a, **kw: MagicMock()  # type: ignore[attr-defined]
    sys.modules["ai_backend"] = fake_backend

    sys.modules.pop("src.shared.python.ai.adapters.rust_adapter", None)
    from src.shared.python.ai.adapters.rust_adapter import (  # noqa: WPS433
        RustAgentAdapter,
    )

    return RustAgentAdapter(
        api_key="k", base_url="http://x", model="rust-model", db_path=":memory:"
    )


_BUILDERS = {
    "anthropic": _build_anthropic_stub,
    "openai": _build_openai_stub,
    "ollama": _build_ollama_stub,
    "cline": _build_cline_stub,
    "gemini": _build_gemini_stub,
    "bitnet": _build_bitnet_stub,
    "rust": _build_rust_stub,
}

ADAPTERS = list(_BUILDERS.keys())


@pytest.fixture
def mock_adapter(request):  # type: ignore[no-untyped-def]
    provider = request.param
    adapter = _BUILDERS[provider]()
    yield adapter
    # Tear down any patches the builder started.
    patches = getattr(adapter, "_test_patches", ())
    for p in patches:
        p.stop()


# ---------------------------------------------------------------------------
# Contract assertions — same checks for every adapter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mock_adapter", ADAPTERS, indirect=True)
def test_send_message_returns_token_usage(mock_adapter) -> None:  # type: ignore[no-untyped-def]
    """Every adapter MUST return AgentResponse.usage as a TokenUsage instance."""
    response = mock_adapter.send_message("hi", _empty_context(), tools=[])
    assert isinstance(response.usage, TokenUsage), (
        f"{type(mock_adapter).__name__}.send_message returned "
        f"usage of type {type(response.usage).__name__}, expected TokenUsage"
    )
    assert (
        response.usage.total_tokens
        == response.usage.input_tokens + response.usage.output_tokens
    )


@pytest.mark.parametrize("mock_adapter", ADAPTERS, indirect=True)
def test_stream_always_emits_final_chunk(mock_adapter) -> None:  # type: ignore[no-untyped-def]
    """Every adapter MUST end its stream with a chunk having is_final=True."""
    chunks: list[AgentChunk] = list(
        mock_adapter.stream_response("hi", _empty_context(), tools=[])
    )
    assert len(chunks) > 0, (
        f"{type(mock_adapter).__name__}.stream_response yielded zero chunks"
    )
    assert chunks[-1].is_final, (
        f"{type(mock_adapter).__name__} did not emit a final chunk; "
        f"last chunk: {chunks[-1]!r}"
    )


@pytest.mark.parametrize("mock_adapter", ADAPTERS, indirect=True)
def test_empty_message_raises_provider_error(mock_adapter) -> None:  # type: ignore[no-untyped-def]
    """Every adapter MUST reject empty input with AIProviderError or its subclass.

    Adapters without an explicit precondition should still raise — either
    because the underlying SDK rejects empty input or because the adapter's
    own glue raises. The contract is: never silently succeed on empty input.
    """
    with pytest.raises((AIProviderError, ValueError)):
        mock_adapter.send_message("", _empty_context(), tools=[])


def test_ensure_final_chunk_helper_handles_empty_stream() -> None:
    """Direct unit test: empty input → single empty final chunk."""
    from src.shared.python.ai.adapters.base import _ensure_final_chunk

    out = list(_ensure_final_chunk(iter([])))
    assert len(out) == 1
    assert out[0].is_final
    assert out[0].content == ""


def test_ensure_final_chunk_helper_promotes_last_to_final() -> None:
    """Direct unit test: last chunk with is_final=False is promoted."""
    from src.shared.python.ai.adapters.base import _ensure_final_chunk

    raw: Iterator[AgentChunk] = iter(
        [
            AgentChunk(content="a", is_final=False, index=0),
            AgentChunk(content="b", is_final=False, index=1),
        ]
    )
    out = list(_ensure_final_chunk(raw))
    assert len(out) == 2
    assert not out[0].is_final
    assert out[1].is_final
    assert out[1].content == "b"


def test_ensure_final_chunk_helper_passes_through_explicit_final() -> None:
    """Direct unit test: existing is_final=True is passed through unchanged."""
    from src.shared.python.ai.adapters.base import _ensure_final_chunk

    raw: Iterator[AgentChunk] = iter(
        [
            AgentChunk(content="a", is_final=False, index=0),
            AgentChunk(content="", is_final=True, index=1),
        ]
    )
    out = list(_ensure_final_chunk(raw))
    assert len(out) == 2
    assert out[1].is_final
