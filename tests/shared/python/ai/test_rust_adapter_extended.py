"""Extended unit tests for RustAgentAdapter.

Supplements test_rust_adapter.py (which covers stream_response generator
behavior from #2752) with:

  - send_message happy path
  - send_message exception swallowing → error AgentResponse
  - index_codebase delegates to rag.index_codebase
  - retrieve_context returns list[str]
  - retrieve_context converts non-str items to str
  - validate_connection always returns True
  - capabilities advertise STREAMING and SYSTEM_MESSAGE
  - constructor builds config / engine / memory correctly
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Bootstrap (identical to test_rust_adapter.py)
# ---------------------------------------------------------------------------

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


_logging_config_stub = sys.modules.setdefault(
    "src.shared.python.logging_pkg.logging_config",
    types.ModuleType("src.shared.python.logging_pkg.logging_config"),
)
_logging_config_stub.get_logger = logging.getLogger  # type: ignore[attr-defined]


def _install_ai_backend_stub() -> types.ModuleType:
    """Install minimal ai_backend stub (matches test_rust_adapter.py pattern)."""
    import time

    stub = types.ModuleType("ai_backend")

    class _AIConfig:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.model = "stub-model"

    class _AIEngine:
        def __init__(self, _config: object) -> None:
            self._response = "default-response"
            self._stream_chunks: list[str] = []
            self._stream_delay: float = 0.0

        def generate_response(self, _prompt: str) -> str:
            return self._response

        def stream_response(self, _prompt: str) -> list[str]:
            if self._stream_delay:
                time.sleep(self._stream_delay)
            return list(self._stream_chunks)

    class _MemoryManager:
        def __init__(self, _path: str) -> None:
            self.initialized = False

        def initialize(self) -> None:
            self.initialized = True

    class _RagPipeline:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self._indexed_path: str = ""
            self._context: list[str] = []

        def index_codebase(self, root: str) -> int:
            self._indexed_path = root
            return len(root)

        def retrieve_context(self, _prompt: str, top_k: int) -> list[str]:
            return self._context[:top_k]

    stub.AIConfig = _AIConfig  # type: ignore[attr-defined]
    stub.AIEngine = _AIEngine  # type: ignore[attr-defined]
    stub.MemoryManager = _MemoryManager  # type: ignore[attr-defined]
    stub.RagPipeline = _RagPipeline  # type: ignore[attr-defined]
    sys.modules["ai_backend"] = stub
    return stub


_install_ai_backend_stub()

from src.shared.python.ai.adapters.rust_adapter import RustAgentAdapter  # noqa: E402
from src.shared.python.ai.types import (  # noqa: E402
    ConversationContext,
    ExpertiseLevel,
    ProviderCapability,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(**kwargs: object) -> RustAgentAdapter:
    return RustAgentAdapter(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="stub-model",
        **kwargs,  # type: ignore[arg-type]
    )


def _make_context() -> ConversationContext:
    return ConversationContext(
        messages=[],
        user_expertise=ExpertiseLevel.INTERMEDIATE,
    )


# ---------------------------------------------------------------------------
# send_message
# ---------------------------------------------------------------------------


class TestSendMessage:
    def test_happy_path_returns_engine_response(self) -> None:
        """send_message returns the Rust engine's generate_response as content."""
        adapter = _make_adapter()
        adapter.engine._response = "Rust says hello"  # type: ignore[attr-defined]

        response = adapter.send_message("prompt", _make_context(), [])

        assert response.content == "Rust says hello"
        assert response.finish_reason != "error"

    def test_context_messages_are_prepended_to_prompt(self) -> None:
        """History messages are concatenated before the new prompt."""
        from src.shared.python.ai.types import Message

        adapter = _make_adapter()
        captured: list[str] = []

        original_gen = adapter.engine.generate_response

        def _capture(prompt: str) -> str:
            captured.append(prompt)
            return original_gen(prompt)

        adapter.engine.generate_response = _capture  # type: ignore[method-assign]

        ctx = _make_context()
        ctx.messages = [
            Message(role="user", content="first"),
            Message(role="assistant", content="second"),
        ]

        adapter.send_message("third", ctx, [])

        assert len(captured) == 1
        full_prompt = captured[0]
        assert "first" in full_prompt
        assert "second" in full_prompt
        assert "third" in full_prompt

    def test_engine_exception_returns_error_response(self) -> None:
        """When the Rust engine raises, send_message returns an error AgentResponse."""
        adapter = _make_adapter()
        adapter.engine.generate_response = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("backend crash")
        )

        response = adapter.send_message("hi", _make_context(), [])

        assert "Error" in response.content or "error" in response.content.lower()
        assert response.finish_reason == "error"

    def test_tools_parameter_is_ignored_without_raising(self) -> None:
        """The Rust adapter does not use tools; passing them must not raise."""
        from src.shared.python.ai.adapters.base import ToolDeclaration

        adapter = _make_adapter()
        tools = [ToolDeclaration(name="noop", description="no-op")]

        # Should not raise
        response = adapter.send_message("hi", _make_context(), tools)
        assert response.content  # some content returned


# ---------------------------------------------------------------------------
# index_codebase
# ---------------------------------------------------------------------------


class TestIndexCodebase:
    def test_delegates_to_rag_pipeline(self) -> None:
        """index_codebase calls rag.index_codebase with the given path."""
        adapter = _make_adapter()
        result = adapter.index_codebase("/tmp/project")

        assert isinstance(result, int)
        # The stub returns len(root_path); non-zero confirms delegation
        assert result > 0
        assert adapter.rag._indexed_path == "/tmp/project"  # type: ignore[attr-defined]

    def test_return_value_is_int(self) -> None:
        """index_codebase always returns an int (document count)."""
        adapter = _make_adapter()
        result = adapter.index_codebase("/some/path")
        assert isinstance(result, int)

    @pytest.mark.parametrize("root", ["/a", "/var/tmp/project", "C:\\project"])
    def test_various_root_paths_are_forwarded(self, root: str) -> None:
        """Arbitrary root path strings are forwarded unchanged."""
        adapter = _make_adapter()
        adapter.index_codebase(root)
        assert adapter.rag._indexed_path == root  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# retrieve_context
# ---------------------------------------------------------------------------


class TestRetrieveContext:
    def test_returns_list_of_strings(self) -> None:
        """retrieve_context always returns list[str]."""
        adapter = _make_adapter()
        adapter.rag._context = ["ctx_a", "ctx_b", "ctx_c"]  # type: ignore[attr-defined]

        result = adapter.retrieve_context("query", top_k=3)

        assert isinstance(result, list)
        assert all(isinstance(item, str) for item in result)

    def test_top_k_limits_results(self) -> None:
        """top_k caps the number of returned context chunks."""
        adapter = _make_adapter()
        adapter.rag._context = ["a", "b", "c", "d", "e"]  # type: ignore[attr-defined]

        result = adapter.retrieve_context("query", top_k=2)

        assert len(result) <= 2

    def test_non_string_items_are_converted_to_str(self) -> None:
        """Non-string items returned by the Rust layer are coerced to str."""
        adapter = _make_adapter()
        # Simulate Rust returning mixed types via the stub
        adapter.rag.retrieve_context = MagicMock(  # type: ignore[method-assign]
            return_value=[42, None, b"bytes"]
        )

        result = adapter.retrieve_context("query", top_k=5)

        assert all(isinstance(item, str) for item in result)

    def test_empty_context_returns_empty_list(self) -> None:
        """No context chunks → empty list."""
        adapter = _make_adapter()
        adapter.rag._context = []  # type: ignore[attr-defined]

        result = adapter.retrieve_context("query")

        assert result == []

    def test_default_top_k_is_five(self) -> None:
        """Default top_k value is 5 (implicit signature contract)."""
        import inspect

        sig = inspect.signature(RustAgentAdapter.retrieve_context)
        top_k_param = sig.parameters.get("top_k")
        assert top_k_param is not None
        assert top_k_param.default == 5


# ---------------------------------------------------------------------------
# validate_connection
# ---------------------------------------------------------------------------


class TestValidateConnection:
    def test_returns_true_when_engine_is_initialized(self) -> None:
        """validate_connection returns (True, message) unconditionally."""
        adapter = _make_adapter()
        ok, msg = adapter.validate_connection()
        assert ok is True
        assert isinstance(msg, str)
        assert len(msg) > 0


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_streaming_is_advertised(self) -> None:
        """STREAMING capability is always present."""
        adapter = _make_adapter()
        assert ProviderCapability.STREAMING in adapter.capabilities.supported

    def test_system_message_is_advertised(self) -> None:
        """SYSTEM_MESSAGE capability is always present."""
        adapter = _make_adapter()
        assert ProviderCapability.SYSTEM_MESSAGE in adapter.capabilities.supported

    def test_provider_name_is_rust(self) -> None:
        """provider_name identifies the backend."""
        adapter = _make_adapter()
        assert adapter.capabilities.provider_name == "rust"

    def test_model_name_comes_from_config(self) -> None:
        """model_name in capabilities reflects the config model."""
        adapter = _make_adapter()
        assert adapter.capabilities.model_name == "stub-model"

    def test_max_tokens_is_positive(self) -> None:
        """max_tokens must be a positive integer."""
        adapter = _make_adapter()
        assert adapter.capabilities.max_tokens > 0


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_initializes_memory_manager(self) -> None:
        """MemoryManager.initialize() is called during construction."""
        adapter = _make_adapter()
        assert adapter.memory.initialized is True  # type: ignore[attr-defined]

    def test_db_path_defaults_to_memory_db(self) -> None:
        """Default db_path is './memory.db'."""
        import inspect

        sig = inspect.signature(RustAgentAdapter.__init__)
        db_path_param = sig.parameters.get("db_path")
        assert db_path_param is not None
        assert db_path_param.default == "./memory.db"
