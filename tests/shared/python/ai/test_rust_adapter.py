# ruff: noqa: E501
"""Unit tests for RustAgentAdapter.stream_response generator behavior.

Issue #2752: the underlying Rust ``AIEngine.stream_response`` is blocking
(uses Tokio ``block_on`` internally) and must be invoked from a worker
thread when used from a Qt UI. The adapter's generator API itself is
unchanged — these tests verify it still yields chunks correctly when the
backend takes time to return.

The bootstrap block mirrors test_bitnet_adapter.py so the adapter module
imports cleanly under a plain pytest run.
"""

from __future__ import annotations

import sys
import time
import types
from typing import Any
from unittest.mock import MagicMock

import pytest


class _FakeSignal:
    """A minimal mock for pyqtSignal."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.callbacks: list[Any] = []

    def connect(self, callback: Any) -> None:
        self.callbacks.append(callback)

    def emit(self, *args: Any) -> None:
        for callback in self.callbacks:
            callback(*args)


class _FakeSignalDescriptor:
    """A descriptor representing pyqtSignal on class level."""

    def __init__(self, *types: Any) -> None:
        self.types = types

    def __get__(self, instance: Any, owner: Any) -> Any:
        if instance is None:
            return self
        name = f"_fake_signal_{id(self)}"
        if not hasattr(instance, name):
            setattr(instance, name, _FakeSignal())
        return getattr(instance, name)


class _FakeQThread:
    """A minimal mock for QThread."""

    currentThread: Any = MagicMock()

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def start(self) -> None:
        self.run()


class _FakeQEventLoop:
    """A minimal mock for QEventLoop."""

    def exec(self) -> None:
        from PyQt6.QtCore import QCoreApplication

        app = QCoreApplication.instance()
        if app is not None:
            app.processEvents()

    def quit(self) -> None:
        pass


class _StubAIConfig:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.model = "stub-model"


class _StubAIEngine:
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


class _StubMemoryManager:
    def __init__(self, _path: str) -> None:
        self.initialized = False

    def initialize(self) -> None:
        self.initialized = True


class _StubRagPipeline:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self._indexed_path: str = ""
        self._context: list[str] = []

    def index_codebase(self, root: str) -> int:
        self._indexed_path = root
        return len(root)

    def retrieve_context(self, _prompt: str, top_k: int) -> list[str]:
        return self._context[:top_k]


def _install_ai_backend_stub() -> types.ModuleType:
    """Install a minimal ai_backend stub module suitable for unit testing."""
    stub = types.ModuleType("ai_backend")
    stub.AIConfig = _StubAIConfig  # type: ignore[attr-defined]
    stub.AIEngine = _StubAIEngine  # type: ignore[attr-defined]
    stub.MemoryManager = _StubMemoryManager  # type: ignore[attr-defined]
    stub.RagPipeline = _StubRagPipeline  # type: ignore[attr-defined]
    sys.modules["ai_backend"] = stub
    return stub


_install_ai_backend_stub()


from src.shared.python.ai.adapters.rust_adapter import (  # noqa: E402
    RustAgentAdapter,
)
from src.shared.python.ai.types import (  # noqa: E402
    AgentChunk,
    ConversationContext,
    ExpertiseLevel,
)


def _make_context() -> ConversationContext:
    return ConversationContext(
        messages=[],
        user_expertise=ExpertiseLevel.INTERMEDIATE,
    )


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


@pytest.fixture()
def adapter() -> RustAgentAdapter:
    return RustAgentAdapter(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="stub-model",
    )


class TestStreamResponseGenerator:
    """Generator contract for stream_response remains intact (#2752)."""

    def test_yields_one_chunk_per_delta(self, adapter: RustAgentAdapter) -> None:
        """Five deltas from the backend produce five AgentChunks in order."""
        chunks_in = ["Hel", "lo, ", "wor", "ld", "!"]
        adapter.engine._stream_chunks = chunks_in

        out = list(adapter.stream_response("hi", _make_context(), []))

        assert [c.content for c in out] == chunks_in
        assert all(isinstance(c, AgentChunk) for c in out)

    def test_only_last_chunk_is_final(self, adapter: RustAgentAdapter) -> None:
        """is_final is True only on the terminal chunk."""
        adapter.engine._stream_chunks = ["a", "b", "c"]

        out = list(adapter.stream_response("hi", _make_context(), []))

        assert [c.is_final for c in out] == [False, False, True]

    def test_blocking_backend_call_does_not_break_generator(
        self, adapter: RustAgentAdapter
    ) -> None:
        """When the Rust call blocks for ~2s and returns 5 chunks, the
        generator still yields all 5 chunks correctly.

        This is the regression scenario from #2752: callers must wrap this
        in a worker thread, but the generator itself must still behave.
        """
        adapter.engine._stream_chunks = ["c1", "c2", "c3", "c4", "c5"]
        adapter.engine._stream_delay = 2.0

        start = time.monotonic()
        out = list(adapter.stream_response("hi", _make_context(), []))
        elapsed = time.monotonic() - start

        assert [c.content for c in out] == ["c1", "c2", "c3", "c4", "c5"]
        assert out[-1].is_final is True
        assert all(not c.is_final for c in out[:-1])
        # Sanity: backend delay was actually exercised.
        assert elapsed >= 1.9

    def test_empty_deltas_yields_single_terminal_chunk(
        self, adapter: RustAgentAdapter
    ) -> None:
        """An empty backend response yields one final empty chunk."""
        adapter.engine._stream_chunks = []

        out = list(adapter.stream_response("hi", _make_context(), []))

        assert len(out) == 1
        assert out[0].content == ""
        assert out[0].is_final is True

    def test_streaming_failure_falls_back_to_generate_response(
        self, adapter: RustAgentAdapter
    ) -> None:
        """If stream_response raises, adapter falls back to generate_response."""
        mock_engine = MagicMock()
        mock_engine.stream_response.side_effect = RuntimeError("SSE not supported")
        mock_engine.generate_response.return_value = "single-shot-result"
        adapter.engine = mock_engine

        out = list(adapter.stream_response("hi", _make_context(), []))

        assert len(out) == 1
        assert out[0].content == "single-shot-result"
        assert out[0].is_final is True

    def test_docstring_documents_blocking_behavior(self) -> None:
        """The docstring must warn callers that the call is blocking (#2752)."""
        doc = RustAgentAdapter.stream_response.__doc__ or ""
        assert "blocking" in doc.lower()
        assert "thread" in doc.lower()

    def test_rust_adapter_stream_response_is_non_blocking_to_qt_events(
        self, adapter: RustAgentAdapter
    ) -> None:
        """Verify that stream_response processes Qt events while waiting for Rust."""
        mock_app = MagicMock()
        mock_thread = MagicMock()
        _MockEventLoop._on_exec = mock_app.processEvents

        orig_pyqt = sys.modules.get("PyQt6")
        orig_pyqt_qtcore = sys.modules.get("PyQt6.QtCore")

        # We need to mock sys.modules for PyQt6 since it might not be installed
        with MagicMock() as mock_pyqt:
            mock_pyqt.QtCore.QCoreApplication.instance.return_value = mock_app
            mock_pyqt.QtCore.QThread = _FakeQThread
            mock_pyqt.QtCore.QThread.currentThread.return_value = mock_thread
            mock_pyqt.QtCore.pyqtSignal = _FakeSignalDescriptor
            mock_pyqt.QtCore.QEventLoop = _FakeQEventLoop
            mock_app.thread.return_value = mock_thread

            sys.modules["PyQt6"] = mock_pyqt
            sys.modules["PyQt6.QtCore"] = mock_pyqt.QtCore

            adapter.engine._stream_chunks = ["chunk1", "chunk2"]
            adapter.engine._stream_delay = 0.1

            chunks = list(adapter.stream_response("test prompt", _make_context(), []))

            assert len(chunks) == 2
            assert chunks[0].content == "chunk1"
            assert chunks[1].content == "chunk2"
            assert mock_app.processEvents.called

        for name, orig in [("PyQt6", orig_pyqt), ("PyQt6.QtCore", orig_pyqt_qtcore)]:
            if orig is not None:
                sys.modules[name] = orig
            else:
                sys.modules.pop(name, None)


class TestSendMessage:
    """Tests for RustAgentAdapter.send_message."""

    @pytest.mark.unit
    def test_send_message_returns_agent_response(
        self, adapter: RustAgentAdapter
    ) -> None:
        """send_message returns an AgentResponse wrapping the engine output."""
        from src.shared.python.ai.types import AgentResponse

        adapter.engine.generate_response = MagicMock(return_value="engine output")
        result = adapter.send_message("hi", _make_context(), [])

        assert isinstance(result, AgentResponse)
        assert result.content == "engine output"

    @pytest.mark.unit
    def test_send_message_combines_context_and_prompt(
        self, adapter: RustAgentAdapter
    ) -> None:
        """Conversation messages are prepended to the prompt passed to the engine."""
        from src.shared.python.ai.types import Message

        ctx = ConversationContext(
            messages=[Message(role="user", content="prior msg")],
            user_expertise=ExpertiseLevel.INTERMEDIATE,
        )
        mock_engine = MagicMock()
        mock_engine.generate_response.return_value = "resp"
        adapter.engine = mock_engine

        adapter.send_message("new msg", ctx, [])

        call_arg = mock_engine.generate_response.call_args[0][0]
        assert "prior msg" in call_arg
        assert "new msg" in call_arg

    @pytest.mark.unit
    def test_send_message_engine_error_returns_error_response(
        self, adapter: RustAgentAdapter
    ) -> None:
        """When the engine raises, send_message returns an error AgentResponse."""
        adapter.engine.generate_response = MagicMock(
            side_effect=RuntimeError("engine crashed")
        )

        result = adapter.send_message("hi", _make_context(), [])

        assert "Error" in result.content
        assert result.finish_reason == "error"


class TestValidateConnection:
    """Tests for RustAgentAdapter.validate_connection."""

    @pytest.mark.unit
    def test_always_returns_true(self, adapter: RustAgentAdapter) -> None:
        """validate_connection always returns (True, msg) once __init__ succeeds."""
        ok, msg = adapter.validate_connection()
        assert ok is True
        assert msg  # non-empty diagnostic


class TestCapabilities:
    """Tests for RustAgentAdapter.capabilities property."""

    @pytest.mark.unit
    def test_provider_name_is_rust(self, adapter: RustAgentAdapter) -> None:
        assert adapter.capabilities.provider_name == "rust"

    @pytest.mark.unit
    def test_streaming_capability_present(self, adapter: RustAgentAdapter) -> None:
        from src.shared.python.ai.types import ProviderCapability

        assert ProviderCapability.STREAMING in adapter.capabilities.supported

    @pytest.mark.unit
    def test_model_name_reflects_config(self, adapter: RustAgentAdapter) -> None:
        assert adapter.capabilities.model_name == "stub-model"


class TestRagMethods:
    """Tests for index_codebase and retrieve_context."""

    @pytest.mark.unit
    def test_index_codebase_returns_int(self, adapter: RustAgentAdapter) -> None:
        result = adapter.index_codebase("/some/path")
        assert isinstance(result, int)

    @pytest.mark.unit
    def test_retrieve_context_returns_list_of_strings(
        self, adapter: RustAgentAdapter
    ) -> None:
        result = adapter.retrieve_context("some query", top_k=3)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)


class TestWheelMissingHint:
    """Tests for the wheel-missing error message."""

    @pytest.mark.unit
    def test_wheel_missing_hint_mentions_maturin(self) -> None:
        """The missing-wheel hint must tell the user how to build the extension."""
        from src.shared.python.ai.adapters.rust_adapter import _WHEEL_MISSING_HINT

        assert "maturin develop" in _WHEEL_MISSING_HINT

    @pytest.mark.unit
    def test_wheel_missing_hint_mentions_ai_backend(self) -> None:
        """The hint references the ai_backend crate name."""
        from src.shared.python.ai.adapters.rust_adapter import _WHEEL_MISSING_HINT

        assert "ai_backend" in _WHEEL_MISSING_HINT


class _MockSignal:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.receivers: list[Any] = []

    def connect(self, receiver: Any) -> None:
        self.receivers.append(receiver)

    def emit(self, *args: object, **kwargs: object) -> None:
        for receiver in self.receivers:
            receiver(*args, **kwargs)


class _MockQThread:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def start(self) -> None:
        self.run()  # type: ignore[attr-defined]


class _MockEventLoop:
    _on_exec: Any = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def exec(self) -> None:
        if _MockEventLoop._on_exec:
            _MockEventLoop._on_exec()

    def quit(self) -> None:
        pass
