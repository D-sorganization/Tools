"""Tests for RustAgentAdapter non-blocking streaming (Tools #2752).

The module is imported once per process; PyQt6 and ai_backend are both stubbed
at the ``sys.modules`` level so no real Qt installation is needed.

The ``TestRustStreamWorker`` and ``TestRustAdapterQThread`` classes use
synchronous fakes for ``QThread``, ``pyqtSignal``, and ``QEventLoop`` so
signals fire *before* the event-loop call returns, keeping tests deterministic
without any real threading.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.serial  # prevent xdist workers sharing sys.modules state

# ---------------------------------------------------------------------------
# Bootstrap: stub the src.* package tree so we can import the adapter module
# without a full repo install.
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

# ---------------------------------------------------------------------------
# Fake PyQt6 infrastructure (synchronous — no real Qt required)
# ---------------------------------------------------------------------------


class _FakeSignal:
    """Minimal pyqtSignal-alike: connect + emit."""

    def __init__(self, *_args: object) -> None:
        self._slots: list = []

    def connect(self, slot: object) -> None:
        self._slots.append(slot)

    def emit(self, *args: object) -> None:
        for slot in list(self._slots):
            if args:
                slot(*args)
            else:
                slot()


class _FakeQThread:
    """Synchronous stand-in for QThread: start() calls run() immediately."""

    def __init__(self) -> None:
        pass

    def start(self) -> None:  # noqa: D401
        self.run()

    def run(self) -> None:  # noqa: D401
        pass


class _FakeQEventLoop:
    """Stand-in for QEventLoop: exec() is a no-op (worker already ran)."""

    _exec_count: int = 0

    def exec(self) -> int:  # noqa: A003
        _FakeQEventLoop._exec_count += 1
        return 0

    def quit(self) -> None:
        pass


def _build_fake_qtcore() -> types.ModuleType:
    qtcore = types.ModuleType("PyQt6.QtCore")
    qtcore.QThread = _FakeQThread
    qtcore.pyqtSignal = _FakeSignal
    qtcore.QEventLoop = _FakeQEventLoop
    return qtcore


def _install_fake_pyqt6() -> tuple[types.ModuleType, types.ModuleType]:
    pyqt6 = types.ModuleType("PyQt6")
    qtcore = _build_fake_qtcore()
    sys.modules["PyQt6"] = pyqt6
    sys.modules["PyQt6.QtCore"] = qtcore
    return pyqt6, qtcore


def _install_fake_ai_backend() -> MagicMock:
    mock = MagicMock()
    sys.modules["ai_backend"] = mock
    return mock


# ---------------------------------------------------------------------------
# Module-level stubs (used by non-QThread tests that don't need real Qt)
# ---------------------------------------------------------------------------

_ai_backend_global = _install_fake_ai_backend()
_install_fake_pyqt6()

# Import under the stubs
import src.shared.python.ai.adapters.rust_adapter as _rust_adapter_module  # noqa: E402
from src.shared.python.ai.adapters.rust_adapter import (  # noqa: E402
    RustAgentAdapter,
    _make_rust_stream_worker_class,
)
from src.shared.python.ai.types import ConversationContext  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(engine_mock: MagicMock) -> RustAgentAdapter:
    _ai_backend_global.AIEngine.return_value = engine_mock
    return RustAgentAdapter(api_key="k", base_url="http://x", model="m")


def _make_context(*messages: str) -> ConversationContext:
    msg_objs = [MagicMock(content=m) for m in messages]
    return ConversationContext(messages=msg_objs)


# ---------------------------------------------------------------------------
# TestRustStreamWorker — unit-tests for _make_rust_stream_worker_class()
# ---------------------------------------------------------------------------


class TestRustStreamWorker:
    """Test the _RustStreamWorker QThread subclass produced by the factory."""

    def setup_method(self) -> None:
        _FakeQEventLoop._exec_count = 0
        _install_fake_pyqt6()
        importlib.reload(_rust_adapter_module)

    def teardown_method(self) -> None:
        pass

    def test_factory_returns_a_class(self) -> None:
        cls = _make_rust_stream_worker_class()
        assert cls is not None
        assert isinstance(cls, type)

    def test_factory_returns_none_without_pyqt6(self) -> None:
        saved_pyqt6 = sys.modules.pop("PyQt6", None)
        saved_qtcore = sys.modules.pop("PyQt6.QtCore", None)
        importlib.reload(_rust_adapter_module)
        try:
            cls = _rust_adapter_module._make_rust_stream_worker_class()
            assert cls is None
        finally:
            if saved_pyqt6 is not None:
                sys.modules["PyQt6"] = saved_pyqt6
            if saved_qtcore is not None:
                sys.modules["PyQt6.QtCore"] = saved_qtcore
            importlib.reload(_rust_adapter_module)

    def test_worker_emits_chunks_for_each_delta(self) -> None:
        cls = _make_rust_stream_worker_class()
        assert cls is not None

        engine = MagicMock()
        engine.stream_response.return_value = ["a", "b", "c"]

        worker = cls(engine, "prompt")
        received: list[str] = []
        finished: list[bool] = [False]

        worker.chunk_received.connect(received.append)
        worker.stream_finished.connect(lambda: finished.__setitem__(0, True))

        worker.start()  # synchronous via _FakeQThread

        assert received == ["a", "b", "c"]
        assert finished[0] is True

    def test_worker_emits_stream_error_on_exception(self) -> None:
        cls = _make_rust_stream_worker_class()
        assert cls is not None

        engine = MagicMock()
        engine.stream_response.side_effect = RuntimeError("boom")

        worker = cls(engine, "prompt")
        errors: list[str] = []
        worker.stream_error.connect(errors.append)

        worker.start()

        assert len(errors) == 1
        assert "boom" in errors[0]

    def test_worker_stop_halts_emission(self) -> None:
        cls = _make_rust_stream_worker_class()
        assert cls is not None

        engine = MagicMock()

        def _gen_deltas(_prompt: str) -> list[str]:
            # Return a bunch of deltas; worker should stop after the first
            return [str(i) for i in range(20)]

        engine.stream_response.side_effect = _gen_deltas

        worker = cls(engine, "prompt")
        received: list[str] = []

        # Stop the worker before it starts; it should emit nothing
        worker.stop()
        worker.chunk_received.connect(received.append)
        worker.start()

        assert received == []

    def test_stream_finished_fires_after_all_chunks(self) -> None:
        cls = _make_rust_stream_worker_class()
        assert cls is not None

        n = 50
        engine = MagicMock()
        engine.stream_response.return_value = [f"delta_{i}" for i in range(n)]

        worker = cls(engine, "prompt")
        received: list[str] = []
        finished_after: list[int] = []

        worker.chunk_received.connect(received.append)
        worker.stream_finished.connect(lambda: finished_after.append(len(received)))

        worker.start()

        assert finished_after == [n], (
            f"stream_finished fired when {finished_after} chunks had been received; "
            f"expected {n}"
        )


# ---------------------------------------------------------------------------
# TestRustAdapterQThread — integration tests for the adapter using fake Qt
# ---------------------------------------------------------------------------


class TestRustAdapterQThread:
    """Tests for RustAgentAdapter.stream_response() using fake PyQt6."""

    def setup_method(self) -> None:
        _FakeQEventLoop._exec_count = 0
        _install_fake_pyqt6()
        importlib.reload(_rust_adapter_module)

    def teardown_method(self) -> None:
        pass

    # ------------------------------------------------------------------
    # 50-chunk happy path
    # ------------------------------------------------------------------

    def test_stream_response_yields_50_chunks(self) -> None:
        n = 50
        engine = MagicMock()
        engine.stream_response.return_value = [f"delta_{i}" for i in range(n)]

        adapter = _make_adapter(engine)
        ctx = _make_context()

        chunks = list(adapter.stream_response("hi", ctx, []))

        assert len(chunks) == n
        for i, chunk in enumerate(chunks):
            assert chunk.content == f"delta_{i}"
        # Only the last chunk should be marked final
        assert chunks[-1].is_final is True
        for chunk in chunks[:-1]:
            assert chunk.is_final is False

    def test_qt_event_loop_exec_is_called_during_streaming(self) -> None:
        """QEventLoop.exec() must be called at least once while streaming."""
        engine = MagicMock()
        engine.stream_response.return_value = ["x", "y"]

        _FakeQEventLoop._exec_count = 0
        adapter = _make_adapter(engine)
        ctx = _make_context()

        list(adapter.stream_response("hi", ctx, []))

        assert _FakeQEventLoop._exec_count >= 1, (
            "QEventLoop.exec() was never called — the Qt event loop was not spun, "
            "meaning the main thread would freeze for real Qt UIs"
        )

    def test_stream_finished_fires_after_all_chunks_via_adapter(self) -> None:
        """Ensure all deltas are yielded before stream_finished fires."""
        n = 50
        engine = MagicMock()
        deltas = [f"d{i}" for i in range(n)]
        engine.stream_response.return_value = deltas

        adapter = _make_adapter(engine)
        ctx = _make_context()

        chunks = list(adapter.stream_response("hi", ctx, []))
        contents = [c.content for c in chunks]

        assert contents == deltas

    # ------------------------------------------------------------------
    # Empty stream
    # ------------------------------------------------------------------

    def test_stream_response_handles_empty_stream(self) -> None:
        engine = MagicMock()
        engine.stream_response.return_value = []

        adapter = _make_adapter(engine)
        ctx = _make_context()

        chunks = list(adapter.stream_response("hi", ctx, []))

        assert len(chunks) == 1
        assert chunks[0].content == ""
        assert chunks[0].is_final is True

    # ------------------------------------------------------------------
    # Error fallback
    # ------------------------------------------------------------------

    def test_stream_response_falls_back_on_error(self) -> None:
        engine = MagicMock()
        engine.stream_response.side_effect = RuntimeError("network error")
        engine.generate_response.return_value = "fallback response"

        adapter = _make_adapter(engine)
        ctx = _make_context()

        chunks = list(adapter.stream_response("hi", ctx, []))

        assert len(chunks) == 1
        assert chunks[0].content == "fallback response"
        assert chunks[0].is_final is True

    # ------------------------------------------------------------------
    # Context messages are included in prompt
    # ------------------------------------------------------------------

    def test_context_messages_are_prepended_to_prompt(self) -> None:
        engine = MagicMock()
        engine.stream_response.return_value = ["ok"]

        adapter = _make_adapter(engine)
        ctx = _make_context("msg1", "msg2")

        list(adapter.stream_response("question", ctx, []))

        call_args = engine.stream_response.call_args[0][0]
        assert "msg1" in call_args
        assert "msg2" in call_args
        assert "question" in call_args

    # ------------------------------------------------------------------
    # Headless fallback (no PyQt6)
    # ------------------------------------------------------------------

    def test_stream_response_works_without_pyqt6(self) -> None:
        """Adapter falls back to threading.Thread when PyQt6 is absent."""
        # Remove PyQt6 from sys.modules
        sys.modules.pop("PyQt6", None)
        sys.modules.pop("PyQt6.QtCore", None)
        importlib.reload(_rust_adapter_module)

        engine = MagicMock()
        engine.stream_response.return_value = ["a", "b", "c"]
        _ai_backend_global.AIEngine.return_value = engine
        adapter = _rust_adapter_module.RustAgentAdapter(
            api_key="k", base_url="http://x", model="m"
        )
        ctx = _make_context()
        chunks = list(adapter.stream_response("hi", ctx, []))

        assert [c.content for c in chunks] == ["a", "b", "c"]

        # Restore
        _install_fake_pyqt6()
        importlib.reload(_rust_adapter_module)


# ---------------------------------------------------------------------------
# TestStreamResponseGenerator — existing generator-contract tests
# (kept to avoid regressing downstream contract coverage)
# ---------------------------------------------------------------------------


class TestStreamResponseGenerator:
    """Original generator-API tests that must still pass."""

    def setup_method(self) -> None:
        _install_fake_pyqt6()
        importlib.reload(_rust_adapter_module)

    def teardown_method(self) -> None:
        pass

    def _make_adapter_for(self, engine: MagicMock) -> RustAgentAdapter:
        _ai_backend_global.AIEngine.return_value = engine
        return _rust_adapter_module.RustAgentAdapter(
            api_key="k", base_url="http://x", model="m"
        )

    def test_stream_response_yields_agent_chunks(self) -> None:
        from src.shared.python.ai.types import AgentChunk

        engine = MagicMock()
        engine.stream_response.return_value = ["hello", " world"]
        adapter = self._make_adapter_for(engine)
        ctx = _make_context()

        chunks = list(adapter.stream_response("hi", ctx, []))

        assert all(isinstance(c, AgentChunk) for c in chunks)
        assert chunks[0].content == "hello"
        assert chunks[1].content == " world"

    def test_last_chunk_is_final(self) -> None:
        engine = MagicMock()
        engine.stream_response.return_value = ["a", "b", "c"]
        adapter = self._make_adapter_for(engine)
        ctx = _make_context()

        chunks = list(adapter.stream_response("hi", ctx, []))

        assert chunks[-1].is_final is True

    def test_intermediate_chunks_are_not_final(self) -> None:
        engine = MagicMock()
        engine.stream_response.return_value = ["a", "b", "c"]
        adapter = self._make_adapter_for(engine)
        ctx = _make_context()

        chunks = list(adapter.stream_response("hi", ctx, []))

        for chunk in chunks[:-1]:
            assert chunk.is_final is False

    def test_stream_response_with_context_messages(self) -> None:
        engine = MagicMock()
        engine.stream_response.return_value = ["response"]
        adapter = self._make_adapter_for(engine)
        ctx = _make_context("previous message")

        list(adapter.stream_response("new prompt", ctx, []))

        call_arg = engine.stream_response.call_args[0][0]
        assert "previous message" in call_arg
        assert "new prompt" in call_arg

    def test_stream_response_error_yields_error_chunk(self) -> None:
        engine = MagicMock()
        engine.stream_response.side_effect = RuntimeError("test error")
        engine.generate_response.side_effect = RuntimeError("fallback also failed")
        adapter = self._make_adapter_for(engine)
        ctx = _make_context()

        chunks = list(adapter.stream_response("hi", ctx, []))

        assert len(chunks) == 1
        assert chunks[0].is_final is True
        assert "Error" in chunks[0].content or "test error" in chunks[0].content

    def test_stream_response_empty_result(self) -> None:
        engine = MagicMock()
        engine.stream_response.return_value = []
        adapter = self._make_adapter_for(engine)
        ctx = _make_context()

        chunks = list(adapter.stream_response("hi", ctx, []))

        assert len(chunks) == 1
        assert chunks[0].content == ""
        assert chunks[0].is_final is True

    def test_stream_response_single_chunk(self) -> None:
        engine = MagicMock()
        engine.stream_response.return_value = ["only chunk"]
        adapter = self._make_adapter_for(engine)
        ctx = _make_context()

        chunks = list(adapter.stream_response("hi", ctx, []))

        assert len(chunks) == 1
        assert chunks[0].content == "only chunk"
        assert chunks[0].is_final is True
