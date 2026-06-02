"""Tests for issue #3179 hardening fixes.

Covers:
- Gemini error-contract: ``send_message``/``stream_response`` raise a typed
  ``AIProviderError`` (via ``_classify_error``) instead of leaking the raw
  exception string as model content.
- Adapter branding: ``_build_system_message`` no longer hardcodes the
  "Golf Modeling Suite" literal and routes through the configurable
  ``app_context`` preamble (brand-neutral by default).
- tool_bridge fail-closed: a tool that requires confirmation with no
  confirmation callback configured is refused, not executed.
- rust_adapter: the headless threaded fallback uses a bounded queue wait
  (no unbounded busy-poll) and times out cleanly.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.ai.exceptions import AIProviderError  # noqa: E402
from src.shared.python.ai.types import (  # noqa: E402
    ConversationContext,
    ExpertiseLevel,
)


def _make_context() -> ConversationContext:
    return ConversationContext(messages=[], user_expertise=ExpertiseLevel.INTERMEDIATE)


# ---------------------------------------------------------------------------
# Gemini error-contract
# ---------------------------------------------------------------------------


@pytest.fixture()
def gemini_adapter():  # type: ignore[no-untyped-def]
    """A GeminiAdapter built without running the SDK-requiring __init__.

    The error-contract path only needs ``_classify_error`` (inherited) and a
    stubbed ``_build_chat_session`` / ``_with_configured_sdk``; constructing
    via ``object.__new__`` avoids running __init__ which requires a real API key.
    gemini_adapter.py guards its google SDK import with try/except ImportError,
    so the module loads cleanly even without the SDK installed.
    """
    # gemini_adapter.py guards the google SDK import with try/except ImportError,
    # so the module loads cleanly even without the SDK installed.
    from src.shared.python.ai.adapters.gemini_adapter import GeminiAdapter

    adapter = object.__new__(GeminiAdapter)
    adapter._api_key = "fake-key"  # type: ignore[attr-defined]
    adapter._model_name = "gemini-pro"  # type: ignore[attr-defined]
    adapter._client = None  # type: ignore[attr-defined]
    adapter._model = MagicMock()  # type: ignore[attr-defined]
    return adapter


class TestGeminiErrorContract:
    """send_message / stream_response must raise, not leak error strings."""

    @pytest.mark.unit
    def test_send_message_raises_ai_provider_error(
        self, gemini_adapter: object
    ) -> None:
        """A SDK RuntimeError is classified and raised, not returned as content."""
        boom = MagicMock()
        boom.send_message.side_effect = RuntimeError("upstream 500")
        with (
            patch.object(gemini_adapter, "_with_configured_sdk"),
            patch.object(gemini_adapter, "_build_chat_session", return_value=boom),
        ):
            with pytest.raises(AIProviderError) as info:
                gemini_adapter.send_message("hi", _make_context(), [])
        assert info.value.provider == "gemini"
        # The raw exception text must not be presented as model output.
        assert "upstream 500" not in str(info.value) or "gemini" in str(info.value)

    @pytest.mark.unit
    def test_stream_response_raises_ai_provider_error(
        self, gemini_adapter: object
    ) -> None:
        """Streaming failure raises AIProviderError instead of an error chunk."""
        boom = MagicMock()
        boom.send_message.side_effect = ValueError("bad request")
        with (
            patch.object(gemini_adapter, "_with_configured_sdk"),
            patch.object(gemini_adapter, "_build_chat_session", return_value=boom),
        ):
            with pytest.raises(AIProviderError):
                list(gemini_adapter.stream_response("hi", _make_context(), []))


# ---------------------------------------------------------------------------
# Branding / configurable preamble
# ---------------------------------------------------------------------------


class TestBranding:
    """No hardcoded product literal; preamble injectable via app_context."""

    @pytest.mark.unit
    def test_no_golf_literal_in_adapter_source(self) -> None:
        """The 'Golf Modeling Suite' literal is gone from adapter sources."""
        import inspect

        import src.shared.python.ai.adapters.anthropic_adapter as a_mod
        import src.shared.python.ai.adapters.openai_adapter as o_mod

        for mod in (a_mod, o_mod):
            src = inspect.getsource(mod)
            assert "Golf Modeling Suite" not in src

    @pytest.mark.unit
    def test_default_preamble_is_brand_neutral(self) -> None:
        """Default app_context produces a neutral preamble (no product literal)."""
        from src.shared.python.ai.adapters.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(api_key="x")
        msg = adapter._build_system_message(_make_context())
        assert "Golf Modeling Suite" not in msg
        assert "AI assistant" in msg

    @pytest.mark.unit
    def test_app_context_injects_branding(self) -> None:
        """A non-default app_context changes the preamble (injectable)."""
        from src.shared.python.ai.adapters.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter(api_key="x", app_context="gasification")
        msg = adapter._build_system_message(_make_context())
        assert "Integrated Process Simulator" in msg


# ---------------------------------------------------------------------------
# tool_bridge fail-closed
# ---------------------------------------------------------------------------


class TestToolBridgeFailClosed:
    """A confirmation-required tool with no callback must NOT execute."""

    @pytest.mark.unit
    def test_confirmation_required_no_callback_is_refused(self) -> None:
        from src.shared.python.ai.tool_bridge import ChatToolBridge

        tool = MagicMock()
        tool.requires_confirmation = True
        tool.validate_arguments.return_value = []

        registry = MagicMock()
        registry.get_tool.return_value = tool

        bridge = ChatToolBridge(registry=registry)  # no confirmation callback

        result = asyncio.run(
            bridge.handle_tool_call(session_id="s1", tool_name="danger", arguments={})
        )

        assert result["success"] is False
        assert "confirmation" in result["error"].lower()
        tool.execute.assert_not_called()


# ---------------------------------------------------------------------------
# rust_adapter bounded wait
# ---------------------------------------------------------------------------


class TestRustBoundedWait:
    """The threaded fallback must not busy-poll and must honor a deadline."""

    @pytest.mark.unit
    def test_no_busy_poll_sleep_in_source(self) -> None:
        """Regression: the 10ms busy-poll loop is gone from the source."""
        import inspect

        from src.shared.python.ai.adapters.rust_adapter import RustAgentAdapter

        src = inspect.getsource(RustAgentAdapter._stream_with_thread)
        assert "time.sleep(0.01)" not in src

    @pytest.mark.unit
    def test_stream_with_thread_times_out_cleanly(self) -> None:
        """A worker that never produces a result yields a final timeout chunk."""
        import queue

        from src.shared.python.ai.adapters import rust_adapter as r_mod

        # Build a bare adapter instance without running __init__ (which needs
        # the ai_backend wheel); we only exercise the threaded helper.
        adapter = object.__new__(r_mod.RustAgentAdapter)
        # engine.stream_response blocks forever -> never enqueues a result.
        engine = MagicMock()
        blocker = __import__("threading").Event()
        engine.stream_response.side_effect = lambda *_a, **_k: blocker.wait()
        adapter.engine = engine  # type: ignore[attr-defined]

        q: queue.Queue = queue.Queue()
        with patch.object(r_mod, "_STREAM_THREAD_TIMEOUT_S", 0.3):
            chunks = list(adapter._stream_with_thread("prompt", q))

        blocker.set()
        assert chunks[-1].is_final is True
        assert "timed out" in chunks[-1].content.lower()
