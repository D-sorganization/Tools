"""Conformance tests for ChatServiceBase abstract contract.

TDD: Tests written to drive the conformance requirement (issue #2937).
DbC: Each test asserts what concrete subclasses MUST guarantee.
DRY: ConformanceTestMixin provides a single suite that runs against ALL
     concrete ChatServiceBase implementations.
LOD: Tests interact with the service through its public API only.

To add a new implementation to the conformance suite:
1. Create a subclass of ConformanceTestMixin
2. Set self.service in setup_method()
3. Pytest will automatically discover and run all conformance tests

Module path: src/shared/python/chat/service_base.py
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# ConformanceTestMixin — reusable suite for every ChatServiceBase subclass
# ---------------------------------------------------------------------------


class ConformanceTestMixin:
    """Mixin with conformance tests that all ChatServiceBase subclasses must pass.

    DRY: Inherit this in each subclass test suite instead of duplicating tests.

    Concrete test classes must set ``self.service`` in ``setup_method()``.
    """

    service: Any  # concrete ChatServiceBase instance

    # -- Structure --

    def test_is_not_abstract(self) -> None:
        """Precondition: service is a concrete subclass instance.
        Postcondition: type is not abstract (can be instantiated)."""
        assert not inspect.isabstract(type(self.service))

    def test_has_stream_response_method(self) -> None:
        """Precondition: service instance exists.
        Postcondition: stream_response is callable."""
        method = getattr(self.service, "stream_response", None)
        assert callable(method), "stream_response must be callable"

    def test_has_get_or_create_session_method(self) -> None:
        """Precondition: service instance exists.
        Postcondition: get_or_create_session is callable."""
        assert callable(getattr(self.service, "get_or_create_session", None))

    def test_has_add_user_message_method(self) -> None:
        """Precondition: service instance exists.
        Postcondition: add_user_message is callable."""
        assert callable(getattr(self.service, "add_user_message", None))

    def test_has_get_session_history_method(self) -> None:
        """Precondition: service instance exists.
        Postcondition: get_session_history is callable."""
        assert callable(getattr(self.service, "get_session_history", None))

    def test_has_list_sessions_method(self) -> None:
        """Precondition: service instance exists.
        Postcondition: list_sessions is callable."""
        assert callable(getattr(self.service, "list_sessions", None))

    # -- Session lifecycle --

    def test_get_or_create_session_with_none_creates_new(self) -> None:
        """Precondition: service exists, session_id=None passed.
        Postcondition: a non-None ChatSession is returned."""
        from chat.service_base import ChatSession

        session = self.service.get_or_create_session(None)
        assert session is not None
        assert isinstance(session, ChatSession)

    def test_get_or_create_session_returns_same_session(self) -> None:
        """Precondition: a session was previously created.
        Postcondition: get_or_create_session(existing_id) returns the same session."""
        session1 = self.service.get_or_create_session(None)
        session2 = self.service.get_or_create_session(session1.session_id)
        assert session2.session_id == session1.session_id

    def test_add_user_message_rejects_empty_session_id(self) -> None:
        """Precondition: empty string passed as session_id.
        Postcondition: ValueError raised (DbC precondition)."""
        with pytest.raises(ValueError, match="non-empty|session_id"):
            self.service.add_user_message("", "hello")

    def test_add_user_message_rejects_too_long_message(self) -> None:
        """Precondition: message exceeds 10000 characters.
        Postcondition: ValueError raised (DbC precondition)."""
        session = self.service.get_or_create_session(None)
        long_message = "x" * 10001
        with pytest.raises(ValueError):
            self.service.add_user_message(session.session_id, long_message)

    def test_add_user_message_returns_string_id(self) -> None:
        """Precondition: valid session exists with a short message.
        Postcondition: add_user_message returns a non-empty string (message ID)."""
        session = self.service.get_or_create_session(None)
        msg_id = self.service.add_user_message(session.session_id, "test input")
        assert isinstance(msg_id, str)
        assert len(msg_id) > 0

    def test_get_session_history_returns_list(self) -> None:
        """Precondition: session exists with one user message.
        Postcondition: get_session_history returns a list with >= 1 entry."""
        session = self.service.get_or_create_session(None)
        self.service.add_user_message(session.session_id, "hello")
        history = self.service.get_session_history(session.session_id)
        assert isinstance(history, list)
        assert len(history) >= 1

    def test_get_session_history_unknown_id_returns_empty(self) -> None:
        """Precondition: session ID does not exist.
        Postcondition: get_session_history returns empty list (not exception)."""
        history = self.service.get_session_history("id_that_does_not_exist")
        assert history == []

    def test_list_sessions_returns_list(self) -> None:
        """Precondition: at least one session was created.
        Postcondition: list_sessions returns a non-empty list."""
        self.service.get_or_create_session(None)
        sessions = self.service.list_sessions()
        assert isinstance(sessions, list)
        assert len(sessions) >= 1

    def test_history_entry_has_role_and_content(self) -> None:
        """Precondition: one message added to a session.
        Postcondition: history[0] has 'role' and 'content' keys."""
        session = self.service.get_or_create_session(None)
        self.service.add_user_message(session.session_id, "check fields")
        history = self.service.get_session_history(session.session_id)
        entry = history[0]
        assert "role" in entry
        assert "content" in entry

    # -- Optional method stubs (default implementations must not crash) --

    def test_condense_session_does_not_raise(self) -> None:
        """Precondition: session exists.
        Postcondition: condense_session default implementation completes
        without error."""
        session = self.service.get_or_create_session(None)

        async def _run() -> None:
            await self.service.condense_session(session.session_id)

        asyncio.run(_run())

    def test_execute_skill_does_not_raise(self) -> None:
        """Precondition: session exists.
        Postcondition: execute_skill default implementation completes without error."""
        session = self.service.get_or_create_session(None)

        async def _run() -> None:
            await self.service.execute_skill(session.session_id, "test-skill")

        asyncio.run(_run())

    def test_request_review_returns_session_id(self) -> None:
        """Precondition: session exists.
        Postcondition: request_review returns a non-empty string (session ID)."""
        session = self.service.get_or_create_session(None)

        async def _run() -> str:
            return await self.service.request_review(session.session_id, "claude")

        result = asyncio.run(_run())
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Abstract contract tests — run directly against ChatServiceBase
# ---------------------------------------------------------------------------


class TestChatServiceBaseContract:
    """Verify the base contract itself (abstract methods declared correctly)."""

    def test_base_class_is_abstract(self) -> None:
        """Precondition: ChatServiceBase imported.
        Postcondition: it is abstract and cannot be instantiated directly."""
        from chat.service_base import ChatServiceBase

        with pytest.raises(TypeError):
            ChatServiceBase()  # type: ignore[abstract]

    def test_stream_response_is_abstract(self) -> None:
        """Precondition: ChatServiceBase class object loaded.
        Postcondition: 'stream_response' is in __abstractmethods__."""
        from chat.service_base import ChatServiceBase

        abstract_methods = getattr(ChatServiceBase, "__abstractmethods__", frozenset())
        assert "stream_response" in abstract_methods

    def test_stream_response_is_async(self) -> None:
        """Precondition: ChatServiceBase.stream_response exists.
        Postcondition: it is an async generator method (coroutine function)."""
        from chat.service_base import ChatServiceBase

        method = ChatServiceBase.__dict__.get("stream_response")
        assert method is not None
        assert asyncio.iscoroutinefunction(method) or inspect.isasyncgenfunction(method)

    def test_non_abstract_subclass_instantiable(self) -> None:
        """Precondition: a concrete subclass implements stream_response.
        Postcondition: it can be instantiated without TypeError."""
        from chat.service_base import ChatServiceBase

        class _ConcreteService(ChatServiceBase):
            async def stream_response(self, session_id: str) -> AsyncIterator[Any]:
                yield "ok"

        svc = _ConcreteService()
        assert svc is not None

    def test_session_ttl_class_attribute_exists(self) -> None:
        """Precondition: ChatServiceBase loaded.
        Postcondition: SESSION_TTL_SECONDS is a positive integer."""
        from chat.service_base import ChatServiceBase

        assert hasattr(ChatServiceBase, "SESSION_TTL_SECONDS")
        assert isinstance(ChatServiceBase.SESSION_TTL_SECONDS, int)
        assert ChatServiceBase.SESSION_TTL_SECONDS > 0

    def test_max_sessions_class_attribute_exists(self) -> None:
        """Precondition: ChatServiceBase loaded.
        Postcondition: MAX_SESSIONS is a positive integer."""
        from chat.service_base import ChatServiceBase

        assert hasattr(ChatServiceBase, "MAX_SESSIONS")
        assert isinstance(ChatServiceBase.MAX_SESSIONS, int)
        assert ChatServiceBase.MAX_SESSIONS > 0

    def test_max_messages_per_session_exists(self) -> None:
        """Precondition: ChatServiceBase loaded.
        Postcondition: MAX_MESSAGES_PER_SESSION is a positive integer."""
        from chat.service_base import ChatServiceBase

        assert hasattr(ChatServiceBase, "MAX_MESSAGES_PER_SESSION")
        assert isinstance(ChatServiceBase.MAX_MESSAGES_PER_SESSION, int)
        assert ChatServiceBase.MAX_MESSAGES_PER_SESSION > 0

    def test_refresh_models_raises_not_implemented(self) -> None:
        """Precondition: concrete subclass that does not override refresh_models.
        Postcondition: refresh_models() raises NotImplementedError."""
        from chat.service_base import ChatServiceBase

        class _MinimalService(ChatServiceBase):
            async def stream_response(self, session_id: str) -> AsyncIterator[Any]:
                yield "ok"

        svc = _MinimalService()

        async def _run() -> None:
            await svc.refresh_models()

        with pytest.raises(NotImplementedError):
            asyncio.run(_run())

    def test_index_codebase_raises_not_implemented(self) -> None:
        """Precondition: concrete subclass that does not override index_codebase.
        Postcondition: index_codebase() raises NotImplementedError."""
        from chat.service_base import ChatServiceBase

        class _MinimalService(ChatServiceBase):
            async def stream_response(self, session_id: str) -> AsyncIterator[Any]:
                yield "ok"

        svc = _MinimalService()

        async def _run() -> None:
            await svc.index_codebase("/some/path")

        with pytest.raises(NotImplementedError):
            asyncio.run(_run())


# ---------------------------------------------------------------------------
# ChatMessage and ChatSession — data model conformance
# ---------------------------------------------------------------------------


class TestChatSessionDataModel:
    """Verify ChatSession and ChatMessage behave as documented."""

    def test_chat_session_has_session_id(self) -> None:
        """Precondition: ChatSession created with no args.
        Postcondition: session_id is a non-empty string."""
        from chat.service_base import ChatSession

        session = ChatSession()
        assert isinstance(session.session_id, str)
        assert len(session.session_id) > 0

    def test_chat_session_message_count_starts_at_zero(self) -> None:
        """Precondition: freshly created ChatSession.
        Postcondition: message_count == 0."""
        from chat.service_base import ChatSession

        session = ChatSession()
        assert session.message_count == 0

    def test_add_message_increments_count(self) -> None:
        """Precondition: empty session.
        Postcondition: add_message() increments message_count by 1."""
        from chat.service_base import ChatSession

        session = ChatSession()
        session.add_message("user", "hello")
        assert session.message_count == 1

    def test_add_message_returns_chat_message(self) -> None:
        """Precondition: session exists.
        Postcondition: add_message returns a ChatMessage with matching role/content."""
        from chat.service_base import ChatMessage, ChatSession

        session = ChatSession()
        msg = session.add_message("user", "test content")
        assert isinstance(msg, ChatMessage)
        assert msg.role == "user"
        assert msg.content == "test content"

    def test_chat_message_has_timestamp(self) -> None:
        """Precondition: ChatMessage created.
        Postcondition: timestamp is a positive float."""
        from chat.service_base import ChatMessage

        msg = ChatMessage(role="user", content="hi")
        assert isinstance(msg.timestamp, float)
        assert msg.timestamp > 0


# ---------------------------------------------------------------------------
# Concrete implementation conformance test
# ---------------------------------------------------------------------------


class _MinimalTestService:
    """A minimal concrete ChatServiceBase for use in conformance tests."""

    @classmethod
    def create(cls) -> Any:
        """Factory that returns a ready-to-use service instance."""
        from chat.service_base import ChatServiceBase

        class _Impl(ChatServiceBase):
            async def stream_response(self, session_id: str) -> AsyncIterator[Any]:
                yield f"response for {session_id}"

        return _Impl()


class TestMinimalConcreteServiceConformance(ConformanceTestMixin):
    """Run the full ConformanceTestMixin against the minimal test implementation.

    DRY: Adding a new implementation is done by creating a new subclass of
    ConformanceTestMixin and setting self.service in setup_method().
    """

    def setup_method(self) -> None:
        """Precondition: _MinimalTestService.create() returns a ChatServiceBase.
        Postcondition: self.service is a concrete, non-abstract instance."""
        self.service = _MinimalTestService.create()


# ---------------------------------------------------------------------------
# LRU session eviction
# ---------------------------------------------------------------------------


class TestChatServiceBaseSessionEviction:
    """Verify LRU eviction when MAX_SESSIONS is exceeded."""

    def test_sessions_evicted_at_limit(self) -> None:
        """Precondition: service created with MAX_SESSIONS=2, 3 sessions created.
        Postcondition: only 2 sessions remain in list_sessions()."""
        from chat.service_base import ChatServiceBase

        class _SmallService(ChatServiceBase):
            MAX_SESSIONS = 2

            async def stream_response(self, session_id: str) -> AsyncIterator[Any]:
                yield "ok"

        svc = _SmallService()
        for _ in range(3):
            svc.get_or_create_session(None)

        assert len(svc.list_sessions()) <= 2
