"""Tests for ai.peer_review.chat_integration (Tools #2738)."""

from __future__ import annotations

import pytest

from shared.python.ai.peer_review.base import Reviewer
from shared.python.ai.peer_review.chat_integration import request_peer_review
from shared.python.ai.peer_review.contracts import (
    ReviewerDescriptor,
    ReviewRequest,
    ReviewSubject,
    ReviewVerdict,
)
from shared.python.ai.peer_review.coordinator import ReviewCoordinator
from shared.python.ai.peer_review.registry import ReviewerRegistry

pytestmark = pytest.mark.unit


class _StubReviewer(Reviewer):
    def __init__(self, *, descriptor: ReviewerDescriptor, verdict: str) -> None:
        super().__init__(descriptor=descriptor)
        self._verdict = verdict

    async def review(
        self,
        request: ReviewRequest,
        subject: ReviewSubject,
    ) -> ReviewVerdict:
        return ReviewVerdict(
            reviewer_agent_id=self.descriptor.agent_id,
            verdict=self._verdict,  # type: ignore[arg-type]
            reasoning="stub",
            suggested_revisions=[],
            confidence_0_to_1=0.8,
            reviewer_role=self.descriptor.role,
        )


def _desc(aid: str, role: str) -> ReviewerDescriptor:
    return ReviewerDescriptor(
        agent_id=aid,
        provider="stub",
        model="stub-1",
        role=role,  # type: ignore[arg-type]
        expertise_tags=[],
    )


class _Session:
    """Minimal chat session double exposing the surface peer_review needs."""

    def __init__(self) -> None:
        self.streamed: list[ReviewVerdict] = []

    async def emit_review_verdict(self, verdict: ReviewVerdict) -> None:
        self.streamed.append(verdict)


class TestRequestPeerReview:
    async def test_streams_verdicts_in_arrival_order(self) -> None:
        registry = ReviewerRegistry()
        registry.register(
            _StubReviewer(descriptor=_desc("c", "critic"), verdict="approve")
        )
        registry.register(
            _StubReviewer(descriptor=_desc("a", "advocate"), verdict="approve")
        )
        coord = ReviewCoordinator(registry=registry)
        session = _Session()
        result = await request_peer_review(
            session=session,
            message_id="msg-1",
            criteria=["correctness"],
            requester_agent_id="user",
            coordinator=coord,
            subject_content="please review",
        )
        assert result.consensus == "approved"
        assert len(session.streamed) == 2
        assert {v.reviewer_agent_id for v in session.streamed} == {"c", "a"}

    async def test_empty_criteria_propagates_as_value_error(self) -> None:
        registry = ReviewerRegistry()
        registry.register(
            _StubReviewer(descriptor=_desc("c", "critic"), verdict="approve")
        )
        registry.register(
            _StubReviewer(descriptor=_desc("a", "advocate"), verdict="approve")
        )
        coord = ReviewCoordinator(registry=registry)
        with pytest.raises(ValueError):
            await request_peer_review(
                session=_Session(),
                message_id="msg-1",
                criteria=[],
                requester_agent_id="user",
                coordinator=coord,
                subject_content="x",
            )

    async def test_session_without_emit_hook_still_returns_result(self) -> None:
        registry = ReviewerRegistry()
        registry.register(
            _StubReviewer(descriptor=_desc("c", "critic"), verdict="approve")
        )
        registry.register(
            _StubReviewer(descriptor=_desc("a", "advocate"), verdict="approve")
        )
        coord = ReviewCoordinator(registry=registry)

        class _Bare:
            pass

        result = await request_peer_review(
            session=_Bare(),
            message_id="msg-1",
            criteria=["correctness"],
            requester_agent_id="user",
            coordinator=coord,
            subject_content="x",
        )
        assert result.consensus == "approved"
