"""Tests for ai.peer_review.coordinator (Tools #2738)."""

from __future__ import annotations

import asyncio

import pytest

from shared.python.ai.peer_review.base import Reviewer
from shared.python.ai.peer_review.contracts import (
    ReviewerDescriptor,
    ReviewRequest,
    ReviewSubject,
    ReviewVerdict,
)
from shared.python.ai.peer_review.coordinator import ReviewCoordinator
from shared.python.ai.peer_review.errors import (
    InsufficientPanelError,
    NoReviewersError,
    ReviewerTimeoutError,
)
from shared.python.ai.peer_review.registry import ReviewerRegistry

pytestmark = pytest.mark.unit


class _ScriptedReviewer(Reviewer):
    def __init__(
        self,
        *,
        descriptor: ReviewerDescriptor,
        verdict: str,
        confidence: float = 0.8,
        delay_s: float = 0.0,
    ) -> None:
        super().__init__(descriptor=descriptor)
        self._verdict = verdict
        self._confidence = confidence
        self._delay_s = delay_s

    async def review(
        self,
        request: ReviewRequest,
        subject: ReviewSubject,
    ) -> ReviewVerdict:
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        return ReviewVerdict(
            reviewer_agent_id=self.descriptor.agent_id,
            verdict=self._verdict,  # type: ignore[arg-type]
            reasoning="scripted",
            suggested_revisions=[],
            confidence_0_to_1=self._confidence,
            reviewer_role=self.descriptor.role,
        )


def _desc(agent_id: str, role: str, tags: list[str]) -> ReviewerDescriptor:
    return ReviewerDescriptor(
        agent_id=agent_id,
        provider="stub",
        model="stub-1",
        role=role,  # type: ignore[arg-type]
        expertise_tags=tags,
    )


def _request(criteria: list[str] | None = None) -> ReviewRequest:
    return ReviewRequest(
        subject_kind="message",
        subject_id="m-1",
        requester_agent_id="requester",
        criteria_set=criteria if criteria is not None else ["correctness"],
    )


def _subject() -> ReviewSubject:
    return ReviewSubject(kind="message", subject_id="m-1", content="hello")


class TestReviewCoordinator:
    async def test_happy_path_all_approve(self) -> None:
        registry = ReviewerRegistry()
        registry.register(
            _ScriptedReviewer(descriptor=_desc("c", "critic", []), verdict="approve")
        )
        registry.register(
            _ScriptedReviewer(descriptor=_desc("a", "advocate", []), verdict="approve")
        )
        registry.register(
            _ScriptedReviewer(
                descriptor=_desc("s", "specialist", ["correctness"]),
                verdict="approve",
            )
        )
        coord = ReviewCoordinator(registry=registry)
        result = await coord.run_review(_request(), _subject())
        assert result.consensus == "approved"
        assert result.final_disposition == "approved"
        assert len(result.verdicts) == 3

    async def test_split_decision(self) -> None:
        registry = ReviewerRegistry()
        registry.register(
            _ScriptedReviewer(
                descriptor=_desc("c", "critic", []),
                verdict="approve",
                confidence=0.9,
            )
        )
        registry.register(
            _ScriptedReviewer(
                descriptor=_desc("a", "advocate", []),
                verdict="approve",
                confidence=0.9,
            )
        )
        registry.register(
            _ScriptedReviewer(
                descriptor=_desc("s", "specialist", ["correctness"]),
                verdict="reject",
                confidence=0.9,
            )
        )
        coord = ReviewCoordinator(registry=registry)
        result = await coord.run_review(_request(), _subject())
        # 2 approves outweigh 1 reject at equal confidence
        assert result.consensus == "approved"

    async def test_no_reviewers_raises(self) -> None:
        registry = ReviewerRegistry()
        coord = ReviewCoordinator(registry=registry)
        with pytest.raises(NoReviewersError):
            await coord.run_review(_request(), _subject())

    async def test_insufficient_panel_raises(self) -> None:
        registry = ReviewerRegistry()
        # Only one reviewer registered → below minimum panel size of 2
        registry.register(
            _ScriptedReviewer(descriptor=_desc("c", "critic", []), verdict="approve")
        )
        coord = ReviewCoordinator(registry=registry, min_panel_size=2)
        with pytest.raises(InsufficientPanelError):
            await coord.run_review(_request(), _subject())

    async def test_timeout(self) -> None:
        registry = ReviewerRegistry()
        registry.register(
            _ScriptedReviewer(
                descriptor=_desc("c", "critic", []),
                verdict="approve",
                delay_s=5.0,
            )
        )
        registry.register(
            _ScriptedReviewer(
                descriptor=_desc("a", "advocate", []),
                verdict="approve",
                delay_s=5.0,
            )
        )
        request = ReviewRequest(
            subject_kind="message",
            subject_id="m-1",
            requester_agent_id="requester",
            criteria_set=["correctness"],
            deadline_seconds=0.05,
        )
        coord = ReviewCoordinator(registry=registry)
        with pytest.raises(ReviewerTimeoutError):
            await coord.run_review(request, _subject())

    async def test_audit_trail_contains_lifecycle_events(self) -> None:
        registry = ReviewerRegistry()
        registry.register(
            _ScriptedReviewer(descriptor=_desc("c", "critic", []), verdict="approve")
        )
        registry.register(
            _ScriptedReviewer(descriptor=_desc("a", "advocate", []), verdict="approve")
        )
        coord = ReviewCoordinator(registry=registry)
        result = await coord.run_review(_request(), _subject())
        kinds = [event["kind"] for event in result.audit_trail]
        assert "started" in kinds
        assert "completed" in kinds

    async def test_empty_criteria_precondition(self) -> None:
        registry = ReviewerRegistry()
        registry.register(
            _ScriptedReviewer(descriptor=_desc("c", "critic", []), verdict="approve")
        )
        registry.register(
            _ScriptedReviewer(descriptor=_desc("a", "advocate", []), verdict="approve")
        )
        coord = ReviewCoordinator(registry=registry)
        # ReviewRequest.criteria_set is enforced non-empty by Pydantic; if we
        # subvert it, the coordinator must still reject defensively.
        request = _request(["correctness"])
        # Manually clear criteria post-validation to verify DbC precondition.
        object.__setattr__(request, "criteria_set", [])
        with pytest.raises(ValueError):
            await coord.run_review(request, _subject())
