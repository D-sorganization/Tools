"""Tests for the builtin advocate reviewer (Tools #2738)."""

from __future__ import annotations

import pytest

from shared.python.ai.peer_review.builtin.advocate_reviewer import (
    AdvocateReviewer,
    StubReviewerLLMClient,
)
from shared.python.ai.peer_review.contracts import ReviewRequest, ReviewSubject

pytestmark = pytest.mark.unit


def _request() -> ReviewRequest:
    return ReviewRequest(
        subject_kind="message",
        subject_id="m-1",
        requester_agent_id="requester",
        criteria_set=["clarity"],
    )


def _subject() -> ReviewSubject:
    return ReviewSubject(kind="message", subject_id="m-1", content="demo")


class TestAdvocateReviewer:
    async def test_default_role_is_advocate(self) -> None:
        reviewer = AdvocateReviewer(llm_client=StubReviewerLLMClient())
        assert reviewer.descriptor.role == "advocate"

    async def test_calls_llm_and_returns_verdict(self) -> None:
        stub = StubReviewerLLMClient(
            canned_verdict="approve",
            canned_reasoning="great work",
            canned_confidence=0.9,
        )
        reviewer = AdvocateReviewer(llm_client=stub)
        verdict = await reviewer.review(_request(), _subject())
        assert stub.call_count == 1
        assert verdict.verdict == "approve"
        assert verdict.reviewer_role == "advocate"
