"""Tests for the builtin specialist reviewer (Tools #2738)."""

from __future__ import annotations

import pytest

from shared.python.ai.peer_review.builtin.specialist_reviewer import (
    SpecialistReviewer,
    StubReviewerLLMClient,
)
from shared.python.ai.peer_review.contracts import ReviewRequest, ReviewSubject

pytestmark = pytest.mark.unit


def _request() -> ReviewRequest:
    return ReviewRequest(
        subject_kind="code_block",
        subject_id="code-1",
        requester_agent_id="requester",
        criteria_set=["physics"],
    )


def _subject() -> ReviewSubject:
    return ReviewSubject(kind="code_block", subject_id="code-1", content="def f(): ...")


class TestSpecialistReviewer:
    async def test_default_role_is_specialist(self) -> None:
        reviewer = SpecialistReviewer(
            llm_client=StubReviewerLLMClient(),
            expertise_tags=["physics"],
        )
        assert reviewer.descriptor.role == "specialist"
        assert reviewer.descriptor.expertise_tags == ["physics"]

    async def test_expertise_tags_required_non_empty(self) -> None:
        with pytest.raises(ValueError):
            SpecialistReviewer(
                llm_client=StubReviewerLLMClient(),
                expertise_tags=[],
            )

    async def test_review_includes_role_in_verdict(self) -> None:
        reviewer = SpecialistReviewer(
            llm_client=StubReviewerLLMClient(canned_verdict="reject"),
            expertise_tags=["physics"],
        )
        verdict = await reviewer.review(_request(), _subject())
        assert verdict.reviewer_role == "specialist"
        assert verdict.verdict == "reject"
