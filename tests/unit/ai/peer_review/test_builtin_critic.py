"""Tests for the builtin critic reviewer (Tools #2738)."""

from __future__ import annotations

import pytest

from shared.python.ai.peer_review.builtin.critic_reviewer import (
    CriticReviewer,
    StubReviewerLLMClient,
)
from shared.python.ai.peer_review.contracts import ReviewRequest, ReviewSubject

pytestmark = pytest.mark.unit


def _request() -> ReviewRequest:
    return ReviewRequest(
        subject_kind="message",
        subject_id="m-1",
        requester_agent_id="requester",
        criteria_set=["correctness"],
    )


def _subject(content: str = "demo") -> ReviewSubject:
    return ReviewSubject(kind="message", subject_id="m-1", content=content)


class TestCriticReviewer:
    async def test_uses_injected_llm(self) -> None:
        stub = StubReviewerLLMClient(
            canned_verdict="request_changes",
            canned_reasoning="needs more evidence",
            canned_confidence=0.7,
        )
        reviewer = CriticReviewer(llm_client=stub)
        verdict = await reviewer.review(_request(), _subject())
        assert stub.call_count == 1
        assert verdict.verdict == "request_changes"
        assert verdict.reasoning == "needs more evidence"
        assert verdict.confidence_0_to_1 == 0.7
        assert verdict.reviewer_role == "critic"

    async def test_default_descriptor_role_is_critic(self) -> None:
        reviewer = CriticReviewer(llm_client=StubReviewerLLMClient())
        assert reviewer.descriptor.role == "critic"
        assert reviewer.descriptor.agent_id

    async def test_propagates_agent_id_into_verdict(self) -> None:
        reviewer = CriticReviewer(
            llm_client=StubReviewerLLMClient(canned_verdict="approve"),
            agent_id="critic-007",
        )
        verdict = await reviewer.review(_request(), _subject())
        assert verdict.reviewer_agent_id == "critic-007"

    async def test_llm_failure_yields_abstain(self) -> None:
        class _FailingLLM:
            async def evaluate(
                self,
                *,
                criteria_set: list[str],
                subject_content: str,
                role: str,
            ) -> dict[str, object]:
                raise RuntimeError("boom")

        reviewer = CriticReviewer(llm_client=_FailingLLM())
        verdict = await reviewer.review(_request(), _subject())
        assert verdict.verdict == "abstain"
        assert "boom" in verdict.reasoning.lower() or verdict.reasoning

    async def test_invalid_verdict_from_llm_yields_abstain(self) -> None:
        class _BadLLM:
            async def evaluate(
                self,
                *,
                criteria_set: list[str],
                subject_content: str,
                role: str,
            ) -> dict[str, object]:
                return {
                    "verdict": "garbled",
                    "reasoning": "noop",
                    "confidence": 0.5,
                }

        reviewer = CriticReviewer(llm_client=_BadLLM())
        verdict = await reviewer.review(_request(), _subject())
        assert verdict.verdict == "abstain"
