"""Tests for ai.peer_review.contracts (Tools #2738)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.python.ai.peer_review.contracts import (
    PeerReviewResult,
    ReviewerDescriptor,
    ReviewRequest,
    ReviewSubject,
    ReviewVerdict,
)

pytestmark = pytest.mark.unit


class TestReviewRequest:
    def test_minimal_round_trip(self) -> None:
        req = ReviewRequest(
            subject_kind="message",
            subject_id="msg-1",
            requester_agent_id="agent-a",
            criteria_set=["correctness"],
        )
        dumped = req.model_dump()
        revived = ReviewRequest.model_validate(dumped)
        assert revived == req

    def test_invalid_subject_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReviewRequest(
                subject_kind="wat",  # type: ignore[arg-type]
                subject_id="x",
                requester_agent_id="a",
                criteria_set=["c"],
            )

    def test_empty_criteria_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReviewRequest(
                subject_kind="message",
                subject_id="x",
                requester_agent_id="a",
                criteria_set=[],
            )

    def test_negative_deadline_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReviewRequest(
                subject_kind="message",
                subject_id="x",
                requester_agent_id="a",
                criteria_set=["c"],
                deadline_seconds=-1.0,
            )

    def test_default_request_id_present(self) -> None:
        req = ReviewRequest(
            subject_kind="code_block",
            subject_id="x",
            requester_agent_id="a",
            criteria_set=["c"],
        )
        assert req.request_id


class TestReviewerDescriptor:
    def test_role_enum_enforced(self) -> None:
        ReviewerDescriptor(
            agent_id="r-1",
            provider="stub",
            model="stub-1",
            role="critic",
            expertise_tags=["physics"],
        )
        with pytest.raises(ValidationError):
            ReviewerDescriptor(
                agent_id="r-2",
                provider="stub",
                model="stub-1",
                role="cheerleader",  # type: ignore[arg-type]
                expertise_tags=[],
            )


class TestReviewVerdict:
    def test_verdict_enum_enforced(self) -> None:
        with pytest.raises(ValidationError):
            ReviewVerdict(
                reviewer_agent_id="r-1",
                verdict="meh",  # type: ignore[arg-type]
                reasoning="x",
                suggested_revisions=[],
                confidence_0_to_1=0.5,
            )

    @pytest.mark.parametrize("conf", [-0.01, 1.01, 2.0, -1.0])
    def test_confidence_out_of_range_rejected(self, conf: float) -> None:
        with pytest.raises(ValidationError):
            ReviewVerdict(
                reviewer_agent_id="r-1",
                verdict="approve",
                reasoning="x",
                suggested_revisions=[],
                confidence_0_to_1=conf,
            )

    @pytest.mark.parametrize("conf", [0.0, 0.5, 1.0])
    def test_confidence_in_range_accepted(self, conf: float) -> None:
        v = ReviewVerdict(
            reviewer_agent_id="r-1",
            verdict="approve",
            reasoning="x",
            suggested_revisions=[],
            confidence_0_to_1=conf,
        )
        assert v.confidence_0_to_1 == conf


class TestReviewSubject:
    def test_round_trip(self) -> None:
        s = ReviewSubject(kind="message", subject_id="m-1", content="hello")
        s2 = ReviewSubject.model_validate(s.model_dump())
        assert s2 == s


class TestPeerReviewResult:
    def test_round_trip(self) -> None:
        verdict = ReviewVerdict(
            reviewer_agent_id="r-1",
            verdict="approve",
            reasoning="ok",
            suggested_revisions=[],
            confidence_0_to_1=1.0,
        )
        result = PeerReviewResult(
            request_id="req-1",
            verdicts=[verdict],
            consensus="approved",
            final_disposition="approved",
            audit_trail=[{"kind": "started"}],
        )
        revived = PeerReviewResult.model_validate(result.model_dump())
        assert revived == result
