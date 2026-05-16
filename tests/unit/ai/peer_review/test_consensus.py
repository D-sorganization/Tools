"""Tests for ai.peer_review.consensus (Tools #2738)."""

from __future__ import annotations

import pytest

from shared.python.ai.peer_review.consensus import compute_consensus
from shared.python.ai.peer_review.contracts import ReviewVerdict

pytestmark = pytest.mark.unit


def _v(
    reviewer: str,
    verdict: str,
    confidence: float = 0.8,
    role: str = "critic",
) -> ReviewVerdict:
    return ReviewVerdict(
        reviewer_agent_id=reviewer,
        verdict=verdict,  # type: ignore[arg-type]
        reasoning="r",
        suggested_revisions=[],
        confidence_0_to_1=confidence,
        reviewer_role=role,  # type: ignore[arg-type]
    )


class TestComputeConsensus:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_consensus([])

    def test_all_approve(self) -> None:
        verdicts = [_v("a", "approve"), _v("b", "approve"), _v("c", "approve")]
        assert compute_consensus(verdicts) == "approved"

    def test_all_reject(self) -> None:
        verdicts = [_v("a", "reject"), _v("b", "reject")]
        assert compute_consensus(verdicts) == "rejected"

    def test_all_request_changes(self) -> None:
        verdicts = [_v("a", "request_changes"), _v("b", "request_changes")]
        assert compute_consensus(verdicts) == "needs_revision"

    def test_two_approve_one_reject(self) -> None:
        verdicts = [
            _v("a", "approve", 0.9),
            _v("b", "approve", 0.9),
            _v("c", "reject", 0.9),
        ]
        assert compute_consensus(verdicts) == "approved"

    def test_split_even_no_consensus(self) -> None:
        # Two roles same weight, opposing verdicts → no_consensus
        verdicts = [
            _v("a", "approve", 0.5, role="critic"),
            _v("b", "reject", 0.5, role="critic"),
        ]
        assert compute_consensus(verdicts) == "no_consensus"

    def test_abstain_ignored(self) -> None:
        verdicts = [
            _v("a", "approve", 1.0),
            _v("b", "abstain", 1.0),
            _v("c", "approve", 1.0),
        ]
        assert compute_consensus(verdicts) == "approved"

    def test_all_abstain_no_consensus(self) -> None:
        verdicts = [_v("a", "abstain"), _v("b", "abstain")]
        assert compute_consensus(verdicts) == "no_consensus"

    def test_confidence_weighting_overrides_count(self) -> None:
        verdicts = [
            _v("a", "reject", 0.99),
            _v("b", "approve", 0.05),
            _v("c", "approve", 0.05),
        ]
        # Two low-confidence approves vs one near-certain reject:
        # confidence-weighted sum favours rejection.
        assert compute_consensus(verdicts) == "rejected"

    def test_specialist_breaks_tie(self) -> None:
        # Same confidence, specialist's vote should outweigh critic's.
        verdicts = [
            _v("c", "reject", 0.5, role="critic"),
            _v("s", "approve", 0.5, role="specialist"),
        ]
        assert compute_consensus(verdicts) == "approved"

    def test_critic_outranks_advocate_in_tie(self) -> None:
        verdicts = [
            _v("a", "approve", 0.5, role="advocate"),
            _v("c", "reject", 0.5, role="critic"),
        ]
        assert compute_consensus(verdicts) == "rejected"

    def test_request_changes_dominates_mixed(self) -> None:
        # Two request_changes vs one approve and one reject
        verdicts = [
            _v("a", "request_changes", 0.9),
            _v("b", "request_changes", 0.9),
            _v("c", "approve", 0.5),
            _v("d", "reject", 0.5),
        ]
        assert compute_consensus(verdicts) == "needs_revision"
