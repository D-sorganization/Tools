"""Tests for ai.peer_review.registry (Tools #2738)."""

from __future__ import annotations

import pytest

from shared.python.ai.peer_review.base import Reviewer
from shared.python.ai.peer_review.contracts import (
    ReviewerDescriptor,
    ReviewRequest,
    ReviewSubject,
    ReviewVerdict,
)
from shared.python.ai.peer_review.registry import ReviewerRegistry

pytestmark = pytest.mark.unit


def _make_reviewer(
    agent_id: str,
    role: str,
    tags: list[str],
    verdict: str = "approve",
) -> Reviewer:
    descriptor = ReviewerDescriptor(
        agent_id=agent_id,
        provider="stub",
        model="stub-1",
        role=role,  # type: ignore[arg-type]
        expertise_tags=tags,
    )

    class _Stub(Reviewer):
        async def review(
            self,
            request: ReviewRequest,
            subject: ReviewSubject,
        ) -> ReviewVerdict:
            return ReviewVerdict(
                reviewer_agent_id=descriptor.agent_id,
                verdict=verdict,  # type: ignore[arg-type]
                reasoning="stub",
                suggested_revisions=[],
                confidence_0_to_1=0.8,
            )

    return _Stub(descriptor=descriptor)


class TestReviewerRegistry:
    def test_register_and_get(self) -> None:
        registry = ReviewerRegistry()
        r = _make_reviewer("r-1", "critic", ["physics"])
        registry.register(r)
        assert registry.get("r-1") is r

    def test_duplicate_register_raises(self) -> None:
        registry = ReviewerRegistry()
        registry.register(_make_reviewer("r-1", "critic", []))
        with pytest.raises(ValueError):
            registry.register(_make_reviewer("r-1", "advocate", []))

    def test_get_unknown_raises(self) -> None:
        registry = ReviewerRegistry()
        with pytest.raises(KeyError):
            registry.get("missing")

    def test_panel_for_criteria_includes_matching_specialist(self) -> None:
        registry = ReviewerRegistry()
        registry.register(_make_reviewer("c", "critic", []))
        registry.register(_make_reviewer("a", "advocate", []))
        registry.register(_make_reviewer("s-phys", "specialist", ["physics"]))
        registry.register(_make_reviewer("s-bio", "specialist", ["biology"]))
        panel = registry.panel_for(["physics"])
        ids = {r.descriptor.agent_id for r in panel}
        assert "s-phys" in ids
        assert "s-bio" not in ids
        # Critics and advocates are always included regardless of criteria
        assert "c" in ids
        assert "a" in ids

    def test_panel_for_no_criteria_match_still_returns_generic_pair(self) -> None:
        registry = ReviewerRegistry()
        registry.register(_make_reviewer("c", "critic", []))
        registry.register(_make_reviewer("a", "advocate", []))
        registry.register(_make_reviewer("s", "specialist", ["chemistry"]))
        panel = registry.panel_for(["physics"])
        ids = {r.descriptor.agent_id for r in panel}
        assert ids == {"c", "a"}

    def test_panel_for_empty_returns_empty(self) -> None:
        registry = ReviewerRegistry()
        assert registry.panel_for(["x"]) == ()
