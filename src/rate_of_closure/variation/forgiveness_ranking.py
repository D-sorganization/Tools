"""Pareto ranking for already summarized chip-shot candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .chip_forgiveness import ChipStudyMetadata, ChipStudySummary


@dataclass(frozen=True)
class ChipCandidateScore:
    """Three declared objectives used for a transparent Pareto comparison."""

    metadata: ChipStudyMetadata
    expected_loss: float
    cvar_loss: float
    clean_probability: float

    def __post_init__(self) -> None:
        """Require finite losses and a bounded probability."""
        values = (self.expected_loss, self.cvar_loss, self.clean_probability)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("candidate score values must be finite")
        if self.expected_loss < 0.0 or self.cvar_loss < 0.0:
            raise ValueError("loss objectives must be >= 0")
        if not 0.0 <= self.clean_probability <= 1.0:
            raise ValueError("clean_probability must be in [0, 1]")

    @classmethod
    def from_summary(cls, summary: ChipStudySummary) -> ChipCandidateScore:
        """Project a complete summary onto the declared Pareto objectives."""
        if not isinstance(summary, ChipStudySummary):
            raise TypeError("summary must be a ChipStudySummary")
        return cls(
            metadata=summary.metadata,
            expected_loss=summary.expected_loss,
            cvar_loss=summary.cvar_loss,
            clean_probability=summary.clean_contact_probability,
        )


def _dominates(first: ChipCandidateScore, second: ChipCandidateScore) -> bool:
    no_worse = (
        first.expected_loss <= second.expected_loss
        and first.cvar_loss <= second.cvar_loss
        and first.clean_probability >= second.clean_probability
    )
    strictly_better = (
        first.expected_loss < second.expected_loss
        or first.cvar_loss < second.cvar_loss
        or first.clean_probability > second.clean_probability
    )
    return no_worse and strictly_better


def pareto_frontier(
    candidates: tuple[ChipCandidateScore, ...],
) -> tuple[ChipCandidateScore, ...]:
    """Return deterministic nondominated candidates without inventing weights."""
    if not candidates:
        raise ValueError("candidates must not be empty")
    identifiers = tuple(candidate.metadata.candidate_id for candidate in candidates)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate IDs must be unique")
    frontier = tuple(
        candidate
        for candidate in candidates
        if not any(
            _dominates(other, candidate)
            for other in candidates
            if other is not candidate
        )
    )
    return tuple(
        sorted(
            frontier,
            key=lambda item: (item.expected_loss, item.metadata.candidate_id),
        )
    )


__all__ = ["ChipCandidateScore", "pareto_frontier"]
