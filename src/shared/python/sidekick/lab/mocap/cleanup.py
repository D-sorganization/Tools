"""Non-destructive worktree cleanup eligibility contracts."""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class CleanupEvidence:
    """Path-free proof requirements for a task-owned worktree cleanup decision."""

    task_owned: bool
    verified_merged: bool
    clean_worktree: bool
    receipts_retained: bool
    contains_user_data: bool
    contains_raw_capture: bool
    shared_directory: bool

    def __post_init__(self) -> None:
        if any(not isinstance(getattr(self, item.name), bool) for item in fields(self)):
            raise TypeError("cleanup evidence fields must be booleans")


@dataclass(frozen=True, slots=True)
class CleanupDecision:
    """Non-destructive eligibility result; it never performs deletion."""

    eligible: bool
    blockers: tuple[str, ...]


def evaluate_cleanup_eligibility(evidence: CleanupEvidence) -> CleanupDecision:
    """Allow cleanup only for owned, merged, clean trees with receipts retained."""
    if not isinstance(evidence, CleanupEvidence):
        raise TypeError("evidence must be CleanupEvidence")
    requirements = (
        (evidence.task_owned, "worktree is not task-owned"),
        (evidence.verified_merged, "merge is not verified"),
        (evidence.clean_worktree, "worktree has uncommitted changes"),
        (evidence.receipts_retained, "required receipts are not retained"),
        (not evidence.contains_user_data, "worktree contains user data"),
        (not evidence.contains_raw_capture, "worktree contains raw capture data"),
        (not evidence.shared_directory, "directory is shared"),
    )
    blockers = tuple(message for condition, message in requirements if not condition)
    return CleanupDecision(not blockers, blockers)
