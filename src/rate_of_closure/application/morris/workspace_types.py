"""Immutable values for lossless Morris workspace persistence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from rate_of_closure.application._workspace_validation import FrozenJsonValue

from ._response_types import MorrisResponseJob
from .contracts import MorrisAuthorityRequest


@dataclass(frozen=True)
class MorrisWorkspaceFactorDraft:
    """One canonical editor draft retaining its raw text and validation state."""

    variable_key: str
    enabled: bool
    lower: str
    upper: str
    validation_error: str | None


@dataclass(frozen=True)
class MorrisWorkspaceSetup:
    """Authority-compatible base plus every user-visible Morris control."""

    export_scope: str
    base: Mapping[str, FrozenJsonValue]
    factor_drafts: tuple[MorrisWorkspaceFactorDraft, ...]
    trajectories: int
    levels: int
    seed: int
    minimum_effects: int
    worker_count: int


@dataclass(frozen=True)
class MorrisCompletedEvidence:
    """Archived completed authority evidence; identifiers are inert provenance."""

    request: MorrisAuthorityRequest
    job: MorrisResponseJob


@dataclass(frozen=True)
class MorrisWorkspace:
    """Strict v1 lossless Morris editor and optional archived-evidence document."""

    schema_id: str
    schema_version: int
    setup: MorrisWorkspaceSetup
    completed_evidence: MorrisCompletedEvidence | None

    def base_config(self):  # type: ignore[no-untyped-def]
        """Reconstruct the pinned authority base without ambient application state."""
        if self.completed_evidence is not None:
            return self.completed_evidence.request.base_config()
        from .workspace_validation import base_config_from_setup

        return base_config_from_setup(self.setup)


__all__ = [
    "MorrisCompletedEvidence",
    "MorrisWorkspace",
    "MorrisWorkspaceFactorDraft",
    "MorrisWorkspaceSetup",
]
