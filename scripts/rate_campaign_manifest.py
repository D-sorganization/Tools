"""Validate the versioned Rate of Closure campaign release authority."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "release" / "rate_of_closure_campaign.v1.json"
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
PLACEHOLDER_PATTERN = re.compile(r"\b(?:FIXME|PLACEHOLDER|TBD|TODO|UNKNOWN)\b", re.I)
ISSUE_URL_PATTERN = re.compile(
    r"^https://github\.com/D-sorganization/Tools/issues/[1-9][0-9]*$"
)
PRIMARY_ISSUES = frozenset(
    (4103, 4120, 4125, 4130, 4142, 4146, 4158, 4181, 4189, 4191)
    + (4201, 4218, 4234, 4260, 4267)
)


class StrictModel(BaseModel):
    """Base contract that rejects undeclared fields and mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ReleaseStage(StrEnum):
    """Mutually exclusive campaign delivery stages."""

    SPECIFIED_ONLY = "specified_only"
    FEATURE_STACK = "implemented_on_feature_stack"
    PROTECTED_PARENT = "protected_merged_to_parent"
    MAIN = "released_to_main"


class CompletionState(StrEnum):
    """Scope completion independent of repository integration."""

    SPECIFIED = "specified"
    PARTIAL = "partial_implementation"
    IMPLEMENTED = "implemented"


class ReleaseStatus(StrEnum):
    """Default-branch release state."""

    NOT_RELEASED = "not_released"
    RELEASED = "released"


class CarrierState(StrEnum):
    """Observed pull-request integration state."""

    OPEN = "open"
    MERGED = "merged"
    CLOSED = "closed"


class ProtectionEvidence(StrEnum):
    """Evidence attached to a carrier merge."""

    NOT_RECORDED = "not_recorded"
    NOT_PROTECTED = "not_protected"
    CHECKS_PASSED = "protected_checks_passed"


class SurfaceState(StrEnum):
    """Truthful implementation state for one product surface."""

    NOT_STARTED = "not_started"
    CONTRACT_ONLY = "contract_only"
    IMPLEMENTED = "implemented_unverified"
    LOCALLY_VERIFIED = "locally_verified"
    RELEASED = "released"


class EvidenceOutcome(StrEnum):
    """Terminal or nonterminal evidence result."""

    PASSED = "passed"
    FAILED = "failed"
    QUEUED = "queued"
    CANCELLED = "cancelled"


class AuthorityRef(StrictModel):
    """Repository or issue authority for a program contract."""

    kind: Literal["repository_path", "github_issue"]
    value: str = Field(min_length=1)


class ReleaseRecord(StrictModel):
    """Default-branch release evidence for a complete program."""

    status: ReleaseStatus
    branch: Literal["main"] = "main"
    sha: str | None

    @model_validator(mode="after")
    def validate_release_sha(self) -> ReleaseRecord:
        """Require an exact SHA only for a claimed main release."""
        if self.status is ReleaseStatus.RELEASED:
            if self.sha is None or SHA_PATTERN.fullmatch(self.sha) is None:
                raise ValueError("released state requires a 40-character release SHA")
        elif self.sha is not None:
            raise ValueError("not_released state cannot carry a release SHA")
        return self


class CarrierRecord(StrictModel):
    """One nonduplicated pull-request carrier record."""

    id: str = Field(pattern=ID_PATTERN.pattern)
    pr: int = Field(gt=0)
    branch: str = Field(min_length=1)
    base_branch: str = Field(min_length=1)
    head_sha: str = Field(pattern=SHA_PATTERN.pattern)
    state: CarrierState
    merge_commit_sha: str | None
    protection_evidence: ProtectionEvidence

    @model_validator(mode="after")
    def validate_merge_fields(self) -> CarrierRecord:
        """Keep open and merged carrier evidence mutually consistent."""
        if self.state is CarrierState.MERGED:
            if self.merge_commit_sha is None:
                raise ValueError("merged carrier requires merge_commit_sha")
            if SHA_PATTERN.fullmatch(self.merge_commit_sha) is None:
                raise ValueError(
                    "merge_commit_sha must contain 40 lowercase hex characters"
                )
        elif self.merge_commit_sha is not None:
            raise ValueError("unmerged carrier cannot carry merge_commit_sha")
        return self


class TestEvidence(StrictModel):
    """Commit-bound test or hosted-check evidence."""

    id: str = Field(pattern=ID_PATTERN.pattern)
    kind: Literal["local", "hosted"]
    commit_sha: str = Field(pattern=SHA_PATTERN.pattern)
    outcome: EvidenceOutcome
    commands: list[str] = Field(min_length=1)
    summary: str = Field(min_length=1)
    source_paths: list[str] = Field(min_length=1)


class ProgramRecord(StrictModel):
    """Release evidence for one primary campaign issue or epic."""

    issue: int = Field(gt=0)
    title: str = Field(min_length=1)
    kind: Literal["epic", "program", "release_gate"]
    delivery_stage: ReleaseStage
    completion: CompletionState
    authorities: list[AuthorityRef] = Field(min_length=1)
    tracked_issue_ids: list[int]
    carrier_ids: list[str]
    supported_surfaces: dict[str, SurfaceState]
    evidence_ids: list[str]
    evidence_gap: str | None
    limitations: list[str] = Field(min_length=1)
    depends_on_issues: list[int]
    release: ReleaseRecord

    @model_validator(mode="after")
    def validate_program_state(self) -> ProgramRecord:
        """Reject delivery-stage and completion contradictions."""
        if self.delivery_stage is ReleaseStage.SPECIFIED_ONLY:
            if self.completion is not CompletionState.SPECIFIED or self.carrier_ids:
                raise ValueError(
                    "specified_only requires specified scope and no carriers"
                )
        elif self.completion is CompletionState.SPECIFIED or not self.carrier_ids:
            raise ValueError(
                "implemented delivery stage requires implementation and carriers"
            )
        if self.delivery_stage is ReleaseStage.MAIN:
            if self.release.status is not ReleaseStatus.RELEASED:
                raise ValueError("released_to_main requires released status")
        elif self.release.status is not ReleaseStatus.NOT_RELEASED:
            raise ValueError("non-main delivery stage cannot claim a release")
        if not self.evidence_ids and not self.evidence_gap:
            raise ValueError(
                "program requires evidence_ids or an explicit evidence_gap"
            )
        expected = {
            "tools.pyqt6",
            "tools.react",
            "upstreamdrift.pyqt6",
            "upstreamdrift.react",
        }
        if set(self.supported_surfaces) != expected:
            raise ValueError(
                "supported_surfaces must declare all four product surfaces"
            )
        if self.issue in self.depends_on_issues:
            raise ValueError("program cannot depend on its own issue")
        return self


class CampaignManifest(StrictModel):
    """Canonical Rate of Closure campaign release-evidence document."""

    schema_version: Literal["rate-of-closure-campaign/v1"]
    repository: Literal["D-sorganization/Tools"]
    as_of: str = Field(pattern=r"^20[0-9]{2}-[0-9]{2}-[0-9]{2}$")
    default_branch: Literal["main"]
    required_main_checks: list[str] = Field(min_length=1)
    release_stage_definitions: list[ReleaseStage]
    campaign_release: ReleaseRecord
    carriers: list[CarrierRecord]
    test_evidence: list[TestEvidence]
    programs: list[ProgramRecord]

    @model_validator(mode="after")
    def validate_authority(self) -> CampaignManifest:
        """Validate scope, references, ordering, and release claims."""
        _require_unique(self.carriers, "id")
        _require_unique(self.carriers, "pr")
        _require_unique(self.test_evidence, "id")
        _require_unique(self.programs, "issue")
        if {program.issue for program in self.programs} != PRIMARY_ISSUES:
            raise ValueError(
                "programs must cover the canonical primary campaign issues"
            )
        if self.release_stage_definitions != list(ReleaseStage):
            raise ValueError(
                "release_stage_definitions must list every stage in canonical order"
            )
        if self.required_main_checks != ["quality-gate", "tests (3.11)"]:
            raise ValueError("required_main_checks must match protected main")
        release_shas = {program.release.sha for program in self.programs}
        release_shas.discard(None)
        if self.campaign_release.status is ReleaseStatus.RELEASED:
            all_released = all(
                program.delivery_stage is ReleaseStage.MAIN for program in self.programs
            )
            if release_shas != {self.campaign_release.sha} or not all_released:
                raise ValueError(
                    "campaign release requires every program at the same main SHA"
                )
        elif release_shas:
            raise ValueError("program release contradicts unreleased campaign")
        pending = {
            program.issue: set(program.depends_on_issues) & PRIMARY_ISSUES
            for program in self.programs
        }
        while pending and (
            ready := {
                issue for issue, dependencies in pending.items() if not dependencies
            }
        ):
            for issue in ready:
                pending.pop(issue)
            for dependencies in pending.values():
                dependencies.difference_update(ready)
        if pending:
            raise ValueError("campaign program dependency cycle is forbidden")
        _validate_references(self)
        _reject_placeholders(self.model_dump(mode="json"), "manifest")
        return self


def _require_unique(records: list[Any], field_name: str) -> None:
    values = [getattr(record, field_name) for record in records]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {field_name} values are not allowed")


def _validate_references(manifest: CampaignManifest) -> None:
    carriers = {carrier.id: carrier for carrier in manifest.carriers}
    evidence_ids = {evidence.id for evidence in manifest.test_evidence}
    for program in manifest.programs:
        if not set(program.carrier_ids).issubset(carriers):
            raise ValueError(f"issue {program.issue} references an undeclared carrier")
        if not set(program.evidence_ids).issubset(evidence_ids):
            raise ValueError(
                f"issue {program.issue} references undeclared test evidence"
            )
        _validate_integrated_stage(program, carriers)


def _validate_integrated_stage(
    program: ProgramRecord, carriers: dict[str, CarrierRecord]
) -> None:
    referenced = [carriers[carrier_id] for carrier_id in program.carrier_ids]
    if program.delivery_stage is ReleaseStage.PROTECTED_PARENT:
        qualified = any(
            carrier.state is CarrierState.MERGED
            and carrier.protection_evidence is ProtectionEvidence.CHECKS_PASSED
            for carrier in referenced
        )
        if not qualified:
            raise ValueError(
                "protected_merged_to_parent requires protected merge evidence"
            )
    if program.delivery_stage is ReleaseStage.MAIN:
        released = [
            carrier
            for carrier in referenced
            if carrier.state is CarrierState.MERGED and carrier.base_branch == "main"
        ]
        if not released or program.release.sha not in {
            carrier.merge_commit_sha for carrier in released
        }:
            raise ValueError("released_to_main SHA must match a main carrier merge")


def _reject_placeholders(value: Any, path: str) -> None:
    if isinstance(value, str) and PLACEHOLDER_PATTERN.search(value):
        raise ValueError(f"placeholder text is forbidden at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_placeholders(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_placeholders(child, f"{path}[{index}]")


def load_campaign_manifest(path: Path = DEFAULT_MANIFEST) -> CampaignManifest:
    """Read and strictly validate a UTF-8 JSON campaign manifest."""
    validated = CampaignManifest.model_validate_json(path.read_text(encoding="utf-8"))
    return cast(CampaignManifest, validated)


def validate_repository_evidence(
    manifest: CampaignManifest, repo_root: Path = REPO_ROOT
) -> None:
    """Validate repository paths and issue references against ``repo_root``."""
    paths = _repository_paths(manifest)
    missing = [path for path in paths if not (repo_root / path).is_file()]
    if missing:
        raise ValueError(f"campaign evidence paths do not exist: {sorted(missing)}")
    issue_urls = _issue_urls(manifest)
    invalid_urls = [
        url for url in issue_urls if ISSUE_URL_PATTERN.fullmatch(url) is None
    ]
    if invalid_urls:
        raise ValueError(f"invalid campaign issue references: {sorted(invalid_urls)}")


def _repository_paths(manifest: CampaignManifest) -> set[str]:
    paths = {
        authority.value
        for program in manifest.programs
        for authority in program.authorities
        if authority.kind == "repository_path"
    }
    return paths | {
        path for evidence in manifest.test_evidence for path in evidence.source_paths
    }


def _issue_urls(manifest: CampaignManifest) -> set[str]:
    return {
        authority.value
        for program in manifest.programs
        for authority in program.authorities
        if authority.kind == "github_issue"
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--schema", action="store_true", help="emit JSON Schema")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate the manifest or emit its deterministic JSON Schema."""
    args = _parser().parse_args(argv)
    if args.schema:
        schema = CampaignManifest.model_json_schema()
        json.dump(schema, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    try:
        manifest = load_campaign_manifest(args.manifest)
        validate_repository_evidence(manifest)
    except (OSError, ValueError, ValidationError) as error:
        logging.error("campaign manifest validation failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
