"""Deterministically enumerate capability declarations from campaign authority."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from rate_of_closure.four_surface_capability import (
    CapabilityCategory,
    CapabilityRecord,
    FourSurfaceCapabilityManifest,
)

CAMPAIGN_SCHEMA = "rate-of-closure-campaign/v1"
CAMPAIGN_SOURCE_PATH = "docs/release/rate_of_closure_campaign.v1.json"
SPEC_PREFIX = "docs/specs/"
RATE_SPEC_PREFIX = "docs/rate_of_closure/"


@dataclass(frozen=True)
class DeclaredCapability:
    """Stable capability declaration derived from a canonical source record."""

    id: str
    category: str
    title: str
    source_path: str
    source_key: str


def derive_declared_capabilities(
    campaign_path: Path, repo_root: Path
) -> tuple[DeclaredCapability, ...]:
    """Enumerate campaign programs and their explicitly linked active specs."""
    payload = _load_campaign_payload(campaign_path)
    programs = _programs(payload)
    campaign_records = tuple(
        _program_declaration(item, index) for index, item in enumerate(programs)
    )
    spec_paths = sorted(_linked_spec_paths(programs))
    spec_records = tuple(_spec_declaration(path, repo_root) for path in spec_paths)
    declared = campaign_records + spec_records
    _require_unique_ids(declared)
    return declared


def validate_declared_scope_completeness(
    manifest: FourSurfaceCapabilityManifest,
    campaign_path: Path,
    repo_root: Path,
) -> None:
    """Require exact manifest coverage of every deterministic declaration."""
    expected = {
        item.id: item for item in derive_declared_capabilities(campaign_path, repo_root)
    }
    declared_categories = {
        CapabilityCategory.CAMPAIGN_PROGRAM,
        CapabilityCategory.ACTIVE_SPECIFICATION,
    }
    actual = {
        item.id: item
        for item in manifest.capabilities
        if item.category in declared_categories
    }
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(
            f"declared capability IDs differ: missing={missing}, extra={extra}"
        )
    for capability_id, declaration in expected.items():
        _validate_record_metadata(actual[capability_id], declaration)
    _validate_inventory_counts(manifest, expected)


def render_declared_scope(campaign_path: Path, repo_root: Path) -> str:
    """Render byte-stable JSON for the deterministically declared scope."""
    records = [
        asdict(item) for item in derive_declared_capabilities(campaign_path, repo_root)
    ]
    return f"{json.dumps(records, indent=2, sort_keys=True, ensure_ascii=False)}\n"


def _load_campaign_payload(path: Path) -> dict[str, Any]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("campaign declaration source must be an object")
    if value.get("schema_version") != CAMPAIGN_SCHEMA:
        raise ValueError("campaign declaration source has unsupported schema")
    return value


def _programs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("programs")
    if not isinstance(value, list):
        raise ValueError("campaign programs must be an array")
    records: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("campaign program must be an object")
        records.append(item)
    return records


def _program_declaration(program: dict[str, Any], index: int) -> DeclaredCapability:
    issue = program.get("issue")
    title = program.get("title")
    if isinstance(issue, bool) or not isinstance(issue, int) or issue < 1:
        raise ValueError("campaign program issue must be a positive integer")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("campaign program title must be non-empty text")
    return DeclaredCapability(
        id=f"campaign.issue-{issue}",
        category=CapabilityCategory.CAMPAIGN_PROGRAM.value,
        title=title,
        source_path=CAMPAIGN_SOURCE_PATH,
        source_key=f"programs[{index}].issue={issue}",
    )


def _linked_spec_paths(programs: list[dict[str, Any]]) -> set[str]:
    paths: set[str] = set()
    for program in programs:
        authorities = program.get("authorities")
        if not isinstance(authorities, list):
            raise ValueError("campaign program authorities must be an array")
        for authority in authorities:
            if not isinstance(authority, dict):
                raise ValueError("campaign authority must be an object")
            value = authority.get("value")
            if authority.get("kind") == "repository_path" and isinstance(value, str):
                is_spec = _is_active_spec_authority(value)
                if is_spec and not _is_normalized_path(value):
                    raise ValueError("active specification path must be normalized")
                if is_spec:
                    paths.add(value)
    return paths


def _is_normalized_path(path: str) -> bool:
    candidate = PurePosixPath(path)
    return "\\" not in path and ".." not in candidate.parts


def _is_active_spec_authority(path: str) -> bool:
    is_markdown_contract = path.endswith(".md") and path.startswith(
        (SPEC_PREFIX, RATE_SPEC_PREFIX)
    )
    return path == "SPEC.md" or is_markdown_contract


def _spec_declaration(path: str, repo_root: Path) -> DeclaredCapability:
    source = repo_root / path
    try:
        heading = source.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError) as error:
        raise ValueError(f"active specification cannot be read: {path}") from error
    title = heading[2:].strip() if heading.startswith("# ") else ""
    if not title:
        raise ValueError(f"active specification requires an H1 title: {path}")
    slug = (
        "repository"
        if path == "SPEC.md"
        else re.sub(r"[^a-z0-9]+", "-", source.stem.lower()).strip("-")
    )
    if not slug:
        raise ValueError(f"active specification requires a stable filename: {path}")
    return DeclaredCapability(
        id=f"spec.{slug}",
        category=CapabilityCategory.ACTIVE_SPECIFICATION.value,
        title=title,
        source_path=path,
        source_key="heading.h1",
    )


def _require_unique_ids(records: tuple[DeclaredCapability, ...]) -> None:
    ids = [item.id for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("derived capability IDs must be unique")


def _validate_record_metadata(
    record: CapabilityRecord, expected: DeclaredCapability
) -> None:
    declaration = record.declaration
    expected_kind = (
        "campaign_program"
        if expected.category == CapabilityCategory.CAMPAIGN_PROGRAM.value
        else "active_release_spec"
    )
    actual = (
        record.category.value,
        record.title,
        declaration.kind,
        declaration.source_path,
        declaration.source_key,
    )
    wanted = (
        expected.category,
        expected.title,
        expected_kind,
        expected.source_path,
        expected.source_key,
    )
    if actual != wanted:
        raise ValueError(f"declared capability metadata differs: {expected.id}")


def _validate_inventory_counts(
    manifest: FourSurfaceCapabilityManifest,
    declared: dict[str, DeclaredCapability],
) -> None:
    programs = sum(
        item.category == CapabilityCategory.CAMPAIGN_PROGRAM.value
        for item in declared.values()
    )
    specs = len(declared) - programs
    curated = len(manifest.capabilities) - len(declared)
    counts = manifest.inventory
    if (
        counts.campaign_program_count != programs
        or counts.active_specification_count != specs
        or counts.curated_capability_count != curated
    ):
        raise ValueError("declared capability inventory counts differ")
