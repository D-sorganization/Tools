"""Strict, versioned contract for Rate of Closure four-surface capability data."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, timedelta
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "four-surface-capability/v1"
SURFACE_IDS = (
    "tools.pyqt6",
    "tools.react",
    "upstreamdrift.pyqt6",
    "upstreamdrift.react",
)
CAPABILITY_CATEGORIES = (
    "model",
    "control",
    "output",
    "view",
    "persistence",
    "export",
)
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$")
PLACEHOLDER_PATTERN = re.compile(r"\b(?:FIXME|PLACEHOLDER|TBD|TODO|UNKNOWN)\b", re.I)
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "release"
    / "four_surface_capability.v1.json"
)


class StrictModel(BaseModel):
    """Immutable contract base that rejects undeclared fields."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, strict=True
    )


# ``StrEnum`` is Python 3.11-only; consumer validation also runs on Python 3.10.
class SurfaceId(str, Enum):  # noqa: UP042
    """Stable public identifiers for the four product surfaces."""

    TOOLS_PYQT6 = "tools.pyqt6"
    TOOLS_REACT = "tools.react"
    UPSTREAMDRIFT_PYQT6 = "upstreamdrift.pyqt6"
    UPSTREAMDRIFT_REACT = "upstreamdrift.react"


class CapabilityCategory(str, Enum):  # noqa: UP042
    """Required inventory categories from issue 4264."""

    MODEL = "model"
    CONTROL = "control"
    OUTPUT = "output"
    VIEW = "view"
    PERSISTENCE = "persistence"
    EXPORT = "export"


class CapabilityState(str, Enum):  # noqa: UP042
    """Mutually exclusive capability classifications."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    DEPRECATED = "deprecated"


class ToolsPin(StrictModel):
    """Exact Tools source snapshot audited by this inventory."""

    repository: Literal["D-sorganization/Tools"]
    branch: str = Field(min_length=1)
    commit_sha: str = Field(pattern=SHA_PATTERN.pattern)


class SchemaReference(StrictModel):
    """Identity and digest for the generated consumer schema."""

    id: Literal["four-surface-capability/v1"]
    path: Literal["docs/release/four_surface_capability.v1.schema.json"]
    sha256: str = Field(pattern=SHA256_PATTERN.pattern)


class FreshnessWindow(StrictModel):
    """Bounded observation window used by CI staleness gates."""

    observed_on: date
    max_age_days: int = Field(gt=0, le=90)
    expires_on: date

    @model_validator(mode="after")
    def validate_expiry(self) -> FreshnessWindow:
        """Require an expiry derived exactly from the declared maximum age."""
        expected = self.observed_on + timedelta(days=self.max_age_days)
        if self.expires_on != expected:
            raise ValueError("expires_on must equal observed_on plus max_age_days")
        return self


class SurfaceDefinition(StrictModel):
    """Repository and runtime identity for one stable surface."""

    id: SurfaceId
    repository: Literal["D-sorganization/Tools", "D-sorganization/UpstreamDrift"]
    runtime: Literal["pyqt6", "react"]
    role: Literal["authority", "consumer"]
    consumer_commit_sha: str | None = Field(default=None, pattern=SHA_PATTERN.pattern)


class EvidenceRecord(StrictModel):
    """Repository- and commit-bound evidence for a supported surface claim."""

    id: str = Field(pattern=ID_PATTERN.pattern)
    repository: Literal["D-sorganization/Tools", "D-sorganization/UpstreamDrift"]
    commit_sha: str = Field(pattern=SHA_PATTERN.pattern)
    source_paths: list[str] = Field(min_length=1)
    test_paths: list[str] = Field(min_length=1)
    assertion: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_paths(self) -> EvidenceRecord:
        """Require unique normalized repository-relative evidence paths."""
        paths = [*self.source_paths, *self.test_paths]
        if len(paths) != len(set(paths)):
            raise ValueError("evidence paths must be unique")
        for path in paths:
            _require_repository_relative_path(path)
        return self


class SurfaceCapability(StrictModel):
    """Truthful state of one capability on one surface."""

    state: CapabilityState
    reason: str | None
    evidence_ids: list[str]

    @model_validator(mode="after")
    def validate_state_evidence(self) -> SurfaceCapability:
        """Keep support evidence and limitation explanations mutually consistent."""
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("surface evidence IDs must be unique")
        if any(ID_PATTERN.fullmatch(item) is None for item in self.evidence_ids):
            raise ValueError("surface evidence IDs must use stable ID syntax")
        if self.state is CapabilityState.SUPPORTED:
            if self.reason is not None:
                raise ValueError("supported state cannot carry a limitation reason")
            if not self.evidence_ids:
                raise ValueError("supported state requires evidence")
        elif not self.reason:
            raise ValueError(f"{self.state.value} state requires a reason")
        return self


class CapabilityRecord(StrictModel):
    """One durable capability key classified across every product surface."""

    id: str = Field(pattern=ID_PATTERN.pattern)
    category: CapabilityCategory
    title: str = Field(min_length=1)
    surfaces: dict[str, SurfaceCapability]

    @model_validator(mode="after")
    def validate_surfaces(self) -> CapabilityRecord:
        """Require every canonical surface key exactly once."""
        if set(self.surfaces) != set(SURFACE_IDS):
            raise ValueError("capability surfaces must use all canonical IDs")
        return self


class CampaignReference(StrictModel):
    """Link to the canonical release authority without implying release."""

    path: Literal["docs/release/rate_of_closure_campaign.v1.json"]
    schema_version: Literal["rate-of-closure-campaign/v1"]
    program_issue: Literal[4260]


class InventoryScope(StrictModel):
    """Explicit boundary for the currently audited capability inventory."""

    status: Literal["partial"]
    included_categories: list[CapabilityCategory]
    excluded_scope_reason: str = Field(min_length=1)


class FourSurfaceCapabilityManifest(StrictModel):
    """Canonical version-1 four-surface capability inventory."""

    schema_version: Literal["four-surface-capability/v1"]
    tools_pin: ToolsPin
    schema_ref: SchemaReference = Field(alias="schema")
    freshness: FreshnessWindow
    campaign: CampaignReference
    inventory: InventoryScope
    surfaces: list[SurfaceDefinition]
    evidence: list[EvidenceRecord]
    capabilities: list[CapabilityRecord]
    limitations: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_authority(self) -> FourSurfaceCapabilityManifest:
        """Validate identities, references, and consumer support boundaries."""
        _require_unique(self.surfaces, "id")
        _require_unique(self.evidence, "id")
        _require_unique(self.capabilities, "id", label="capability id")
        _validate_surface_definitions(self.surfaces)
        _validate_inventory(self.inventory, self.capabilities)
        _validate_evidence_references(self)
        _reject_placeholders(self.model_dump(mode="json"), "manifest")
        return self


def _require_unique(
    records: list[Any], field_name: str, *, label: str | None = None
) -> None:
    values = [getattr(record, field_name) for record in records]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label or field_name} values are not allowed")


def _validate_surface_definitions(surfaces: list[SurfaceDefinition]) -> None:
    expected = (
        ("tools.pyqt6", "D-sorganization/Tools", "pyqt6", "authority"),
        ("tools.react", "D-sorganization/Tools", "react", "authority"),
        ("upstreamdrift.pyqt6", "D-sorganization/UpstreamDrift", "pyqt6", "consumer"),
        ("upstreamdrift.react", "D-sorganization/UpstreamDrift", "react", "consumer"),
    )
    actual = tuple(
        (surface.id.value, surface.repository, surface.runtime, surface.role)
        for surface in surfaces
    )
    if actual != expected:
        raise ValueError("surface definitions must match the canonical four surfaces")


def _validate_inventory(
    inventory: InventoryScope, capabilities: list[CapabilityRecord]
) -> None:
    if tuple(item.value for item in inventory.included_categories) != (
        CAPABILITY_CATEGORIES
    ):
        raise ValueError("inventory must list every capability category in order")
    observed = {capability.category.value for capability in capabilities}
    if observed != set(CAPABILITY_CATEGORIES):
        raise ValueError("capabilities must include every declared category")


def _validate_evidence_references(manifest: FourSurfaceCapabilityManifest) -> None:
    evidence = {item.id: item for item in manifest.evidence}
    surface_definitions = {item.id.value: item for item in manifest.surfaces}
    for item in evidence.values():
        is_tools_evidence = item.repository == manifest.tools_pin.repository
        if is_tools_evidence and item.commit_sha != manifest.tools_pin.commit_sha:
            raise ValueError("evidence must match the exact Tools pin")
    for capability in manifest.capabilities:
        for surface_id, state in capability.surfaces.items():
            if not set(state.evidence_ids).issubset(evidence):
                raise ValueError(f"{capability.id} references undeclared evidence")
            definition = surface_definitions[surface_id]
            if state.state is CapabilityState.SUPPORTED:
                is_unpinned_consumer = (
                    definition.repository != manifest.tools_pin.repository
                    and definition.consumer_commit_sha is None
                )
                if is_unpinned_consumer:
                    raise ValueError(
                        "consumer support requires an installed consumer pin"
                    )
                expected_commit = (
                    manifest.tools_pin.commit_sha
                    if definition.repository == manifest.tools_pin.repository
                    else definition.consumer_commit_sha
                )
                referenced = [evidence[item] for item in state.evidence_ids]
                matches_surface = any(
                    item.repository == definition.repository
                    and item.commit_sha == expected_commit
                    for item in referenced
                )
                if not matches_surface:
                    raise ValueError(
                        "supported state requires evidence from the surface pin"
                    )


def _require_repository_relative_path(path: str) -> None:
    candidate = PurePosixPath(path)
    invalid = (
        not path
        or "\\" in path
        or candidate.is_absolute()
        or any(part in {".", ".."} for part in candidate.parts)
    )
    if invalid:
        raise ValueError("evidence paths must be normalized repository-relative paths")


def _reject_placeholders(value: Any, path: str) -> None:
    if isinstance(value, str) and PLACEHOLDER_PATTERN.search(value):
        raise ValueError(f"placeholder text is forbidden at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_placeholders(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_placeholders(child, f"{path}[{index}]")


def load_four_surface_capability(
    path: Path = DEFAULT_MANIFEST_PATH,
) -> FourSurfaceCapabilityManifest:
    """Load and strictly validate one UTF-8 capability inventory."""
    value = FourSurfaceCapabilityManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    return cast(FourSurfaceCapabilityManifest, value)


def render_json_schema() -> bytes:
    """Generate deterministic UTF-8 JSON Schema bytes for consumers."""
    schema = FourSurfaceCapabilityManifest.model_json_schema()
    rendered = json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False)
    return f"{rendered}\n".encode()


def canonical_manifest_json(manifest: FourSurfaceCapabilityManifest) -> str:
    """Render a validated manifest in deterministic canonical presentation."""
    payload = manifest.model_dump(mode="json", by_alias=True)
    return f"{json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)}\n"


def validate_freshness(
    manifest: FourSurfaceCapabilityManifest, *, on_date: date | None = None
) -> None:
    """Fail closed when the evidence window is not current on ``on_date``."""
    current = on_date or date.today()
    if current < manifest.freshness.observed_on:
        raise ValueError("capability evidence is not yet current")
    if current > manifest.freshness.expires_on:
        raise ValueError("capability evidence is stale")


def validate_repository_evidence(
    manifest: FourSurfaceCapabilityManifest, repo_root: Path
) -> None:
    """Validate checked-in schema and commit-bound evidence paths."""
    schema_path = repo_root / manifest.schema_ref.path
    if not schema_path.is_file():
        raise ValueError(
            f"capability schema does not exist: {manifest.schema_ref.path}"
        )
    schema_bytes = schema_path.read_bytes()
    if schema_bytes != render_json_schema():
        raise ValueError("checked-in capability schema is not generator-current")
    schema_digest = hashlib.sha256(schema_bytes).hexdigest()
    if schema_digest != manifest.schema_ref.sha256:
        raise ValueError("capability schema digest does not match the manifest")
    paths = {
        path
        for evidence in manifest.evidence
        if evidence.repository == manifest.tools_pin.repository
        for path in (*evidence.source_paths, *evidence.test_paths)
    }
    missing = sorted(path for path in paths if not (repo_root / path).is_file())
    if missing:
        raise ValueError(f"capability evidence paths do not exist: {missing}")
