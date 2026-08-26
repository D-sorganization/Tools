"""Stable values for the Tools design-manual governance contract."""

from __future__ import annotations

from dataclasses import dataclass

EXPECTED_POLICY_FIELDS = {
    "schema_version",
    "program",
    "canonical_source",
    "contracts",
    "calculation_inventory",
    "generated_outputs",
    "freshness",
    "publication",
    "quality",
    "license_boundary",
    "git",
    "agent_context",
}
EXPECTED_CONTRACTS = {
    "owner_repository": "D-sorganization/Engineering-Design-Manuals",
    "calculation_registry": (
        "https://schemas.d-sorganization.org/calculation-registry/1.0.0.json"
    ),
    "publication_projection": (
        "https://schemas.d-sorganization.org/engineering-manuals/"
        "publication-projection/1.0.0"
    ),
}
GENERATED_SUFFIXES = frozenset({".docx", ".html", ".pdf", ".tex"})
REQUIRED_FORMATS = ["html", "latex", "pdf", "docx"]
REQUIRED_EVIDENCE = [
    "immutable_source_commit",
    "source_tree_sha256",
    "calculation_registry_sha256",
    "toolchain_lock_sha256",
    "artifact_sha256",
    "semantic_parity",
    "pdf_page_review",
    "docx_page_review",
    "accessibility_review",
    "human_approval",
]
IMPACTED_PATHS = [
    "src",
    "scripts",
    "schemas",
    "config",
    "rust",
    "manuals/tools",
]
REQUIRED_UPDATE_FILES = [
    "config/design_manual_governance.json",
    "manuals/tools/calculation-registry.json",
    "manuals/tools/manifests/module-inventory.json",
    "manuals/tools/schemas/module-inventory.schema.json",
    "manuals/tools/schemas/module-inventory-shard.schema.json",
    "SPEC.md",
    "AGENT_HANDOFF.md",
]


class DesignManualGovernanceError(RuntimeError):
    """Raised when design-manual authority violates its versioned contract."""


def is_valid_revision(value: object) -> bool:
    """Return whether a value is a lowercase immutable Git revision."""
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class DesignManualGovernanceSummary:
    """Deterministic summary of the accepted repository governance state."""

    manual_id: str
    canonical_qmd_count: int
    calculation_count: int
    release_status: str
    public_projection_allowed: bool
