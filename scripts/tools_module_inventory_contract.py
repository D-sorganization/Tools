"""Strict consumer contract for the Tools module inventory extension."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import cast

MODULE_INVENTORY_SCHEMA_VERSION = "tools-module-inventory/1.0.0"
AUTHORITY = "D-sorganization/Tools"
RELEASE_STATUS = "blocked-pathways-required"
CLASSIFICATIONS = frozenset({"calculation", "non-calculation"})
AUTHORITY_STATUSES = frozenset({"blocked", "not-applicable", "provisional"})
REVIEW_STATUSES = frozenset({"blocked", "inventory-baseline", "review-required"})
STATE_VALUES = frozenset(
    {
        "blocked",
        "mapped",
        "not-applicable",
        "unavailable",
        "unmapped-pending-TOOLS-D3",
        "unmapped-pending-TOOLS-D7",
    }
)
ENVELOPE_FIELDS = {
    "authority",
    "blockers",
    "entries",
    "families",
    "hash_contract",
    "producer",
    "release_status",
    "schema_version",
    "scope",
    "source_tree_sha256",
    "summary",
}
ENTRY_FIELDS = {
    "authority_status",
    "bytes_lf",
    "classification",
    "classification_basis",
    "content_sha256_lf",
    "family",
    "id",
    "language",
    "maintainer",
    "path",
    "purpose",
    "review_status",
    "risk_tags",
    "states",
    "traceability",
}
STATE_FIELDS = {
    "artifacts",
    "adrs",
    "chapters",
    "citations",
    "equation_pathway",
    "publication",
    "public_surfaces",
    "routes",
    "tests",
    "units",
    "validation",
}
TRACEABILITY_FIELDS = {
    "adr_paths",
    "artifact_sha256",
    "chapter_paths",
    "citation_refs",
    "equation_refs",
    "public_surfaces",
    "public_routes",
    "test_paths",
    "unit_mentions",
    "validation_paths",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_PATTERN = re.compile(r"^TOOLS-MODULE-[A-Z0-9-]+$")


class ToolsModuleInventoryError(RuntimeError):
    """Raised when a producer payload violates the public consumer contract."""


@dataclass(frozen=True)
class ToolsModuleInventoryView:
    """Bounded immutable summary exposed to downstream inventory consumers."""

    schema_version: str
    source_tree_sha256: str
    module_count: int
    calculation_count: int
    non_calculation_count: int
    blocked_count: int


def _object(value: object, label: str, fields: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ToolsModuleInventoryError(f"{label} must be an object")
    result = cast(dict[str, object], value)
    actual = set(result)
    if actual != fields:
        raise ToolsModuleInventoryError(
            f"{label} fields differ: missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )
    return result


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ToolsModuleInventoryError(f"{label} must be an array")
    return cast(list[object], value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolsModuleInventoryError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ToolsModuleInventoryError(f"{label} must be a non-negative integer")
    return value


def _safe_path(value: object, label: str) -> PurePosixPath:
    text = _text(value, label)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in text
        or path.as_posix() != text
    ):
        raise ToolsModuleInventoryError(f"{label} must be a normalized relative path")
    return path


def _sha256(value: object, label: str) -> str:
    text = _text(value, label)
    if SHA256_PATTERN.fullmatch(text) is None:
        raise ToolsModuleInventoryError(f"{label} must be a lowercase SHA-256")
    return text


def _sorted_unique_texts(value: object, label: str) -> tuple[str, ...]:
    items = tuple(_text(item, label) for item in _array(value, label))
    if items != tuple(sorted(set(items))):
        raise ToolsModuleInventoryError(f"{label} must be sorted and unique")
    return items


def _verify_traceability(value: object, module_id: str) -> None:
    trace = _object(value, f"{module_id} traceability", TRACEABILITY_FIELDS)
    for field in TRACEABILITY_FIELDS - {"public_surfaces"}:
        values = _sorted_unique_texts(trace[field], f"{module_id} {field}")
        if field.endswith("_paths"):
            for item in values:
                _safe_path(item, f"{module_id} {field} item")
        if field == "artifact_sha256":
            for item in values:
                _sha256(item, f"{module_id} artifact SHA-256")
    surfaces = _array(trace["public_surfaces"], f"{module_id} public surfaces")
    keys: list[str] = []
    for surface_value in surfaces:
        surface = _object(
            surface_value,
            f"{module_id} public surface",
            {"kind", "name", "signature"},
        )
        kind = _text(surface["kind"], f"{module_id} surface kind")
        name = _text(surface["name"], f"{module_id} surface name")
        signature = surface["signature"]
        if signature is not None and not isinstance(signature, str):
            raise ToolsModuleInventoryError(
                f"{module_id} surface signature must be text or null"
            )
        keys.append(f"{kind}:{name}:{signature or ''}")
    if keys != sorted(set(keys)):
        raise ToolsModuleInventoryError(
            f"{module_id} public surfaces must be sorted and unique"
        )


def _verify_states(value: object, module_id: str) -> None:
    states = _object(value, f"{module_id} states", STATE_FIELDS)
    for field, raw in states.items():
        state = _text(raw, f"{module_id} {field} state")
        if state not in STATE_VALUES:
            raise ToolsModuleInventoryError(f"{module_id} {field} state is unsupported")


def _verify_entry(value: object) -> tuple[str, str, str]:
    entry = _object(value, "module entry", ENTRY_FIELDS)
    identifier = _text(entry["id"], "module ID")
    if IDENTIFIER_PATTERN.fullmatch(identifier) is None:
        raise ToolsModuleInventoryError("module ID is invalid")
    _safe_path(entry["path"], f"{identifier} path")
    classification = _text(entry["classification"], f"{identifier} classification")
    if classification not in CLASSIFICATIONS:
        raise ToolsModuleInventoryError(f"{identifier} classification is unsupported")
    authority = _text(entry["authority_status"], f"{identifier} authority status")
    if authority not in AUTHORITY_STATUSES:
        raise ToolsModuleInventoryError(f"{identifier} authority status is unsupported")
    review = _text(entry["review_status"], f"{identifier} review status")
    if review not in REVIEW_STATUSES:
        raise ToolsModuleInventoryError(f"{identifier} review status is unsupported")
    _text(entry["classification_basis"], f"{identifier} classification basis")
    _text(entry["family"], f"{identifier} family")
    _text(entry["language"], f"{identifier} language")
    _text(entry["maintainer"], f"{identifier} maintainer")
    purpose = entry["purpose"]
    if purpose is not None and not isinstance(purpose, str):
        raise ToolsModuleInventoryError(f"{identifier} purpose must be text or null")
    _integer(entry["bytes_lf"], f"{identifier} normalized byte count")
    _sha256(entry["content_sha256_lf"], f"{identifier} content SHA-256")
    _sorted_unique_texts(entry["risk_tags"], f"{identifier} risk tags")
    _verify_states(entry["states"], identifier)
    _verify_traceability(entry["traceability"], identifier)
    return identifier, classification, authority


def _count_map(value: object, label: str) -> dict[str, int]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ToolsModuleInventoryError(f"{label} must be an object")
    return {key: _integer(item, f"{label} {key}") for key, item in value.items()}


def _verify_blockers(value: object) -> None:
    blockers = _array(value, "blockers")
    if not blockers:
        raise ToolsModuleInventoryError("blocked inventory requires blockers")
    for blocker in blockers:
        item = _object(blocker, "blocker", {"id", "owner", "resolution"})
        for field, raw in item.items():
            _text(raw, f"blocker {field}")


def _verify_families(value: object) -> None:
    identifiers: list[str] = []
    for family_value in _array(value, "families"):
        family = _object(
            family_value,
            "family",
            {"classification", "id", "maintainer", "rationale"},
        )
        identifier = _text(family["id"], "family ID")
        classification = _text(family["classification"], f"{identifier} classification")
        if classification not in CLASSIFICATIONS:
            raise ToolsModuleInventoryError(
                f"{identifier} classification is unsupported"
            )
        _text(family["maintainer"], f"{identifier} maintainer")
        _text(family["rationale"], f"{identifier} rationale")
        identifiers.append(identifier)
    if identifiers != sorted(set(identifiers)):
        raise ToolsModuleInventoryError("family IDs must be sorted and unique")


def load_inventory(payload: object) -> ToolsModuleInventoryView:
    """Validate and summarize one inventory payload without touching producer internals."""
    document = _object(payload, "module inventory", ENVELOPE_FIELDS)
    if document["schema_version"] != MODULE_INVENTORY_SCHEMA_VERSION:
        raise ToolsModuleInventoryError(
            "module inventory schema version is unsupported"
        )
    if document["authority"] != AUTHORITY:
        raise ToolsModuleInventoryError("module inventory authority is unsupported")
    if document["release_status"] != RELEASE_STATUS:
        raise ToolsModuleInventoryError(
            "module inventory release status is unsupported"
        )
    source_tree = _sha256(document["source_tree_sha256"], "source tree SHA-256")
    scope = _object(
        document["scope"], "scope", {"discovery", "exclusions", "roots", "suffixes"}
    )
    if (
        scope["discovery"]
        != "tracked-implementation-and-governed-configuration-modules"
    ):
        raise ToolsModuleInventoryError("scope discovery is unsupported")
    for field in ("exclusions", "roots", "suffixes"):
        _sorted_unique_texts(scope[field], f"scope {field}")
    hash_contract = _object(
        document["hash_contract"],
        "hash contract",
        {"algorithm", "line_endings", "tree_encoding"},
    )
    if hash_contract != {
        "algorithm": "sha256",
        "line_endings": "CRLF-and-CR-normalized-to-LF",
        "tree_encoding": "path-colon-content_sha256_lf-newline",
    }:
        raise ToolsModuleInventoryError("hash contract is unsupported")
    producer = _object(
        document["producer"], "producer", {"generator_path", "schema_path"}
    )
    if producer != {
        "generator_path": "scripts/build_tools_module_inventory.py",
        "schema_path": "manuals/tools/schemas/module-inventory.schema.json",
    }:
        raise ToolsModuleInventoryError("producer contract is unsupported")
    _verify_blockers(document["blockers"])
    _verify_families(document["families"])
    entries = _array(document["entries"], "entries")
    identifiers: list[str] = []
    classifications: list[str] = []
    authorities: list[str] = []
    for value in entries:
        identifier, classification, authority = _verify_entry(value)
        identifiers.append(identifier)
        classifications.append(classification)
        authorities.append(authority)
    if len(identifiers) != len(set(identifiers)):
        raise ToolsModuleInventoryError("duplicate module ID")
    summary = _object(
        document["summary"],
        "summary",
        {
            "authority_status_counts",
            "blocked_count",
            "calculation_count",
            "classification_counts",
            "family_counts",
            "module_count",
            "non_calculation_count",
            "provisional_count",
            "review_status_counts",
        },
    )
    module_count = _integer(summary["module_count"], "module count")
    calculation_count = _integer(summary["calculation_count"], "calculation count")
    non_calculation_count = _integer(
        summary["non_calculation_count"], "non-calculation count"
    )
    blocked_count = _integer(summary["blocked_count"], "blocked count")
    classification_counts = _count_map(
        summary["classification_counts"], "classification counts"
    )
    authority_counts = _count_map(
        summary["authority_status_counts"], "authority status counts"
    )
    _count_map(summary["family_counts"], "family counts")
    _count_map(summary["review_status_counts"], "review status counts")
    if (
        module_count != len(entries)
        or sum(classification_counts.values()) != module_count
    ):
        raise ToolsModuleInventoryError("summary module denominator differs")
    if sum(authority_counts.values()) != module_count:
        raise ToolsModuleInventoryError("summary authority denominator differs")
    if calculation_count != classifications.count("calculation"):
        raise ToolsModuleInventoryError("summary calculation count differs")
    if non_calculation_count != classifications.count("non-calculation"):
        raise ToolsModuleInventoryError("summary non-calculation count differs")
    if blocked_count != authorities.count("blocked"):
        raise ToolsModuleInventoryError("summary blocked count differs")
    return ToolsModuleInventoryView(
        schema_version=MODULE_INVENTORY_SCHEMA_VERSION,
        source_tree_sha256=source_tree,
        module_count=module_count,
        calculation_count=calculation_count,
        non_calculation_count=non_calculation_count,
        blocked_count=blocked_count,
    )
