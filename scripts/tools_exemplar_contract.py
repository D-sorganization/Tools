"""Strict TOOLS-D4 exemplar coverage and cross-registry consumer contract."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from scripts.tools_textbook_chapter_contract import (
    TextbookChapterError,
    load_chapter_contract,
    load_chapter_registry,
)

COVERAGE_PATH = PurePosixPath("manuals/tools/exemplar-coverage.json")
COVERAGE_VERSION = "tools-exemplar-coverage/1.0.0"
CALCULATION_REGISTRY_PATH = PurePosixPath("manuals/tools/calculation-registry.json")
CHAPTER_CONTRACT_PATH = PurePosixPath("manuals/tools/textbook-chapter-contract.json")
CHAPTER_REGISTRY_PATH = PurePosixPath("manuals/tools/textbook-chapters.json")
MODULE_MANIFEST_PATH = PurePosixPath("manuals/tools/manifests/module-inventory.json")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class ExemplarContractError(RuntimeError):
    """Raised when exemplar evidence is incomplete, unsafe, or overclaimed."""


@dataclass(frozen=True)
class WorkedExample:
    """One golden-fixture-backed worked example."""

    example_id: str
    fixture_path: PurePosixPath
    fixture_case_id: str


@dataclass(frozen=True)
class ExemplarDescriptor:
    """One available or explicitly blocked exemplar pathway."""

    exemplar_id: str
    title: str
    status: str
    chapter_id: str | None
    calculation_ids: tuple[str, ...]
    module_ids: tuple[str, ...]
    source_paths: tuple[PurePosixPath, ...]
    test_paths: tuple[str, ...]
    consumer_paths: tuple[PurePosixPath, ...]
    worked_examples: tuple[WorkedExample, ...]
    blockers: tuple[str, ...]
    dependency_evidence: tuple[str, ...]
    owner: str
    review_owner: str
    approval_state: str


@dataclass(frozen=True)
class ExemplarCoverage:
    """Versioned repository exemplar coverage envelope."""

    schema_version: str
    release_status: str
    entries: tuple[ExemplarDescriptor, ...]


@dataclass(frozen=True)
class ExemplarRepositorySummary:
    """Deterministic evidence returned after cross-registry verification."""

    verified_exemplar_count: int
    blocked_exemplar_count: int
    calculation_ids: tuple[str, ...]
    chapter_ids: tuple[str, ...]
    worked_example_ids: tuple[str, ...]


def _object(value: object, label: str, fields: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ExemplarContractError(f"{label} must be an object")
    document = cast(dict[str, object], value)
    actual = set(document)
    if actual != fields:
        raise ExemplarContractError(
            f"{label} fields differ: missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )
    return document


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ExemplarContractError(f"{label} must be an array")
    return cast(list[object], value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ExemplarContractError(f"{label} must be a trimmed non-empty string")
    return value


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _texts(value: object, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    result = tuple(_text(item, label) for item in _array(value, label))
    if not allow_empty and not result:
        raise ExemplarContractError(f"{label} must not be empty")
    if len(set(result)) != len(result):
        raise ExemplarContractError(f"{label} must contain unique values")
    if result != tuple(sorted(result)):
        raise ExemplarContractError(f"{label} must be sorted")
    return result


def _safe_path(value: object, label: str) -> PurePosixPath:
    text = _text(value, label)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in text
        or text != path.as_posix()
    ):
        raise ExemplarContractError(f"{label} must be a normalized relative path")
    return path


def _paths(
    value: object, label: str, *, allow_empty: bool = False
) -> tuple[PurePosixPath, ...]:
    result = tuple(_safe_path(item, label) for item in _array(value, label))
    if not allow_empty and not result:
        raise ExemplarContractError(f"{label} must not be empty")
    if len(set(result)) != len(result) or result != tuple(sorted(result)):
        raise ExemplarContractError(f"{label} must be sorted and unique")
    return result


def _worked_example(value: object) -> WorkedExample:
    item = _object(
        value,
        "worked example",
        {"example_id", "fixture_path", "fixture_case_id"},
    )
    return WorkedExample(
        example_id=_text(item["example_id"], "worked example ID"),
        fixture_path=_safe_path(item["fixture_path"], "worked example fixture"),
        fixture_case_id=_text(item["fixture_case_id"], "fixture case ID"),
    )


def _descriptor(value: object) -> ExemplarDescriptor:
    item = _object(
        value,
        "exemplar descriptor",
        {
            "exemplar_id",
            "title",
            "status",
            "chapter_id",
            "calculation_ids",
            "module_ids",
            "source_paths",
            "test_paths",
            "consumer_paths",
            "worked_examples",
            "blockers",
            "dependency_evidence",
            "owner",
            "review_owner",
            "approval_state",
        },
    )
    status = _text(item["status"], "exemplar status")
    descriptor = ExemplarDescriptor(
        exemplar_id=_text(item["exemplar_id"], "exemplar ID"),
        title=_text(item["title"], "exemplar title"),
        status=status,
        chapter_id=_optional_text(item["chapter_id"], "chapter ID"),
        calculation_ids=_texts(
            item["calculation_ids"], "calculation IDs", allow_empty=True
        ),
        module_ids=_texts(item["module_ids"], "module IDs", allow_empty=True),
        source_paths=_paths(item["source_paths"], "source paths", allow_empty=True),
        test_paths=_texts(item["test_paths"], "test paths", allow_empty=True),
        consumer_paths=_paths(
            item["consumer_paths"], "consumer paths", allow_empty=True
        ),
        worked_examples=tuple(
            _worked_example(example)
            for example in _array(item["worked_examples"], "worked examples")
        ),
        blockers=_texts(item["blockers"], "blockers", allow_empty=True),
        dependency_evidence=_texts(item["dependency_evidence"], "dependency evidence"),
        owner=_text(item["owner"], "owner"),
        review_owner=_text(item["review_owner"], "review owner"),
        approval_state=_text(item["approval_state"], "approval state"),
    )
    evidence = (
        descriptor.chapter_id is not None,
        bool(descriptor.calculation_ids),
        bool(descriptor.module_ids),
        bool(descriptor.source_paths),
        bool(descriptor.test_paths),
        bool(descriptor.consumer_paths),
        bool(descriptor.worked_examples),
    )
    if status == "verified-unapproved":
        if not all(evidence) or descriptor.blockers:
            raise ExemplarContractError(
                "verified exemplar requires complete evidence and no blockers"
            )
    elif status == "blocked":
        if any(evidence) or not descriptor.blockers:
            raise ExemplarContractError(
                "blocked exemplar must expose no chapter or calculation evidence"
            )
    else:
        raise ExemplarContractError("exemplar status is unsupported")
    if descriptor.approval_state != "blocked-pending-human-approval":
        raise ExemplarContractError("approval state must remain blocked")
    return descriptor


def load_exemplar_coverage(value: object) -> ExemplarCoverage:
    """Load the exact D4 coverage contract or fail deterministically."""
    document = _object(
        value,
        "exemplar coverage",
        {"schema_version", "manual_id", "release_status", "owner_subepic", "entries"},
    )
    version = _text(document["schema_version"], "coverage schema version")
    if version != COVERAGE_VERSION:
        raise ExemplarContractError(
            f"coverage schema version must be {COVERAGE_VERSION}"
        )
    if _text(document["manual_id"], "manual ID") != "tools":
        raise ExemplarContractError("manual ID must be tools")
    if document["owner_subepic"] != 4720:
        raise ExemplarContractError("owner subepic must be 4720")
    release_status = _text(document["release_status"], "release status")
    if release_status != "provisional":
        raise ExemplarContractError("release status must remain provisional")
    entries = tuple(
        _descriptor(item) for item in _array(document["entries"], "entries")
    )
    identifiers = tuple(item.exemplar_id for item in entries)
    if not entries or len(set(identifiers)) != len(identifiers):
        raise ExemplarContractError("entries must have unique IDs")
    if identifiers != tuple(sorted(identifiers)):
        raise ExemplarContractError("entries must be sorted by exemplar ID")
    return ExemplarCoverage(version, release_status, entries)


def _load(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExemplarContractError(
            f"governed JSON cannot be loaded: {path}"
        ) from error


def _calculation_ids(value: object) -> tuple[str, ...]:
    registry = _object(
        value,
        "calculation registry",
        {
            "schema_version",
            "manual_id",
            "repository",
            "release_status",
            "inventory_commit",
            "blockers",
            "calculations",
        },
    )
    if (
        registry["schema_version"] != "1.0.0"
        or registry["release_status"] != "provisional"
    ):
        raise ExemplarContractError(
            "calculation registry must remain provisional at schema 1.0.0"
        )
    revision = registry["inventory_commit"]
    if not isinstance(revision, str) or not REVISION_PATTERN.fullmatch(revision):
        raise ExemplarContractError("calculation inventory commit must be immutable")
    ids: list[str] = []
    required = {
        "schema_version",
        "calculation_id",
        "title",
        "owner",
        "method_version",
        "implementation",
        "status",
        "inputs",
        "outputs",
        "contracts",
        "equations",
        "numerical_method",
        "sources",
        "tests",
        "manual",
        "limitations",
        "approval",
    }
    for raw in _array(registry["calculations"], "calculations"):
        item = _object(raw, "calculation", required)
        calculation_id = _text(item["calculation_id"], "calculation ID")
        approval = _object(
            item["approval"],
            "calculation approval",
            {"state", "required_reviewers", "blocking_items"},
        )
        if approval["state"] == "approved":
            raise ExemplarContractError("D4 calculation cannot claim approval")
        implementation = _object(
            item["implementation"], "implementation", {"symbols", "commit"}
        )
        if not isinstance(
            implementation["commit"], str
        ) or not REVISION_PATTERN.fullmatch(implementation["commit"]):
            raise ExemplarContractError("implementation commit must be immutable")
        ids.append(calculation_id)
    if len(set(ids)) != len(ids) or ids != sorted(ids):
        raise ExemplarContractError("calculation IDs must be sorted and unique")
    return tuple(ids)


def _module_ids(root: Path) -> set[str]:
    manifest = _object(
        _load(root.joinpath(*MODULE_MANIFEST_PATH.parts)),
        "module manifest",
        {
            "authority",
            "blockers",
            "hash_contract",
            "producer",
            "release_status",
            "schema_version",
            "scope",
            "shards",
        },
    )
    identifiers: set[str] = set()
    for shard in _array(manifest["shards"], "module shards"):
        item = cast(dict[str, object], shard)
        path = _safe_path(item.get("path"), "module shard path")
        document = cast(dict[str, object], _load(root.joinpath(*path.parts)))
        for entry in _array(document.get("entries"), "module entries"):
            record = cast(dict[str, object], entry)
            identifiers.add(_text(record.get("id"), "module ID"))
    return identifiers


def _verify_example(root: Path, example: WorkedExample) -> None:
    path = root.joinpath(*example.fixture_path.parts)
    fixture = _object(
        _load(path),
        "worked example fixture",
        {"schema_version", "frame_id", "tolerance_deg", "cases"},
    )
    cases = [
        case
        for case in _array(fixture["cases"], "fixture cases")
        if isinstance(case, dict) and case.get("id") == example.fixture_case_id
    ]
    if len(cases) != 1:
        raise ExemplarContractError(
            f"worked example fixture case must resolve exactly once: {example.example_id}"
        )


def verify_exemplar_traceability(
    repository_root: Path,
    coverage: ExemplarCoverage,
) -> ExemplarRepositorySummary:
    """Verify a loaded coverage contract against all repository authorities."""
    root = repository_root.resolve()
    calculation_ids = set(
        _calculation_ids(_load(root.joinpath(*CALCULATION_REGISTRY_PATH.parts)))
    )
    try:
        contract = load_chapter_contract(
            _load(root.joinpath(*CHAPTER_CONTRACT_PATH.parts))
        )
        chapters = load_chapter_registry(
            _load(root.joinpath(*CHAPTER_REGISTRY_PATH.parts)), contract
        )
    except TextbookChapterError as error:
        raise ExemplarContractError(str(error)) from error
    chapter_ids = {chapter.chapter_id for chapter in chapters.chapters}
    module_ids = _module_ids(root)
    verified: list[ExemplarDescriptor] = []
    worked_examples: list[str] = []
    for descriptor in coverage.entries:
        if descriptor.status == "blocked":
            continue
        verified.append(descriptor)
        if descriptor.chapter_id not in chapter_ids:
            raise ExemplarContractError("registered exemplar chapter is missing")
        if not set(descriptor.calculation_ids).issubset(calculation_ids):
            raise ExemplarContractError("registered exemplar calculation is missing")
        if not set(descriptor.module_ids).issubset(module_ids):
            raise ExemplarContractError("registered exemplar module is missing")
        targets = [*descriptor.source_paths, *descriptor.consumer_paths]
        targets.extend(
            PurePosixPath(item.split("::", 1)[0]) for item in descriptor.test_paths
        )
        for target in targets:
            if not root.joinpath(*target.parts).is_file():
                raise ExemplarContractError(
                    f"traceability target is missing: {target.as_posix()}"
                )
        for example in descriptor.worked_examples:
            _verify_example(root, example)
            worked_examples.append(example.example_id)
    return ExemplarRepositorySummary(
        verified_exemplar_count=len(verified),
        blocked_exemplar_count=len(coverage.entries) - len(verified),
        calculation_ids=tuple(
            sorted({item for entry in verified for item in entry.calculation_ids})
        ),
        chapter_ids=tuple(sorted(cast(str, entry.chapter_id) for entry in verified)),
        worked_example_ids=tuple(sorted(worked_examples)),
    )


def verify_exemplar_repository(repository_root: Path) -> ExemplarRepositorySummary:
    """Load D4 coverage and verify every cross-registry traceability link."""
    root = repository_root.resolve()
    coverage = load_exemplar_coverage(_load(root.joinpath(*COVERAGE_PATH.parts)))
    return verify_exemplar_traceability(root, coverage)
