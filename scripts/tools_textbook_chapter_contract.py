"""Strict consumer and content contracts for governed textbook chapters."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import cast

CONTRACT_PATH = PurePosixPath("manuals/tools/textbook-chapter-contract.json")
REGISTRY_PATH = PurePosixPath("manuals/tools/textbook-chapters.json")
CONTRACT_VERSION = "tools-textbook-chapter-contract/1.0.0"
REGISTRY_VERSION = "tools-textbook-chapter-registry/1.0.0"
CHAPTER_PATH_ROOT = PurePosixPath("manuals/tools/chapters")
SEMVER_PATTERN = re.compile(r"^[1-9]\d*\.\d+\.\d+$|^0\.\d+\.\d+$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
REQUIRED_SECTIONS_SHA256 = (
    # Public deterministic contract digest, not a credential.
    "e03dd54ba296e458a0b4a5562358b77ee2073a1d9cc98aa78195d62c198aac80"  # pragma: allowlist secret
)


class TextbookChapterError(RuntimeError):
    """Raised when a chapter contract, registry, or source fails closed."""


@dataclass(frozen=True)
class RequiredSection:
    """One ordered section required in every registered textbook chapter."""

    section_id: str
    heading: str
    purpose: str
    required_subheadings: tuple[str, ...]


@dataclass(frozen=True)
class TextbookChapterContract:
    """Versioned structural requirements applied to registered chapters."""

    schema_version: str
    manual_id: str
    owner_subepic: int
    status: str
    minimum_section_characters: int
    forbidden_tokens: tuple[str, ...]
    required_sections: tuple[RequiredSection, ...]


@dataclass(frozen=True)
class RevisionRecord:
    """One immutable version-history row declared by a chapter descriptor."""

    version: str
    date: str
    summary: str


@dataclass(frozen=True)
class TextbookChapterDescriptor:
    """Machine-readable owner and traceability boundary for one QMD chapter."""

    chapter_id: str
    path: PurePosixPath
    title: str
    chapter_version: str
    authority_status: str
    owner: str
    review_owner: str
    calculation_ids: tuple[str, ...]
    module_ids: tuple[str, ...]
    verification_tests: tuple[str, ...]
    source_citations: tuple[str, ...]
    limitations: tuple[str, ...]
    revision_history: tuple[RevisionRecord, ...]


@dataclass(frozen=True)
class TextbookChapterRegistry:
    """Strict registry envelope for all chapters governed by this contract."""

    schema_version: str
    manual_id: str
    contract_schema_version: str
    release_status: str
    owner: str
    review_owner: str
    blockers: tuple[str, ...]
    chapters: tuple[TextbookChapterDescriptor, ...]


def _object(value: object, label: str, fields: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TextbookChapterError(f"{label} must be an object")
    document = cast(dict[str, object], value)
    actual = set(document)
    if actual != fields:
        raise TextbookChapterError(
            f"{label} fields differ: missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )
    return document


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise TextbookChapterError(f"{label} must be an array")
    return cast(list[object], value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise TextbookChapterError(f"{label} must be a trimmed non-empty string")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise TextbookChapterError(f"{label} must be an integer >= {minimum}")
    return value


def _safe_chapter_path(value: object) -> PurePosixPath:
    text = _text(value, "chapter path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in text
        or path.as_posix() != text
        or path.parent != CHAPTER_PATH_ROOT
        or path.suffix != ".qmd"
    ):
        raise TextbookChapterError(
            "chapter path must be a normalized relative path directly under "
            "manuals/tools/chapters"
        )
    return path


def _unique_texts(
    value: object, label: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    texts = tuple(_text(item, label) for item in _array(value, label))
    if not allow_empty and not texts:
        raise TextbookChapterError(f"{label} must not be empty")
    if len(set(texts)) != len(texts):
        raise TextbookChapterError(f"{label} must contain unique values")
    if texts != tuple(sorted(texts)):
        raise TextbookChapterError(f"{label} must be sorted")
    return texts


def _load_section(value: object) -> RequiredSection:
    item = _object(
        value,
        "required section",
        {"id", "heading", "purpose", "required_subheadings"},
    )
    return RequiredSection(
        section_id=_text(item["id"], "section ID"),
        heading=_text(item["heading"], "section heading"),
        purpose=_text(item["purpose"], "section purpose"),
        required_subheadings=tuple(
            _text(entry, "required subheading")
            for entry in _array(item["required_subheadings"], "required subheadings")
        ),
    )


def load_chapter_contract(value: object) -> TextbookChapterContract:
    """Load the exact D3 contract or raise a deterministic typed failure."""
    document = _object(
        value,
        "textbook chapter contract",
        {
            "schema_version",
            "manual_id",
            "owner_subepic",
            "status",
            "minimum_section_characters",
            "forbidden_tokens",
            "required_sections",
        },
    )
    version = _text(document["schema_version"], "contract schema version")
    if version != CONTRACT_VERSION:
        raise TextbookChapterError(
            f"contract schema version must be {CONTRACT_VERSION}"
        )
    manual_id = _text(document["manual_id"], "manual ID")
    if manual_id != "tools":
        raise TextbookChapterError("manual ID must be tools")
    owner = _integer(document["owner_subepic"], "owner subepic", 1)
    if owner != 4717:
        raise TextbookChapterError("owner subepic must be 4717")
    status = _text(document["status"], "contract status")
    if status != "qualified-generated-unapproved":
        raise TextbookChapterError("contract status must remain generated-unapproved")
    sections = tuple(
        _load_section(item)
        for item in _array(document["required_sections"], "required sections")
    )
    section_shape = [
        {
            "heading": section.heading,
            "id": section.section_id,
            "purpose": section.purpose,
            "required_subheadings": list(section.required_subheadings),
        }
        for section in sections
    ]
    section_digest = hashlib.sha256(
        json.dumps(
            section_shape,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if section_digest != REQUIRED_SECTIONS_SHA256:
        raise TextbookChapterError("contract required sections differ")
    forbidden = _unique_texts(document["forbidden_tokens"], "forbidden tokens")
    return TextbookChapterContract(
        schema_version=version,
        manual_id=manual_id,
        owner_subepic=owner,
        status=status,
        minimum_section_characters=_integer(
            document["minimum_section_characters"],
            "minimum section characters",
            20,
        ),
        forbidden_tokens=forbidden,
        required_sections=sections,
    )


def _revision(value: object) -> RevisionRecord:
    item = _object(value, "revision record", {"version", "date", "summary"})
    version = _text(item["version"], "revision version")
    date = _text(item["date"], "revision date")
    if not SEMVER_PATTERN.fullmatch(version):
        raise TextbookChapterError("revision version must be semantic versioning")
    if not DATE_PATTERN.fullmatch(date):
        raise TextbookChapterError("revision date must use YYYY-MM-DD")
    return RevisionRecord(version, date, _text(item["summary"], "revision summary"))


def _descriptor(value: object) -> TextbookChapterDescriptor:
    item = _object(
        value,
        "chapter descriptor",
        {
            "chapter_id",
            "path",
            "title",
            "chapter_version",
            "authority_status",
            "owner",
            "review_owner",
            "calculation_ids",
            "module_ids",
            "verification_tests",
            "source_citations",
            "limitations",
            "revision_history",
        },
    )
    status = _text(item["authority_status"], "chapter authority status")
    if status == "approved":
        raise TextbookChapterError("approved chapter requires later approval evidence")
    if status not in {"provisional", "verified-unapproved"}:
        raise TextbookChapterError("chapter authority status is unsupported")
    chapter_version = _text(item["chapter_version"], "chapter version")
    if not SEMVER_PATTERN.fullmatch(chapter_version):
        raise TextbookChapterError("chapter version must use semantic versioning")
    revisions = tuple(
        _revision(entry)
        for entry in _array(item["revision_history"], "revision history")
    )
    if not revisions or revisions[-1].version != chapter_version:
        raise TextbookChapterError("revision history must end at the chapter version")
    revision_keys = tuple((revision.date, revision.version) for revision in revisions)
    if len(set(revision_keys)) != len(revision_keys):
        raise TextbookChapterError("revision history must contain unique records")
    if revision_keys != tuple(sorted(revision_keys)):
        raise TextbookChapterError("revision history must be chronological")
    return TextbookChapterDescriptor(
        chapter_id=_text(item["chapter_id"], "chapter ID"),
        path=_safe_chapter_path(item["path"]),
        title=_text(item["title"], "chapter title"),
        chapter_version=chapter_version,
        authority_status=status,
        owner=_text(item["owner"], "chapter owner"),
        review_owner=_text(item["review_owner"], "chapter review owner"),
        calculation_ids=_unique_texts(item["calculation_ids"], "calculation IDs"),
        module_ids=_unique_texts(item["module_ids"], "module IDs"),
        verification_tests=_unique_texts(
            item["verification_tests"], "verification tests"
        ),
        source_citations=_unique_texts(item["source_citations"], "source citations"),
        limitations=_unique_texts(item["limitations"], "limitations"),
        revision_history=revisions,
    )


def load_chapter_registry(
    value: object,
    contract: TextbookChapterContract,
) -> TextbookChapterRegistry:
    """Load the registry through the contract's public, fail-closed seam."""
    document = _object(
        value,
        "textbook chapter registry",
        {
            "schema_version",
            "manual_id",
            "contract_schema_version",
            "release_status",
            "owner",
            "review_owner",
            "blockers",
            "chapters",
        },
    )
    version = _text(document["schema_version"], "registry schema version")
    if version != REGISTRY_VERSION:
        raise TextbookChapterError(
            f"registry schema version must be {REGISTRY_VERSION}"
        )
    manual_id = _text(document["manual_id"], "registry manual ID")
    if manual_id != contract.manual_id:
        raise TextbookChapterError("registry manual ID must match the contract")
    contract_version = _text(document["contract_schema_version"], "contract version")
    if contract_version != contract.schema_version:
        raise TextbookChapterError("registry contract version must match the contract")
    status = _text(document["release_status"], "registry release status")
    if status != "provisional":
        raise TextbookChapterError("registry release status must remain provisional")
    blockers = _unique_texts(document["blockers"], "registry blockers")
    chapters = tuple(
        _descriptor(item) for item in _array(document["chapters"], "chapters")
    )
    chapter_ids = tuple(chapter.chapter_id for chapter in chapters)
    paths = tuple(chapter.path for chapter in chapters)
    if len(set(chapter_ids)) != len(chapter_ids):
        raise TextbookChapterError("duplicate chapter ID")
    if len(set(paths)) != len(paths):
        raise TextbookChapterError("duplicate chapter path")
    if chapter_ids != tuple(sorted(chapter_ids)):
        raise TextbookChapterError("chapters must be sorted by chapter ID")
    return TextbookChapterRegistry(
        schema_version=version,
        manual_id=manual_id,
        contract_schema_version=contract_version,
        release_status=status,
        owner=_text(document["owner"], "registry owner"),
        review_owner=_text(document["review_owner"], "registry review owner"),
        blockers=blockers,
        chapters=chapters,
    )
