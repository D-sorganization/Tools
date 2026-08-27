"""Deterministic QMD content linter for registered textbook chapters."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from scripts.tools_textbook_chapter_contract import (
    CONTRACT_PATH,
    REGISTRY_PATH,
    TextbookChapterContract,
    TextbookChapterDescriptor,
    TextbookChapterError,
    load_chapter_contract,
    load_chapter_registry,
)

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class TextbookChapterEvidence:
    """Deterministic content evidence returned after a chapter passes lint."""

    chapter_id: str
    source_sha256_lf: str
    section_sha256_lf: tuple[str, ...]


@dataclass(frozen=True)
class TextbookChapterSummary:
    """Repository-wide result returned by the chapter governance gate."""

    chapter_count: int
    release_status: str
    evidence: tuple[TextbookChapterEvidence, ...]


def _sha256_lf(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _section_blocks(text: str) -> tuple[list[str], list[str]]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("## ")]
    headings = [lines[index][3:].strip() for index in starts]
    blocks = [
        "\n".join(
            lines[
                start : starts[offset + 1] if offset + 1 < len(starts) else len(lines)
            ]
        )
        for offset, start in enumerate(starts)
    ]
    return headings, blocks


def verify_chapter_source(
    repository_root: Path,
    descriptor: TextbookChapterDescriptor,
    contract: TextbookChapterContract,
) -> TextbookChapterEvidence:
    """Verify ordered chapter content and return LF-normalized evidence hashes."""
    path = repository_root.joinpath(*descriptor.path.parts)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise TextbookChapterError(
            f"chapter source is not UTF-8: {descriptor.path}"
        ) from error
    except OSError as error:
        raise TextbookChapterError(
            f"chapter source cannot be read: {descriptor.path}: {error.strerror}"
        ) from error
    for token in contract.forbidden_tokens:
        if token.casefold() in text.casefold():
            raise TextbookChapterError(f"chapter contains forbidden token: {token}")
    headings, blocks = _section_blocks(text)
    expected = [section.heading for section in contract.required_sections]
    if headings != expected:
        raise TextbookChapterError(
            f"required section headings differ: expected={expected}, actual={headings}"
        )
    section_hashes: list[str] = []
    for section, block in zip(contract.required_sections, blocks, strict=True):
        matches = [HEADING_PATTERN.match(line) for line in block.splitlines()]
        subheadings = [
            match.group(2)
            for match in matches
            if match is not None and match.group(1) == "###"
        ]
        if subheadings != list(section.required_subheadings):
            raise TextbookChapterError(
                f"required subheadings differ for section {section.section_id}"
            )
        body = " ".join(
            line.strip() for line in block.splitlines()[1:] if not line.startswith("#")
        )
        if len(body) < contract.minimum_section_characters:
            raise TextbookChapterError(
                f"section {section.section_id} has insufficient governed content"
            )
        section_hashes.append(_sha256_lf(block))
    return TextbookChapterEvidence(
        chapter_id=descriptor.chapter_id,
        source_sha256_lf=_sha256_lf(text),
        section_sha256_lf=tuple(section_hashes),
    )


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_textbook_chapters(repository_root: Path) -> TextbookChapterSummary:
    """Verify the repository-owned contract, registry, and every registered QMD."""
    root = repository_root.resolve()
    contract = load_chapter_contract(_load_json(root.joinpath(*CONTRACT_PATH.parts)))
    registry = load_chapter_registry(
        _load_json(root.joinpath(*REGISTRY_PATH.parts)),
        contract,
    )
    evidence = tuple(
        verify_chapter_source(root, descriptor, contract)
        for descriptor in registry.chapters
    )
    return TextbookChapterSummary(
        chapter_count=len(registry.chapters),
        release_status=registry.release_status,
        evidence=evidence,
    )
