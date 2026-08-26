"""Consumer contracts for required Tools textbook chapters."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from scripts.tools_textbook_chapter_contract import (
    TextbookChapterError,
    load_chapter_contract,
    load_chapter_registry,
)
from scripts.tools_textbook_chapter_lint import (
    verify_chapter_source,
    verify_textbook_chapters,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "manuals" / "tools" / "textbook-chapter-contract.json"
REGISTRY_PATH = REPO_ROOT / "manuals" / "tools" / "textbook-chapters.json"
CONTRACT_SCHEMA_PATH = (
    REPO_ROOT
    / "manuals"
    / "tools"
    / "schemas"
    / "textbook-chapter-contract.schema.json"
)
REGISTRY_SCHEMA_PATH = (
    REPO_ROOT
    / "manuals"
    / "tools"
    / "schemas"
    / "textbook-chapter-registry.schema.json"
)


def _json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _descriptor() -> dict[str, Any]:
    return {
        "chapter_id": "tools.textbook.fixture",
        "path": "manuals/tools/chapters/fixture.qmd",
        "title": "Fixture Calculation",
        "chapter_version": "1.0.0",
        "authority_status": "provisional",
        "owner": "Fixture module maintainer",
        "review_owner": "TOOLS-D7 reviewer",
        "calculation_ids": ["tools.calculation.fixture"],
        "module_ids": ["tools.module.fixture"],
        "verification_tests": ["tests/test_fixture.py::test_fixture"],
        "source_citations": ["fixture-source-2026"],
        "limitations": ["Fixture evidence is not engineering approval."],
        "revision_history": [
            {
                "version": "1.0.0",
                "date": "2026-08-26",
                "summary": "Initial governed fixture.",
            }
        ],
    }


def _valid_source() -> str:
    contract = _json(CONTRACT_PATH)
    lines = ["# Fixture Calculation", ""]
    for section in contract["required_sections"]:
        lines.extend(
            [f"## {section['heading']}", "Governed evidence for this section.", ""]
        )
        for heading in section["required_subheadings"]:
            lines.extend([f"### {heading}", "Governed subsection evidence.", ""])
    return "\n".join(lines)


def test_owned_schemas_are_strict_and_current_documents_conform() -> None:
    pairs = (
        (CONTRACT_SCHEMA_PATH, CONTRACT_PATH),
        (REGISTRY_SCHEMA_PATH, REGISTRY_PATH),
    )
    for schema_path, document_path in pairs:
        schema = _json(schema_path)
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(_json(document_path))
        assert schema["additionalProperties"] is False


def test_current_contract_and_registry_are_versioned_and_fail_closed() -> None:
    contract = load_chapter_contract(_json(CONTRACT_PATH))
    registry = load_chapter_registry(_json(REGISTRY_PATH), contract)

    assert contract.schema_version == "tools-textbook-chapter-contract/1.0.0"
    assert len(contract.required_sections) == 14
    assert registry.schema_version == "tools-textbook-chapter-registry/1.0.0"
    assert registry.release_status == "provisional"
    assert tuple(chapter.chapter_id for chapter in registry.chapters) == (
        "tools.textbook.swing-rate-of-closure-dplane",
    )
    assert registry.blockers[0].startswith("Markerless mocap")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["required_sections"][0].update(
            heading="Purpose and Scope Drift"
        ),
        lambda payload: payload["required_sections"][1][
            "required_subheadings"
        ].reverse(),
    ],
)
def test_contract_loader_rejects_shape_drift_without_a_version_change(
    mutation: Any,
) -> None:
    payload = _json(CONTRACT_PATH)
    mutation(payload)

    with pytest.raises(TextbookChapterError, match="required section"):
        load_chapter_contract(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(extra=True), "fields differ"),
        (
            lambda payload: payload.update(
                schema_version="tools-textbook-chapter-registry/9.0.0"
            ),
            "schema version",
        ),
        (
            lambda payload: payload["chapters"][0].update(path="../outside.qmd"),
            "normalized relative path",
        ),
        (
            lambda payload: payload["chapters"].append(
                copy.deepcopy(payload["chapters"][0])
            ),
            "duplicate chapter ID",
        ),
        (
            lambda payload: payload["chapters"][0].update(authority_status="approved"),
            "approved chapter",
        ),
    ],
)
def test_registry_loader_rejects_unsafe_or_unapproved_claims(
    mutation: Any,
    message: str,
) -> None:
    contract = load_chapter_contract(_json(CONTRACT_PATH))
    payload = _json(REGISTRY_PATH)
    payload["chapters"] = [_descriptor()]
    mutation(payload)

    with pytest.raises(TextbookChapterError, match=message):
        load_chapter_registry(payload, contract)


@pytest.mark.parametrize("section_index", range(14))
def test_linter_rejects_every_missing_required_section(
    tmp_path: Path,
    section_index: int,
) -> None:
    contract = load_chapter_contract(_json(CONTRACT_PATH))
    descriptor_payload = _descriptor()
    registry_payload = _json(REGISTRY_PATH)
    registry_payload["chapters"] = [descriptor_payload]
    registry = load_chapter_registry(registry_payload, contract)
    source = _valid_source().splitlines()
    missing_heading = f"## {contract.required_sections[section_index].heading}"
    start = source.index(missing_heading)
    end = next(
        (
            index
            for index in range(start + 1, len(source))
            if source[index].startswith("## ")
        ),
        len(source),
    )
    del source[start:end]
    chapter_path = tmp_path / descriptor_payload["path"]
    chapter_path.parent.mkdir(parents=True)
    chapter_path.write_text("\n".join(source), encoding="utf-8")

    with pytest.raises(TextbookChapterError, match="required section headings"):
        verify_chapter_source(tmp_path, registry.chapters[0], contract)


def test_linter_rejects_order_placeholders_and_missing_subheadings(
    tmp_path: Path,
) -> None:
    contract = load_chapter_contract(_json(CONTRACT_PATH))
    registry_payload = _json(REGISTRY_PATH)
    registry_payload["chapters"] = [_descriptor()]
    descriptor = load_chapter_registry(registry_payload, contract).chapters[0]
    chapter_path = tmp_path / descriptor.path
    chapter_path.parent.mkdir(parents=True)

    headings = [section.heading for section in contract.required_sections]
    reordered = _valid_source().replace(headings[0], "SWAP", 1)
    reordered = reordered.replace(headings[1], headings[0], 1).replace(
        "SWAP", headings[1], 1
    )
    chapter_path.write_text(reordered, encoding="utf-8")
    with pytest.raises(TextbookChapterError, match="required section headings"):
        verify_chapter_source(tmp_path, descriptor, contract)

    chapter_path.write_text(
        _valid_source().replace("Governed evidence", "TODO", 1), encoding="utf-8"
    )
    with pytest.raises(TextbookChapterError, match="forbidden token"):
        verify_chapter_source(tmp_path, descriptor, contract)

    required_subheading = contract.required_sections[1].required_subheadings[0]
    chapter_path.write_text(
        _valid_source().replace(f"### {required_subheading}\n", "", 1),
        encoding="utf-8",
    )
    with pytest.raises(TextbookChapterError, match="required subheadings"):
        verify_chapter_source(tmp_path, descriptor, contract)


def test_linter_reports_missing_source_through_typed_contract(tmp_path: Path) -> None:
    contract = load_chapter_contract(_json(CONTRACT_PATH))
    registry_payload = _json(REGISTRY_PATH)
    registry_payload["chapters"] = [_descriptor()]
    descriptor = load_chapter_registry(registry_payload, contract).chapters[0]

    with pytest.raises(TextbookChapterError, match="chapter source cannot be read"):
        verify_chapter_source(tmp_path, descriptor, contract)


def test_registry_rejects_nonchronological_revision_history() -> None:
    contract = load_chapter_contract(_json(CONTRACT_PATH))
    registry_payload = _json(REGISTRY_PATH)
    descriptor = _descriptor()
    descriptor["chapter_version"] = "2.0.0"
    descriptor["revision_history"] = [
        {"version": "1.0.0", "date": "2026-08-27", "summary": "Later row."},
        {"version": "2.0.0", "date": "2026-08-26", "summary": "Earlier row."},
    ]
    registry_payload["chapters"] = [descriptor]

    with pytest.raises(TextbookChapterError, match="chronological"):
        load_chapter_registry(registry_payload, contract)


def test_valid_chapter_and_repository_registry_are_deterministic(
    tmp_path: Path,
) -> None:
    contract = load_chapter_contract(_json(CONTRACT_PATH))
    registry_payload = _json(REGISTRY_PATH)
    registry_payload["chapters"] = [_descriptor()]
    registry = load_chapter_registry(registry_payload, contract)
    chapter_path = tmp_path / registry.chapters[0].path
    chapter_path.parent.mkdir(parents=True)
    chapter_path.write_text(_valid_source(), encoding="utf-8")

    evidence = verify_chapter_source(tmp_path, registry.chapters[0], contract)
    summary = verify_textbook_chapters(REPO_ROOT)

    assert evidence.chapter_id == "tools.textbook.fixture"
    assert len(evidence.section_sha256_lf) == 14
    assert summary.chapter_count == 1
    assert summary.release_status == "provisional"
