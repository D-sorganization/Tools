"""Consumer contracts for TOOLS-D7 render, semantic, and accessibility QA."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from scripts.tools_manual_qa_contract import (
    EXPECTED_PDF_PAGES,
    QA_SCHEMA_VERSION,
    ManualQAError,
    inspect_docx_artifact,
    inspect_html_artifact,
    inspect_pdf_artifact,
    inspect_tex_artifact,
    load_qa_ledger,
    verify_manual_qa,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANUAL_ROOT = REPO_ROOT / "manuals" / "tools"
LEDGER_PATH = MANUAL_ROOT / "manual-qa.json"
SCHEMA_PATH = MANUAL_ROOT / "schemas" / "manual-qa.schema.json"
DIST_DIR = MANUAL_ROOT / "dist"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_manual_qa_schema_is_strict_and_ledger_conforms() -> None:
    schema = _json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    ledger_doc = _json(LEDGER_PATH)
    Draft202012Validator(schema).validate(ledger_doc)

    assert schema["$id"].endswith("/tools/manual-qa/1.0.0.json")
    assert schema["additionalProperties"] is False


def test_qa_ledger_loader_validates_complete_zero_sampling() -> None:
    ledger = load_qa_ledger(_json(LEDGER_PATH))

    assert ledger.schema_version == QA_SCHEMA_VERSION
    assert ledger.manual_id == "tools"
    assert ledger.owner_subepic == 4725
    assert ledger.inspection_mode == "complete-zero-sampling"
    assert ledger.sampling_rate == 1.0

    # PDF verification
    assert ledger.pdf.page_count == EXPECTED_PDF_PAGES
    assert ledger.pdf.uninspected_pages == 0
    assert len(ledger.pdf.pages) == EXPECTED_PDF_PAGES
    assert len(ledger.pdf.fonts) >= 10
    assert ledger.pdf.outline_item_count >= 50
    assert ledger.pdf.total_images == 1

    # DOCX verification
    assert ledger.docx.paragraph_count >= 150
    assert ledger.docx.heading_count >= 50
    assert ledger.docx.math_element_count >= 40
    assert ledger.docx.table_count >= 1
    assert ledger.docx.drawing_count >= 1
    assert ledger.docx.bookmark_count >= 50
    assert ledger.docx.unresolved_reference_count == 0

    # HTML verification
    assert ledger.html.has_lang is True
    assert ledger.html.has_viewport is True
    assert ledger.html.mathml_block_count >= 40
    assert ledger.html.table_count >= 1
    assert ledger.html.image_count >= 1
    assert ledger.html.images_missing_alt == 0
    assert ledger.html.unresolved_reference_count == 0

    # TeX verification
    assert ledger.tex.has_geometry is True
    assert ledger.tex.has_microtype is True
    assert ledger.tex.figure_count >= 1
    assert ledger.tex.csl_reference_block_present is True
    assert ledger.tex.unresolved_reference_count == 0

    # Cross format integrity
    assert ledger.cross_format_integrity.figure_presence_verified is True
    assert ledger.cross_format_integrity.table_presence_verified is True
    assert ledger.cross_format_integrity.math_representation_verified is True
    assert ledger.cross_format_integrity.status == "verified-semantic-parity"

    # Blockers
    assert len(ledger.blockers) >= 2


def test_inspect_pdf_artifact_extracts_all_pages_without_sampling() -> None:
    pdf_path = DIST_DIR / "tools-engineering-design-manual.pdf"
    result = inspect_pdf_artifact(pdf_path)

    assert result.page_count == 10
    assert result.uninspected_pages == 0
    assert len(result.pages) == 10
    for idx, page_record in enumerate(result.pages):
        assert page_record.page_number == idx + 1
        assert page_record.character_count > 1000
        assert page_record.line_count >= 20
        assert len(page_record.first_line_prefix) > 5


def test_inspect_docx_artifact_finds_math_headings_and_tables() -> None:
    docx_path = DIST_DIR / "tools-engineering-design-manual.docx"
    result = inspect_docx_artifact(docx_path)

    assert result.paragraph_count == 188
    assert result.heading_count == 69
    assert result.math_element_count == 52
    assert result.table_count == 1
    assert result.drawing_count == 1
    assert result.unresolved_reference_count == 0


def test_inspect_html_artifact_verifies_accessibility_attributes() -> None:
    html_path = DIST_DIR / "tools-engineering-design-manual.html"
    result = inspect_html_artifact(html_path)

    assert result.has_lang is True
    assert result.has_viewport is True
    assert result.mathml_block_count == 52
    assert result.image_count == 1
    assert result.images_with_valid_alt == 1
    assert result.images_missing_alt == 0
    assert result.unresolved_reference_count == 0


def test_inspect_tex_artifact_verifies_required_packages() -> None:
    tex_path = DIST_DIR / "tools-engineering-design-manual.tex"
    result = inspect_tex_artifact(tex_path)

    assert result.has_geometry is True
    assert result.has_microtype is True
    assert result.figure_count == 2
    assert result.csl_reference_block_present is True
    assert result.unresolved_reference_count == 0


def test_verify_manual_qa_succeeds_on_repo_root() -> None:
    ledger = verify_manual_qa(REPO_ROOT)
    assert ledger.schema_version == QA_SCHEMA_VERSION
    assert ledger.owner_subepic == 4725


@pytest.mark.parametrize(
    ("mutation", "error_message"),
    [
        (
            lambda v: v.update(schema_version="invalid/1.0.0"),
            "Unsupported QA ledger schema version",
        ),
        (lambda v: v.update(manual_id="other"), "manual_id must be 'tools'"),
        (lambda v: v.update(owner_subepic=9999), "owner_subepic must be 4725"),
        (
            lambda v: v.update(inspection_mode="sampled-50-percent"),
            "inspection_mode must be 'complete-zero-sampling'",
        ),
        (lambda v: v.update(sampling_rate=0.5), "sampling_rate must be 1.0"),
        (
            lambda v: v["inspections"]["pdf"].update(page_count=5),
            "PDF page count must be 10",
        ),
        (
            lambda v: v["inspections"]["pdf"].update(uninspected_pages=1),
            "uninspected_pages must be 0",
        ),
        (
            lambda v: v["inspections"]["docx"].update(unresolved_reference_count=2),
            "DOCX unresolved reference count must be 0",
        ),
        (
            lambda v: v["inspections"]["html"].update(images_missing_alt=1),
            "HTML images missing alt text",
        ),
        (
            lambda v: v["inspections"]["html"].update(has_lang=False),
            "HTML missing lang or viewport metadata",
        ),
        (
            lambda v: v["inspections"]["tex"].update(has_geometry=False),
            "TeX missing geometry or microtype package",
        ),
        (
            lambda v: v["inspections"]["tex"].update(csl_reference_block_present=False),
            "TeX missing CSL reference block",
        ),
        (
            lambda v: v["cross_format_integrity"].update(status="drifted"),
            "Cross-format integrity status must be 'verified-semantic-parity'",
        ),
        (lambda v: v.update(blockers=[]), "Blockers must not be empty"),
    ],
)
def test_loader_rejects_mutations_and_failures(
    mutation: Any, error_message: str
) -> None:
    payload = copy.deepcopy(_json(LEDGER_PATH))
    mutation(payload)

    with pytest.raises(ManualQAError, match=error_message):
        load_qa_ledger(payload)
