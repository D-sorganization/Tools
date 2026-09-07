"""Consumer contracts for TOOLS-D8 immutable public publication projection."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from scripts.tools_publication_projection_contract import (
    PROJECTION_SCHEMA_VERSION,
    PublicationProjectionError,
    load_publication_projection,
    verify_publication_projection,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANUAL_ROOT = REPO_ROOT / "manuals" / "tools"
MANIFEST_PATH = MANUAL_ROOT / "publication-projection.json"
SCHEMA_PATH = MANUAL_ROOT / "schemas" / "publication-projection.schema.json"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_publication_projection_schema_is_strict_and_manifest_conforms() -> None:
    schema = _json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    manifest_doc = _json(MANIFEST_PATH)
    Draft202012Validator(schema).validate(manifest_doc)

    assert schema["$id"].endswith("/tools/publication-projection/1.0.0.json")
    assert schema["additionalProperties"] is False


def test_publication_projection_loader_validates_evidence() -> None:
    ledger = load_publication_projection(_json(MANIFEST_PATH))

    assert ledger.schema_version == PROJECTION_SCHEMA_VERSION
    assert ledger.manual_id == "tools"
    assert ledger.owner_subepic == 4728
    assert ledger.authority_repository == "D-sorganization/Tools"
    assert ledger.catalog_repository == "D-sorganization/Engineering-Design-Manuals"
    assert ledger.repository_license == "MIT"
    assert ledger.private_content_allowed is False
    assert ledger.release_status == "unapproved-projection-verified"

    # Evidence verification
    assert len(ledger.evidence.immutable_source_commit) == 40
    assert len(ledger.evidence.source_tree_sha256) == 40
    assert len(ledger.evidence.calculation_registry_sha256) == 64
    assert len(ledger.evidence.toolchain_lock_sha256) == 64
    assert ledger.evidence.semantic_parity == "verified-semantic-parity"

    # Artifacts
    for fmt in ("docx", "html", "pdf", "tex"):
        assert fmt in ledger.evidence.artifact_sha256
        rec = ledger.evidence.artifact_sha256[fmt]
        assert len(rec.sha256) == 64
        assert rec.bytes > 0

    # Reviews
    assert ledger.evidence.pdf_page_review.page_count == 10
    assert ledger.evidence.pdf_page_review.uninspected_pages == 0
    assert ledger.evidence.docx_page_review.unresolved_reference_count == 0
    assert ledger.evidence.accessibility_review.images_missing_alt == 0
    assert ledger.evidence.accessibility_review.has_lang is True

    # Approval and Blockers
    assert ledger.evidence.human_approval.state == "blocked-pending-human-approval"
    assert len(ledger.blockers) >= 2


def test_verify_publication_projection_succeeds_on_current_tree() -> None:
    ledger = verify_publication_projection(REPO_ROOT)
    assert ledger.release_status == "unapproved-projection-verified"


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    [
        (
            lambda doc: doc.update(authority_repository="D-sorganization/Unknown"),
            ValidationError,
            "was expected",
        ),
        (
            lambda doc: doc.update(catalog_repository="D-sorganization/Other"),
            ValidationError,
            "was expected",
        ),
        (
            lambda doc: doc.update(private_content_allowed=True),
            ValidationError,
            "was expected",
        ),
        (
            lambda doc: doc.update(release_status="approved-for-publication"),
            PublicationProjectionError,
            "Approved projection requires human_approval state 'approved'",
        ),
        (
            lambda doc: doc["evidence"].update(
                immutable_source_commit="not-a-40-char-sha"
            ),
            ValidationError,
            "does not match",
        ),
        (
            lambda doc: doc["evidence"].update(calculation_registry_sha256="0" * 64),
            PublicationProjectionError,
            "calculation_registry_sha256 mismatch",
        ),
        (
            lambda doc: doc["evidence"].update(toolchain_lock_sha256="0" * 64),
            PublicationProjectionError,
            "toolchain_lock_sha256 mismatch",
        ),
        (
            lambda doc: doc["evidence"]["artifact_sha256"]["pdf"].update(
                sha256="0" * 64
            ),
            PublicationProjectionError,
            "Artifact SHA-256 mismatch for pdf",
        ),
    ],
)
def test_verify_publication_projection_fails_closed_on_tampering(
    mutation: Any, error_type: type[Exception], message: str, tmp_path: Path
) -> None:
    manifest_doc = copy.deepcopy(_json(MANIFEST_PATH))
    mutation(manifest_doc)

    temp_manual = tmp_path / "manuals" / "tools"
    temp_manual.mkdir(parents=True, exist_ok=True)
    (temp_manual / "schemas").mkdir(parents=True, exist_ok=True)
    (temp_manual / "manifests").mkdir(parents=True, exist_ok=True)
    (temp_manual / "dist").mkdir(parents=True, exist_ok=True)

    (temp_manual / "publication-projection.json").write_text(
        json.dumps(manifest_doc, indent=2), encoding="utf-8", newline="\n"
    )
    (temp_manual / "schemas" / "publication-projection.schema.json").write_bytes(
        SCHEMA_PATH.read_bytes()
    )
    (temp_manual / "calculation-registry.json").write_bytes(
        (MANUAL_ROOT / "calculation-registry.json").read_bytes()
    )
    (temp_manual / "toolchain-lock.json").write_bytes(
        (MANUAL_ROOT / "toolchain-lock.json").read_bytes()
    )
    (temp_manual / "manual-qa.json").write_bytes(
        (MANUAL_ROOT / "manual-qa.json").read_bytes()
    )
    (temp_manual / "manifests" / "artifacts.json").write_bytes(
        (MANUAL_ROOT / "manifests" / "artifacts.json").read_bytes()
    )
    # Copy dist files
    for dist_file in (MANUAL_ROOT / "dist").glob("*"):
        (temp_manual / "dist" / dist_file.name).write_bytes(dist_file.read_bytes())

    with pytest.raises(error_type, match=message):
        verify_publication_projection(tmp_path)
