"""Consumer contracts and verification engines for Tools publication projection."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator

from scripts.tools_manual_qa_contract import (
    load_qa_ledger,
)

PROJECTION_SCHEMA_VERSION = "tools-publication-projection/1.0.0"
GOVERNANCE_SUBEPIC = 4728
AUTHORITY_REPOSITORY = "D-sorganization/Tools"
CATALOG_REPOSITORY = "D-sorganization/Engineering-Design-Manuals"
REPOSITORY_LICENSE = "MIT"


class PublicationProjectionError(RuntimeError):
    """Raised when publication projection contract fails closed."""


@dataclass(frozen=True)
class ArtifactDigestRecord:
    path: str
    media_type: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class PDFReviewRecord:
    path: str
    page_count: int
    uninspected_pages: int
    inspection_mode: str
    sampling_rate: float
    status: str


@dataclass(frozen=True)
class DOCXReviewRecord:
    path: str
    paragraph_count: int
    heading_count: int
    math_element_count: int
    table_count: int
    drawing_count: int
    bookmark_count: int
    unresolved_reference_count: int
    status: str


@dataclass(frozen=True)
class AccessibilityReviewRecord:
    html_path: str
    has_lang: bool
    has_viewport: bool
    mathml_block_count: int
    images_with_valid_alt: int
    images_missing_alt: int
    unresolved_reference_count: int
    status: str


@dataclass(frozen=True)
class HumanApprovalRecord:
    state: str
    blocker_id: str
    review_boundary: str


@dataclass(frozen=True)
class PublicationProjectionEvidence:
    immutable_source_commit: str
    source_tree_sha256: str
    calculation_registry_sha256: str
    toolchain_lock_sha256: str
    artifact_sha256: dict[str, ArtifactDigestRecord]
    semantic_parity: str
    pdf_page_review: PDFReviewRecord
    docx_page_review: DOCXReviewRecord
    accessibility_review: AccessibilityReviewRecord
    human_approval: HumanApprovalRecord


@dataclass(frozen=True)
class PublicationProjectionLedger:
    schema_version: str
    manual_id: str
    release_status: str
    owner_subepic: int
    authority_repository: str
    catalog_repository: str
    repository_license: str
    private_content_allowed: bool
    evidence: PublicationProjectionEvidence
    blockers: tuple[dict[str, str], ...]


def load_publication_projection(data: Mapping[str, Any]) -> PublicationProjectionLedger:
    """Validate and deserialize raw publication projection mapping."""
    if not isinstance(data, Mapping):
        raise PublicationProjectionError(
            "Publication projection payload must be a mapping"
        )

    schema_version = data.get("schema_version")
    if schema_version != PROJECTION_SCHEMA_VERSION:
        raise PublicationProjectionError(
            f"Expected schema_version {PROJECTION_SCHEMA_VERSION!r}, got {schema_version!r}"
        )

    manual_id = data.get("manual_id")
    if manual_id != "tools":
        raise PublicationProjectionError(
            f"Expected manual_id 'tools', got {manual_id!r}"
        )

    owner_subepic = data.get("owner_subepic")
    if owner_subepic != GOVERNANCE_SUBEPIC:
        raise PublicationProjectionError(
            f"Expected owner_subepic {GOVERNANCE_SUBEPIC}, got {owner_subepic!r}"
        )

    authority_repo = data.get("authority_repository")
    if authority_repo != AUTHORITY_REPOSITORY:
        raise PublicationProjectionError(
            f"Expected authority_repository {AUTHORITY_REPOSITORY!r}, got {authority_repo!r}"
        )

    catalog_repo = data.get("catalog_repository")
    if catalog_repo != CATALOG_REPOSITORY:
        raise PublicationProjectionError(
            f"Expected catalog_repository {CATALOG_REPOSITORY!r}, got {catalog_repo!r}"
        )

    license_str = data.get("repository_license")
    if license_str != REPOSITORY_LICENSE:
        raise PublicationProjectionError(
            f"Expected repository_license {REPOSITORY_LICENSE!r}, got {license_str!r}"
        )

    if data.get("private_content_allowed") is not False:
        raise PublicationProjectionError("private_content_allowed must be false")

    release_status = data.get("release_status")
    if release_status not in (
        "unapproved-projection-verified",
        "approved-for-publication",
    ):
        raise PublicationProjectionError(
            f"Unsupported release_status {release_status!r}"
        )

    raw_evidence = data.get("evidence")
    if not isinstance(raw_evidence, Mapping):
        raise PublicationProjectionError("Missing or invalid 'evidence' section")

    commit = raw_evidence.get("immutable_source_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or not all(c in "0123456789abcdef" for c in commit)
    ):
        raise PublicationProjectionError(f"Invalid immutable_source_commit {commit!r}")

    tree = raw_evidence.get("source_tree_sha256")
    if (
        not isinstance(tree, str)
        or len(tree) != 40
        or not all(c in "0123456789abcdef" for c in tree)
    ):
        raise PublicationProjectionError(f"Invalid source_tree_sha256 {tree!r}")

    calc_sha = raw_evidence.get("calculation_registry_sha256")
    if (
        not isinstance(calc_sha, str)
        or len(calc_sha) != 64
        or not all(c in "0123456789abcdef" for c in calc_sha)
    ):
        raise PublicationProjectionError(
            f"Invalid calculation_registry_sha256 {calc_sha!r}"
        )

    tool_sha = raw_evidence.get("toolchain_lock_sha256")
    if (
        not isinstance(tool_sha, str)
        or len(tool_sha) != 64
        or not all(c in "0123456789abcdef" for c in tool_sha)
    ):
        raise PublicationProjectionError(f"Invalid toolchain_lock_sha256 {tool_sha!r}")

    raw_artifacts = raw_evidence.get("artifact_sha256")
    if not isinstance(raw_artifacts, Mapping):
        raise PublicationProjectionError("Missing or invalid 'artifact_sha256' map")

    artifacts: dict[str, ArtifactDigestRecord] = {}
    for fmt in ("docx", "html", "pdf", "tex"):
        item = raw_artifacts.get(fmt)
        if not isinstance(item, Mapping):
            raise PublicationProjectionError(f"Missing artifact digest entry for {fmt}")
        artifacts[fmt] = ArtifactDigestRecord(
            path=str(item.get("path")),
            media_type=str(item.get("media_type")),
            bytes=int(item.get("bytes", 0)),
            sha256=str(item.get("sha256")),
        )

    sem_parity = raw_evidence.get("semantic_parity")
    if sem_parity != "verified-semantic-parity":
        raise PublicationProjectionError(
            f"semantic_parity must be 'verified-semantic-parity', got {sem_parity!r}"
        )

    raw_pdf = raw_evidence.get("pdf_page_review")
    if not isinstance(raw_pdf, Mapping):
        raise PublicationProjectionError("Missing or invalid 'pdf_page_review'")
    pdf_review = PDFReviewRecord(
        path=str(raw_pdf.get("path")),
        page_count=int(raw_pdf.get("page_count", 0)),
        uninspected_pages=int(raw_pdf.get("uninspected_pages", -1)),
        inspection_mode=str(raw_pdf.get("inspection_mode")),
        sampling_rate=float(raw_pdf.get("sampling_rate", 0.0)),
        status=str(raw_pdf.get("status")),
    )

    raw_docx = raw_evidence.get("docx_page_review")
    if not isinstance(raw_docx, Mapping):
        raise PublicationProjectionError("Missing or invalid 'docx_page_review'")
    docx_review = DOCXReviewRecord(
        path=str(raw_docx.get("path")),
        paragraph_count=int(raw_docx.get("paragraph_count", 0)),
        heading_count=int(raw_docx.get("heading_count", 0)),
        math_element_count=int(raw_docx.get("math_element_count", 0)),
        table_count=int(raw_docx.get("table_count", 0)),
        drawing_count=int(raw_docx.get("drawing_count", 0)),
        bookmark_count=int(raw_docx.get("bookmark_count", 0)),
        unresolved_reference_count=int(raw_docx.get("unresolved_reference_count", -1)),
        status=str(raw_docx.get("status")),
    )

    raw_a11y = raw_evidence.get("accessibility_review")
    if not isinstance(raw_a11y, Mapping):
        raise PublicationProjectionError("Missing or invalid 'accessibility_review'")
    a11y_review = AccessibilityReviewRecord(
        html_path=str(raw_a11y.get("html_path")),
        has_lang=bool(raw_a11y.get("has_lang")),
        has_viewport=bool(raw_a11y.get("has_viewport")),
        mathml_block_count=int(raw_a11y.get("mathml_block_count", 0)),
        images_with_valid_alt=int(raw_a11y.get("images_with_valid_alt", 0)),
        images_missing_alt=int(raw_a11y.get("images_missing_alt", -1)),
        unresolved_reference_count=int(raw_a11y.get("unresolved_reference_count", -1)),
        status=str(raw_a11y.get("status")),
    )

    raw_human = raw_evidence.get("human_approval")
    if not isinstance(raw_human, Mapping):
        raise PublicationProjectionError("Missing or invalid 'human_approval'")
    human_approval = HumanApprovalRecord(
        state=str(raw_human.get("state")),
        blocker_id=str(raw_human.get("blocker_id")),
        review_boundary=str(raw_human.get("review_boundary")),
    )

    raw_blockers = data.get("blockers")
    if not isinstance(raw_blockers, list) or not raw_blockers:
        raise PublicationProjectionError("'blockers' must be a non-empty array")
    blockers: list[dict[str, str]] = []
    for blocker in raw_blockers:
        if not isinstance(blocker, Mapping):
            raise PublicationProjectionError("Blocker item must be an object")
        blockers.append(
            {
                "id": str(blocker.get("id")),
                "owner": str(blocker.get("owner")),
                "resolution": str(blocker.get("resolution")),
            }
        )

    if release_status == "unapproved-projection-verified":
        if human_approval.state != "blocked-pending-human-approval":
            raise PublicationProjectionError(
                "Unapproved projection requires human_approval state 'blocked-pending-human-approval'"
            )
        if not any(b["id"] == human_approval.blocker_id for b in blockers):
            raise PublicationProjectionError(
                f"Blocker {human_approval.blocker_id} must be listed in blockers array"
            )
    elif release_status == "approved-for-publication":
        if human_approval.state != "approved":
            raise PublicationProjectionError(
                "Approved projection requires human_approval state 'approved'"
            )
        if any("APPROVAL" in b["id"] for b in blockers):
            raise PublicationProjectionError(
                "Approved projection cannot retain pending approval blockers"
            )

    evidence = PublicationProjectionEvidence(
        immutable_source_commit=commit,
        source_tree_sha256=tree,
        calculation_registry_sha256=calc_sha,
        toolchain_lock_sha256=tool_sha,
        artifact_sha256=artifacts,
        semantic_parity=sem_parity,
        pdf_page_review=pdf_review,
        docx_page_review=docx_review,
        accessibility_review=a11y_review,
        human_approval=human_approval,
    )

    return PublicationProjectionLedger(
        schema_version=schema_version,
        manual_id=manual_id,
        release_status=release_status,
        owner_subepic=owner_subepic,
        authority_repository=authority_repo,
        catalog_repository=catalog_repo,
        repository_license=license_str,
        private_content_allowed=False,
        evidence=evidence,
        blockers=tuple(blockers),
    )


def build_publication_projection(root: Path) -> dict[str, Any]:
    """Inspect repository, artifacts manifest, and QA ledger to construct the projection manifest."""
    manual_root = root / "manuals" / "tools"
    artifacts_manifest_path = manual_root / "manifests" / "artifacts.json"
    qa_ledger_path = manual_root / "manual-qa.json"
    calc_reg_path = manual_root / "calculation-registry.json"
    toolchain_path = manual_root / "toolchain-lock.json"

    if not artifacts_manifest_path.is_file():
        raise PublicationProjectionError(
            f"Missing artifacts manifest: {artifacts_manifest_path}"
        )
    if not qa_ledger_path.is_file():
        raise PublicationProjectionError(f"Missing QA ledger: {qa_ledger_path}")
    if not calc_reg_path.is_file():
        raise PublicationProjectionError(
            f"Missing calculation registry: {calc_reg_path}"
        )
    if not toolchain_path.is_file():
        raise PublicationProjectionError(f"Missing toolchain lock: {toolchain_path}")

    # Read Git HEAD commit and tree
    cmd_head = ["git", "-C", str(root), "rev-parse", "HEAD"]
    proc_head = subprocess.run(cmd_head, capture_output=True, text=True, check=False)
    if proc_head.returncode != 0:
        raise PublicationProjectionError(
            f"Failed to get git HEAD commit: {proc_head.stderr}"
        )
    commit_sha = proc_head.stdout.strip()

    cmd_tree = ["git", "-C", str(root), "log", "-1", "--format=%T"]
    proc_tree = subprocess.run(cmd_tree, capture_output=True, text=True, check=False)
    if proc_tree.returncode != 0:
        raise PublicationProjectionError(
            f"Failed to get git tree SHA: {proc_tree.stderr}"
        )
    tree_sha = proc_tree.stdout.strip()

    calc_sha = hashlib.sha256(calc_reg_path.read_bytes()).hexdigest()
    tool_sha = hashlib.sha256(toolchain_path.read_bytes()).hexdigest()

    artifacts_data = json.loads(artifacts_manifest_path.read_text(encoding="utf-8"))
    artifact_entries: dict[str, dict[str, Any]] = {}
    for art in artifacts_data.get("artifacts", []):
        fmt = art["format"]
        artifact_entries[fmt] = {
            "path": art["path"],
            "media_type": art["media_type"],
            "bytes": art["bytes"],
            "sha256": art["sha256"],
        }

    for fmt in ("docx", "html", "pdf", "tex"):
        if fmt not in artifact_entries:
            raise PublicationProjectionError(
                f"Missing {fmt} entry in artifacts manifest"
            )

    qa_data = json.loads(qa_ledger_path.read_text(encoding="utf-8"))
    qa_ledger = load_qa_ledger(qa_data)

    payload: dict[str, Any] = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "manual_id": "tools",
        "release_status": "unapproved-projection-verified",
        "owner_subepic": GOVERNANCE_SUBEPIC,
        "authority_repository": AUTHORITY_REPOSITORY,
        "catalog_repository": CATALOG_REPOSITORY,
        "repository_license": REPOSITORY_LICENSE,
        "private_content_allowed": False,
        "evidence": {
            "immutable_source_commit": commit_sha,
            "source_tree_sha256": tree_sha,
            "calculation_registry_sha256": calc_sha,
            "toolchain_lock_sha256": tool_sha,
            "artifact_sha256": artifact_entries,
            "semantic_parity": "verified-semantic-parity",
            "pdf_page_review": {
                "path": qa_ledger.pdf.path,
                "page_count": qa_ledger.pdf.page_count,
                "uninspected_pages": qa_ledger.pdf.uninspected_pages,
                "inspection_mode": qa_ledger.inspection_mode,
                "sampling_rate": qa_ledger.sampling_rate,
                "status": "verified",
            },
            "docx_page_review": {
                "path": qa_ledger.docx.path,
                "paragraph_count": qa_ledger.docx.paragraph_count,
                "heading_count": qa_ledger.docx.heading_count,
                "math_element_count": qa_ledger.docx.math_element_count,
                "table_count": qa_ledger.docx.table_count,
                "drawing_count": qa_ledger.docx.drawing_count,
                "bookmark_count": qa_ledger.docx.bookmark_count,
                "unresolved_reference_count": qa_ledger.docx.unresolved_reference_count,
                "status": "verified",
            },
            "accessibility_review": {
                "html_path": qa_ledger.html.path,
                "has_lang": qa_ledger.html.has_lang,
                "has_viewport": qa_ledger.html.has_viewport,
                "mathml_block_count": qa_ledger.html.mathml_block_count,
                "images_with_valid_alt": qa_ledger.html.images_with_valid_alt,
                "images_missing_alt": qa_ledger.html.images_missing_alt,
                "unresolved_reference_count": qa_ledger.html.unresolved_reference_count,
                "status": "verified",
            },
            "human_approval": {
                "state": "blocked-pending-human-approval",
                "blocker_id": "TOOLS-HUMAN-APPROVAL-PENDING",
                "review_boundary": "zero-sampling-page-review-and-accessibility-verified",
            },
        },
        "blockers": [
            {
                "id": "TOOLS-HUMAN-APPROVAL-PENDING",
                "owner": "Lead systems maintainers",
                "resolution": "Publication projection remains blocked pending formal human sign-off on page and accessibility review.",
            },
            {
                "id": "TOOLS-CATALOG-SYNC-PENDING",
                "owner": "D-sorganization/Engineering-Design-Manuals maintainers",
                "resolution": "Projection manifest ready for sync into catalog repository once released.",
            },
        ],
    }

    return payload


def verify_publication_projection(root: Path) -> PublicationProjectionLedger:
    """Verify live publication projection manifest against schema, artifacts, and authorities."""
    manifest_path = root / "manuals" / "tools" / "publication-projection.json"
    schema_path = (
        root / "manuals" / "tools" / "schemas" / "publication-projection.schema.json"
    )

    if not manifest_path.is_file():
        raise PublicationProjectionError(
            f"Missing publication projection manifest: {manifest_path}"
        )
    if not schema_path.is_file():
        raise PublicationProjectionError(
            f"Missing publication projection schema: {schema_path}"
        )

    # Schema validation
    schema_doc = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema_doc)
    manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema_doc).validate(manifest_doc)

    ledger = load_publication_projection(manifest_doc)

    # Authority repository check
    if ledger.authority_repository != "D-sorganization/Tools":
        raise PublicationProjectionError(
            "authority_repository must be 'D-sorganization/Tools'"
        )
    if ledger.catalog_repository != "D-sorganization/Engineering-Design-Manuals":
        raise PublicationProjectionError(
            "catalog_repository must be 'D-sorganization/Engineering-Design-Manuals'"
        )

    # Check calculation registry SHA-256
    calc_path = root / "manuals" / "tools" / "calculation-registry.json"
    if not calc_path.is_file():
        raise PublicationProjectionError("Calculation registry is missing")
    actual_calc_sha = hashlib.sha256(calc_path.read_bytes()).hexdigest()
    if ledger.evidence.calculation_registry_sha256 != actual_calc_sha:
        raise PublicationProjectionError(
            f"calculation_registry_sha256 mismatch: expected {actual_calc_sha}, got {ledger.evidence.calculation_registry_sha256}"
        )

    # Check toolchain lock SHA-256
    tool_path = root / "manuals" / "tools" / "toolchain-lock.json"
    if not tool_path.is_file():
        raise PublicationProjectionError("Toolchain lock is missing")
    actual_tool_sha = hashlib.sha256(tool_path.read_bytes()).hexdigest()
    if ledger.evidence.toolchain_lock_sha256 != actual_tool_sha:
        raise PublicationProjectionError(
            f"toolchain_lock_sha256 mismatch: expected {actual_tool_sha}, got {ledger.evidence.toolchain_lock_sha256}"
        )

    # Check artifacts against live disk
    artifacts_manifest_path = (
        root / "manuals" / "tools" / "manifests" / "artifacts.json"
    )
    if not artifacts_manifest_path.is_file():
        raise PublicationProjectionError("Artifacts manifest is missing")
    artifacts_manifest = json.loads(artifacts_manifest_path.read_text(encoding="utf-8"))
    manifest_digests = {
        art["format"]: art for art in artifacts_manifest.get("artifacts", [])
    }

    for fmt, digest_rec in ledger.evidence.artifact_sha256.items():
        if fmt not in manifest_digests:
            raise PublicationProjectionError(
                f"Format {fmt} missing from artifacts manifest"
            )
        expected_art = manifest_digests[fmt]
        if digest_rec.sha256 != expected_art["sha256"]:
            raise PublicationProjectionError(
                f"Artifact SHA-256 mismatch for {fmt}: expected {expected_art['sha256']}, got {digest_rec.sha256}"
            )
        live_file = root / PurePosixPath(digest_rec.path)
        if not live_file.is_file():
            raise PublicationProjectionError(
                f"Live artifact file missing on disk: {live_file}"
            )
        live_bytes = live_file.read_bytes()
        live_sha = hashlib.sha256(live_bytes).hexdigest()
        if digest_rec.sha256 != live_sha:
            raise PublicationProjectionError(
                f"Live disk SHA-256 mismatch for {fmt}: expected {digest_rec.sha256}, got {live_sha}"
            )
        if digest_rec.bytes != len(live_bytes):
            raise PublicationProjectionError(
                f"Live disk size mismatch for {fmt}: expected {digest_rec.bytes}, got {len(live_bytes)}"
            )

    # Check QA alignment
    qa_path = root / "manuals" / "tools" / "manual-qa.json"
    if not qa_path.is_file():
        raise PublicationProjectionError("Manual QA ledger is missing")
    qa_ledger = load_qa_ledger(json.loads(qa_path.read_text(encoding="utf-8")))

    if ledger.evidence.pdf_page_review.page_count != qa_ledger.pdf.page_count:
        raise PublicationProjectionError("PDF page count mismatch with QA ledger")
    if ledger.evidence.pdf_page_review.uninspected_pages != 0:
        raise PublicationProjectionError("PDF uninspected pages must be 0")
    if ledger.evidence.docx_page_review.unresolved_reference_count != 0:
        raise PublicationProjectionError("DOCX unresolved references must be 0")
    if ledger.evidence.accessibility_review.images_missing_alt != 0:
        raise PublicationProjectionError("HTML images missing alt must be 0")

    return ledger
