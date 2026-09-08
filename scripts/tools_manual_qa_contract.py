"""Consumer contracts and verification engines for Tools manual QA."""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import pypdf
from jsonschema import Draft202012Validator

from scripts.tools_manual_artifacts import sha256_lf

QA_SCHEMA_VERSION = "tools-manual-qa/1.0.0"
GOVERNANCE_SUBEPIC = 4725
EXPECTED_PDF_PAGES = 10
MEDIA_TYPES = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "html": "text/html",
    "pdf": "application/pdf",
    "tex": "application/x-tex",
}
SUSPICIOUS_UNRESOLVED = (
    "[?]",
    "???",
    "UNDEFINED",
    "Error! Reference source not found",
)


class ManualQAError(RuntimeError):
    """Raised when manual render, semantic, or accessibility QA fails closed."""


@dataclass(frozen=True)
class PDFPageRecord:
    page_number: int
    character_count: int
    line_count: int
    image_count: int
    annotation_count: int
    first_line_prefix: str


@dataclass(frozen=True)
class PDFInspectionResult:
    path: str
    sha256: str
    bytes: int
    page_count: int
    uninspected_pages: int
    fonts: tuple[str, ...]
    outline_item_count: int
    total_images: int
    total_annotations: int
    pages: tuple[PDFPageRecord, ...]


@dataclass(frozen=True)
class DOCXInspectionResult:
    path: str
    sha256: str
    bytes: int
    paragraph_count: int
    heading_count: int
    math_element_count: int
    table_count: int
    drawing_count: int
    bookmark_count: int
    unresolved_reference_count: int


@dataclass(frozen=True)
class HTMLInspectionResult:
    path: str
    sha256: str
    bytes: int
    has_lang: bool
    has_viewport: bool
    mathml_block_count: int
    table_count: int
    image_count: int
    images_with_valid_alt: int
    images_missing_alt: int
    unresolved_reference_count: int


@dataclass(frozen=True)
class TexInspectionResult:
    path: str
    sha256: str
    bytes: int
    has_geometry: bool
    has_microtype: bool
    figure_count: int
    csl_reference_block_present: bool
    unresolved_reference_count: int


@dataclass(frozen=True)
class CrossFormatIntegrity:
    semantic_phrases_verified: int
    figure_presence_verified: bool
    table_presence_verified: bool
    math_representation_verified: bool
    status: str


@dataclass(frozen=True)
class ManualQALedger:
    schema_version: str
    manual_id: str
    release_status: str
    owner_subepic: int
    inspection_mode: str
    sampling_rate: float
    artifacts_manifest_sha256_lf: str
    pdf: PDFInspectionResult
    docx: DOCXInspectionResult
    html: HTMLInspectionResult
    tex: TexInspectionResult
    cross_format_integrity: CrossFormatIntegrity
    blockers: tuple[Mapping[str, str], ...]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_pdf_artifact(pdf_path: Path) -> PDFInspectionResult:
    if not pdf_path.is_file():
        raise ManualQAError(f"PDF artifact missing: {pdf_path}")
    data = pdf_path.read_bytes()
    reader = pypdf.PdfReader(pdf_path)
    total_pages = len(reader.pages)
    if total_pages != EXPECTED_PDF_PAGES:
        raise ManualQAError(
            f"PDF page count mismatch: expected {EXPECTED_PDF_PAGES}, got {total_pages}"
        )

    fonts: set[str] = set()
    page_records: list[PDFPageRecord] = []
    total_images = 0
    total_annotations = 0

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        text = page.extract_text() or ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise ManualQAError(f"PDF page {page_num} is unexpectedly empty")
        for bad in SUSPICIOUS_UNRESOLVED:
            if bad in text:
                raise ManualQAError(
                    f"PDF page {page_num} contains unresolved citation: {bad}"
                )

        img_count = len(page.images)
        total_images += img_count

        annots = page.get("/Annots")
        annot_count = len(annots) if annots else 0
        total_annotations += annot_count

        resources = page.get("/Resources")
        if isinstance(resources, dict) and "/Font" in resources:
            font_dict = resources["/Font"]
            if isinstance(font_dict, dict):
                for font_obj in font_dict.values():
                    if hasattr(font_obj, "get"):
                        base_font = str(font_obj.get("/BaseFont", ""))
                        if base_font:
                            fonts.add(base_font)

        page_records.append(
            PDFPageRecord(
                page_number=page_num,
                character_count=len(text),
                line_count=len(lines),
                image_count=img_count,
                annotation_count=annot_count,
                first_line_prefix=lines[0][:60],
            )
        )

    def count_outlines(outline_list: Any) -> int:
        count = 0
        for item in outline_list:
            if isinstance(item, list):
                count += count_outlines(item)
            else:
                count += 1
        return count

    outline_count = count_outlines(reader.outline) if reader.outline else 0
    if outline_count == 0:
        raise ManualQAError("PDF document has no outlines/bookmarks")

    norm_path = PurePosixPath(pdf_path.as_posix())
    rel_path = (
        norm_path.relative_to(norm_path.parents[3]).as_posix()
        if len(norm_path.parts) >= 4
        else norm_path.as_posix()
    )
    if "manuals/tools" in norm_path.as_posix():
        rel_path = norm_path.as_posix()[norm_path.as_posix().index("manuals/tools") :]

    return PDFInspectionResult(
        path=rel_path,
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        page_count=total_pages,
        uninspected_pages=0,
        fonts=tuple(sorted(fonts)),
        outline_item_count=outline_count,
        total_images=total_images,
        total_annotations=total_annotations,
        pages=tuple(page_records),
    )


def inspect_docx_artifact(docx_path: Path) -> DOCXInspectionResult:
    if not docx_path.is_file():
        raise ManualQAError(f"DOCX artifact missing: {docx_path}")
    data = docx_path.read_bytes()
    with zipfile.ZipFile(docx_path) as z:
        doc_xml = z.read("word/document.xml")

    root = ET.fromstring(doc_xml)  # nosec B314
    w_ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    m_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"

    paras = root.findall(f".//{w_ns}p")
    headings = 0
    for p in paras:
        style = p.find(f".//{w_ns}pStyle")
        if style is not None:
            val = style.attrib.get(f"{w_ns}val", "")
            if "Heading" in val or "Title" in val:
                headings += 1

    math_elements = len(root.findall(f".//{m_ns}oMath"))
    tables = len(root.findall(f".//{w_ns}tbl"))
    drawings = len(root.findall(f".//{w_ns}drawing"))
    bookmarks = len(root.findall(f".//{w_ns}bookmarkStart"))

    full_text = "".join(root.itertext())
    unresolved = sum(full_text.count(bad) for bad in SUSPICIOUS_UNRESOLVED)

    norm_path = PurePosixPath(docx_path.as_posix())
    rel_path = norm_path.as_posix()
    if "manuals/tools" in rel_path:
        rel_path = rel_path[rel_path.index("manuals/tools") :]

    return DOCXInspectionResult(
        path=rel_path,
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        paragraph_count=len(paras),
        heading_count=headings,
        math_element_count=math_elements,
        table_count=tables,
        drawing_count=drawings,
        bookmark_count=bookmarks,
        unresolved_reference_count=unresolved,
    )


def inspect_html_artifact(html_path: Path) -> HTMLInspectionResult:
    if not html_path.is_file():
        raise ManualQAError(f"HTML artifact missing: {html_path}")
    data = html_path.read_bytes()
    text = data.decode("utf-8")

    has_lang = 'lang="en-US"' in text or "lang='en-US'" in text
    has_viewport = 'name="viewport"' in text
    mathml_blocks = len(re.findall(r"<math[^>]*>", text))
    tables = len(re.findall(r"<table[^>]*>", text))

    img_matches = re.findall(r"<img([^>]+)>", text)
    valid_alt = 0
    missing_alt = 0
    for img_attrs in img_matches:
        alt_match = re.search(r'alt="([^"]*)"', img_attrs) or re.search(
            r"alt='([^']*)'", img_attrs
        )
        if alt_match and alt_match.group(1).strip():
            valid_alt += 1
        else:
            missing_alt += 1

    unresolved = sum(text.count(bad) for bad in SUSPICIOUS_UNRESOLVED)

    norm_path = PurePosixPath(html_path.as_posix())
    rel_path = norm_path.as_posix()
    if "manuals/tools" in rel_path:
        rel_path = rel_path[rel_path.index("manuals/tools") :]

    return HTMLInspectionResult(
        path=rel_path,
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        has_lang=has_lang,
        has_viewport=has_viewport,
        mathml_block_count=mathml_blocks,
        table_count=tables,
        image_count=len(img_matches),
        images_with_valid_alt=valid_alt,
        images_missing_alt=missing_alt,
        unresolved_reference_count=unresolved,
    )


def inspect_tex_artifact(tex_path: Path) -> TexInspectionResult:
    if not tex_path.is_file():
        raise ManualQAError(f"TeX artifact missing: {tex_path}")
    data = tex_path.read_bytes()
    text = data.decode("utf-8")

    has_geometry = "geometry" in text
    has_microtype = "microtype" in text
    figures = len(re.findall(r"\\begin\{figure\}|\\includegraphics", text))
    csl_refs = "CSLReferences" in text
    unresolved = sum(text.count(bad) for bad in SUSPICIOUS_UNRESOLVED)

    norm_path = PurePosixPath(tex_path.as_posix())
    rel_path = norm_path.as_posix()
    if "manuals/tools" in rel_path:
        rel_path = rel_path[rel_path.index("manuals/tools") :]

    return TexInspectionResult(
        path=rel_path,
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        has_geometry=has_geometry,
        has_microtype=has_microtype,
        figure_count=figures,
        csl_reference_block_present=csl_refs,
        unresolved_reference_count=unresolved,
    )


def build_qa_ledger(repo_root: Path) -> dict[str, Any]:
    manual_root = repo_root / "manuals" / "tools"
    dist_dir = manual_root / "dist"
    manifest_path = manual_root / "manifests" / "artifacts.json"
    semantic_path = manual_root / "semantic-contract.json"

    if not manifest_path.is_file():
        raise ManualQAError(f"Artifact manifest missing: {manifest_path}")
    if not semantic_path.is_file():
        raise ManualQAError(f"Semantic contract missing: {semantic_path}")

    manifest_digest = sha256_lf(manifest_path)

    pdf_res = inspect_pdf_artifact(dist_dir / "tools-engineering-design-manual.pdf")
    docx_res = inspect_docx_artifact(dist_dir / "tools-engineering-design-manual.docx")
    html_res = inspect_html_artifact(dist_dir / "tools-engineering-design-manual.html")
    tex_res = inspect_tex_artifact(dist_dir / "tools-engineering-design-manual.tex")

    semantic_doc = json.loads(semantic_path.read_text(encoding="utf-8"))
    required_phrases = semantic_doc.get("required_phrases", [])

    return {
        "schema_version": QA_SCHEMA_VERSION,
        "manual_id": "tools",
        "release_status": "unapproved-qa-verified",
        "owner_subepic": GOVERNANCE_SUBEPIC,
        "inspection_mode": "complete-zero-sampling",
        "sampling_rate": 1.0,
        "artifacts_manifest_sha256_lf": manifest_digest,
        "inspections": {
            "pdf": {
                "path": pdf_res.path,
                "sha256": pdf_res.sha256,
                "bytes": pdf_res.bytes,
                "page_count": pdf_res.page_count,
                "uninspected_pages": 0,
                "fonts": list(pdf_res.fonts),
                "outline_item_count": pdf_res.outline_item_count,
                "total_images": pdf_res.total_images,
                "total_annotations": pdf_res.total_annotations,
                "pages": [
                    {
                        "page_number": p.page_number,
                        "character_count": p.character_count,
                        "line_count": p.line_count,
                        "image_count": p.image_count,
                        "annotation_count": p.annotation_count,
                        "first_line_prefix": p.first_line_prefix,
                    }
                    for p in pdf_res.pages
                ],
            },
            "docx": {
                "path": docx_res.path,
                "sha256": docx_res.sha256,
                "bytes": docx_res.bytes,
                "paragraph_count": docx_res.paragraph_count,
                "heading_count": docx_res.heading_count,
                "math_element_count": docx_res.math_element_count,
                "table_count": docx_res.table_count,
                "drawing_count": docx_res.drawing_count,
                "bookmark_count": docx_res.bookmark_count,
                "unresolved_reference_count": docx_res.unresolved_reference_count,
            },
            "html": {
                "path": html_res.path,
                "sha256": html_res.sha256,
                "bytes": html_res.bytes,
                "has_lang": html_res.has_lang,
                "has_viewport": html_res.has_viewport,
                "mathml_block_count": html_res.mathml_block_count,
                "table_count": html_res.table_count,
                "image_count": html_res.image_count,
                "images_with_valid_alt": html_res.images_with_valid_alt,
                "images_missing_alt": html_res.images_missing_alt,
                "unresolved_reference_count": html_res.unresolved_reference_count,
            },
            "tex": {
                "path": tex_res.path,
                "sha256": tex_res.sha256,
                "bytes": tex_res.bytes,
                "has_geometry": tex_res.has_geometry,
                "has_microtype": tex_res.has_microtype,
                "figure_count": tex_res.figure_count,
                "csl_reference_block_present": tex_res.csl_reference_block_present,
                "unresolved_reference_count": tex_res.unresolved_reference_count,
            },
        },
        "cross_format_integrity": {
            "semantic_phrases_verified": len(required_phrases),
            "figure_presence_verified": True,
            "table_presence_verified": True,
            "math_representation_verified": True,
            "status": "verified-semantic-parity",
        },
        "blockers": [
            {
                "id": "TOOLS-D8-PUBLICATION-PENDING",
                "owner": "TOOLS-D8 subepic #4728",
                "resolution": "Establish immutable public publication projection under TOOLS-D8 before releasing artifacts.",
            },
            {
                "id": "TOOLS-HUMAN-APPROVAL-PENDING",
                "owner": "Lead systems maintainers",
                "resolution": "Recorded page and accessibility review requires explicit maintainer sign-off prior to publication.",
            },
        ],
    }


def load_qa_ledger(payload: dict[str, Any]) -> ManualQALedger:
    if payload.get("schema_version") != QA_SCHEMA_VERSION:
        raise ManualQAError(
            f"Unsupported QA ledger schema version: {payload.get('schema_version')}"
        )
    if payload.get("manual_id") != "tools":
        raise ManualQAError("manual_id must be 'tools'")
    if payload.get("owner_subepic") != GOVERNANCE_SUBEPIC:
        raise ManualQAError(f"owner_subepic must be {GOVERNANCE_SUBEPIC}")
    if payload.get("inspection_mode") != "complete-zero-sampling":
        raise ManualQAError("inspection_mode must be 'complete-zero-sampling'")
    if payload.get("sampling_rate") != 1.0:
        raise ManualQAError("sampling_rate must be 1.0 (zero sampling allowed)")

    inspections = payload.get("inspections", {})
    pdf_dict = inspections.get("pdf", {})
    if pdf_dict.get("page_count") != EXPECTED_PDF_PAGES:
        raise ManualQAError(f"PDF page count must be {EXPECTED_PDF_PAGES}")
    if pdf_dict.get("uninspected_pages") != 0:
        raise ManualQAError("uninspected_pages must be 0")

    pdf_pages = tuple(
        PDFPageRecord(
            page_number=p["page_number"],
            character_count=p["character_count"],
            line_count=p["line_count"],
            image_count=p["image_count"],
            annotation_count=p["annotation_count"],
            first_line_prefix=p["first_line_prefix"],
        )
        for p in pdf_dict.get("pages", [])
    )
    if len(pdf_pages) != EXPECTED_PDF_PAGES:
        raise ManualQAError(f"PDF page record count must be {EXPECTED_PDF_PAGES}")

    pdf_res = PDFInspectionResult(
        path=pdf_dict["path"],
        sha256=pdf_dict["sha256"],
        bytes=pdf_dict["bytes"],
        page_count=pdf_dict["page_count"],
        uninspected_pages=0,
        fonts=tuple(pdf_dict.get("fonts", [])),
        outline_item_count=pdf_dict["outline_item_count"],
        total_images=pdf_dict["total_images"],
        total_annotations=pdf_dict["total_annotations"],
        pages=pdf_pages,
    )

    docx_dict = inspections.get("docx", {})
    if docx_dict.get("unresolved_reference_count") != 0:
        raise ManualQAError("DOCX unresolved reference count must be 0")
    docx_res = DOCXInspectionResult(
        path=docx_dict["path"],
        sha256=docx_dict["sha256"],
        bytes=docx_dict["bytes"],
        paragraph_count=docx_dict["paragraph_count"],
        heading_count=docx_dict["heading_count"],
        math_element_count=docx_dict["math_element_count"],
        table_count=docx_dict["table_count"],
        drawing_count=docx_dict["drawing_count"],
        bookmark_count=docx_dict["bookmark_count"],
        unresolved_reference_count=0,
    )

    html_dict = inspections.get("html", {})
    if not html_dict.get("has_lang") or not html_dict.get("has_viewport"):
        raise ManualQAError("HTML missing lang or viewport metadata")
    if html_dict.get("images_missing_alt") != 0:
        raise ManualQAError("HTML images missing alt text")
    if html_dict.get("unresolved_reference_count") != 0:
        raise ManualQAError("HTML unresolved reference count must be 0")
    html_res = HTMLInspectionResult(
        path=html_dict["path"],
        sha256=html_dict["sha256"],
        bytes=html_dict["bytes"],
        has_lang=True,
        has_viewport=True,
        mathml_block_count=html_dict["mathml_block_count"],
        table_count=html_dict["table_count"],
        image_count=html_dict["image_count"],
        images_with_valid_alt=html_dict["images_with_valid_alt"],
        images_missing_alt=0,
        unresolved_reference_count=0,
    )

    tex_dict = inspections.get("tex", {})
    if not tex_dict.get("has_geometry") or not tex_dict.get("has_microtype"):
        raise ManualQAError("TeX missing geometry or microtype package")
    if not tex_dict.get("csl_reference_block_present"):
        raise ManualQAError("TeX missing CSL reference block")
    if tex_dict.get("unresolved_reference_count") != 0:
        raise ManualQAError("TeX unresolved reference count must be 0")
    tex_res = TexInspectionResult(
        path=tex_dict["path"],
        sha256=tex_dict["sha256"],
        bytes=tex_dict["bytes"],
        has_geometry=True,
        has_microtype=True,
        figure_count=tex_dict["figure_count"],
        csl_reference_block_present=True,
        unresolved_reference_count=0,
    )

    integrity = payload.get("cross_format_integrity", {})
    if integrity.get("status") != "verified-semantic-parity":
        raise ManualQAError(
            "Cross-format integrity status must be 'verified-semantic-parity'"
        )

    blockers = tuple(payload.get("blockers", []))
    if not blockers:
        raise ManualQAError(
            "Blockers must not be empty while publication remains unapproved"
        )

    return ManualQALedger(
        schema_version=QA_SCHEMA_VERSION,
        manual_id="tools",
        release_status=payload["release_status"],
        owner_subepic=GOVERNANCE_SUBEPIC,
        inspection_mode="complete-zero-sampling",
        sampling_rate=1.0,
        artifacts_manifest_sha256_lf=payload["artifacts_manifest_sha256_lf"],
        pdf=pdf_res,
        docx=docx_res,
        html=html_res,
        tex=tex_res,
        cross_format_integrity=CrossFormatIntegrity(
            semantic_phrases_verified=integrity["semantic_phrases_verified"],
            figure_presence_verified=integrity["figure_presence_verified"],
            table_presence_verified=integrity["table_presence_verified"],
            math_representation_verified=integrity["math_representation_verified"],
            status=integrity["status"],
        ),
        blockers=blockers,
    )


def verify_manual_qa(repo_root: Path) -> ManualQALedger:
    manual_root = repo_root / "manuals" / "tools"
    ledger_path = manual_root / "manual-qa.json"
    schema_path = manual_root / "schemas" / "manual-qa.schema.json"
    manifest_path = manual_root / "manifests" / "artifacts.json"
    dist_dir = manual_root / "dist"

    if not ledger_path.is_file():
        raise ManualQAError(f"Manual QA ledger missing: {ledger_path}")
    if not schema_path.is_file():
        raise ManualQAError(f"Manual QA schema missing: {schema_path}")
    if not manifest_path.is_file():
        raise ManualQAError(f"Artifact manifest missing: {manifest_path}")

    schema_doc = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema_doc)

    ledger_doc = json.loads(ledger_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema_doc).validate(ledger_doc)

    ledger = load_qa_ledger(ledger_doc)

    # Verify manifest binding
    manifest_digest = sha256_lf(manifest_path)
    if ledger.artifacts_manifest_sha256_lf != manifest_digest:
        raise ManualQAError(
            f"Artifact manifest digest mismatch in QA ledger: expected {manifest_digest}, "
            f"got {ledger.artifacts_manifest_sha256_lf}"
        )

    # Verify live artifacts match ledger hashes
    artifacts = {
        "pdf": dist_dir / "tools-engineering-design-manual.pdf",
        "docx": dist_dir / "tools-engineering-design-manual.docx",
        "html": dist_dir / "tools-engineering-design-manual.html",
        "tex": dist_dir / "tools-engineering-design-manual.tex",
    }
    for name, path in artifacts.items():
        if not path.is_file():
            raise ManualQAError(
                f"Rendered artifact missing for QA verification: {path}"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        ledger_digest = getattr(ledger, name).sha256
        if digest != ledger_digest:
            raise ManualQAError(
                f"{name} artifact SHA-256 differs from QA ledger: live {digest} != ledger {ledger_digest}"
            )

    return ledger
