"""Deterministic artifact normalization and semantic verification helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from xml.etree import ElementTree

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    ByteStringObject,
    DictionaryObject,
    NameObject,
)

from scripts.tools_manual_renderer_contract import (
    ArtifactManifest,
    ManualRendererError,
    ToolchainLock,
    canonical_semantic_text,
    load_artifact_manifest,
)

FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
FIXED_PDF_ID = bytes.fromhex("00" * 16)
FONT_SUBSET_PATTERN = re.compile(r"^/[A-Z]{6}\+")
MEDIA_TYPES = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "html": "text/html",
    "pdf": "application/pdf",
    "tex": "application/x-tex",
}


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest for bytes."""
    return hashlib.sha256(value).hexdigest()


def sha256_lf(path: Path) -> str:
    """Hash a text file after CRLF/bare-CR to LF normalization."""
    value = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return sha256_bytes(value)


def run_checked(
    command: Sequence[str], *, cwd: Path, environment: Mapping[str, str]
) -> str:
    """Run one bounded renderer command and return stdout or fail typed."""
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env={**os.environ, **environment},
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        raise ManualRendererError(
            f"renderer command unavailable: {command[0]}"
        ) from exc
    if result.returncode != 0:
        detail = canonical_semantic_text(result.stderr or result.stdout)
        raise ManualRendererError(
            f"renderer command failed ({result.returncode}): {command[0]}: {detail[:800]}"
        )
    return result.stdout


def normalize_text_artifact(path: Path) -> None:
    """Normalize line endings and terminal newline for deterministic text output."""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def _normalize_core_properties(value: bytes) -> bytes:
    fixed = b"2026-08-26T00:00:00Z"
    result = re.sub(
        rb"(<dcterms:(?:created|modified)[^>]*>)[^<]*(</dcterms:(?:created|modified)>)",
        rb"\g<1>" + fixed + rb"\g<2>",
        value,
    )
    return re.sub(rb"(<cp:revision>)[^<]*(</cp:revision>)", rb"\g<1>1\g<2>", result)


def canonicalize_docx(path: Path) -> None:
    """Rewrite OOXML ordering, timestamps, and core properties deterministically."""
    temporary = path.with_suffix(".canonical.docx")
    with (
        zipfile.ZipFile(path, "r") as source,
        zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as target,
    ):
        for name in sorted(source.namelist()):
            value = source.read(name)
            if name == "docProps/core.xml":
                value = _normalize_core_properties(value)
            info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0
            target.writestr(
                info, value, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
            )
    temporary.replace(path)


def canonicalize_pdf(path: Path) -> None:
    """Rewrite PDF metadata, object ordering, and trailer ID deterministically."""
    reader = PdfReader(path)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    _normalize_pdf_font_names(writer)
    writer.metadata = None
    writer.add_metadata(
        {
            "/Title": "Tools Engineering Design Manual",
            "/Creator": "D-sorganization deterministic renderer",
            "/Producer": "pypdf canonicalizer",
            "/CreationDate": "D:20260826000000Z",
            "/ModDate": "D:20260826000000Z",
        }
    )
    writer._ID = ArrayObject(  # noqa: SLF001 - deterministic trailer is the contract
        [ByteStringObject(FIXED_PDF_ID), ByteStringObject(FIXED_PDF_ID)]
    )
    temporary = path.with_suffix(".canonical.pdf")
    with temporary.open("wb") as stream:
        writer.write(stream)
    temporary.replace(path)


def _normalize_pdf_font_names(writer: PdfWriter) -> None:
    """Replace XeTeX's random six-letter font subset tags in dictionaries."""

    def normalize(value: object) -> None:
        if isinstance(value, DictionaryObject):
            for key, item in list(value.items()):
                if isinstance(item, NameObject) and FONT_SUBSET_PATTERN.match(
                    str(item)
                ):
                    value[key] = NameObject(
                        FONT_SUBSET_PATTERN.sub("/DSTOOL+", str(item))
                    )
                else:
                    normalize(item)
        elif isinstance(value, ArrayObject):
            for item in value:
                normalize(item)

    for item in writer._objects:  # noqa: SLF001 - complete PDF graph is required
        normalize(item)


def extract_visible_text(path: Path, lock: ToolchainLock, workspace: Path) -> str:
    """Extract format-visible text through Pandoc or pypdf."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    header_text = ""
    if path.suffix.lower() == ".tex":
        header_text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".docx":
        with zipfile.ZipFile(path) as package:
            parts = sorted(
                name
                for name in package.namelist()
                if re.fullmatch(r"word/header\d+\.xml", name)
            )
            values: list[str] = []
            for name in parts:
                root = ElementTree.fromstring(package.read(name))
                values.extend(
                    node.text or ""
                    for node in root.iter(
                        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
                    )
                )
            header_text = "\n".join(values)
    command = lock.commands["pandoc"].executable
    body_text = run_checked(
        [command, str(path), "--to=plain"],
        cwd=workspace,
        environment={"SOURCE_DATE_EPOCH": str(lock.source_date_epoch)},
    )
    return f"{header_text}\n{body_text}"


def semantic_digest_for_artifacts(
    artifacts: Mapping[str, Path],
    lock: ToolchainLock,
    semantic_contract: Path,
    workspace: Path,
) -> str:
    """Require every representation to retain each governed semantic phrase."""
    payload = json.loads(semantic_contract.read_text(encoding="utf-8"))
    if set(payload) != {"schema_version", "required_phrases"}:
        raise ManualRendererError("semantic contract fields differ")
    if payload["schema_version"] != "tools-manual-semantics/1.0.0":
        raise ManualRendererError("unsupported semantic contract schema version")
    phrases = payload["required_phrases"]
    if (
        not isinstance(phrases, list)
        or not phrases
        or not all(isinstance(item, str) and item.strip() for item in phrases)
    ):
        raise ManualRendererError("required_phrases must be non-empty strings")
    canonical_phrases = tuple(canonical_semantic_text(item) for item in phrases)
    for name, path in artifacts.items():
        visible = canonical_semantic_text(extract_visible_text(path, lock, workspace))
        missing = [phrase for phrase in canonical_phrases if phrase not in visible]
        if missing:
            raise ManualRendererError(
                f"{name} semantic parity failed; missing phrases: {missing}"
            )
    joined = "\n".join(canonical_phrases).encode("utf-8")
    return sha256_bytes(joined)


def artifact_payload(
    artifacts: Mapping[str, Path],
    semantic_sha256: str,
    source_sha256_lf: str,
    toolchain_sha256_lf: str,
) -> dict[str, object]:
    """Build the strict unapproved artifact-manifest payload."""
    rows = []
    for name in sorted(artifacts):
        path = artifacts[name]
        value = path.read_bytes()
        rows.append(
            {
                "format": name,
                "path": f"manuals/tools/dist/tools-engineering-design-manual.{name}",
                "media_type": MEDIA_TYPES[name],
                "bytes": len(value),
                "sha256": sha256_bytes(value),
                "semantic_sha256": semantic_sha256,
            }
        )
    return {
        "schema_version": "tools-manual-artifacts/1.0.0",
        "manual_id": "tools",
        "release_status": "generated-unapproved",
        "source_commit": None,
        "source_sha256_lf": source_sha256_lf,
        "toolchain_lock_sha256_lf": toolchain_sha256_lf,
        "semantic_sha256": semantic_sha256,
        "semantic_parity": "verified",
        "owner": "Tools documentation epic #4707",
        "review_owner": "TOOLS-D7 and TOOLS-D8 reviewers",
        "publication_approval": "blocked-pending-TOOLS-D7-D8",
        "blockers": [
            "Stable calculation pathways remain pending TOOLS-D3.",
            "Accessibility and page-review approval remain pending TOOLS-D7.",
            "Public projection and human approval remain pending TOOLS-D8.",
        ],
        "artifacts": rows,
    }


def write_manifest(payload: Mapping[str, object], path: Path) -> ArtifactManifest:
    """Validate then write one LF-normalized deterministic manifest."""
    view = load_artifact_manifest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return view


def reset_directory(path: Path) -> None:
    """Replace a renderer-owned output directory without touching other paths."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
