"""Consumer contracts for the reproducible Tools design-manual renderer."""

from __future__ import annotations

import builtins
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from jsonschema import Draft202012Validator

from scripts.render_tools_design_manual import build_manual, main
from scripts.tools_manual_artifacts import canonicalize_docx
from scripts.tools_manual_renderer_contract import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    TOOLCHAIN_LOCK_SCHEMA_VERSION,
    ManualRendererError,
    canonical_semantic_text,
    load_artifact_manifest,
    load_toolchain_lock,
    materialize_canonical_source,
    verify_toolchain,
)

ROOT = Path(__file__).resolve().parents[2]
MANUAL_ROOT = ROOT / "manuals" / "tools"
DIST = MANUAL_ROOT / "dist"
MANIFEST = MANUAL_ROOT / "manifests" / "artifacts.json"
MANIFEST_SCHEMA = MANUAL_ROOT / "schemas" / "artifact-manifest.schema.json"
LOCK = MANUAL_ROOT / "toolchain-lock.json"
LOCK_SCHEMA = MANUAL_ROOT / "schemas" / "toolchain-lock.schema.json"
REQUIRED_FORMATS = ("docx", "html", "pdf", "tex")


def _payload(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_toolchain_lock_is_strict_versioned_and_complete() -> None:
    schema = _payload(LOCK_SCHEMA)
    payload = _payload(LOCK)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)

    view = load_toolchain_lock(payload)
    assert view.schema_version == TOOLCHAIN_LOCK_SCHEMA_VERSION
    assert view.canonical_source == "manuals/tools/index.qmd"
    assert view.reference_docx == "manuals/tools/styles/tools-reference.docx"
    assert set(view.commands) == {"pandoc", "pdflatex", "quarto"}
    assert all(command.exact_version for command in view.commands.values())
    assert view.source_date_epoch == 1_787_724_102
    assert set(view.input_sha256) == {
        view.bibliography,
        view.semantic_contract,
        view.reference_docx,
        *view.style_files,
        *view.figure_files,
    }


def test_toolchain_lock_rejects_unknown_fields_versions_and_unsafe_paths() -> None:
    payload = _payload(LOCK)
    with pytest.raises(ManualRendererError, match="schema version"):
        load_toolchain_lock({**payload, "schema_version": "tools/toolchain/2.0.0"})
    with pytest.raises(ManualRendererError, match="fields differ"):
        load_toolchain_lock({**payload, "fallback_renderer": "word"})
    with pytest.raises(ManualRendererError, match="normalized relative path"):
        load_toolchain_lock({**payload, "canonical_source": "../private.qmd"})


def test_toolchain_verification_is_exact_and_fail_closed() -> None:
    lock = load_toolchain_lock(_payload(LOCK))
    exact = {name: command.version_output for name, command in lock.commands.items()}
    verify_toolchain(lock, lambda name, _args: exact[name])

    drifted = dict(exact)
    drifted["pandoc"] = "pandoc 3.7.0\n"
    with pytest.raises(ManualRendererError, match="pandoc version mismatch"):
        verify_toolchain(lock, lambda name, _args: drifted[name])
    with pytest.raises(ManualRendererError, match="quarto unavailable"):
        verify_toolchain(
            lock,
            lambda name, _args: (
                exact[name]
                if name != "quarto"
                else (_ for _ in ()).throw(FileNotFoundError(name))
            ),
        )


def test_canonical_source_materialization_is_ordered_and_bounded(
    tmp_path: Path,
) -> None:
    lock = load_toolchain_lock(_payload(LOCK))
    output = tmp_path / "manual.qmd"
    sources = materialize_canonical_source(ROOT, lock, output)

    assert sources == (
        "manuals/tools/index.qmd",
        "manuals/tools/chapters/00-governance.qmd",
        "manuals/tools/chapters/01-module-inventory.qmd",
        "manuals/tools/chapters/02-reproducible-rendering.qmd",
    )
    text = output.read_text(encoding="utf-8")
    assert "{{< include" not in text
    assert text.index("# Documentation Authority") < text.index("# Module and")
    assert text.index("# Module and") < text.index("# Reproducible Rendering")


def test_materializer_rejects_missing_or_traversing_includes(tmp_path: Path) -> None:
    manual_root = tmp_path / "manuals" / "tools"
    manual_root.mkdir(parents=True)
    (manual_root / "index.qmd").write_text(
        "{{< include ../../private.qmd >}}\n", encoding="utf-8"
    )
    lock_payload = _payload(LOCK)
    lock_payload["canonical_source"] = "manuals/tools/index.qmd"
    lock = load_toolchain_lock(lock_payload)
    with pytest.raises(ManualRendererError, match="include path"):
        materialize_canonical_source(tmp_path, lock, tmp_path / "out.qmd")


def test_artifact_manifest_is_strict_complete_and_digest_bound() -> None:
    schema = _payload(MANIFEST_SCHEMA)
    payload = _payload(MANIFEST)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    view = load_artifact_manifest(payload)

    assert view.schema_version == ARTIFACT_MANIFEST_SCHEMA_VERSION
    assert tuple(sorted(view.artifacts)) == REQUIRED_FORMATS
    assert view.release_status == "generated-unapproved"
    assert view.semantic_parity == "verified"
    assert view.publication_approval == "blocked-pending-TOOLS-D7-D8"
    assert view.source_commit is None
    assert view.owner == "Tools documentation epic #4707"
    assert view.review_owner == "TOOLS-D7 and TOOLS-D8 reviewers"

    semantic_digests = set()
    for name, artifact in view.artifacts.items():
        path = ROOT / artifact.path
        assert path == DIST / f"tools-engineering-design-manual.{name}"
        assert path.is_file() and path.stat().st_size == artifact.bytes
        assert _digest(path) == artifact.sha256
        semantic_digests.add(artifact.semantic_sha256)
    assert semantic_digests == {view.semantic_sha256}


def test_artifact_manifest_rejects_drift_and_authority_promotion() -> None:
    payload = _payload(MANIFEST)
    with pytest.raises(ManualRendererError, match="schema version"):
        load_artifact_manifest({**payload, "schema_version": "tools/artifacts/2"})
    with pytest.raises(ManualRendererError, match="fields differ"):
        load_artifact_manifest({**payload, "approved": True})
    with pytest.raises(ManualRendererError, match="generated-unapproved"):
        load_artifact_manifest({**payload, "release_status": "approved"})

    missing = {**payload, "artifacts": payload["artifacts"][:-1]}
    with pytest.raises(ManualRendererError, match="exactly"):
        load_artifact_manifest(missing)


def test_semantic_normalization_preserves_units_warnings_and_figure_text() -> None:
    left = "Speed: 10 m/s\r\nWARNING: provisional\nFigure 1: Pipeline"
    right = "  Speed: 10 m/s  WARNING: provisional Figure 1: Pipeline  "
    expected = "Speed: 10 m/s WARNING: provisional Figure 1: Pipeline"
    assert canonical_semantic_text(left) == expected
    assert canonical_semantic_text(right) == expected


def test_docx_canonicalization_removes_workspace_bibliography_path(
    tmp_path: Path,
) -> None:
    paths = []
    for name, bibliography in (
        ("left", r"C:\worktree-a\manuals\tools\references.bib"),
        ("right", "/worktree-b/manuals/tools/references.bib"),
    ):
        path = tmp_path / f"{name}.docx"
        custom = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/'
            'officeDocument/2006/custom-properties" '
            'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/'
            '2006/docPropsVTypes">'
            '<property fmtid="{D5CDD505-2E9C-101B-9397-08002B2CF9AE}" '
            'pid="2" name="bibliography"><vt:lpwstr>'
            f"{bibliography}"
            "</vt:lpwstr></property></Properties>"
        )
        with ZipFile(path, "w", compression=ZIP_DEFLATED) as package:
            package.writestr("docProps/custom.xml", custom)
        canonicalize_docx(path)
        paths.append(path)

    assert paths[0].read_bytes() == paths[1].read_bytes()
    with ZipFile(paths[0]) as package:
        custom = package.read("docProps/custom.xml").decode("utf-8")
    assert "manuals/tools/references.bib" in custom
    assert "worktree-a" not in custom


def test_docx_artifact_helpers_import_without_optional_pdf_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-PDF consumers must not require the documentation-only PDF stack."""
    module_name = "scripts.tools_manual_artifacts"
    original_import = builtins.__import__

    def reject_pypdf(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "pypdf" or name.startswith("pypdf."):
            raise ModuleNotFoundError("pypdf intentionally unavailable")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_pypdf)
    sys.modules.pop(module_name, None)
    imported = importlib.import_module(module_name)
    assert callable(imported.canonicalize_docx)


def test_checked_in_artifacts_are_fresh_and_semantically_equivalent() -> None:
    assert main(["--check"]) == 0


@pytest.mark.integration
def test_renderer_is_byte_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = build_manual(ROOT, first)
    second_manifest = build_manual(ROOT, second)

    assert first_manifest.semantic_sha256 == second_manifest.semantic_sha256
    for name in REQUIRED_FORMATS:
        first_path = first / f"tools-engineering-design-manual.{name}"
        second_path = second / f"tools-engineering-design-manual.{name}"
        assert first_path.read_bytes() == second_path.read_bytes()
