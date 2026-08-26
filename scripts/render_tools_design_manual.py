#!/usr/bin/env python3
"""Render and verify the canonical Tools manual in four governed formats."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from scripts.tools_manual_artifacts import (
    artifact_payload,
    canonicalize_docx,
    canonicalize_pdf,
    normalize_text_artifact,
    reset_directory,
    run_checked,
    semantic_digest_for_artifacts,
    sha256_lf,
    write_manifest,
)
from scripts.tools_manual_renderer_contract import (
    ArtifactManifest,
    ManualRendererError,
    ToolchainLock,
    load_artifact_manifest,
    load_toolchain_lock,
    materialize_canonical_source,
    verify_toolchain,
)

ARTIFACT_STEM = "tools-engineering-design-manual"


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ManualRendererError(f"JSON document must be an object: {path}")
    return value


def _version_runner(executable: str, args: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            [executable, *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        raise OSError(executable) from exc
    return result.stdout or result.stderr


def _paths(root: Path) -> tuple[Path, Path, Path]:
    manual_root = root / "manuals" / "tools"
    return (
        manual_root,
        manual_root / "toolchain-lock.json",
        manual_root / "semantic-contract.json",
    )


def _pandoc_common(root: Path, lock: ToolchainLock, source: Path) -> list[str]:
    return [
        lock.commands["pandoc"].executable,
        str(source),
        "--standalone",
        "--from=markdown+tex_math_single_backslash",
        "--citeproc",
        f"--bibliography={root / lock.bibliography}",
        f"--resource-path={root / 'manuals' / 'tools'}",
        "--metadata=lang:en-US",
    ]


def _render_formats(
    root: Path, lock: ToolchainLock, source: Path, output: Path
) -> dict[str, Path]:
    manual_root, _, _ = _paths(root)
    environment = {
        "SOURCE_DATE_EPOCH": str(lock.source_date_epoch),
        "TZ": "UTC",
    }
    common = _pandoc_common(root, lock, source)
    artifacts = {
        name: output / f"{ARTIFACT_STEM}.{name}"
        for name in ("docx", "html", "pdf", "tex")
    }
    run_checked(
        [
            *common,
            "--to=html5",
            "--embed-resources",
            "--mathml",
            f"--css={manual_root / 'styles' / 'tools.css'}",
            f"--output={artifacts['html']}",
        ],
        cwd=manual_root,
        environment=environment,
    )
    run_checked(
        [
            *common,
            "--to=latex",
            f"--include-in-header={manual_root / 'styles' / 'tools-header.tex'}",
            f"--output={artifacts['tex']}",
        ],
        cwd=output,
        environment=environment,
    )
    os.utime(
        artifacts["tex"],
        (lock.source_date_epoch, lock.source_date_epoch),
    )
    run_checked(
        [
            *common,
            "--to=docx",
            f"--reference-doc={root / lock.reference_docx}",
            f"--output={artifacts['docx']}",
        ],
        cwd=output,
        environment=environment,
    )
    run_checked(
        [
            lock.commands["pdflatex"].executable,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-no-shell-escape",
            f"-job-time={artifacts['tex']}",
            f"-output-directory={output}",
            str(artifacts["tex"]),
        ],
        cwd=manual_root,
        environment=environment,
    )
    generated_pdf = output / f"{ARTIFACT_STEM}.pdf"
    if not generated_pdf.is_file():
        raise ManualRendererError("pdflatex did not produce the required PDF")
    for suffix in (".aux", ".log", ".out"):
        auxiliary = output / f"{ARTIFACT_STEM}{suffix}"
        if auxiliary.exists():
            auxiliary.unlink()
    return artifacts


def _verify_locked_inputs(root: Path, lock: ToolchainLock) -> None:
    for relative, expected in lock.input_sha256.items():
        path = root / relative
        if not path.is_file():
            raise ManualRendererError(f"locked renderer input is missing: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ManualRendererError(
                f"locked renderer input digest drifted: {relative}"
            )


def build_manual(repository_root: Path, output_dir: Path) -> ArtifactManifest:
    """Build deterministic outputs after exact-toolchain qualification."""
    root = repository_root.resolve()
    manual_root, lock_path, semantic_contract = _paths(root)
    lock = load_toolchain_lock(_read_json(lock_path))
    verify_toolchain(lock, _version_runner)
    _verify_locked_inputs(root, lock)
    reset_directory(output_dir)
    with tempfile.TemporaryDirectory(prefix="tools-manual-") as temporary_name:
        temporary = Path(temporary_name)
        source = temporary / "manual.qmd"
        materialize_canonical_source(root, lock, source)
        artifacts = _render_formats(root, lock, source, output_dir)
        normalize_text_artifact(artifacts["html"])
        normalize_text_artifact(artifacts["tex"])
        canonicalize_docx(artifacts["docx"])
        canonicalize_pdf(artifacts["pdf"])
        semantic_sha256 = semantic_digest_for_artifacts(
            artifacts, lock, semantic_contract, temporary
        )
        payload = artifact_payload(
            artifacts,
            semantic_sha256,
            sha256_lf(source),
            sha256_lf(lock_path),
        )
    return write_manifest(payload, output_dir / "artifacts.json")


def check_manual(repository_root: Path) -> ArtifactManifest:
    """Fail closed when checked-in artifacts or their semantic binding drift."""
    root = repository_root.resolve()
    manual_root, lock_path, semantic_contract = _paths(root)
    lock = load_toolchain_lock(_read_json(lock_path))
    _verify_locked_inputs(root, lock)
    manifest_path = manual_root / "manifests" / "artifacts.json"
    manifest = load_artifact_manifest(_read_json(manifest_path))
    artifacts = {name: root / item.path for name, item in manifest.artifacts.items()}
    for name, item in manifest.artifacts.items():
        path = artifacts[name]
        if not path.is_file() or path.stat().st_size != item.bytes:
            raise ManualRendererError(f"{name} artifact is missing or size-stale")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item.sha256:
            raise ManualRendererError(f"{name} artifact digest is stale")
    with tempfile.TemporaryDirectory(prefix="tools-manual-check-") as temporary_name:
        temporary = Path(temporary_name)
        source = temporary / "manual.qmd"
        materialize_canonical_source(root, lock, source)
        if sha256_lf(source) != manifest.source_sha256_lf:
            raise ManualRendererError("canonical QMD source digest is stale")
        if sha256_lf(lock_path) != manifest.toolchain_lock_sha256_lf:
            raise ManualRendererError("toolchain lock digest is stale")
        semantic = semantic_digest_for_artifacts(
            artifacts, lock, semantic_contract, temporary
        )
    if semantic != manifest.semantic_sha256:
        raise ManualRendererError("artifact semantic digest is stale")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--render", action="store_true")
    mode.add_argument("--verify-toolchain", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a renderer mode and return a process-compatible status."""
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        if args.check:
            check_manual(root)
        elif args.verify_toolchain:
            _, lock_path, _ = _paths(root)
            verify_toolchain(
                load_toolchain_lock(_read_json(lock_path)), _version_runner
            )
        else:
            output = args.output_dir or root / "manuals" / "tools" / "dist"
            manifest = build_manual(root, output)
            if output.resolve() == (root / "manuals" / "tools" / "dist").resolve():
                generated_manifest = output / "artifacts.json"
                target = root / "manuals" / "tools" / "manifests" / "artifacts.json"
                target.write_bytes(generated_manifest.read_bytes())
                generated_manifest.unlink()
            if manifest.release_status != "generated-unapproved":
                raise ManualRendererError("renderer attempted approval promotion")
    except (ManualRendererError, OSError, ValueError) as exc:
        sys.stderr.write(f"Tools manual renderer failed: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
