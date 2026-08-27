"""Strict consumer contracts for Tools manual rendering and artifacts."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

TOOLCHAIN_LOCK_SCHEMA_VERSION = "tools-manual-toolchain/1.0.0"
ARTIFACT_MANIFEST_SCHEMA_VERSION = "tools-manual-artifacts/1.0.0"
REQUIRED_ARTIFACT_FORMATS = ("docx", "html", "pdf", "tex")
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
INCLUDE_PATTERN = re.compile(r"^\s*\{\{<\s*include\s+([^ >]+)\s*>\}\}\s*$")


class ManualRendererError(RuntimeError):
    """Raised when a renderer input or generated artifact fails closed."""


@dataclass(frozen=True)
class ToolCommand:
    """One exact external-tool command contract."""

    executable: str
    version_args: tuple[str, ...]
    exact_version: str
    version_output: str


@dataclass(frozen=True)
class ToolchainLock:
    """Validated immutable renderer-input and external-tool contract."""

    schema_version: str
    canonical_source: str
    bibliography: str
    semantic_contract: str
    reference_docx: str
    style_files: tuple[str, ...]
    figure_files: tuple[str, ...]
    input_sha256: Mapping[str, str]
    source_date_epoch: int
    commands: Mapping[str, ToolCommand]


@dataclass(frozen=True)
class Artifact:
    """Digest and semantic binding for one rendered representation."""

    format: str
    path: str
    media_type: str
    bytes: int
    sha256: str
    semantic_sha256: str


@dataclass(frozen=True)
class ArtifactManifest:
    """Validated view of a generated-but-unapproved artifact set."""

    schema_version: str
    manual_id: str
    release_status: str
    source_commit: str | None
    source_sha256_lf: str
    toolchain_lock_sha256_lf: str
    semantic_sha256: str
    semantic_parity: str
    owner: str
    review_owner: str
    publication_approval: str
    blockers: tuple[str, ...]
    artifacts: Mapping[str, Artifact]


def _require_fields(
    payload: Mapping[str, object], expected: set[str], label: str
) -> None:
    actual = set(payload)
    if actual != expected:
        raise ManualRendererError(
            f"{label} fields differ: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ManualRendererError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManualRendererError(f"{label} must be non-empty text")
    return value


def _relative_path(value: object, label: str) -> str:
    text = _text(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise ManualRendererError(f"{label} must be a normalized relative path")
    return text


def _sha256(value: object, label: str) -> str:
    text = _text(value, label)
    if HASH_PATTERN.fullmatch(text) is None:
        raise ManualRendererError(f"{label} must be lowercase SHA-256")
    return text


def _command(value: object, label: str) -> ToolCommand:
    payload = _mapping(value, label)
    _require_fields(
        payload,
        {"executable", "version_args", "exact_version", "version_output"},
        label,
    )
    args = payload["version_args"]
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise ManualRendererError(f"{label}.version_args must be a string array")
    return ToolCommand(
        executable=_text(payload["executable"], f"{label}.executable"),
        version_args=tuple(args),
        exact_version=_text(payload["exact_version"], f"{label}.exact_version"),
        version_output=_text(payload["version_output"], f"{label}.version_output"),
    )


def load_toolchain_lock(payload: Mapping[str, object]) -> ToolchainLock:
    """Validate and return the only supported toolchain-lock version."""
    expected = {
        "schema_version",
        "canonical_source",
        "bibliography",
        "semantic_contract",
        "reference_docx",
        "style_files",
        "figure_files",
        "input_sha256",
        "source_date_epoch",
        "commands",
    }
    _require_fields(payload, expected, "toolchain lock")
    if payload["schema_version"] != TOOLCHAIN_LOCK_SCHEMA_VERSION:
        raise ManualRendererError("unsupported toolchain lock schema version")
    style_files = payload["style_files"]
    if not isinstance(style_files, list) or not style_files:
        raise ManualRendererError("style_files must be a non-empty array")
    styles = tuple(_relative_path(item, "style_files entry") for item in style_files)
    figure_files = payload["figure_files"]
    if not isinstance(figure_files, list) or not figure_files:
        raise ManualRendererError("figure_files must be a non-empty array")
    figures = tuple(_relative_path(item, "figure_files entry") for item in figure_files)
    input_payload = _mapping(payload["input_sha256"], "input_sha256")
    inputs = {
        _relative_path(path, "input_sha256 path"): _sha256(
            digest, "input_sha256 digest"
        )
        for path, digest in input_payload.items()
    }
    expected_inputs = {
        _relative_path(payload["bibliography"], "bibliography"),
        _relative_path(payload["semantic_contract"], "semantic_contract"),
        _relative_path(payload["reference_docx"], "reference_docx"),
        *styles,
        *figures,
    }
    if set(inputs) != expected_inputs:
        raise ManualRendererError("input_sha256 must bind every non-QMD renderer input")
    epoch = payload["source_date_epoch"]
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= 0:
        raise ManualRendererError("source_date_epoch must be a positive integer")
    command_payload = _mapping(payload["commands"], "commands")
    required_commands = {"pandoc", "pdflatex", "quarto"}
    if set(command_payload) != required_commands:
        raise ManualRendererError(
            "commands must define exactly pandoc, pdflatex, quarto"
        )
    commands = {
        name: _command(command_payload[name], f"commands.{name}")
        for name in sorted(command_payload)
    }
    return ToolchainLock(
        schema_version=TOOLCHAIN_LOCK_SCHEMA_VERSION,
        canonical_source=_relative_path(
            payload["canonical_source"], "canonical_source"
        ),
        bibliography=_relative_path(payload["bibliography"], "bibliography"),
        semantic_contract=_relative_path(
            payload["semantic_contract"], "semantic_contract"
        ),
        reference_docx=_relative_path(payload["reference_docx"], "reference_docx"),
        style_files=styles,
        figure_files=figures,
        input_sha256=inputs,
        source_date_epoch=epoch,
        commands=commands,
    )


VersionRunner = Callable[[str, Sequence[str]], str]


def verify_toolchain(lock: ToolchainLock, runner: VersionRunner) -> None:
    """Require every external tool to report its locked exact version."""
    for name, command in lock.commands.items():
        try:
            output = runner(command.executable, command.version_args)
        except (FileNotFoundError, OSError) as exc:
            raise ManualRendererError(f"{name} unavailable") from exc
        first_line = output.replace("\r\n", "\n").splitlines()[0].strip()
        if first_line != command.version_output:
            raise ManualRendererError(
                f"{name} version mismatch: expected {command.exact_version!r}, "
                f"received {first_line!r}"
            )


def _artifact(value: object, label: str) -> Artifact:
    payload = _mapping(value, label)
    expected = {"format", "path", "media_type", "bytes", "sha256", "semantic_sha256"}
    _require_fields(payload, expected, label)
    size = payload["bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ManualRendererError(f"{label}.bytes must be positive")
    return Artifact(
        format=_text(payload["format"], f"{label}.format"),
        path=_relative_path(payload["path"], f"{label}.path"),
        media_type=_text(payload["media_type"], f"{label}.media_type"),
        bytes=size,
        sha256=_sha256(payload["sha256"], f"{label}.sha256"),
        semantic_sha256=_sha256(payload["semantic_sha256"], f"{label}.semantic_sha256"),
    )


def load_artifact_manifest(payload: Mapping[str, object]) -> ArtifactManifest:
    """Validate a complete unapproved artifact manifest for consumers."""
    expected = {
        "schema_version",
        "manual_id",
        "release_status",
        "source_commit",
        "source_sha256_lf",
        "toolchain_lock_sha256_lf",
        "semantic_sha256",
        "semantic_parity",
        "owner",
        "review_owner",
        "publication_approval",
        "blockers",
        "artifacts",
    }
    _require_fields(payload, expected, "artifact manifest")
    if payload["schema_version"] != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise ManualRendererError("unsupported artifact manifest schema version")
    if payload["release_status"] != "generated-unapproved":
        raise ManualRendererError("release_status must remain generated-unapproved")
    if payload["semantic_parity"] != "verified":
        raise ManualRendererError("semantic_parity must be verified")
    source_commit = payload["source_commit"]
    if source_commit is not None and (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
    ):
        raise ManualRendererError("source_commit must be null or lowercase Git SHA")
    artifacts_value = payload["artifacts"]
    if not isinstance(artifacts_value, list):
        raise ManualRendererError("artifacts must be an array")
    artifacts = {
        artifact.format: artifact
        for index, item in enumerate(artifacts_value)
        for artifact in [_artifact(item, f"artifacts[{index}]")]
    }
    if tuple(sorted(artifacts)) != REQUIRED_ARTIFACT_FORMATS or len(artifacts) != len(
        artifacts_value
    ):
        raise ManualRendererError("artifacts must contain exactly docx, html, pdf, tex")
    semantic_sha256 = _sha256(payload["semantic_sha256"], "semantic_sha256")
    if any(item.semantic_sha256 != semantic_sha256 for item in artifacts.values()):
        raise ManualRendererError("artifact semantic digests must match the manifest")
    blockers = payload["blockers"]
    if (
        not isinstance(blockers, list)
        or not blockers
        or not all(isinstance(item, str) and item for item in blockers)
    ):
        raise ManualRendererError("blockers must be a non-empty string array")
    return ArtifactManifest(
        schema_version=ARTIFACT_MANIFEST_SCHEMA_VERSION,
        manual_id=_text(payload["manual_id"], "manual_id"),
        release_status="generated-unapproved",
        source_commit=source_commit,
        source_sha256_lf=_sha256(payload["source_sha256_lf"], "source_sha256_lf"),
        toolchain_lock_sha256_lf=_sha256(
            payload["toolchain_lock_sha256_lf"], "toolchain_lock_sha256_lf"
        ),
        semantic_sha256=semantic_sha256,
        semantic_parity="verified",
        owner=_text(payload["owner"], "owner"),
        review_owner=_text(payload["review_owner"], "review_owner"),
        publication_approval=_text(
            payload["publication_approval"], "publication_approval"
        ),
        blockers=tuple(blockers),
        artifacts=artifacts,
    )


def canonical_semantic_text(value: str) -> str:
    """Return the LF/whitespace-normalized visible semantic representation."""
    if not isinstance(value, str):
        raise TypeError("value must be str")
    return " ".join(value.replace("\r\n", "\n").replace("\r", "\n").split())


def materialize_canonical_source(
    repository_root: Path, lock: ToolchainLock, output_path: Path
) -> tuple[str, ...]:
    """Expand bounded QMD includes into one deterministic Pandoc input."""
    root = repository_root.resolve()
    manual_root = (root / "manuals" / "tools").resolve()
    visited: list[str] = []

    def expand(relative: str) -> str:
        source = (root / relative).resolve()
        if not source.is_relative_to(manual_root) or source.suffix != ".qmd":
            raise ManualRendererError(f"include path escapes canonical QMD: {relative}")
        normalized = source.relative_to(root).as_posix()
        if normalized in visited:
            raise ManualRendererError(f"duplicate or cyclic include path: {normalized}")
        if not source.is_file():
            raise ManualRendererError(f"missing canonical QMD: {normalized}")
        visited.append(normalized)
        rendered: list[str] = []
        for line in source.read_text(encoding="utf-8").splitlines():
            match = INCLUDE_PATTERN.fullmatch(line)
            if match is None:
                rendered.append(line)
                continue
            include = (source.parent / match.group(1)).resolve()
            if not include.is_relative_to(manual_root):
                raise ManualRendererError(
                    f"include path escapes canonical QMD: {match.group(1)}"
                )
            rendered.append(expand(include.relative_to(root).as_posix()))
        return "\n".join(rendered).rstrip() + "\n"

    text = expand(lock.canonical_source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="\n")
    return tuple(visited)
