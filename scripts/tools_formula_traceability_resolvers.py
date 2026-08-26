"""Repository-path, public-symbol, and exact-test resolvers for manuals."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

CALCULATION_REGISTRY_PATH = PurePosixPath("manuals/tools/calculation-registry.json")
CHAPTER_CONTRACT_PATH = PurePosixPath("manuals/tools/textbook-chapter-contract.json")
CHAPTER_REGISTRY_PATH = PurePosixPath("manuals/tools/textbook-chapters.json")
EXEMPLAR_COVERAGE_PATH = PurePosixPath("manuals/tools/exemplar-coverage.json")
ARTIFACT_MANIFEST_PATH = PurePosixPath("manuals/tools/manifests/artifacts.json")


class FormulaTraceabilityError(RuntimeError):
    """Raised when a formula family is incomplete, stale, or orphaned."""


@dataclass(frozen=True)
class TraceabilityDocuments:
    """Injectable authority documents used by the deterministic verifier."""

    chapter_contract: dict[str, object]
    chapter_registry: dict[str, object]
    calculation_registry: dict[str, object]
    exemplar_coverage: dict[str, object]
    artifact_manifest: dict[str, object]


@dataclass(frozen=True)
class ChapterTraceabilityContext:
    """Repository and chapter-level authorities shared by family resolvers."""

    root: Path
    chapter_text: str
    examples: frozenset[str]
    rendered: frozenset[str]


@dataclass(frozen=True)
class FormulaResolverContext:
    """Family-local authorities shared by formula resolvers."""

    root: Path
    claims: frozenset[str]
    anchors: frozenset[str]
    examples: frozenset[str]
    rendered: frozenset[str]


def load_authority_document(path: Path) -> dict[str, object]:
    """Load one UTF-8 JSON object or return a typed failure."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormulaTraceabilityError(f"authority cannot be loaded: {path}") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise FormulaTraceabilityError("authority must be an object")
    return value


def load_traceability_documents(repository_root: Path) -> TraceabilityDocuments:
    """Load all owning authorities from normalized repository paths."""
    root = repository_root.resolve()

    def load(relative: PurePosixPath) -> dict[str, object]:
        return load_authority_document(root.joinpath(*relative.parts))

    return TraceabilityDocuments(
        chapter_contract=load(CHAPTER_CONTRACT_PATH),
        chapter_registry=load(CHAPTER_REGISTRY_PATH),
        calculation_registry=load(CALCULATION_REGISTRY_PATH),
        exemplar_coverage=load(EXEMPLAR_COVERAGE_PATH),
        artifact_manifest=load(ARTIFACT_MANIFEST_PATH),
    )


def safe_repository_path(value: object, label: str) -> PurePosixPath:
    """Return a normalized relative path or fail closed."""
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise FormulaTraceabilityError(f"{label} must be a trimmed non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise FormulaTraceabilityError(f"{label} must be a normalized relative path")
    if path.as_posix() != value:
        raise FormulaTraceabilityError(f"{label} must be a normalized relative path")
    return path


def assert_public_symbol(root: Path, path: str, symbol: str) -> None:
    """Resolve one public top-level symbol through static AST evidence."""
    source = root.joinpath(*PurePosixPath(path).parts)
    if symbol.startswith("_"):
        raise FormulaTraceabilityError(f"public symbol is private: {path}::{symbol}")
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise FormulaTraceabilityError(
            f"public symbol source is unreadable: {path}"
        ) from error
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if symbol not in names:
        raise FormulaTraceabilityError(f"public symbol is missing: {path}::{symbol}")


def assert_test_target(root: Path, target: str) -> None:
    """Resolve an exact module, function, or class-method pytest node ID."""
    parts = target.split("::")
    path = safe_repository_path(parts[0], "test path")
    if len(parts) not in {2, 3}:
        raise FormulaTraceabilityError(f"test target is missing: {target}")
    try:
        tree = ast.parse(root.joinpath(*path.parts).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise FormulaTraceabilityError(f"test target is missing: {target}") from error
    nodes: list[ast.AST] = list(tree.body)
    for name in parts[1:]:
        match = next(
            (
                node
                for node in nodes
                if isinstance(
                    node,
                    (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                )
                and node.name == name
            ),
            None,
        )
        if match is None:
            raise FormulaTraceabilityError(f"test target is missing: {target}")
        nodes = list(match.body) if isinstance(match, ast.ClassDef) else []
