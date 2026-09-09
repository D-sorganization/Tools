"""Validated semantic catalog with explicit source and interaction evidence."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import CatalogError, read_text, safe_path

CATALOG_PATH = "docs/agent_context/catalog.json"
CONTRACT_HEADINGS = (
    "Responsibilities",
    "Data Contract",
    "Lifecycle and Failures",
    "Evidence",
    "Rationale",
)


@dataclass(frozen=True)
class Reference:
    """An exact Python public symbol whose declaration is read statically."""

    path: str
    symbol: str


@dataclass(frozen=True)
class Component:
    """A maintained subsystem, with mechanical facts linked to original files."""

    id: str
    title: str
    summary: str
    owner: str
    status: str
    sources: tuple[str, ...]
    documentation: tuple[str, ...]
    tests: tuple[str, ...]
    entrypoints: tuple[Reference, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True)
class Relation:
    """A directed semantic interaction; source imports alone do not prove it."""

    id: str
    provider: str
    consumer: str
    kind: str
    contract: str
    inputs: tuple[str, ...]
    tests: tuple[str, ...]


@dataclass(frozen=True)
class Catalog:
    """Validated configuration for one explicit repository checkout."""

    root: Path
    path: str
    repository: str
    components: tuple[Component, ...]
    relations: tuple[Relation, ...]
    inventories: tuple[dict[str, str], ...]
    dependencies: tuple[str, ...]
    source_hash: str


def known_fields(row: dict[str, Any], allowed: set[str]) -> None:
    """Reject misspelled metadata rather than silently weakening coverage."""
    unknown = set(row) - allowed
    if unknown:
        raise CatalogError(f"Unknown catalog fields: {sorted(unknown)}")


def text(value: Any, field: str) -> str:
    """Require a nonempty single value instead of silently coercing metadata."""
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{field} must be a nonempty string")
    return value


def strings(value: Any, field: str, *, required: bool = False) -> tuple[str, ...]:
    """Require unique string lists, optionally with at least one entry."""
    if not isinstance(value, list) or (required and not value):
        raise CatalogError(f"{field} must be {'nonempty ' if required else ''}list")
    result = tuple(text(item, field) for item in value)
    if len(set(result)) != len(result):
        raise CatalogError(f"duplicate value in {field}")
    return result


def records(value: Any, field: str) -> list[dict[str, Any]]:
    """Require a list of objects before processing catalog records."""
    if not isinstance(value, list) or any(not isinstance(x, dict) for x in value):
        raise CatalogError(f"{field} must be a list of objects")
    return value


def identifier(value: Any) -> str:
    """Stable IDs are safe in anchors, CLI selectors and graph references."""
    result = text(value, "id")
    if not re.fullmatch(r"[a-z][a-z0-9_.-]*", result):
        raise CatalogError(f"Invalid id: {result}")
    return result


def symbol_lines(root: Path, ref: Reference) -> tuple[int, int]:
    """Locate declarations through AST inspection, never source imports."""
    path = safe_path(root, ref.path)
    if path.suffix != ".py":
        raise CatalogError(f"Symbol resolution supports Python only: {ref.path}")
    try:
        tree = ast.parse(read_text(path), filename=ref.path)
    except SyntaxError as exc:
        raise CatalogError(f"Cannot parse {ref.path}: {exc.msg}") from exc
    body = tree.body
    found: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for part in ref.symbol.split("."):
        found = next(
            (
                n
                for n in body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == part
            ),
            None,
        )
        if found is None:
            raise CatalogError(f"Missing symbol {ref.symbol} in {ref.path}")
        body = getattr(found, "body", [])
    assert found is not None
    return found.lineno, found.end_lineno or found.lineno


def _component(root: Path, row: dict[str, Any]) -> Component:
    known_fields(row, set(Component.__dataclass_fields__))
    args = {
        key: text(row.get(key), key) for key in ("title", "summary", "owner", "status")
    }
    if args["status"] not in {"implemented", "partial", "proposed", "deprecated"}:
        raise CatalogError(f"Unknown component status: {args['status']}")
    paths = {
        key: strings(row.get(key), key, required=True)
        for key in ("sources", "documentation", "tests")
    }
    for key, values in paths.items():
        for value in values:
            safe_path(root, value, directory=key == "sources")
    refs = tuple(
        Reference(text(r.get("path"), "path"), text(r.get("symbol"), "symbol"))
        for r in records(row.get("entrypoints"), "entrypoints")
    )
    for ref in refs:
        symbol_lines(root, ref)
    return Component(
        identifier(row.get("id")),
        title=args["title"],
        summary=args["summary"],
        owner=args["owner"],
        status=args["status"],
        sources=paths["sources"],
        documentation=paths["documentation"],
        tests=paths["tests"],
        entrypoints=refs,
        tags=strings(row.get("tags"), "tags"),
    )


def _relation(root: Path, row: dict[str, Any], ids: set[str]) -> Relation:
    known_fields(row, set(Relation.__dataclass_fields__))
    args = {
        key: text(row.get(key), key)
        for key in ("provider", "consumer", "kind", "contract")
    }
    for endpoint in (args["provider"], args["consumer"]):
        if endpoint not in ids:
            raise CatalogError(f"Relation has unknown endpoint: {endpoint}")
    if args["provider"] == args["consumer"]:
        raise CatalogError("A relation must connect distinct components")
    contract = read_text(safe_path(root, args["contract"]))
    for heading in CONTRACT_HEADINGS:
        match = re.search(rf"(?m)^## {re.escape(heading)}\s*\n([^#]+)", contract)
        if match is None or not match.group(1).strip():
            raise CatalogError(
                f"{args['contract']}: missing populated {heading} section"
            )
    paths = {
        key: strings(row.get(key), key, required=True) for key in ("inputs", "tests")
    }
    for values in paths.values():
        for value in values:
            safe_path(root, value)
    return Relation(
        identifier(row.get("id")),
        args["provider"],
        args["consumer"],
        args["kind"],
        args["contract"],
        paths["inputs"],
        paths["tests"],
    )


def _inventories(root: Path, data: dict[str, Any]) -> list[dict[str, str]]:
    """Validate references to existing machine-readable registries."""
    inventories = []
    for row in records(data.get("inventories", []), "inventories"):
        entry = {key: text(row.get(key), key) for key in ("path", "key", "title")}
        safe_path(root, entry["path"])
        inventories.append(entry)
    return inventories


def load_catalog(root: Path, path: str = CATALOG_PATH) -> Catalog:
    """Load a complete validated catalog or fail with an actionable error."""
    root = root.resolve()
    content = read_text(safe_path(root, path))
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise CatalogError(f"Invalid catalog JSON: {exc}") from exc
    if (
        not isinstance(data, dict)
        or type(data.get("version")) is not int
        or data["version"] != 1
    ):
        raise CatalogError("Unsupported catalog version; expected integer 1")
    known_fields(
        data,
        {
            "version",
            "repository",
            "components",
            "relations",
            "inventories",
            "dependencies",
        },
    )
    components = tuple(
        _component(root, row) for row in records(data.get("components"), "components")
    )
    ids = {c.id for c in components}
    if not ids or len(ids) != len(components):
        raise CatalogError("Empty catalog or duplicate component id")
    relations = tuple(
        _relation(root, row, ids) for row in records(data.get("relations"), "relations")
    )
    if len({r.id for r in relations}) != len(relations):
        raise CatalogError("duplicate relation id")
    inventories = _inventories(root, data)
    dependencies = strings(data.get("dependencies", []), "dependencies")
    for dependency in dependencies:
        safe_path(root, dependency, directory=True)
    return Catalog(
        root,
        path,
        text(data.get("repository"), "repository"),
        components,
        relations,
        tuple(inventories),
        dependencies,
        hashlib.sha256(content.encode()).hexdigest(),
    )
