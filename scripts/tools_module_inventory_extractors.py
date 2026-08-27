"""Repository discovery and static extraction for the Tools module inventory."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

LANGUAGES = {
    ".c": "c",
    ".cc": "cpp",
    ".cjs": "javascript",
    ".cpp": "cpp",
    ".css": "css",
    ".cxx": "cpp",
    ".h": "c-header",
    ".hpp": "cpp-header",
    ".html": "html",
    ".hxx": "cpp-header",
    ".ini": "ini",
    ".ino": "arduino",
    ".js": "javascript",
    ".json": "json",
    ".m": "matlab",
    ".mjs": "javascript",
    ".prisma": "prisma",
    ".ps1": "powershell",
    ".py": "python",
    ".pyi": "python-stub",
    ".rs": "rust",
    ".sh": "shell",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "typescript-react",
    ".urdf": "urdf",
    ".webmanifest": "webmanifest",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
}
CONFIG_SUFFIXES = frozenset(
    {".ini", ".json", ".toml", ".urdf", ".xml", ".yaml", ".yml"}
)
CODE_SUFFIXES = frozenset(LANGUAGES) - CONFIG_SUFFIXES
EXCLUDED_PARTS = frozenset(
    {
        ".agents",
        ".gaai",
        ".github",
        ".venv",
        "archive",
        "build",
        "dist",
        "docs",
        "examples",
        "experimental",
        "legacy",
        "manuals",
        "node_modules",
        "replicants",
        "resources",
        "test",
        "tests",
        "vendor",
        "venv",
    }
)
TEST_PARTS = frozenset({"test", "tests"})
EXPORT_PATTERN = re.compile(
    r"\bexport\s+(?:default\s+)?(?:async\s+)?"
    r"(?:function|class|const|let|var|interface|type|enum)\s+"
    r"([A-Za-z_$][\w$]*)"
)
RUST_PUBLIC_PATTERN = re.compile(
    r"^\s*pub\s+(?:async\s+)?(fn|struct|enum|trait|type|const)\s+"
    r"([A-Za-z_]\w*)",
    re.MULTILINE,
)
MATLAB_FUNCTION_PATTERN = re.compile(
    r"^\s*function\s+(?:\[[^]]+\]|\w+\s*=\s*)?([A-Za-z_]\w*)", re.MULTILINE
)
COMMON_STEMS = frozenset(
    {
        "app",
        "base",
        "cli",
        "config",
        "constants",
        "core",
        "main",
        "models",
        "types",
        "utils",
    }
)


@dataclass(frozen=True)
class TestIndex:
    """Read-once lookup maps for conservative module-to-test traceability."""

    by_stem: dict[str, tuple[str, ...]]
    by_token: dict[str, tuple[str, ...]]


def _unique_surfaces(items: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return public surfaces in the canonical consumer key order."""
    keyed = {
        f"{item['kind']}:{item['name']}:{item['signature'] or ''}": item
        for item in items
    }
    return [keyed[key] for key in sorted(keyed)]


def tracked_files(root: Path) -> tuple[Path, ...]:
    """Return tracked paths; untracked workspace content cannot enter authority."""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return tuple(
        Path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item
    )


def is_test(path: Path) -> bool:
    """Return whether a path is test evidence rather than an implementation module."""
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return (
        bool(parts & TEST_PARTS)
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _is_governed(path: Path) -> bool:
    lower_parts = {part.lower() for part in path.parts}
    if lower_parts & EXCLUDED_PARTS or is_test(path):
        return False
    suffix = path.suffix.lower()
    if suffix in CODE_SUFFIXES:
        return True
    return suffix in CONFIG_SUFFIXES and path.parts[0].lower() in {"config", "schemas"}


def discover_governed_paths(root: Path) -> tuple[Path, ...]:
    """Discover every tracked implementation or governed configuration module."""
    paths = {
        path
        for path in tracked_files(root)
        if _is_governed(path) and (root / path).is_file()
    }
    return tuple(sorted(paths, key=lambda item: item.as_posix()))


def normalized_bytes(path: Path) -> bytes:
    """Normalize all text line endings before hashing or byte accounting."""
    raw = path.read_bytes()
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _public_python(
    text: str,
) -> tuple[list[dict[str, object]], str | None, bool]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [], None, True
    surfaces: list[dict[str, object]] = []
    imported: dict[str, dict[str, object]] = {}
    declared_exports: set[str] | None = None
    for node in tree.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and not node.name.startswith("_"):
            surfaces.append(
                {
                    "kind": "function",
                    "name": node.name,
                    "signature": ast.unparse(node.args),
                }
            )
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            surfaces.append({"kind": "class", "name": node.name, "signature": None})
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.asname or alias.name.rsplit(".", 1)[-1]
                if not name.startswith("_"):
                    imported[name] = {
                        "kind": "re-export",
                        "name": name,
                        "signature": None,
                    }
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            value = node.value
            if isinstance(target, ast.Name) and target.id == "__all__":
                if isinstance(value, (ast.List, ast.Tuple)):
                    declared_exports = {
                        element.value
                        for element in value.elts
                        if isinstance(element, ast.Constant)
                        and isinstance(element.value, str)
                    }
            elif isinstance(target, ast.Name) and target.id.isupper():
                surfaces.append(
                    {"kind": "constant", "name": target.id, "signature": None}
                )
    if declared_exports is not None:
        surfaces = [item for item in surfaces if item["name"] in declared_exports]
        existing = {str(item["name"]) for item in surfaces}
        surfaces.extend(
            imported[name]
            for name in sorted(declared_exports - existing)
            if name in imported
        )
    purpose = ast.get_docstring(tree)
    first_line = purpose.splitlines()[0].strip() if purpose else None
    return _unique_surfaces(surfaces), first_line, False


def public_surfaces(
    path: Path, text: str
) -> tuple[list[dict[str, object]], str | None, bool]:
    """Extract public surfaces statically and report parse blockers."""
    suffix = path.suffix.lower()
    if suffix in {".py", ".pyi"}:
        return _public_python(text)
    if suffix in {".js", ".mjs", ".cjs", ".ts", ".tsx"}:
        items = [
            {"kind": "export", "name": name, "signature": None}
            for name in EXPORT_PATTERN.findall(text)
        ]
        return _unique_surfaces(items), None, False
    if suffix == ".rs":
        items = [
            {"kind": kind, "name": name, "signature": None}
            for kind, name in RUST_PUBLIC_PATTERN.findall(text)
        ]
        return _unique_surfaces(items), None, False
    if suffix == ".m":
        items = [
            {"kind": "function", "name": name, "signature": None}
            for name in MATLAB_FUNCTION_PATTERN.findall(text)
        ]
        return _unique_surfaces(items), None, False
    if suffix == ".json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return [], None, True
        if isinstance(value, dict) and isinstance(value.get("$id"), str):
            title = value.get("title") if isinstance(value.get("title"), str) else None
            surface = {"kind": "schema", "name": value["$id"], "signature": None}
            return [surface], title, False
    return [], None, False


def build_test_index(root: Path) -> TestIndex:
    """Index test stems and import-like tokens once for bounded lookup."""
    by_stem: dict[str, set[str]] = defaultdict(set)
    by_token: dict[str, set[str]] = defaultdict(set)
    for path in tracked_files(root):
        if not is_test(path) or path.suffix.lower() not in CODE_SUFFIXES:
            continue
        if not (root / path).is_file():
            continue
        relative = path.as_posix()
        stem = path.stem.lower().replace("-", "_")
        by_stem[stem.removeprefix("test_").removesuffix("_test")].add(relative)
        text = (root / path).read_text(encoding="utf-8", errors="replace")
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_./-]{3,}", text):
            by_token[token.replace("/", ".").replace("-", "_")].add(relative)
    return TestIndex(
        by_stem={key: tuple(sorted(value)) for key, value in by_stem.items()},
        by_token={key: tuple(sorted(value)) for key, value in by_token.items()},
    )


def _module_tokens(path: Path) -> set[str]:
    parts = list(path.with_suffix("").parts)
    tokens = {".".join(parts)}
    for marker in ("src", "python"):
        if marker in parts:
            tokens.add(".".join(parts[parts.index(marker) + 1 :]))
    if "shared" in parts:
        tokens.add(".".join(parts[parts.index("shared") :]))
    return {token.replace("-", "_") for token in tokens if token}


def mapped_test_paths(path: Path, index: TestIndex) -> list[str]:
    """Map conservative exact stem and import-token evidence to a module."""
    matches: set[str] = set()
    stem = path.stem.lower().replace("-", "_")
    if len(stem) >= 4 and stem not in COMMON_STEMS and stem != "__init__":
        matches.update(index.by_stem.get(stem, ()))
    for token in _module_tokens(path):
        matches.update(index.by_token.get(token, ()))
    return sorted(matches)
