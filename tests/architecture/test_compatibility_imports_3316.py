"""Import-boundary guard for the #3316 compatibility canonicalization slice."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = (
    REPO_ROOT / "src" / "p1am_control_system",
    REPO_ROOT / "src" / "shared" / "python" / "ai",
    REPO_ROOT / "src" / "shared" / "python" / "calc_backend",
)


def _source_files() -> list[Path]:
    files: list[Path] = []
    for source_root in SOURCE_ROOTS:
        files.extend(source_root.rglob("*.py"))
    return sorted(files)


def _compatibility_alias_violations() -> list[str]:
    violations: list[str] = []
    for py_file in _source_files():
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "compatibility":
                violations.append(
                    f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}: "
                    "from compatibility import ..."
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "compatibility":
                        violations.append(
                            f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}: "
                            "import compatibility"
                        )
            elif (
                isinstance(node, ast.Constant)
                and node.value == "src.shared.python.compatibility"
            ):
                violations.append(
                    f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}: "
                    "dynamic import of src.shared.python.compatibility"
                )
    return violations


def test_internal_sources_use_canonical_compatibility_imports() -> None:
    """Internal code should avoid duplicate compatibility module aliases."""
    assert not _compatibility_alias_violations()
