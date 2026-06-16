"""Import-boundary guard for the #3316 sidekick canonicalization slice."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

_ALLOWED_PREFIXES = {
    Path("src/shared/python/sidekick"),
    Path("src/shared/python/upstream_drift_tools"),
}

_FORBIDDEN_IMPORT_ROOTS = (
    "sidekick",
    "upstream_drift_tools",
    "src.shared.python.sidekick",
)


def _is_allowed_path(path: Path) -> bool:
    relative = path.relative_to(REPO_ROOT)
    if "tests" in relative.parts:
        return True
    return any(relative.is_relative_to(prefix) for prefix in _ALLOWED_PREFIXES)


def _root_matches(module: str) -> bool:
    return any(
        module == root or module.startswith(f"{root}.")
        for root in _FORBIDDEN_IMPORT_ROOTS
    )


def _sidekick_import_violations() -> list[str]:
    violations: list[str] = []
    for py_file in SRC_ROOT.rglob("*.py"):
        if _is_allowed_path(py_file):
            continue

        source = py_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if _root_matches(module):
                    violations.append(
                        f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}: {module}"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if _root_matches(alias.name):
                        violations.append(
                            f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}: "
                            f"{alias.name}"
                        )
    return violations


def test_external_production_code_uses_shared_python_sidekick_imports() -> None:
    """External production consumers must use one canonical Sidekick spelling."""
    assert not _sidekick_import_violations()
