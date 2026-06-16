"""Import-boundary guard for the #3316 AI core canonicalization slice."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_CORE_FILES = (
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "access_policy.py",
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "config.py",
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "memory_manager.py",
)


def _src_shared_import_violations() -> list[str]:
    violations: list[str] = []
    for py_file in AI_CORE_FILES:
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("src.shared.python"):
                    violations.append(
                        f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}: {module}"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("src.shared.python"):
                        violations.append(
                            f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}: "
                            f"{alias.name}"
                        )
    return violations


def test_ai_core_modules_use_canonical_shared_imports() -> None:
    """Selected AI core modules should avoid the duplicate src.shared alias."""
    assert not _src_shared_import_violations()
