"""Import-boundary guard for the #3316 AI adapter canonicalization slice."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_ROOT = REPO_ROOT / "src" / "shared" / "python" / "ai" / "adapters"

_FORBIDDEN_PREFIXES = (
    "src.shared.python.ai",
    "src.shared.python.config",
    "src.shared.python.contracts",
    "src.shared.python.logging_pkg",
)


def _is_forbidden(module_name: str) -> bool:
    return any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in _FORBIDDEN_PREFIXES
    )


def _forbidden_adapter_imports() -> list[str]:
    violations: list[str] = []
    for py_file in ADAPTER_ROOT.glob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if _is_forbidden(module):
                    violations.append(
                        f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}: {module}"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_forbidden(alias.name):
                        violations.append(
                            f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}: "
                            f"{alias.name}"
                        )
    return violations


def test_ai_adapters_do_not_import_through_src_shared_python_alias() -> None:
    """AI adapters should import canonical shared modules, not duplicate aliases."""
    assert not _forbidden_adapter_imports()
