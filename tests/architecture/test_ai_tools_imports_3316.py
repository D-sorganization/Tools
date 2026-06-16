"""Import-boundary guard for the #3316 AI tools canonicalization slice."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_TOOL_FILES = (
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "tools" / "__init__.py",
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "tools" / "agent_control.py",
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "tools" / "cli_tools.py",
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "tools" / "codemap_tools.py",
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "tools" / "file_ops.py",
)


def _src_shared_import_violations() -> list[str]:
    violations: list[str] = []
    for py_file in AI_TOOL_FILES:
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


def test_ai_tools_modules_use_canonical_shared_imports() -> None:
    """Selected AI tools modules should avoid the duplicate src.shared alias."""
    assert not _src_shared_import_violations()


def test_ai_tools_modules_import_from_canonical_package() -> None:
    """Canonical package imports should still load the selected AI tools modules."""
    for module_name in (
        "shared.python.ai.tools",
        "shared.python.ai.tools.agent_control",
        "shared.python.ai.tools.cli_tools",
        "shared.python.ai.tools.codemap_tools",
        "shared.python.ai.tools.file_ops",
    ):
        assert importlib.import_module(module_name)
