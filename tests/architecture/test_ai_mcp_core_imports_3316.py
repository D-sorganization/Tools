"""Import-boundary guard for the #3316 AI MCP core slice."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_MCP_CORE_FILES = (
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "mcp" / "__init__.py",
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "mcp" / "client.py",
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "mcp" / "config_loader.py",
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "mcp" / "config_writer.py",
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "mcp" / "pool.py",
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "mcp" / "presets.py",
)


def _src_shared_import_violations() -> list[str]:
    violations: list[str] = []
    for py_file in AI_MCP_CORE_FILES:
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


def test_ai_mcp_core_uses_canonical_shared_imports() -> None:
    """Selected MCP core modules should avoid the duplicate src.shared alias."""
    assert not _src_shared_import_violations()


def test_ai_mcp_core_imports_from_canonical_package() -> None:
    """Canonical package imports should still load the selected MCP modules."""
    for module_name in (
        "shared.python.ai.mcp",
        "shared.python.ai.mcp.client",
        "shared.python.ai.mcp.config_loader",
        "shared.python.ai.mcp.config_writer",
        "shared.python.ai.mcp.contracts",
        "shared.python.ai.mcp.pool",
        "shared.python.ai.mcp.presets",
    ):
        assert importlib.import_module(module_name)
