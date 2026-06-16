"""Import-boundary guard for the #3316 AI integrations slice."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_INTEGRATION_FILES = (
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "integrations" / "affine.py",
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "integrations" / "linear.py",
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "integrations" / "notion.py",
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "integrations" / "obsidian.py",
    REPO_ROOT
    / "src"
    / "shared"
    / "python"
    / "ai"
    / "integrations"
    / "github_mcp"
    / "__init__.py",
    REPO_ROOT
    / "src"
    / "shared"
    / "python"
    / "ai"
    / "integrations"
    / "github_mcp"
    / "discovery.py",
    REPO_ROOT
    / "src"
    / "shared"
    / "python"
    / "ai"
    / "integrations"
    / "github_mcp"
    / "integration.py",
    REPO_ROOT
    / "src"
    / "shared"
    / "python"
    / "ai"
    / "integrations"
    / "github_mcp"
    / "server_config.py",
)


def _src_shared_import_violations() -> list[str]:
    violations: list[str] = []
    for py_file in AI_INTEGRATION_FILES:
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


def test_ai_integrations_use_canonical_shared_imports() -> None:
    """Selected AI integration modules should avoid the duplicate src.shared alias."""
    assert not _src_shared_import_violations()


def test_ai_integrations_import_from_canonical_package() -> None:
    """Canonical package imports should still load the selected integrations."""
    for module_name in (
        "shared.python.ai.integrations.affine",
        "shared.python.ai.integrations.linear",
        "shared.python.ai.integrations.notion",
        "shared.python.ai.integrations.obsidian",
        "shared.python.ai.integrations.github_mcp",
        "shared.python.ai.integrations.github_mcp.discovery",
        "shared.python.ai.integrations.github_mcp.integration",
        "shared.python.ai.integrations.github_mcp.server_config",
    ):
        assert importlib.import_module(module_name)
