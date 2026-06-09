from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "mypy_autofix_agent.py"


def _script_tree() -> ast.Module:
    return ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))


def test_script_delegates_to_canonical_package_entrypoint() -> None:
    """The scripts shim must not duplicate the package implementation."""
    tree = _script_tree()
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    assert "run_agent" not in function_names
    assert "parse_mypy_output" not in function_names
    assert "write_file_lines" not in function_names
    assert "main" in function_names


def test_script_help_uses_canonical_cli() -> None:
    """The compatibility script must remain directly executable from repo root."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0
    assert "Mypy Autofix Agent" in result.stdout
    assert "--max-fixes" in result.stdout
