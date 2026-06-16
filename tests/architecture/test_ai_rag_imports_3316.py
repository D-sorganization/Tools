"""Import-boundary guard for the #3316 AI RAG canonicalization slice."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_RAG_FILES = (
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "rag" / "simple_rag.py",
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "rag" / "indexer_worker.py",
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "rag" / "context_provider.py",
)


def _src_shared_import_violations() -> list[str]:
    violations: list[str] = []
    for py_file in AI_RAG_FILES:
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


def test_ai_rag_modules_use_canonical_shared_imports() -> None:
    """Selected AI RAG modules should avoid the duplicate src.shared alias."""
    assert not _src_shared_import_violations()
