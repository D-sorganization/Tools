"""Layering guard for the AI/chat contract boundary (Tools #3331)."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = REPO_ROOT / "src" / "shared" / "python" / "ai"
CHAT_ROOT = REPO_ROOT / "src" / "shared" / "python" / "chat"
ALLOWED_CONTRACT_PREFIXES = (
    "chat_contracts",
    "chat_contracts",
)


def _python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts and "tests" not in path.parts
    )


def _module_name(node: ast.AST) -> str:
    if isinstance(node, ast.ImportFrom):
        return node.module or ""
    if isinstance(node, ast.Import):
        return ",".join(alias.name for alias in node.names)
    return ""


def _is_type_checking_guard(node: ast.AST) -> bool:
    return isinstance(node, ast.If) and (
        isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
    )


def _forbidden_imports(root: Path, forbidden_roots: tuple[str, ...]) -> list[str]:
    failures: list[str] = []
    for path in _python_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        def visit(
            node: ast.AST,
            *,
            current_path: Path = path,
            type_checking_only: bool = False,
        ) -> None:
            next_type_checking_only = type_checking_only or _is_type_checking_guard(
                node
            )
            if isinstance(node, ast.Import | ast.ImportFrom):
                module_name = _module_name(node)
                if not next_type_checking_only and _is_forbidden(
                    module_name, forbidden_roots
                ):
                    rel = current_path.relative_to(REPO_ROOT).as_posix()
                    failures.append(f"{rel}:{node.lineno}: {module_name}")
            for child in ast.iter_child_nodes(node):
                visit(
                    child,
                    current_path=current_path,
                    type_checking_only=next_type_checking_only,
                )

        visit(tree)
    return failures


def _is_forbidden(module_name: str, forbidden_roots: tuple[str, ...]) -> bool:
    if module_name.startswith(ALLOWED_CONTRACT_PREFIXES):
        return False
    return any(
        module_name == root or module_name.startswith(f"{root}.")
        for root in forbidden_roots
    )


def test_ai_production_code_does_not_import_chat_package() -> None:
    """AI code may share chat_contracts, but not depend on chat internals."""
    assert (
        _forbidden_imports(
            AI_ROOT,
            ("chat", "src.shared.python.chat"),
        )
        == []
    )


def test_chat_production_code_does_not_import_ai_package() -> None:
    """Chat code may use injected AI factories, but not static AI imports."""
    assert (
        _forbidden_imports(
            CHAT_ROOT,
            ("ai", "src.shared.python.ai"),
        )
        == []
    )
