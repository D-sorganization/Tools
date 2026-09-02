"""Regression tests for Tools #3331 chat/AI import boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CHAT_DOCK_WIDGET = (
    ROOT / "src" / "shared" / "python" / "chat" / "_chat_dock_widget_qt.py"
)


def test_chat_dock_widget_has_no_top_level_ai_session_manager_import() -> None:
    """The concrete AI session manager must stay lazy or injected."""
    tree = ast.parse(CHAT_DOCK_WIDGET.read_text(encoding="utf-8"))
    top_level_imports = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "ai.gui.session_manager" not in top_level_imports
    assert "src.shared.python.ai.gui.session_manager" not in top_level_imports


def test_chat_dock_widget_constructor_exposes_session_manager_injection() -> None:
    """Dependency inversion keeps host-owned session persistence out of chat.

    Tools #4896: ``session_manager`` moved from a flat ``ChatDockWidget.__init__``
    keyword-only param into a field of the ``ChatIntegrationHooks`` dataclass
    accepted via the ``integrations`` param. The injection guarantee this test
    protects — hosts can supply their own session manager instead of chat
    importing the concrete AI implementation — still holds; only the shape of
    where that injection point lives has changed.
    """
    tree = ast.parse(CHAT_DOCK_WIDGET.read_text(encoding="utf-8"))
    widget_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ChatDockWidget"
    )
    init_method = next(
        node
        for node in widget_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    init_arg_names = [arg.arg for arg in init_method.args.args]
    assert "integrations" in init_arg_names

    hooks_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ChatIntegrationHooks"
    )
    hooks_field_names = [
        stmt.target.id
        for stmt in hooks_class.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    ]
    assert "session_manager" in hooks_field_names
