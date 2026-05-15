"""AST-based regression guard: tab_context_menu.py must not access private
attributes on its ``sidebar`` argument.

This test walks the AST of ``tab_context_menu.py`` and asserts that no
``Attribute`` node reads an underscore-prefixed name from the ``sidebar``
parameter.  It prevents accidental re-introduction of Law-of-Demeter
violations that were removed in issue #2771.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_TAB_CONTEXT_MENU_PATH = (
    Path(__file__).resolve().parents[5]
    / "src"
    / "shared"
    / "python"
    / "upstream_drift_tools"
    / "ui"
    / "tools_sidebar"
    / "tab_context_menu.py"
)

_SIDEBAR_PARAM_NAMES = {"sidebar"}


def _collect_private_sidebar_accesses(source: str) -> list[tuple[int, str]]:
    """Return (lineno, attr) pairs where sidebar._<attr> is accessed."""
    tree = ast.parse(source)
    violations: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not node.attr.startswith("_"):
            continue
        # Check the object being accessed is the sidebar argument.
        value = node.value
        if isinstance(value, ast.Name) and value.id in _SIDEBAR_PARAM_NAMES:
            violations.append((node.lineno, node.attr))

    return violations


def test_tab_context_menu_has_no_private_sidebar_accesses() -> None:
    """tab_context_menu.py must not access any _-prefixed attribute on sidebar."""
    assert _TAB_CONTEXT_MENU_PATH.exists(), (
        f"Expected source file not found: {_TAB_CONTEXT_MENU_PATH}"
    )
    source = _TAB_CONTEXT_MENU_PATH.read_text(encoding="utf-8")
    violations = _collect_private_sidebar_accesses(source)

    if violations:
        details = "\n".join(
            f"  line {lineno}: sidebar.{attr}" for lineno, attr in violations
        )
        pytest.fail(
            f"tab_context_menu.py accesses {len(violations)} private "
            f"sidebar attribute(s) — use the public API instead:\n{details}"
        )


def test_tab_context_menu_uses_public_api_methods() -> None:
    """Confirm the expected public-API call sites are present after the refactor."""
    assert _TAB_CONTEXT_MENU_PATH.exists()
    source = _TAB_CONTEXT_MENU_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    public_calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        value = node.value
        if isinstance(value, ast.Name) and value.id in _SIDEBAR_PARAM_NAMES:
            if not node.attr.startswith("_"):
                public_calls.add(node.attr)

    required_public_api = {
        "get_tab_definition",
        "get_tab_id_at",
        "get_tab_display_name",
        "prompt_rename_tab",
    }
    missing = required_public_api - public_calls
    assert not missing, (
        f"Expected public API methods not found in tab_context_menu.py: {missing}"
    )
