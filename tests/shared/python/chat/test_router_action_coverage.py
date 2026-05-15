"""AST contract guard: every chat action a widget sends must be handled
by the router (Tools issue #2751).

The dock widget and quick bar send WebSocket actions to the shared chat
router.  When a sender adds a new action without a matching ``elif
action == "..."`` branch in ``router_factory.py`` the server returns
``"Unknown action: ..."`` and the feature is silently broken.

This test walks the AST of the sender modules, collects every literal
action string, and asserts that ``router_factory.py`` has a handler
branch for it.  Actions that the router knowingly does not yet handle
can be listed in ``KNOWN_UNHANDLED`` with a tracking issue.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

CHAT_DIR = Path(__file__).resolve().parents[4] / "src" / "shared" / "python" / "chat"
SENDERS = [CHAT_DIR / "_chat_dock_widget_qt.py", CHAT_DIR / "quick_bar.py"]
ROUTER = CHAT_DIR / "router_factory.py"

# Actions that senders emit but the router intentionally does not handle
# yet.  Each entry must reference a tracking issue.
KNOWN_UNHANDLED: dict[str, str] = {
    # file_upload: tracked separately (not part of #2751 scope)
    "file_upload": "tracked separately",
}


def _collect_sent_actions(path: Path) -> set[str]:
    """Return the set of literal action strings sent from a sender module."""
    if not path.exists():
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    actions: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=False):
            if (
                isinstance(key, ast.Constant)
                and key.value == "action"
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                actions.add(value.value)
    return actions


def _collect_router_branches(path: Path) -> set[str]:
    """Return the set of action strings handled by elif branches in router."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    branches: set[str] = set()
    for node in ast.walk(tree):
        # Match: ``action == "..."`` and ``"..." == action``
        if not isinstance(node, ast.Compare):
            continue
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
            continue
        left, right = node.left, node.comparators[0]
        for a, b in ((left, right), (right, left)):
            if (
                isinstance(a, ast.Name)
                and a.id == "action"
                and isinstance(b, ast.Constant)
                and isinstance(b.value, str)
            ):
                branches.add(b.value)
    return branches


def test_every_sent_action_has_router_branch() -> None:
    """Guard against sender/router drift (Tools issue #2751)."""
    sent: set[str] = set()
    for sender in SENDERS:
        sent |= _collect_sent_actions(sender)

    handled = _collect_router_branches(ROUTER)
    missing = sent - handled - set(KNOWN_UNHANDLED)

    assert not missing, (
        f"Widget(s) send action(s) {sorted(missing)} but "
        f"router_factory.py has no matching ``elif action == '...'`` "
        f"branch. Either add a handler or list the action in "
        f"KNOWN_UNHANDLED with a tracking issue."
    )


def test_known_unhandled_actions_are_actually_sent() -> None:
    """KNOWN_UNHANDLED entries should not bit-rot — every listed action
    must still be sent by some widget. Otherwise the entry is stale."""
    sent: set[str] = set()
    for sender in SENDERS:
        sent |= _collect_sent_actions(sender)

    stale = set(KNOWN_UNHANDLED) - sent
    if stale:
        pytest.fail(
            f"KNOWN_UNHANDLED lists action(s) {sorted(stale)} that no "
            f"widget sends. Remove the stale entries."
        )
