"""Phase 1 safety tests: all integration tool functions must raise NotImplementedError.

These tests guard against the safety hazard described in issue #2759 where
integration tools returned hardcoded fake placeholder data, causing users to
believe they had received real API responses.

Phase 2 will replace the NotImplementedError stubs with real API clients.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: add the repo root to sys.path so that ``src.*`` imports resolve,
# and stub out heavy transitive dependencies that the integration modules pull
# in via tool_registry (logging_pkg, exceptions, types).
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PACKAGE_STUBS: list[tuple[str, str | None]] = [
    ("src", "src"),
    ("src.shared", "src/shared"),
    ("src.shared.python", "src/shared/python"),
    ("src.shared.python.ai", "src/shared/python/ai"),
    ("src.shared.python.ai.integrations", "src/shared/python/ai/integrations"),
    ("src.shared.python.logging_pkg", None),
    ("src.shared.python.logging_pkg.logging_config", None),
]
for _mod_name, _rel_path in _PACKAGE_STUBS:
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        if _rel_path is not None:
            _stub.__path__ = [str(ROOT / _rel_path)]  # type: ignore[attr-defined]
        sys.modules[_mod_name] = _stub

# Stub logging_pkg so tool_registry can import get_logger without extra deps.
_logging_config_stub = sys.modules["src.shared.python.logging_pkg.logging_config"]
_logging_config_stub.get_logger = logging.getLogger  # type: ignore[attr-defined]
_logging_config_stub.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]


# Stub exceptions and types modules that tool_registry imports.
def _make_stub(name: str) -> types.ModuleType:
    stub = types.ModuleType(name)
    sys.modules[name] = stub
    return stub


_exc_stub = _make_stub("src.shared.python.ai.exceptions")
_exc_stub.ToolExecutionError = Exception  # type: ignore[attr-defined]

_types_stub = _make_stub("src.shared.python.ai.types")
_types_stub.ToolResult = dict  # type: ignore[attr-defined]

# Now it is safe to import the tool_registry and integration modules.
from src.shared.python.ai.tool_registry import ToolRegistry  # noqa: E402

# Provide a fresh registry so module-level @registry.register() calls do not
# collide with any globally registered tools from other test modules.
_fresh_registry = ToolRegistry()


def _get_global_registry_stub() -> ToolRegistry:
    return _fresh_registry


# Patch get_global_registry before the integration modules are imported.
import src.shared.python.ai.tool_registry as _tr_mod  # noqa: E402

_tr_mod.get_global_registry = _get_global_registry_stub  # type: ignore[attr-defined]

from src.shared.python.ai.integrations.affine import affine_sync_notes  # noqa: E402
from src.shared.python.ai.integrations.linear import (  # noqa: E402
    linear_create_issue,
    linear_query_issues,
)
from src.shared.python.ai.integrations.notion import (  # noqa: E402
    notion_push_report,
    notion_read_knowledge_base,
)
from src.shared.python.ai.integrations.obsidian import (  # noqa: E402
    obsidian_read_note,
    obsidian_write_note,
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn, kwargs",
    [
        # Linear
        pytest.param(
            linear_query_issues,
            {"query": "auth bug", "status": "open"},
            id="linear_query_issues",
        ),
        pytest.param(
            linear_create_issue,
            {"title": "Test issue", "description": "desc", "team_id": "TEAM-1"},
            id="linear_create_issue",
        ),
        # Notion
        pytest.param(
            notion_push_report,
            {
                "title": "Q1 Report",
                "markdown_content": "# Q1\ncontent",
                "parent_page_id": "abc123",
            },
            id="notion_push_report",
        ),
        pytest.param(
            notion_read_knowledge_base,
            {"query": "onboarding"},
            id="notion_read_knowledge_base",
        ),
        # Affine
        pytest.param(
            affine_sync_notes,
            {
                "title": "Meeting Notes",
                "markdown_content": "# Meeting\nnotes",
                "workspace_id": "ws-1",
            },
            id="affine_sync_notes",
        ),
        # Obsidian
        pytest.param(
            obsidian_read_note,
            {"note_name": "Daily Note"},
            id="obsidian_read_note",
        ),
        pytest.param(
            obsidian_write_note,
            {
                "note_name": "New Note",
                "markdown_content": "# New\ncontent",
                "overwrite": False,
            },
            id="obsidian_write_note",
        ),
    ],
)
def test_integration_tool_raises_not_implemented(fn, kwargs):
    """Every integration tool function must raise NotImplementedError (Phase 1).

    Precondition: fn is callable with kwargs.
    Postcondition: NotImplementedError is raised — no fake success data returned.
    """
    with pytest.raises(NotImplementedError) as exc_info:
        fn(**kwargs)

    # The error message must mention the issue number so users know where to look.
    assert "#2759" in str(exc_info.value), (
        f"{fn.__name__} NotImplementedError message must reference issue #2759"
    )
