"""Phase 1 safety tests: Linear integration tools must raise NotImplementedError.

These tests guard against the safety hazard described in issue #2759 where
integration tools returned hardcoded fake placeholder data. Notion, Affine,
and Obsidian have been promoted to Phase 2 (real API clients). Linear remains
in Phase 1 until its real GraphQL client lands.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

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

_logging_config_stub = sys.modules["src.shared.python.logging_pkg.logging_config"]
_logging_config_stub.get_logger = logging.getLogger  # type: ignore[attr-defined]
_logging_config_stub.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]


def _make_stub(name: str) -> types.ModuleType:
    stub = types.ModuleType(name)
    sys.modules[name] = stub
    return stub


_exc_stub = _make_stub("src.shared.python.ai.exceptions")
_exc_stub.ToolExecutionError = Exception  # type: ignore[attr-defined]

_types_stub = _make_stub("src.shared.python.ai.types")
_types_stub.ToolResult = dict  # type: ignore[attr-defined]

from src.shared.python.ai.tool_registry import ToolRegistry  # noqa: E402

_fresh_registry = ToolRegistry()


def _get_global_registry_stub() -> ToolRegistry:
    return _fresh_registry


import src.shared.python.ai.tool_registry as _tr_mod  # noqa: E402

_tr_mod.get_global_registry = _get_global_registry_stub  # type: ignore[attr-defined]

from src.shared.python.ai.integrations.linear import (  # noqa: E402
    linear_create_issue,
    linear_query_issues,
)


@pytest.mark.parametrize(
    "fn, kwargs",
    [
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
    ],
)
def test_integration_tool_raises_not_implemented(fn, kwargs):
    """Linear integration tools must raise NotImplementedError (Phase 1)."""
    with pytest.raises(NotImplementedError) as exc_info:
        fn(**kwargs)

    assert "#2759" in str(exc_info.value), (
        f"{fn.__name__} NotImplementedError message must reference issue #2759"
    )
