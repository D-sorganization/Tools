"""Tests for src.shared.python.ai.system_prompts.

Verifies the prompt registry is brand-neutral by default and preserves
golf-themed branding only when ``app_context="upstream_drift"`` is used.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: ensure the package chain is importable in a plain pytest run.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PACKAGE_STUBS: list[tuple[str, str | None]] = [
    ("src", "src"),
    ("src.shared", "src/shared"),
    ("src.shared.python", "src/shared/python"),
    ("src.shared.python.ai", "src/shared/python/ai"),
    ("src.shared.python.ai.adapters", "src/shared/python/ai/adapters"),
    ("src.shared.python.logging_pkg", None),
    ("src.shared.python.logging_pkg.logging_config", None),
]
for _mod_name, _rel_path in _PACKAGE_STUBS:
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        if _rel_path is not None:
            _stub.__path__ = [str(ROOT / _rel_path)]  # type: ignore[attr-defined]
        sys.modules[_mod_name] = _stub

# Stub logging_pkg so adapter modules can import get_logger.
_logging_config_stub = sys.modules["src.shared.python.logging_pkg.logging_config"]
_logging_config_stub.get_logger = logging.getLogger  # type: ignore[attr-defined]

# Stub contracts used by base adapter.
if "src.shared.python.contracts" not in sys.modules:
    _contracts_stub = types.ModuleType("src.shared.python.contracts")

    def _precondition(predicate, message="precondition failed"):  # type: ignore[misc]
        """No-op precondition decorator stub."""

        def decorator(func):  # type: ignore[misc]
            return func

        return decorator

    _contracts_stub.precondition = _precondition  # type: ignore[attr-defined]
    sys.modules["src.shared.python.contracts"] = _contracts_stub

# Stub memory_manager used by base adapter.
if "src.shared.python.ai.memory_manager" not in sys.modules:
    _mem_stub = types.ModuleType("src.shared.python.ai.memory_manager")
    _mem_stub.build_memory_prompt_section = lambda **_: ""  # type: ignore[attr-defined]
    _mem_stub.load_agents_md = lambda _root=None: None  # type: ignore[attr-defined]
    sys.modules["src.shared.python.ai.memory_manager"] = _mem_stub

# ---------------------------------------------------------------------------
# Now import the modules under test.
# ---------------------------------------------------------------------------

from src.shared.python.ai.system_prompts import (  # noqa: E402
    GENERIC_SYSTEM_PROMPT,
    GOLF_MODELING_SYSTEM_PROMPT,
    get_prompt,
)

# ── GENERIC_SYSTEM_PROMPT contract ───────────────────────────────────


def test_generic_prompt_contains_no_golf_mentions() -> None:
    assert "Golf" not in GENERIC_SYSTEM_PROMPT
    assert "golf" not in GENERIC_SYSTEM_PROMPT


def test_generic_prompt_contains_no_mujoco() -> None:
    assert "MuJoCo" not in GENERIC_SYSTEM_PROMPT
    assert "Drake" not in GENERIC_SYSTEM_PROMPT
    assert "Pinocchio" not in GENERIC_SYSTEM_PROMPT


def test_generic_prompt_is_non_empty() -> None:
    assert len(GENERIC_SYSTEM_PROMPT.strip()) > 20


# ── GOLF_MODELING_SYSTEM_PROMPT contract ─────────────────────────────


def test_golf_prompt_mentions_golf_modeling_suite() -> None:
    assert "Golf Modeling Suite" in GOLF_MODELING_SYSTEM_PROMPT


def test_golf_prompt_mentions_physics_engines() -> None:
    assert "MuJoCo" in GOLF_MODELING_SYSTEM_PROMPT
    assert "Drake" in GOLF_MODELING_SYSTEM_PROMPT
    assert "Pinocchio" in GOLF_MODELING_SYSTEM_PROMPT


# ── get_prompt() registry ────────────────────────────────────────────


@pytest.mark.parametrize(
    "app_context",
    [None, ""],
    ids=["none", "empty_string"],
)
def test_get_prompt_default_is_generic(app_context: str | None) -> None:
    prompt = get_prompt(app_context)
    assert "Golf" not in prompt
    assert "MuJoCo" not in prompt
    assert prompt == GENERIC_SYSTEM_PROMPT


def test_get_prompt_upstream_drift_returns_golf_prompt() -> None:
    prompt = get_prompt("upstream_drift")
    assert "Golf Modeling Suite" in prompt
    assert "MuJoCo" in prompt


def test_get_prompt_unknown_app_returns_generic_not_error() -> None:
    prompt = get_prompt("unknown_app_xyz")
    assert "Golf" not in prompt
    assert prompt == GENERIC_SYSTEM_PROMPT


def test_get_prompt_tools_returns_generic() -> None:
    prompt = get_prompt("tools")
    assert "Golf" not in prompt
    assert "MuJoCo" not in prompt


# ── BaseAgentAdapter.build_system_prompt smoke test ──────────────────


def _make_adapter():  # type: ignore[misc]
    """Construct a minimal concrete BaseAgentAdapter for smoke testing."""
    from src.shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration

    class ConcreteAdapter(BaseAgentAdapter):
        def send_message(self, message, context, tools):  # type: ignore[override]
            raise NotImplementedError

        def stream_response(self, message, context, tools):  # type: ignore[override]
            raise NotImplementedError

        @property
        def capabilities(self):  # type: ignore[override]
            raise NotImplementedError

        def validate_connection(self):  # type: ignore[override]
            raise NotImplementedError

    return ConcreteAdapter(), ToolDeclaration


def test_build_system_prompt_default_no_golf() -> None:
    adapter, ToolDeclaration = _make_adapter()
    tools = [ToolDeclaration(name="ping", description="Check connectivity")]
    prompt = adapter.build_system_prompt(tools, app_context=None)
    assert "Golf" not in prompt
    assert "MuJoCo" not in prompt


def test_build_system_prompt_upstream_drift_has_golf() -> None:
    adapter, ToolDeclaration = _make_adapter()
    tools = [ToolDeclaration(name="ping", description="Check connectivity")]
    prompt = adapter.build_system_prompt(tools, app_context="upstream_drift")
    assert "Golf Modeling Suite" in prompt
    assert "MuJoCo" in prompt


def test_build_system_prompt_includes_tool_names() -> None:
    adapter, ToolDeclaration = _make_adapter()
    tools = [
        ToolDeclaration(name="run_sim", description="Run simulation"),
        ToolDeclaration(name="export_csv", description="Export to CSV"),
    ]
    prompt = adapter.build_system_prompt(tools, app_context=None)
    assert "run_sim" in prompt
    assert "export_csv" in prompt
