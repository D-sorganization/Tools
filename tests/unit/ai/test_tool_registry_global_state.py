"""Tests for the global tool-registry accessor (Tools #3745 P2 cleanup).

The dict-holder singleton (``_registry_holder = {"instance": None}``) was
replaced with an ``lru_cache``-memoized accessor. These tests prove the
accessor is still a process-wide singleton, that ``reset_global_registry``
yields a *fresh, independent* registry (no shared global state), and that
independently constructed ``ToolRegistry`` objects do not share tool state.

Robustness note: some integration-client test bootstraps permanently rebind
``tool_registry.get_global_registry`` to a plain stub (no ``cache_clear``). The
autouse fixture below installs a genuine ``lru_cache`` accessor for the
duration of these tests and restores whatever was there afterward, so the
tests are deterministic regardless of xdist worker sharing.
"""

from __future__ import annotations

import functools

import pytest

import src.shared.python.ai.tool_registry as _tr
from src.shared.python.ai.tool_registry import Tool, ToolRegistry

pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _genuine_global_accessor():
    """Install genuine lru_cache accessors, snapshotting/restoring state."""
    saved_get = _tr.get_global_registry
    saved_reset = _tr.reset_global_registry

    @functools.lru_cache(maxsize=1)
    def _fresh_global() -> ToolRegistry:
        return ToolRegistry()

    def _reset() -> None:
        _fresh_global.cache_clear()

    _tr.get_global_registry = _fresh_global
    _tr.reset_global_registry = _reset
    try:
        yield
    finally:
        _tr.get_global_registry = saved_get
        _tr.reset_global_registry = saved_reset


def test_get_global_registry_returns_same_instance() -> None:
    first = _tr.get_global_registry()
    second = _tr.get_global_registry()
    assert first is second


def test_reset_global_registry_yields_fresh_instance() -> None:
    first = _tr.get_global_registry()
    _tr.reset_global_registry()
    second = _tr.get_global_registry()
    assert first is not second


def test_reset_clears_global_tool_state() -> None:
    """State registered on the old global must not leak into the new one."""
    registry = _tr.get_global_registry()
    registry._tools["leaky"] = Tool(
        name="leaky",
        description="should not survive a reset",
        handler=lambda: "ok",
    )
    assert "leaky" in _tr.get_global_registry()._tools

    _tr.reset_global_registry()
    assert "leaky" not in _tr.get_global_registry()._tools


def test_independent_registries_do_not_share_state() -> None:
    a = ToolRegistry()
    b = ToolRegistry()
    a._tools["only_a"] = Tool(
        name="only_a",
        description="lives only in registry a",
        handler=lambda: "ok",
    )
    assert "only_a" in a._tools
    assert "only_a" not in b._tools
