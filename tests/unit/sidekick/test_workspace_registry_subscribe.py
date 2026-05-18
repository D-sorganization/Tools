"""Tests for ``WorkspaceRegistry.subscribe`` (UpstreamDrift #5616).

The MATLAB-style workspace table auto-refreshes from registry events; the
subscribe API is the seam.
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.ui.tools_sidebar.registry import WorkspaceRegistry


def test_subscribe_fires_on_set() -> None:
    registry = WorkspaceRegistry()
    events: list[tuple[str, str]] = []

    registry.subscribe(lambda event, name: events.append((event, name)))

    registry.set("x", 3)
    assert ("set", "x") in events


def test_subscribe_fires_on_remove() -> None:
    registry = WorkspaceRegistry(initial={"x": 1})
    events: list[tuple[str, str]] = []

    registry.subscribe(lambda event, name: events.append((event, name)))

    assert registry.remove("x") is True
    assert ("remove", "x") in events


def test_unsubscribe_stops_firing() -> None:
    registry = WorkspaceRegistry()
    events: list[tuple[str, str]] = []
    sub = registry.subscribe(lambda event, name: events.append((event, name)))

    sub.unsubscribe()
    registry.set("y", 1)

    assert events == []


def test_multiple_subscribers_all_fire() -> None:
    registry = WorkspaceRegistry()
    seen_a: list[str] = []
    seen_b: list[str] = []
    registry.subscribe(lambda event, name: seen_a.append(name))
    registry.subscribe(lambda event, name: seen_b.append(name))

    registry.set("z", 2)

    assert seen_a == ["z"]
    assert seen_b == ["z"]


def test_subscribe_none_raises_type_error() -> None:
    registry = WorkspaceRegistry()
    with pytest.raises(TypeError):
        registry.subscribe(None)  # type: ignore[arg-type]


def test_subscribe_non_callable_raises_type_error() -> None:
    registry = WorkspaceRegistry()
    with pytest.raises(TypeError):
        registry.subscribe(42)  # type: ignore[arg-type]


def test_reentrant_set_during_callback_does_not_loop() -> None:
    registry = WorkspaceRegistry()
    events: list[str] = []

    def reentrant(event: str, name: str) -> None:
        events.append(name)
        if name == "a":
            registry.set("b", 2)

    registry.subscribe(reentrant)
    registry.set("a", 1)

    # Both notifications fire exactly once; no recursion explosion.
    assert events.count("a") == 1
    assert events.count("b") == 1
    assert len(events) == 2


def test_clear_fires_remove_for_each() -> None:
    registry = WorkspaceRegistry(initial={"a": 1, "b": 2})
    removed: list[str] = []
    registry.subscribe(
        lambda event, name: removed.append(name) if event == "remove" else None
    )

    registry.clear()

    assert sorted(removed) == ["a", "b"]
