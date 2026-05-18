"""Tests for WorkspaceRegistry — the shared variable store used by all Sidekick tabs.

DbC: Each test states preconditions and postconditions.
LOD: Tests interact through the WorkspaceRegistry public API only.
TDD: These tests prove the registry contract that consumers depend on.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestWorkspaceRegistryCore:
    """Core CRUD operations: set, get, remove, clear, list."""

    def test_empty_registry_list_is_empty(self) -> None:
        """Precondition: freshly constructed WorkspaceRegistry.
        Postcondition: list() returns empty list."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        assert reg.list() == []

    def test_set_and_get_scalar(self) -> None:
        """Precondition: registry is empty.
        Postcondition: set('x', 42) makes get('x') return 42."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        reg.set("x", 42)
        assert reg.get("x") == 42

    def test_set_returns_workspace_variable(self) -> None:
        """Precondition: registry is empty.
        Postcondition: set() returns a WorkspaceVariable with the correct name."""
        from sidekick.ui.tools_sidebar.registry import (
            WorkspaceRegistry,
            WorkspaceVariable,
        )

        reg = WorkspaceRegistry()
        var = reg.set("pi", 3.14)
        assert isinstance(var, WorkspaceVariable)
        assert var.name == "pi"

    def test_get_missing_returns_default(self) -> None:
        """Precondition: variable 'z' was never set.
        Postcondition: get('z') returns None, get('z', 99) returns 99."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        assert reg.get("z") is None
        assert reg.get("z", 99) == 99

    def test_remove_existing_returns_true(self) -> None:
        """Precondition: variable was previously set.
        Postcondition: remove() returns True and variable is gone."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        reg.set("y", "hello")
        assert reg.remove("y") is True
        assert reg.get("y") is None

    def test_remove_nonexistent_returns_false(self) -> None:
        """Precondition: variable was never set.
        Postcondition: remove() returns False without error."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        assert reg.remove("never_existed") is False

    def test_clear_removes_all_variables(self) -> None:
        """Precondition: multiple variables are set.
        Postcondition: list() returns empty list after clear()."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        for name, value in [("a", 1), ("b", 2), ("c", 3)]:
            reg.set(name, value)
        reg.clear()
        assert reg.list() == []

    def test_list_returns_sorted_names(self) -> None:
        """Precondition: variables set in reverse order.
        Postcondition: list() returns alphabetically sorted names."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        reg.set("z", 1)
        reg.set("a", 2)
        reg.set("m", 3)
        assert reg.list() == ["a", "m", "z"]

    def test_list_names_alias_matches_list(self) -> None:
        """Precondition: variables are set.
        Postcondition: list_names() returns identical result to list()."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        reg.set("alpha", 1)
        reg.set("beta", 2)
        assert reg.list() == reg.list_names()

    def test_set_empty_name_raises(self) -> None:
        """Precondition: empty string passed as name.
        Postcondition: ValueError raised (DbC precondition)."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        with pytest.raises(ValueError, match="non-empty"):
            reg.set("", 1)

    def test_initial_dict_populates_registry(self) -> None:
        """Precondition: WorkspaceRegistry constructed with initial dict.
        Postcondition: variables from the initial dict are accessible."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry(initial={"a": 1, "b": 2})
        assert reg.get("a") == 1
        assert reg.get("b") == 2


class TestWorkspaceVariable:
    """WorkspaceVariable metadata and serialization."""

    def test_describe_returns_correct_type_name(self) -> None:
        """Precondition: variable of type float is set.
        Postcondition: describe().type_name == 'float'."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        reg.set("val", 3.14)
        var = reg.describe("val")
        assert var.type_name == "float"

    def test_describe_string_type_name(self) -> None:
        """Precondition: variable of type str is set.
        Postcondition: describe().type_name == 'str'."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        reg.set("label", "hello")
        var = reg.describe("label")
        assert var.type_name == "str"

    def test_describe_json_safe_for_scalar(self) -> None:
        """Precondition: a float value is set.
        Postcondition: describe().json_safe is True."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        reg.set("pi", 3.14159)
        var = reg.describe("pi")
        assert var.json_safe is True

    def test_describe_missing_raises_key_error(self) -> None:
        """Precondition: variable does not exist in registry.
        Postcondition: describe() raises KeyError."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        with pytest.raises(KeyError):
            reg.describe("does_not_exist")

    def test_to_metadata_has_required_keys(self) -> None:
        """Precondition: a simple JSON-safe variable is set.
        Postcondition: to_metadata() dict has 'name', 'type', 'summary',
        'json_safe', 'preview' keys."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        reg.set("v", 42)
        meta = reg.describe("v").to_metadata()
        for key in ("name", "type", "summary", "json_safe", "preview"):
            assert key in meta, f"Missing key: {key}"

    def test_variables_returns_list_of_workspace_variable(self) -> None:
        """Precondition: registry has two variables.
        Postcondition: variables() returns a list with exactly two WorkspaceVariable."""
        from sidekick.ui.tools_sidebar.registry import (
            WorkspaceRegistry,
            WorkspaceVariable,
        )

        reg = WorkspaceRegistry()
        reg.set("a", 1)
        reg.set("b", 2)
        result = reg.variables()
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(v, WorkspaceVariable) for v in result)


class TestWorkspaceRegistrySubscriptions:
    """Event subscription and notification behavior."""

    def test_subscribe_receives_set_event(self) -> None:
        """Precondition: callback registered via subscribe().
        Postcondition: callback called with event='set' and correct name."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        events: list[tuple[str, str]] = []
        reg = WorkspaceRegistry()
        reg.subscribe(lambda event, name: events.append((event, name)))
        reg.set("x", 10)

        assert len(events) == 1
        assert events[0] == ("set", "x")

    def test_subscribe_receives_remove_event(self) -> None:
        """Precondition: callback registered and variable exists.
        Postcondition: callback called with event='remove' on remove()."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        events: list[tuple[str, str]] = []
        reg = WorkspaceRegistry()
        reg.set("y", 5)
        reg.subscribe(lambda event, name: events.append((event, name)))
        reg.remove("y")

        assert ("remove", "y") in events

    def test_subscribe_clear_emits_remove_for_each_variable(self) -> None:
        """Precondition: two variables set, callback registered.
        Postcondition: clear() emits two 'remove' events."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        events: list[tuple[str, str]] = []
        reg = WorkspaceRegistry()
        reg.set("a", 1)
        reg.set("b", 2)
        reg.subscribe(lambda event, name: events.append((event, name)))
        reg.clear()

        remove_names = {name for event, name in events if event == "remove"}
        assert remove_names == {"a", "b"}

    def test_unsubscribe_stops_further_callbacks(self) -> None:
        """Precondition: subscription is active.
        Postcondition: after unsubscribe(), callback not called on subsequent set()."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        called: list[str] = []
        reg = WorkspaceRegistry()
        sub = reg.subscribe(lambda event, name: called.append(name))
        reg.set("first", 1)
        sub.unsubscribe()
        reg.set("second", 2)

        assert "first" in called
        assert "second" not in called

    def test_unsubscribe_is_idempotent(self) -> None:
        """Precondition: subscription was already unsubscribed.
        Postcondition: calling unsubscribe() again does not raise."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        sub = reg.subscribe(lambda event, name: None)
        sub.unsubscribe()
        sub.unsubscribe()  # should not raise

    def test_subscribe_null_callback_raises_type_error(self) -> None:
        """Precondition: None passed as callback.
        Postcondition: TypeError raised (DbC)."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        with pytest.raises(TypeError):
            reg.subscribe(None)  # type: ignore[arg-type]

    def test_subscribe_non_callable_raises_type_error(self) -> None:
        """Precondition: a non-callable string passed as callback.
        Postcondition: TypeError raised."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        with pytest.raises(TypeError):
            reg.subscribe("not_callable")  # type: ignore[arg-type]


class TestWorkspaceRegistryPersistence:
    """JSON serialization and deserialization round-trips."""

    def test_save_and_load_json_preserves_scalars(self, tmp_path: Path) -> None:
        """Precondition: registry has JSON-safe scalar variables.
        Postcondition: load_json restores all names and JSON-safe values."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        reg.set("x", 1.5)
        reg.set("label", "test")
        reg.set("flag", True)

        json_path = tmp_path / "ws.json"
        reg.save_json(json_path)

        loaded = WorkspaceRegistry.load_json(json_path)
        assert sorted(loaded.list()) == sorted(reg.list())
        assert loaded.get("x") == pytest.approx(1.5)
        assert loaded.get("label") == "test"
        assert loaded.get("flag") is True

    def test_to_dict_version_is_1(self) -> None:
        """Precondition: registry has at least one variable.
        Postcondition: to_dict()['version'] == 1."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        reg.set("v", 42)
        payload = reg.to_dict()
        assert payload["version"] == 1

    def test_to_dict_variables_is_list(self) -> None:
        """Precondition: registry has variables.
        Postcondition: to_dict()['variables'] is a list."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        reg.set("v", 42)
        payload = reg.to_dict()
        assert isinstance(payload["variables"], list)

    def test_export_environment_produces_strings(self) -> None:
        """Precondition: registry has a JSON-safe variable.
        Postcondition: export_environment() returns a dict[str, str]."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        reg.set("count", 7)
        env = reg.export_environment()
        assert isinstance(env, dict)
        for key, value in env.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    def test_update_from_merges_variables(self) -> None:
        """Precondition: two separate registries with distinct variables.
        Postcondition: update_from() merges variables from other into self."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg1 = WorkspaceRegistry()
        reg1.set("a", 1)

        reg2 = WorkspaceRegistry()
        reg2.set("b", 2)

        reg1.update_from(reg2)
        assert reg1.get("a") == 1
        assert reg1.get("b") == 2
