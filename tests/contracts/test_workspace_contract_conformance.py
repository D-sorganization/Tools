"""Conformance tests for WorkspaceContract Protocol.

DbC: All tests assert both preconditions and postconditions.
TDD: Tests written before implementation to drive the Protocol design.

WorkspaceContract defines the minimal interface that any calculator-local
workspace facade must satisfy.  CalculatorWorkspaceFacade is the canonical
implementation; these tests verify structural subtyping is satisfied.
"""

from __future__ import annotations

import pytest
from sidekick.ui.tools_sidebar.calculator_workspace import CalculatorWorkspaceFacade
from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry
from sidekick.workspace_contract import WorkspaceContract


@pytest.fixture()
def local_registry() -> WorkspaceRegistry:
    """Return a fresh calculator-local registry."""
    return WorkspaceRegistry()


@pytest.fixture()
def global_registry() -> WorkspaceRegistry:
    """Return a fresh global registry."""
    return WorkspaceRegistry()


@pytest.fixture()
def facade(
    local_registry: WorkspaceRegistry, global_registry: WorkspaceRegistry
) -> CalculatorWorkspaceFacade:
    """Return a fully-initialised CalculatorWorkspaceFacade."""
    return CalculatorWorkspaceFacade(
        local_registry=local_registry,
        global_registry=global_registry,
    )


class TestWorkspaceContractProtocol:
    """Protocol conformance: CalculatorWorkspaceFacade satisfies WorkspaceContract."""

    def test_protocol_is_importable(self) -> None:
        """Precondition: workspace_contract module exists.
        Postcondition: WorkspaceContract is a non-None runtime-checkable Protocol."""
        # Postcondition
        assert WorkspaceContract is not None

    def test_facade_satisfies_protocol(self, facade: CalculatorWorkspaceFacade) -> None:
        """Precondition: CalculatorWorkspaceFacade and WorkspaceContract both importable.
        Postcondition: isinstance() returns True (structural subtyping)."""
        # Precondition: facade is a CalculatorWorkspaceFacade
        assert isinstance(facade, CalculatorWorkspaceFacade)
        # Postcondition: structural subtyping holds
        assert isinstance(facade, WorkspaceContract)

    def test_protocol_has_required_methods(self) -> None:
        """Precondition: Protocol is defined.
        Postcondition: all required method names are declared on the Protocol."""
        required = {
            "set_variable",
            "get_variable",
            "list_variable_names",
            "clear_local",
        }
        protocol_methods = {m for m in dir(WorkspaceContract) if not m.startswith("_")}
        missing = required - protocol_methods
        assert not missing, f"Protocol missing methods: {missing}"

    def test_set_variable_returns_variable_metadata(
        self, facade: CalculatorWorkspaceFacade
    ) -> None:
        """Precondition: facade is WorkspaceContract-compatible.
        Postcondition: set_variable stores value and returns non-None metadata."""
        # Precondition
        assert isinstance(facade, WorkspaceContract)
        # Exercise
        result = facade.set_variable("x", 42)
        # Postcondition: metadata is returned, name matches
        assert result is not None
        assert result.name == "x"

    def test_get_variable_returns_stored_value(
        self, facade: CalculatorWorkspaceFacade
    ) -> None:
        """Precondition: a variable was stored via set_variable.
        Postcondition: get_variable returns the same value."""
        facade.set_variable("y", 3.14)
        # Postcondition
        value = facade.get_variable("y")
        assert value == 3.14

    def test_get_variable_missing_returns_default(
        self, facade: CalculatorWorkspaceFacade
    ) -> None:
        """Precondition: 'missing_var' does not exist in the workspace.
        Postcondition: get_variable returns the supplied default."""
        sentinel = object()
        result = facade.get_variable("missing_var", sentinel)
        assert result is sentinel

    def test_list_variable_names_returns_sorted_list(
        self, facade: CalculatorWorkspaceFacade
    ) -> None:
        """Precondition: two variables stored.
        Postcondition: list_variable_names returns a sorted list of their names."""
        facade.set_variable("b", 2)
        facade.set_variable("a", 1)
        names = facade.list_variable_names()
        assert isinstance(names, list)
        assert names == sorted(names)
        assert set(names) >= {"a", "b"}

    def test_clear_local_removes_all_variables(
        self, facade: CalculatorWorkspaceFacade
    ) -> None:
        """Precondition: at least one variable stored.
        Postcondition: clear_local empties the local workspace."""
        facade.set_variable("z", 99)
        assert "z" in facade.list_variable_names()
        facade.clear_local()
        # Postcondition: workspace is now empty
        assert facade.list_variable_names() == []

    def test_clear_local_does_not_affect_global(
        self,
        facade: CalculatorWorkspaceFacade,
        global_registry: WorkspaceRegistry,
    ) -> None:
        """Precondition: global registry has its own variable.
        Postcondition: clear_local does not remove global variables."""
        global_registry.set("global_x", 100)
        facade.set_variable("local_x", 1)
        facade.clear_local()
        # Postcondition: global variable is unaffected
        assert global_registry.get("global_x") == 100


class TestWorkspaceContractIsolation:
    """Workspace isolation: different facade instances do not share local state."""

    def test_two_facades_have_independent_locals(
        self,
        global_registry: WorkspaceRegistry,
    ) -> None:
        """Precondition: two facades share the same global registry but distinct local registries.
        Postcondition: setting a local variable in one facade is invisible to the other."""
        local_a = WorkspaceRegistry()
        local_b = WorkspaceRegistry()
        facade_a = CalculatorWorkspaceFacade(
            local_registry=local_a,
            global_registry=global_registry,
            calculator_scope_id="calc_a",
        )
        facade_b = CalculatorWorkspaceFacade(
            local_registry=local_b,
            global_registry=global_registry,
            calculator_scope_id="calc_b",
        )

        facade_a.set_variable("result", 42)

        # Postcondition: facade_b cannot see the local variable
        assert "result" not in facade_b.list_variable_names()

    def test_clear_on_one_facade_leaves_other_intact(
        self,
        global_registry: WorkspaceRegistry,
    ) -> None:
        """Precondition: both facades have local variables.
        Postcondition: clearing facade_a does not remove facade_b's local variables."""
        local_a = WorkspaceRegistry()
        local_b = WorkspaceRegistry()
        facade_a = CalculatorWorkspaceFacade(
            local_registry=local_a,
            global_registry=global_registry,
            calculator_scope_id="calc_a",
        )
        facade_b = CalculatorWorkspaceFacade(
            local_registry=local_b,
            global_registry=global_registry,
            calculator_scope_id="calc_b",
        )

        facade_a.set_variable("a_var", 1)
        facade_b.set_variable("b_var", 2)

        facade_a.clear_local()

        # Postcondition
        assert "a_var" not in facade_a.list_variable_names()
        assert "b_var" in facade_b.list_variable_names()
