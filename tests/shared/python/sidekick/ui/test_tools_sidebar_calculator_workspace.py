"""Integration tests for CalculatorWorkspaceFacade workspace isolation.

DbC: Each test states preconditions and postconditions.
LOD: Tests use only the WorkspaceContract public surface.
TDD: Tests drive the CalculatorWorkspaceFacade Protocol design.

These tests verify that the facade correctly isolates calculator-local
state from the shared global workspace, and that the WorkspaceContract
Protocol surface (`set_variable`, `get_variable`, `list_variable_names`,
`clear_local`) is the sole interaction point.
"""

from __future__ import annotations

import pytest
from sidekick.ui.tools_sidebar.calculator_workspace import (
    CalculatorWorkspaceFacade,
)
from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry
from sidekick.workspace_contract import WorkspaceContract

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def global_registry() -> WorkspaceRegistry:
    """Return a fresh shared global registry."""
    return WorkspaceRegistry()


@pytest.fixture()
def calculator_facade(global_registry: WorkspaceRegistry) -> CalculatorWorkspaceFacade:
    """Return a CalculatorWorkspaceFacade with a fresh local registry."""
    return CalculatorWorkspaceFacade(
        local_registry=WorkspaceRegistry(),
        global_registry=global_registry,
    )


# ---------------------------------------------------------------------------
# WorkspaceContract surface tests
# ---------------------------------------------------------------------------


class TestWorkspaceContractSurface:
    """CalculatorWorkspaceFacade satisfies WorkspaceContract at runtime."""

    def test_facade_is_workspace_contract(
        self, calculator_facade: CalculatorWorkspaceFacade
    ) -> None:
        """Precondition: facade was constructed successfully.
        Postcondition: isinstance check against WorkspaceContract passes."""
        assert isinstance(calculator_facade, WorkspaceContract)

    def test_set_variable_stores_and_returns_metadata(
        self, calculator_facade: CalculatorWorkspaceFacade
    ) -> None:
        """Precondition: workspace is empty.
        Postcondition: set_variable returns metadata with the correct name."""
        meta = calculator_facade.set_variable("pi", 3.14159)
        assert meta.name == "pi"
        assert meta.value == 3.14159

    def test_get_variable_retrieves_stored_value(
        self, calculator_facade: CalculatorWorkspaceFacade
    ) -> None:
        """Precondition: 'result' is set to 99.
        Postcondition: get_variable returns 99 for 'result'."""
        calculator_facade.set_variable("result", 99)
        assert calculator_facade.get_variable("result") == 99

    def test_get_variable_absent_returns_default(
        self, calculator_facade: CalculatorWorkspaceFacade
    ) -> None:
        """Precondition: 'missing' is not in the workspace.
        Postcondition: get_variable returns the supplied default, not a KeyError."""
        sentinel = object()
        assert calculator_facade.get_variable("missing", sentinel) is sentinel

    def test_list_variable_names_returns_sorted_names(
        self, calculator_facade: CalculatorWorkspaceFacade
    ) -> None:
        """Precondition: variables 'c', 'a', 'b' set in non-alphabetical order.
        Postcondition: list_variable_names returns them sorted."""
        calculator_facade.set_variable("c", 3)
        calculator_facade.set_variable("a", 1)
        calculator_facade.set_variable("b", 2)
        names = calculator_facade.list_variable_names()
        assert names == ["a", "b", "c"]

    def test_clear_local_empties_workspace(
        self, calculator_facade: CalculatorWorkspaceFacade
    ) -> None:
        """Precondition: workspace has one variable.
        Postcondition: clear_local leaves workspace empty."""
        calculator_facade.set_variable("temp", 42)
        calculator_facade.clear_local()
        assert calculator_facade.list_variable_names() == []


# ---------------------------------------------------------------------------
# Workspace isolation tests
# ---------------------------------------------------------------------------


class TestCalculatorWorkspaceIsolation:
    """Local calculator state does not bleed into global or sibling calculators."""

    def test_local_variable_not_visible_in_global(
        self,
        calculator_facade: CalculatorWorkspaceFacade,
        global_registry: WorkspaceRegistry,
    ) -> None:
        """Precondition: global registry is empty.
        Postcondition: setting a local variable does not appear in global_registry."""
        calculator_facade.set_variable("local_only", 7)
        assert "local_only" not in global_registry.list_names()

    def test_global_variable_not_visible_via_get_variable(
        self,
        calculator_facade: CalculatorWorkspaceFacade,
        global_registry: WorkspaceRegistry,
    ) -> None:
        """Precondition: 'global_result' is in the global registry only.
        Postcondition: get_variable (local-only mode) returns default, not the global value."""
        global_registry.set("global_result", 100)
        sentinel = object()
        result = calculator_facade.get_variable("global_result", sentinel)
        assert result is sentinel

    def test_two_facades_do_not_share_locals(
        self,
        global_registry: WorkspaceRegistry,
    ) -> None:
        """Precondition: two facades share a global registry but have separate local registries.
        Postcondition: set_variable in facade_a is invisible to facade_b."""
        facade_a = CalculatorWorkspaceFacade(
            local_registry=WorkspaceRegistry(),
            global_registry=global_registry,
            calculator_scope_id="calc_a",
        )
        facade_b = CalculatorWorkspaceFacade(
            local_registry=WorkspaceRegistry(),
            global_registry=global_registry,
            calculator_scope_id="calc_b",
        )
        facade_a.set_variable("a_exclusive", 42)
        assert "a_exclusive" not in facade_b.list_variable_names()

    def test_clear_local_does_not_clear_sibling_facade(
        self,
        global_registry: WorkspaceRegistry,
    ) -> None:
        """Precondition: facade_a and facade_b each have a local variable.
        Postcondition: clear_local on facade_a leaves facade_b's variable intact."""
        facade_a = CalculatorWorkspaceFacade(
            local_registry=WorkspaceRegistry(),
            global_registry=global_registry,
            calculator_scope_id="calc_a",
        )
        facade_b = CalculatorWorkspaceFacade(
            local_registry=WorkspaceRegistry(),
            global_registry=global_registry,
            calculator_scope_id="calc_b",
        )
        facade_a.set_variable("a_var", 1)
        facade_b.set_variable("b_var", 2)

        facade_a.clear_local()

        assert facade_a.list_variable_names() == []
        assert "b_var" in facade_b.list_variable_names()

    def test_overwrite_local_replaces_value(
        self, calculator_facade: CalculatorWorkspaceFacade
    ) -> None:
        """Precondition: 'x' is set to 1.
        Postcondition: setting 'x' again to 2 replaces the stored value."""
        calculator_facade.set_variable("x", 1)
        calculator_facade.set_variable("x", 2)
        assert calculator_facade.get_variable("x") == 2
        assert calculator_facade.list_variable_names().count("x") == 1


# ---------------------------------------------------------------------------
# Protocol type narrowing
# ---------------------------------------------------------------------------


class TestWorkspaceContractTypeNarrowing:
    """Functions typed as WorkspaceContract accept CalculatorWorkspaceFacade."""

    def test_function_accepting_contract_works_with_facade(
        self,
        calculator_facade: CalculatorWorkspaceFacade,
    ) -> None:
        """Precondition: function annotated with WorkspaceContract is defined.
        Postcondition: calling it with a CalculatorWorkspaceFacade succeeds."""

        def store_and_retrieve(
            ws: WorkspaceContract, name: str, value: object
        ) -> object:
            """Precondition: ws satisfies WorkspaceContract.
            Postcondition: stored value is retrievable."""
            ws.set_variable(name, value)
            return ws.get_variable(name)

        result = store_and_retrieve(calculator_facade, "typed_input", 3.14)
        assert result == 3.14
