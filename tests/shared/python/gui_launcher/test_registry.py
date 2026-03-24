"""Tests for gui_launcher.registry module.

Covers:
- GUIRegistry singleton pattern
- register / unregister / get / list_tools / list_categories
- Category filtering
- get_config and get_available_gui_types
- register_gui convenience function
- clear functionality
"""

from __future__ import annotations

import pytest
from gui_launcher.launcher import GUIType, LaunchConfig
from gui_launcher.registry import (
    GUIRegistry,
    get_registry,
    register_gui,
)


@pytest.fixture
def registry() -> GUIRegistry:
    """Provide a fresh registry instance (not the singleton)."""
    r = GUIRegistry()
    return r


def _sample_config() -> dict[GUIType, LaunchConfig]:
    return {
        GUIType.PYQT6: LaunchConfig(
            tool_name="test",
            gui_type=GUIType.PYQT6,
            module_path="my_tool.main",
            entry_point="MyWidget",
        ),
    }


# ── Registration ─────────────────────────────────────────────────────────


class TestRegistration:
    """Test register and unregister."""

    def test_register_and_get(self, registry: GUIRegistry) -> None:
        registry.register(
            tool_name="test_tool",
            display_name="Test Tool",
            description="A test tool",
            gui_configs=_sample_config(),
        )
        entry = registry.get("test_tool")
        assert entry is not None
        assert entry.tool_name == "test_tool"
        assert entry.display_name == "Test Tool"

    def test_get_nonexistent_returns_none(self, registry: GUIRegistry) -> None:
        assert registry.get("no_such_tool") is None

    def test_unregister(self, registry: GUIRegistry) -> None:
        registry.register(
            tool_name="tool_a",
            display_name="A",
            description="desc",
            gui_configs=_sample_config(),
        )
        assert registry.unregister("tool_a") is True
        assert registry.get("tool_a") is None

    def test_unregister_nonexistent(self, registry: GUIRegistry) -> None:
        assert registry.unregister("ghost_tool") is False

    def test_clear(self, registry: GUIRegistry) -> None:
        registry.register(
            tool_name="x",
            display_name="X",
            description="d",
            gui_configs=_sample_config(),
        )
        registry.clear()
        assert registry.list_tools() == []


# ── Listing ──────────────────────────────────────────────────────────────


class TestListing:
    """Test list_tools and list_categories."""

    def test_list_tools(self, registry: GUIRegistry) -> None:
        registry.register("a", "A", "da", _sample_config(), category="Cat1")
        registry.register("b", "B", "db", _sample_config(), category="Cat2")
        tools = registry.list_tools()
        names = [t.tool_name for t in tools]
        assert "a" in names
        assert "b" in names

    def test_list_tools_by_category(self, registry: GUIRegistry) -> None:
        registry.register("a", "A", "da", _sample_config(), category="Engineering")
        registry.register("b", "B", "db", _sample_config(), category="Data")
        eng_tools = registry.list_tools(category="Engineering")
        assert len(eng_tools) == 1
        assert eng_tools[0].tool_name == "a"

    def test_list_categories(self, registry: GUIRegistry) -> None:
        registry.register("x", "X", "dx", _sample_config(), category="Alpha")
        registry.register("y", "Y", "dy", _sample_config(), category="Beta")
        cats = registry.list_categories()
        assert "Alpha" in cats
        assert "Beta" in cats
        assert cats == sorted(cats)


# ── Config Access ────────────────────────────────────────────────────────


class TestConfigAccess:
    """Test get_config and get_available_gui_types."""

    def test_get_config(self, registry: GUIRegistry) -> None:
        configs = {
            GUIType.PYQT6: LaunchConfig(
                tool_name="multi",
                gui_type=GUIType.PYQT6,
                module_path="m",
                entry_point="W",
            ),
            GUIType.REACT: LaunchConfig(
                tool_name="multi",
                gui_type=GUIType.REACT,
                module_path="m",
                web_path="web",
            ),
        }
        registry.register("multi", "Multi", "d", configs)
        cfg = registry.get_config("multi", GUIType.PYQT6)
        assert cfg is not None
        assert cfg.entry_point == "W"

    def test_get_config_wrong_type(self, registry: GUIRegistry) -> None:
        registry.register("pyonly", "PyOnly", "d", _sample_config())
        assert registry.get_config("pyonly", GUIType.REACT) is None

    def test_get_config_missing_tool(self, registry: GUIRegistry) -> None:
        assert registry.get_config("nope", GUIType.PYQT6) is None

    def test_available_gui_types(self, registry: GUIRegistry) -> None:
        configs = {
            GUIType.PYQT6: LaunchConfig(
                tool_name="dual",
                gui_type=GUIType.PYQT6,
                module_path="m",
                entry_point="W",
            ),
            GUIType.BROWSER: LaunchConfig(
                tool_name="dual",
                gui_type=GUIType.BROWSER,
                module_path="m",
            ),
        }
        registry.register("dual", "Dual", "d", configs)
        types = registry.get_available_gui_types("dual")
        assert GUIType.PYQT6 in types
        assert GUIType.BROWSER in types
        assert GUIType.REACT not in types

    def test_available_gui_types_missing_tool(self, registry: GUIRegistry) -> None:
        assert registry.get_available_gui_types("nope") == []


# ── Singleton & Convenience ──────────────────────────────────────────────


class TestSingleton:
    """Test singleton pattern and convenience functions."""

    def test_singleton_returns_same_instance(self) -> None:
        r1 = GUIRegistry.instance()
        r2 = GUIRegistry.instance()
        assert r1 is r2

    def test_get_registry_returns_singleton(self) -> None:
        r = get_registry()
        assert r is GUIRegistry.instance()

    def test_register_gui_adds_to_singleton(self) -> None:
        singleton = get_registry()
        # Clean up before test
        singleton.unregister("conv_test")

        register_gui(
            tool_name="conv_test",
            display_name="Conv Test",
            description="Convenience test",
            gui_configs=_sample_config(),
        )
        entry = singleton.get("conv_test")
        assert entry is not None
        assert entry.tool_name == "conv_test"

        # Cleanup
        singleton.unregister("conv_test")


# ── DbC Contract Violations ──────────────────────────────────────────────


class TestGUIRegistryContracts:
    """Tests that verify DbC preconditions on GUIRegistry methods."""

    @pytest.fixture()
    def registry(self) -> GUIRegistry:
        """Fresh (non-singleton) registry for isolation."""
        return GUIRegistry()

    # register contracts

    def test_register_empty_tool_name(self, registry: GUIRegistry) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            registry.register("", "My Tool", "desc", _sample_config())

    def test_register_non_string_tool_name(self, registry: GUIRegistry) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            registry.register(123, "My Tool", "desc", _sample_config())  # type: ignore[arg-type]

    def test_register_empty_display_name(self, registry: GUIRegistry) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            registry.register("tool_x", "", "desc", _sample_config())

    def test_register_non_dict_configs(self, registry: GUIRegistry) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            registry.register("tool_x", "Name", "desc", [])  # type: ignore[arg-type]

    def test_register_empty_configs(self, registry: GUIRegistry) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            registry.register("tool_x", "Name", "desc", {})

    def test_register_non_string_description(self, registry: GUIRegistry) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            registry.register("tool_x", "Name", None, _sample_config())  # type: ignore[arg-type]

    def test_register_empty_category(self, registry: GUIRegistry) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            registry.register("tool_x", "Name", "desc", _sample_config(), category="")

    # unregister contracts

    def test_unregister_empty_name(self, registry: GUIRegistry) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            registry.unregister("")

    def test_unregister_non_string(self, registry: GUIRegistry) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            registry.unregister(None)  # type: ignore[arg-type]

    # get contracts

    def test_get_empty_name(self, registry: GUIRegistry) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            registry.get("")

    def test_get_non_string(self, registry: GUIRegistry) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            registry.get(42)  # type: ignore[arg-type]

    # get_config contracts

    def test_get_config_empty_tool_name(self, registry: GUIRegistry) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            registry.get_config("", GUIType.PYQT6)

    def test_get_config_non_guitype(self, registry: GUIRegistry) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            registry.get_config("tool_x", "pyqt6")  # type: ignore[arg-type]

    # auto_discover_guis contracts

    def test_auto_discover_non_list(self) -> None:
        from contracts import PreconditionError

        from src.shared.python.gui_launcher.registry import auto_discover_guis

        with pytest.raises(PreconditionError):
            auto_discover_guis("/not/a/list")  # type: ignore[arg-type]

    def test_auto_discover_empty_list_returns_zero(self) -> None:
        from src.shared.python.gui_launcher.registry import auto_discover_guis

        count = auto_discover_guis([])
        assert count == 0
