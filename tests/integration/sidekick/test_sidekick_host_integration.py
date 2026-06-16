"""Integration tests proving the shared Sidekick host components work
across consumers (UpstreamDrift, Gasification_Model, Tools launchers).

DbC: Each test states preconditions and postconditions in its docstring.
LOD: Tests interact with Sidekick host through its public API only.
TDD: Tests are written to the real API discovered from source inspection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# bootstrap.py
# ---------------------------------------------------------------------------


class TestBootstrapEnsurePaths:
    """Prove the sys.path bootstrap works without import errors."""

    def test_bootstrap_importable(self) -> None:
        """Precondition: bootstrap module exists in sidekick package.
        Postcondition: ensure_paths is callable."""
        from sidekick.bootstrap import ensure_paths

        assert callable(ensure_paths)

    def test_ensure_paths_returns_path(self, tmp_path: Path) -> None:
        """Precondition: a repo-root-like directory is provided.
        Postcondition: returns a resolved Path without adding duplicate shared roots."""
        # Use tmp_path as a fake repo root — ensure_paths is idempotent about
        # missing directories, so no extra dirs required.
        from sidekick.bootstrap import ensure_paths

        result = ensure_paths(repo_root=tmp_path)
        assert isinstance(result, Path)
        assert result == tmp_path.resolve()

    def test_ensure_paths_idempotent(self, tmp_path: Path) -> None:
        """Precondition: ensure_paths called twice with the same root.
        Postcondition: sys.path contains no duplicates from repeated calls."""
        import sys

        from sidekick.bootstrap import ensure_paths

        # Create a src dir so the canonical package root is inserted.
        src_dir = tmp_path / "src"
        src_dir.mkdir(parents=True)

        before = len(sys.path)
        ensure_paths(repo_root=tmp_path)
        after_first = len(sys.path)

        ensure_paths(repo_root=tmp_path)
        after_second = len(sys.path)

        # A second call must not add more path entries than the first
        assert after_second == after_first
        # First call may add up to 2 entries (src, src/python/src).
        assert after_first <= before + 2


# ---------------------------------------------------------------------------
# protocols.py — Calculator, ProcessCalculator, DataTransformer, etc.
# ---------------------------------------------------------------------------


class TestSidekickProtocols:
    """Prove core Protocol interfaces are importable and structurally sound."""

    def test_protocols_importable(self) -> None:
        """Precondition: sidekick package is installed/on sys.path.
        Postcondition: all protocol types can be imported from the public API."""
        from sidekick import (
            CalculationResult,
            Calculator,
            DataTransformer,
            InputValidator,
            ProcessCalculator,
            StateSerializable,
            UnitConverter,
            ValidationResult,
        )

        for obj in (
            CalculationResult,
            Calculator,
            DataTransformer,
            InputValidator,
            ProcessCalculator,
            StateSerializable,
            UnitConverter,
            ValidationResult,
        ):
            assert obj is not None

    def test_calculation_result_defaults(self) -> None:
        """Precondition: CalculationResult is instantiated with no args.
        Postcondition: all collections are empty and types are correct."""
        from sidekick import CalculationResult

        result = CalculationResult()
        assert isinstance(result.values, dict)
        assert isinstance(result.units, dict)
        assert isinstance(result.warnings, list)
        assert isinstance(result.metadata, dict)

    def test_validation_result_defaults(self) -> None:
        """Precondition: ValidationResult is instantiated with no args.
        Postcondition: valid=True, errors and warnings are empty lists."""
        from sidekick import ValidationResult

        result = ValidationResult()
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_calculator_protocol_is_runtime_checkable(self) -> None:
        """Precondition: Calculator is decorated with @runtime_checkable.
        Postcondition: isinstance() checks work on conforming objects."""
        from sidekick import CalculationResult, Calculator, ValidationResult

        class _MinimalCalc:
            @property
            def name(self) -> str:
                return "test"

            @property
            def version(self) -> str:
                return "0.1.0"

            def calculate(self, inputs: dict) -> CalculationResult:
                return CalculationResult(values={"result": 0.0})

            def validate_inputs(self, inputs: dict) -> ValidationResult:
                return ValidationResult(valid=True)

        calc = _MinimalCalc()
        assert isinstance(calc, Calculator)

    def test_input_validator_require_positive(self) -> None:
        """Precondition: InputValidator.require_positive is called with negative value.
        Postcondition: raises ValueError."""
        from sidekick import InputValidator

        v = InputValidator()
        with pytest.raises(ValueError, match="positive"):
            v.require_positive("flow_rate", -1.0)

    def test_input_validator_require_positive_passes(self) -> None:
        """Precondition: InputValidator.require_positive called with strictly positive.
        Postcondition: no exception raised."""
        from sidekick import InputValidator

        v = InputValidator()
        v.require_positive("flow_rate", 1.5)  # should not raise

    def test_input_validator_composition_bad_fraction(self) -> None:
        """Precondition: validate_composition given negative fraction.
        Postcondition: raises ValueError."""
        from sidekick import InputValidator

        v = InputValidator()
        with pytest.raises(ValueError):
            v.validate_composition({"N2": -0.1, "CO2": 1.1})


# ---------------------------------------------------------------------------
# WorkspaceRegistry — sidekick.ui.tools_sidebar.registry
# ---------------------------------------------------------------------------


class TestWorkspaceRegistryIntegration:
    """Prove WorkspaceRegistry set/get/subscribe lifecycle works end-to-end."""

    def test_registry_set_and_get(self) -> None:
        """Precondition: WorkspaceRegistry is empty.
        Postcondition: set() stores the value and get() retrieves it."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        reg.set("x", 42)
        assert reg.get("x") == 42

    def test_registry_remove(self) -> None:
        """Precondition: variable was previously set.
        Postcondition: remove() returns True and get() returns default."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        reg.set("y", "hello")
        removed = reg.remove("y")
        assert removed is True
        assert reg.get("y") is None

    def test_registry_remove_nonexistent(self) -> None:
        """Precondition: variable was never set.
        Postcondition: remove() returns False without error."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        assert reg.remove("no_such_var") is False

    def test_registry_list_sorted(self) -> None:
        """Precondition: multiple variables set in arbitrary order.
        Postcondition: list() returns sorted names."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        reg.set("z", 1)
        reg.set("a", 2)
        reg.set("m", 3)
        assert reg.list() == ["a", "m", "z"]

    def test_registry_subscribe_receives_set_event(self) -> None:
        """Precondition: callback registered via subscribe().
        Postcondition: callback is invoked with ('set', name) on set()."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        events: list[tuple[str, str]] = []
        reg = WorkspaceRegistry()
        reg.subscribe(lambda event, name: events.append((event, name)))
        reg.set("alpha", 99)

        assert len(events) == 1
        assert events[0] == ("set", "alpha")

    def test_registry_subscribe_receives_remove_event(self) -> None:
        """Precondition: callback registered and variable exists.
        Postcondition: callback invoked with ('remove', name) on remove()."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        events: list[tuple[str, str]] = []
        reg = WorkspaceRegistry()
        reg.set("beta", 7)
        reg.subscribe(lambda event, name: events.append((event, name)))
        reg.remove("beta")

        assert ("remove", "beta") in events

    def test_registry_subscription_unsubscribe(self) -> None:
        """Precondition: subscription is active.
        Postcondition: after unsubscribe(), callback is no longer called."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        events: list[str] = []
        reg = WorkspaceRegistry()
        sub = reg.subscribe(lambda event, name: events.append(name))
        reg.set("c", 1)
        sub.unsubscribe()
        reg.set("d", 2)

        assert "c" in events
        assert "d" not in events

    def test_registry_describe_known_variable(self) -> None:
        """Precondition: variable is set.
        Postcondition: describe() returns WorkspaceVariable with correct name."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        reg.set("x_val", 3.14)
        var = reg.describe("x_val")
        assert var.name == "x_val"
        assert var.value == pytest.approx(3.14)

    def test_registry_describe_missing_raises(self) -> None:
        """Precondition: variable not in registry.
        Postcondition: describe() raises KeyError."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        with pytest.raises(KeyError):
            reg.describe("nonexistent")

    def test_registry_json_round_trip(self, tmp_path: Path) -> None:
        """Precondition: registry has JSON-safe variables.
        Postcondition: save_json + load_json restores all variable names."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        reg.set("pi", 3.14159)
        reg.set("label", "test")
        reg.set("count", 5)

        json_path = tmp_path / "workspace.json"
        reg.save_json(json_path)

        loaded = WorkspaceRegistry.load_json(json_path)
        assert sorted(loaded.list()) == sorted(reg.list())
        assert loaded.get("label") == "test"

    def test_registry_to_dict_structure(self) -> None:
        """Precondition: registry has at least one variable.
        Postcondition: to_dict() returns dict with 'version' and 'variables' keys."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        reg.set("x", 1)
        payload = reg.to_dict()

        assert "version" in payload
        assert "variables" in payload
        assert isinstance(payload["variables"], list)

    def test_registry_subscribe_null_callback_raises(self) -> None:
        """Precondition: None passed as callback.
        Postcondition: TypeError raised (DbC precondition)."""
        from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

        reg = WorkspaceRegistry()
        with pytest.raises(TypeError):
            reg.subscribe(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FileNavigationController
# ---------------------------------------------------------------------------


class TestFileNavigationControllerIntegration:
    """Prove FileNavigationController history and containment work correctly."""

    def test_initial_path_is_project_root(self, tmp_path: Path) -> None:
        """Precondition: controller created with tmp_path as project root.
        Postcondition: current_path == project_root."""
        from sidekick.ui.tools_sidebar.file_navigation import (
            FileNavigationController,
        )

        ctrl = FileNavigationController(tmp_path)
        assert ctrl.current_path == tmp_path.resolve()

    def test_navigate_to_subdirectory(self, tmp_path: Path) -> None:
        """Precondition: a subdirectory exists inside the project root.
        Postcondition: navigate_to() returns True and current_path changes."""
        from sidekick.ui.tools_sidebar.file_navigation import (
            FileNavigationController,
        )

        subdir = tmp_path / "src"
        subdir.mkdir()
        ctrl = FileNavigationController(tmp_path)
        result = ctrl.navigate_to(subdir)

        assert result is True
        assert ctrl.current_path == subdir

    def test_back_restores_previous_path(self, tmp_path: Path) -> None:
        """Precondition: navigation occurred to a subdirectory.
        Postcondition: back() returns True and current_path returns to root."""
        from sidekick.ui.tools_sidebar.file_navigation import (
            FileNavigationController,
        )

        subdir = tmp_path / "sub"
        subdir.mkdir()
        ctrl = FileNavigationController(tmp_path)
        ctrl.navigate_to(subdir)
        went_back = ctrl.back()

        assert went_back is True
        assert ctrl.current_path == tmp_path.resolve()

    def test_forward_after_back(self, tmp_path: Path) -> None:
        """Precondition: navigated forward then backward.
        Postcondition: forward() returns True and current_path is the subdir."""
        from sidekick.ui.tools_sidebar.file_navigation import (
            FileNavigationController,
        )

        subdir = tmp_path / "fwd"
        subdir.mkdir()
        ctrl = FileNavigationController(tmp_path)
        ctrl.navigate_to(subdir)
        ctrl.back()
        went_forward = ctrl.forward()

        assert went_forward is True
        assert ctrl.current_path == subdir

    def test_state_flags_on_fresh_controller(self, tmp_path: Path) -> None:
        """Precondition: controller is newly created.
        Postcondition: can_go_back=False, can_go_forward=False."""
        from sidekick.ui.tools_sidebar.file_navigation import (
            FileNavigationController,
        )

        ctrl = FileNavigationController(tmp_path)
        state = ctrl.state()

        assert state.can_go_back is False
        assert state.can_go_forward is False
        assert state.current_path == tmp_path.resolve()

    def test_containment_blocks_outside_project(self, tmp_path: Path) -> None:
        """Precondition: allow_outside_project=False (default).
        Postcondition: navigate_to() returns False for paths outside root."""
        from sidekick.ui.tools_sidebar.file_navigation import (
            FileNavigationController,
        )

        outside = tmp_path.parent  # definitely outside tmp_path
        ctrl = FileNavigationController(tmp_path)
        result = ctrl.navigate_to(outside)

        assert result is False
        assert ctrl.current_path == tmp_path.resolve()

    def test_invalid_project_root_raises(self) -> None:
        """Precondition: project_root path does not exist as a directory.
        Postcondition: ValueError is raised."""
        from sidekick.ui.tools_sidebar.file_navigation import (
            FileNavigationController,
        )

        with pytest.raises(ValueError, match="not a directory"):
            FileNavigationController("/definitely/does/not/exist/xyz_abc")


# ---------------------------------------------------------------------------
# CalculatorStartupConfig
# ---------------------------------------------------------------------------


class TestCalculatorStartupIntegration:
    """Prove CalculatorStartupConfig validates and applies imports correctly."""

    def test_default_config_has_numpy_scipy(self) -> None:
        """Precondition: default_calculator_startup_config() called.
        Postcondition: includes numpy and scipy entries."""
        from sidekick.ui.tools_sidebar.calculator_startup import (
            default_calculator_startup_config,
        )

        cfg = default_calculator_startup_config()
        modules = [imp.module for imp in cfg.imports]
        assert "numpy" in modules
        assert "scipy" in modules

    def test_enabled_imports_filters_disabled(self) -> None:
        """Precondition: config has one enabled and one disabled import.
        Postcondition: enabled_imports() returns only the enabled one."""
        from sidekick.ui.tools_sidebar.calculator_startup import (
            CalculatorStartupConfig,
            CalculatorStartupImport,
        )

        imports = (
            CalculatorStartupImport("math", "math", enabled=True),
            CalculatorStartupImport("cmath", "cmath", enabled=False),
        )
        cfg = CalculatorStartupConfig(imports)
        enabled = cfg.enabled_imports()
        modules = [imp.module for imp in enabled]
        assert "math" in modules
        assert "cmath" not in modules

    def test_duplicate_alias_raises(self) -> None:
        """Precondition: two imports share the same alias.
        Postcondition: ValueError raised by CalculatorStartupConfig."""
        from sidekick.ui.tools_sidebar.calculator_startup import (
            CalculatorStartupConfig,
            CalculatorStartupImport,
        )

        imports = (
            CalculatorStartupImport("os", "myalias"),
            CalculatorStartupImport("sys", "myalias"),
        )
        with pytest.raises(ValueError, match="duplicate"):
            CalculatorStartupConfig(imports)

    def test_apply_imports_loads_stdlib(self) -> None:
        """Precondition: stdlib module (math) requested.
        Postcondition: namespace contains alias pointing to the module."""
        from sidekick.ui.tools_sidebar.calculator_startup import (
            CalculatorStartupConfig,
            CalculatorStartupImport,
            apply_calculator_startup_imports,
        )

        imports = (CalculatorStartupImport("math", "math"),)
        cfg = CalculatorStartupConfig(imports)
        namespace: dict = {}
        result = apply_calculator_startup_imports(namespace, cfg)

        import math as _math

        assert namespace.get("math") is _math
        assert "math" in result.loaded_modules

    def test_apply_imports_warns_on_missing_module(self) -> None:
        """Precondition: a non-existent module is requested.
        Postcondition: result has a warning; namespace does not contain alias."""
        from sidekick.ui.tools_sidebar.calculator_startup import (
            CalculatorStartupConfig,
            CalculatorStartupImport,
            apply_calculator_startup_imports,
        )

        imports = (CalculatorStartupImport("definitely_not_real_xyz", "xyz"),)
        cfg = CalculatorStartupConfig(imports)
        namespace: dict = {}
        result = apply_calculator_startup_imports(namespace, cfg)

        assert len(result.warnings) == 1
        assert "xyz" not in namespace


# ---------------------------------------------------------------------------
# SidekickStateProfileStore
# ---------------------------------------------------------------------------


class TestStateProfileStoreIntegration:
    """Prove SidekickStateProfileStore round-trips profiles through the filesystem."""

    def test_save_and_load_profile(self, tmp_path: Path) -> None:
        """Precondition: a valid SidebarState can be created and saved.
        Postcondition: load_profile returns ok=True with equivalent state."""
        from sidekick.ui.tools_sidebar.state import SidebarState
        from sidekick.ui.tools_sidebar.state_profiles import (
            SidekickStateProfileStore,
        )

        store = SidekickStateProfileStore(tmp_path / "sidekick_data")
        state = SidebarState()
        result = store.save_profile("my-profile", state)

        assert result.ok is True
        assert result.profile_name == "my-profile"

        loaded = store.load_profile("my-profile")
        assert loaded.ok is True
        assert loaded.state is not None

    def test_load_missing_profile_returns_not_ok(self, tmp_path: Path) -> None:
        """Precondition: profile file does not exist.
        Postcondition: load_profile returns ok=False."""
        from sidekick.ui.tools_sidebar.state_profiles import (
            SidekickStateProfileStore,
        )

        store = SidekickStateProfileStore(tmp_path / "sidekick_data")
        result = store.load_profile("nonexistent-profile")

        assert result.ok is False
        assert "not found" in result.message.lower()

    def test_clear_data_requires_confirmation(self, tmp_path: Path) -> None:
        """Precondition: no confirmation string provided.
        Postcondition: clear_data() returns ok=False with warning."""
        from sidekick.ui.tools_sidebar.state_profiles import (
            SidekickStateProfileStore,
        )

        store = SidekickStateProfileStore(tmp_path / "sidekick_data")
        result = store.clear_data()

        assert result.ok is False
        assert result.warning is not None

    def test_clear_data_with_confirmation(self, tmp_path: Path) -> None:
        """Precondition: correct confirmation string provided.
        Postcondition: clear_data() returns ok=True."""
        from sidekick.ui.tools_sidebar.state_profiles import (
            CLEAR_SIDEKICK_DATA_CONFIRMATION,
            SidekickStateProfileStore,
        )

        store = SidekickStateProfileStore(tmp_path / "sidekick_data")
        result = store.clear_data(confirmation=CLEAR_SIDEKICK_DATA_CONFIRMATION)

        assert result.ok is True

    def test_invalid_profile_name_raises(self, tmp_path: Path) -> None:
        """Precondition: profile name contains path-unsafe characters.
        Postcondition: validate_profile_name raises ValueError."""
        from sidekick.ui.tools_sidebar.state_profiles import validate_profile_name

        with pytest.raises(ValueError):
            validate_profile_name("../../evil")

    def test_valid_profile_name_accepted(self) -> None:
        """Precondition: alphanumeric name with spaces and dots.
        Postcondition: validate_profile_name returns the same name."""
        from sidekick.ui.tools_sidebar.state_profiles import validate_profile_name

        name = validate_profile_name("My Profile 1.0")
        assert name == "My Profile 1.0"


# ---------------------------------------------------------------------------
# CommandHistoryController
# ---------------------------------------------------------------------------


class TestCommandHistoryControllerIntegration:
    """Prove command history appends, deduplicates, and navigates correctly."""

    def test_submit_and_retrieve(self) -> None:
        """Precondition: empty controller.
        Postcondition: submitted command appears in commands tuple."""
        from sidekick.ui.tools_sidebar.command_history import (
            CommandHistoryController,
        )

        ctrl = CommandHistoryController()
        ctrl.submit("x = 1")
        assert "x = 1" in ctrl.commands

    def test_submit_deduplicates_consecutive(self) -> None:
        """Precondition: same command submitted twice in a row.
        Postcondition: only one entry exists in commands."""
        from sidekick.ui.tools_sidebar.command_history import (
            CommandHistoryController,
        )

        ctrl = CommandHistoryController()
        ctrl.submit("x = 1")
        ctrl.submit("x = 1")
        assert ctrl.commands.count("x = 1") == 1

    def test_navigation_previous_and_next(self) -> None:
        """Precondition: two commands submitted.
        Postcondition: previous_preview returns last command, next restores draft."""
        from sidekick.ui.tools_sidebar.command_history import (
            CommandHistoryController,
        )

        ctrl = CommandHistoryController()
        ctrl.submit("x = 1")
        ctrl.submit("y = 2")

        prev = ctrl.previous_preview("")
        assert prev == "y = 2"

        ctrl.previous_preview("")
        nxt = ctrl.next_preview()
        assert nxt == "y = 2"

    def test_max_entries_bounded(self) -> None:
        """Precondition: max_entries=3, four commands submitted.
        Postcondition: only three most recent entries are kept."""
        from sidekick.ui.tools_sidebar.command_history import (
            CommandHistoryController,
        )

        ctrl = CommandHistoryController(max_entries=3)
        for cmd in ["a = 1", "b = 2", "c = 3", "d = 4"]:
            ctrl.submit(cmd)

        assert len(ctrl.commands) == 3
        assert "a = 1" not in ctrl.commands

    def test_max_entries_below_one_raises(self) -> None:
        """Precondition: max_entries=0.
        Postcondition: ValueError raised at construction."""
        from sidekick.ui.tools_sidebar.command_history import (
            CommandHistoryController,
        )

        with pytest.raises(ValueError, match="at least 1"):
            CommandHistoryController(max_entries=0)


# ---------------------------------------------------------------------------
# SidekickDesignTokens
# ---------------------------------------------------------------------------


class TestDesignTokensIntegration:
    """Prove SidekickDesignTokens can be constructed and queried."""

    def test_default_tokens_importable(self) -> None:
        """Precondition: sidekick.ui.tools_sidebar.design_tokens is importable.
        Postcondition: SIDEKICK_DESIGN_TOKENS is a SidekickDesignTokens instance."""
        from sidekick.ui.tools_sidebar.design_tokens import (
            SIDEKICK_DESIGN_TOKENS,
            SidekickDesignTokens,
        )

        assert isinstance(SIDEKICK_DESIGN_TOKENS, SidekickDesignTokens)

    def test_default_tokens_cover_all_names(self) -> None:
        """Precondition: SIDEKICK_DESIGN_TOKENS is created.
        Postcondition: every SIDEKICK_TOKEN_NAMES entry is present."""
        from sidekick.ui.tools_sidebar.design_tokens import (
            SIDEKICK_DESIGN_TOKENS,
            SIDEKICK_TOKEN_NAMES,
        )

        for name in SIDEKICK_TOKEN_NAMES:
            assert SIDEKICK_DESIGN_TOKENS[name], f"Token {name!r} is empty/missing"

    def test_css_variables_prefixed_correctly(self) -> None:
        """Precondition: SidekickDesignTokens with default values.
        Postcondition: css_variables keys start with --sidekick-."""
        from sidekick.ui.tools_sidebar.design_tokens import SIDEKICK_DESIGN_TOKENS

        css_vars = SIDEKICK_DESIGN_TOKENS.css_variables()
        assert all(key.startswith("--sidekick-") for key in css_vars)

    def test_with_overrides_returns_new_instance(self) -> None:
        """Precondition: SIDEKICK_DESIGN_TOKENS has a default accent color.
        Postcondition: with_overrides returns a different instance,
        original unchanged."""
        from sidekick.ui.tools_sidebar.design_tokens import (
            SIDEKICK_DESIGN_TOKENS,
        )

        original_accent = SIDEKICK_DESIGN_TOKENS["color.accent"]
        custom = SIDEKICK_DESIGN_TOKENS.with_overrides(**{"color.accent": "#ff0000"})

        assert custom["color.accent"] == "#ff0000"
        assert SIDEKICK_DESIGN_TOKENS["color.accent"] == original_accent
