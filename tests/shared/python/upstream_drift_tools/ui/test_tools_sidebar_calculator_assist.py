"""Tests for Sidekick calculator help and predictive text contracts."""

from __future__ import annotations

import os
import subprocess
import sys
import types

import pytest
from upstream_drift_tools.ui.tools_sidebar import (
    CALCULATOR_HELP,
    CalculatorPredictiveText,
    SidebarState,
    StaticCalculatorPredictionProvider,
    WorkspaceRegistry,
)
from upstream_drift_tools.ui.tools_sidebar.calculator_startup import (
    CalculatorStartupConfig,
    CalculatorStartupImport,
    apply_calculator_startup_imports,
    default_calculator_startup_config,
)


def test_calculator_help_metadata_includes_examples_and_tips() -> None:
    metadata = CALCULATOR_HELP.to_metadata()

    assert metadata["title"] == "Calculator"
    assert "symbolic" in metadata["summary"]
    assert "solve(x**2 - 4, x)" in metadata["examples"]
    assert "assignments" in metadata["tips"]
    assert "Workspace" in metadata["tips"]
    assert "latex(expression)" in metadata["tips"]


def test_prediction_provider_uses_allowlisted_context_without_execution() -> None:
    registry = WorkspaceRegistry({"mass_flow": 12.5, "matrix_case": [[1, 2]]})
    provider = StaticCalculatorPredictionProvider()

    suggestions = provider.suggest(
        "m",
        workspace_variables=registry.variables(),
        loaded_dependencies=("numpy",),
    )

    assert [suggestion.label for suggestion in suggestions] == [
        "Matrix(",
        "mass_flow",
        "matrix_case",
    ]

    dependency_suggestions = provider.suggest(
        "np",
        workspace_variables=registry.variables(),
        loaded_dependencies=("numpy",),
    )
    assert [suggestion.label for suggestion in dependency_suggestions] == ["np"]


def test_predictive_text_preference_and_missing_provider_suppress_suggestions() -> None:
    provider = StaticCalculatorPredictionProvider()
    enabled = CalculatorPredictiveText(enabled=True, provider=provider)
    disabled = CalculatorPredictiveText(enabled=False, provider=provider)
    missing_provider = CalculatorPredictiveText(enabled=True, provider=None)

    assert enabled.suggest("sol")
    assert disabled.suggest("sol") == ()
    assert missing_provider.suggest("sol") == ()


def test_startup_import_config_defaults_to_optional_scientific_aliases() -> None:
    config = default_calculator_startup_config()

    assert [item.to_dict() for item in config.imports] == [
        {
            "module": "numpy",
            "alias": "np",
            "enabled": True,
            "allow_private": False,
        },
        {
            "module": "scipy",
            "alias": "scipy",
            "enabled": True,
            "allow_private": False,
        },
    ]
    assert SidebarState().calculator_startup_imports == config.to_list()


def test_startup_config_import_is_qt_and_scientific_dependency_lazy() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(("src", "src/shared/python"))
    script = """
import sys
from upstream_drift_tools.ui.tools_sidebar.calculator_startup import (
    default_calculator_startup_config,
)
assert default_calculator_startup_config().imports
loaded = set(sys.modules)
assert "numpy" not in loaded
assert "scipy" not in loaded
assert not any(name in loaded for name in {"PyQt6", "PySide6", "PyQt5", "PySide2"})
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_startup_imports_load_available_modules_transactionally(monkeypatch) -> None:
    imported: list[str] = []

    def fake_import_module(module_name: str) -> types.ModuleType:
        imported.append(module_name)
        if module_name == "missing_package":
            raise ImportError("not installed")
        module = types.ModuleType(module_name)
        module.MARKER = module_name
        return module

    monkeypatch.setattr(
        "upstream_drift_tools.ui.tools_sidebar.calculator_startup."
        "importlib.import_module",
        fake_import_module,
    )
    namespace: dict[str, object] = {"existing": object()}
    config = CalculatorStartupConfig(
        (
            CalculatorStartupImport("numpy", "np"),
            CalculatorStartupImport("missing_package", "missing"),
            CalculatorStartupImport("scipy", "scipy", enabled=False),
        )
    )

    result = apply_calculator_startup_imports(namespace, config)

    assert imported == ["numpy", "missing_package"]
    assert namespace["np"] is namespace["numpy"]
    assert namespace["np"].MARKER == "numpy"
    assert "missing" not in namespace
    assert "scipy" not in namespace
    assert result.loaded_modules == ("numpy",)
    assert result.warnings[0].module == "missing_package"
    assert "Install optional dependency" in result.warnings[0].message


@pytest.mark.parametrize(
    "module, alias",
    [
        (".numpy", "np"),
        ("numpy..linalg", "np"),
        ("_private_package", "private"),
        ("numpy", "__dict__"),
        ("numpy", "for"),
        ("numpy", "bad-name"),
    ],
)
def test_startup_import_config_rejects_unsafe_entries(
    module: str,
    alias: str,
) -> None:
    with pytest.raises(ValueError):
        CalculatorStartupImport(module, alias)


def test_user_added_startup_imports_round_trip_through_sidebar_state() -> None:
    config = CalculatorStartupConfig(
        (
            CalculatorStartupImport("numpy", "np"),
            CalculatorStartupImport("statistics", "stats"),
        )
    )
    state = SidebarState(calculator_startup_imports=config.to_list())
    restored = SidebarState.from_dict(state.to_dict())

    assert restored.calculator_startup_imports == config.to_list()
    assert (
        CalculatorStartupConfig.from_list(restored.calculator_startup_imports) == config
    )
