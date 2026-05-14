"""Tests for Sidekick calculator help and predictive text contracts."""

from __future__ import annotations

from upstream_drift_tools.ui.tools_sidebar import (
    CALCULATOR_HELP,
    CalculatorPredictiveText,
    StaticCalculatorPredictionProvider,
    WorkspaceRegistry,
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
