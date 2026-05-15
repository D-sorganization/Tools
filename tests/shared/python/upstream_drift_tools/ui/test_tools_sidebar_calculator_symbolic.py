"""Tests for Sidekick symbolic calculator workflow metadata."""

from __future__ import annotations

from upstream_drift_tools.ui.tools_sidebar import (
    CALCULATOR_HELP,
    SYMBOLIC_CALCULATOR_WORKFLOWS,
    symbolic_calculator_workflow_metadata,
)


def test_symbolic_workflow_metadata_is_static_and_guided() -> None:
    metadata = symbolic_calculator_workflow_metadata()

    assert {workflow["id"] for workflow in metadata} == {
        "equation",
        "system",
        "substitution",
    }
    assert metadata[0]["steps"][0].startswith("Enter")
    assert "timeout" in metadata[0]["limits"]
    assert SYMBOLIC_CALCULATOR_WORKFLOWS[0].example == "solve x**2 - 4 for x"


def test_calculator_help_mentions_symbolic_limits_and_latex() -> None:
    metadata = CALCULATOR_HELP.to_metadata()

    assert "LaTeX" in metadata["tips"]
    assert "bounded" in metadata["tips"]
