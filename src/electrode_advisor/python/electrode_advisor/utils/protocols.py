"""Protocol classes for electrode advisor mixin interface contracts.

Issue #1438: The 13-mixin composition relies on shared attributes and methods
that are only defined at runtime when all mixins are composed into
ElectrodeAdvisorWidget.  These Protocol classes document the expected
interface so that type-checkers and developers can verify contracts
without running the full application.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from upstream_drift_tools.calculators.electrical import ElectrodeConfig


@runtime_checkable
class SupportsElectrodeConfig(Protocol):
    """Mixin host must expose an ElectrodeConfig instance."""

    config: ElectrodeConfig


@runtime_checkable
class SupportsVisualization(Protocol):
    """Mixin host must expose the visualization object and results dict."""

    visualization: Any
    calculation_results: dict[str, Any]


@runtime_checkable
class SupportsCalculation(Protocol):
    """Mixin host must expose the electrical model and depth inputs."""

    electrical_model: Any
    depth_inputs: dict[int, Any]
    phase_inputs: dict[str, dict[str, Any]]

    def _calculate_system(self) -> None: ...
