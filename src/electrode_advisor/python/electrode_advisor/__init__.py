"""Electrode Advisor - AC Electrode Advancement Module.

Provides models, UI components, and utilities for electrode system analysis.
Uses the shared electrical model engine from upstream_drift_tools.
"""

from __future__ import annotations

# Re-export the shared engine components
from upstream_drift_tools.calculators.electrical import (
    ElectrodeConfig,
    GlassPropertiesInterface,
    ThreePhaseElectricalModelEnhanced,
)

# Import UI components when available
try:
    from .ui.pyqt6.main_window import ElectrodeAdvisorWidget
except ImportError:
    ElectrodeAdvisorWidget = None  # type: ignore[assignment, misc]

__all__ = [
    "ElectrodeAdvisorWidget",
    "ElectrodeConfig",
    "GlassPropertiesInterface",
    "ThreePhaseElectricalModelEnhanced",
]
