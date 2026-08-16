"""Clubhead and scenario publication methods for the main window."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from rate_of_closure.club import ClubSpec, head_cog, hosel_point, parametric_head_mesh
from rate_of_closure.model import ImpactScenario, closure_metrics, solve
from rate_of_closure.ui.pyqt6.main_window_contracts import (
    _METRIC_ROWS,
    _QUANTITY_ROWS,
    _RESULT_ROWS,
    _UNITS,
)
from rate_of_closure.units import convert_from_canonical

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QStatusBar

    from rate_of_closure.ui.pyqt6.club_view import Club3DView
    from rate_of_closure.ui.pyqt6.controls_panel import ControlsPanel
    from rate_of_closure.ui.pyqt6.derivation_view import DerivationView
    from rate_of_closure.ui.pyqt6.morris_tab import MorrisScreeningTab
    from rate_of_closure.ui.pyqt6.plots_tab import PlotsTab
    from rate_of_closure.ui.pyqt6.result_row import ResultRow
    from rate_of_closure.ui.pyqt6.simulation_tab import SimulationTab
    from rate_of_closure.ui.pyqt6.variation_tab import VariationTab


class MainWindowClubMixin:
    """Publish club and scenario changes through their guarded view boundaries."""

    if TYPE_CHECKING:
        _controls: ControlsPanel
        _club_view: Club3DView
        _rows: dict[str, ResultRow]
        _plots_tab: PlotsTab
        _derivation_view: DerivationView
        _simulation_tab: SimulationTab
        _variation_tab: VariationTab
        _morris_tab: MorrisScreeningTab
        statusBar: Callable[[], QStatusBar | None]

    def _format_row(self, field: str, value: float) -> str:
        if not math.isfinite(value):
            return "∞ (not closing)"
        quantity = _QUANTITY_ROWS.get(field)
        if quantity is None:
            return f"{value:+.2f}{_UNITS[field]}"
        unit = self._controls.unit_for(quantity)
        displayed = convert_from_canonical(quantity, unit, value)
        return f"{displayed:+.2f} {unit}"

    def _on_club_head(self, spec: ClubSpec) -> None:
        """Build and guard publication of one representative generated head."""
        report = head_cog(spec)
        adopted = self._club_view.try_set_head_mesh(
            parametric_head_mesh(spec),
            hosel_point=hosel_point(spec),
            cog_point=report.cog,
            label=spec.name,
        )
        status_bar = self.statusBar()
        if status_bar is not None and adopted:
            face = (
                "curved face (bulge "
                f"{spec.face_bulge_radius_m * 1000.0:.0f} mm, roll "
                f"{spec.face_roll_radius_m * 1000.0:.0f} mm)"
                if spec.face_bulge_radius_m is not None
                and spec.face_roll_radius_m is not None
                else "flat face"
            )
            status_bar.showMessage(
                f"Representative head generated: {spec.name} — loft "
                f"{spec.loft_deg:.1f}°, {face}"
            )

    def _on_scenario(self, scenario: ImpactScenario) -> None:
        result = solve(scenario)
        metrics = closure_metrics(scenario)
        for field, _ in _RESULT_ROWS:
            self._rows[field].value_label.setText(
                self._format_row(field, getattr(result, field))
            )
        for field, _ in _METRIC_ROWS:
            self._rows[field].value_label.setText(
                self._format_row(field, getattr(metrics, field))
            )
        self._club_view.try_set_scenario(scenario)
        self._plots_tab.set_scenario(scenario)
        self._derivation_view.set_scenario(scenario)
        self._simulation_tab.set_club_spec(self._controls.club_spec())
        self._simulation_tab.set_scenario(scenario)
        self._variation_tab.set_scenario(scenario)
        config = self._simulation_tab.config()
        self._variation_tab.set_simulation_config(config)
        self._morris_tab.set_simulation_config(config)
        status_bar = self.statusBar()
        if status_bar is not None:
            side = "left" if result.path_deviation_deg < 0 else "right"
            status_bar.showMessage(
                f"Reference {result.reference_speed_mph:.1f} mph — impact point "
                f"path {result.path_deviation_deg:+.2f}° ({side}), "
                f"AoA {result.aoa_deviation_deg:+.2f}°, "
                f"CCV {result.closure_rate_dps:.0f} °/s "
                f"({result.normalized_closure_deg_per_ft:.1f} °/ft)"
            )
