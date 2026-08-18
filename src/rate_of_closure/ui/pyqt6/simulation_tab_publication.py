"""Successful-run publication for the PyQt simulation session."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from rate_of_closure.simulation import SimulationConfig, SimulationRun, run_simulation
from rate_of_closure.ui.pyqt6.result_row import ResultRow
from rate_of_closure.ui.pyqt6.simulation_specs import LAUNCH_ROWS
from rate_of_closure.units import format_distance_m

if TYPE_CHECKING:
    from rate_of_closure.ui.pyqt6.flight_view import FlightView
    from rate_of_closure.ui.pyqt6.inspector_view import InspectorView
    from rate_of_closure.ui.pyqt6.kinetics_panel import KineticsPanel
    from rate_of_closure.ui.pyqt6.simulation_view import SimulationView
    from rate_of_closure.ui.pyqt6.solver_panel import SolverPanel
    from rate_of_closure.ui.pyqt6.strike_view import StrikeView
    from rate_of_closure.ui.pyqt6.torque_profile_panel import TorqueProfilePanel

logger = logging.getLogger(__name__)


class SimulationTabPublicationMixin:
    """Own result publication after the scientific kernel succeeds."""

    _flight_view: FlightView
    _inspector: InspectorView
    _kinetics_panel: KineticsPanel
    _rows: dict[str, ResultRow]
    _run: SimulationRun | None
    _solver_panel: SolverPanel
    _strike_view: StrikeView
    _tau: float | None
    _torque_profile_panel: TorqueProfilePanel
    _view: SimulationView
    runCompleted: Any

    if TYPE_CHECKING:

        def _set_completed_status(self, run: SimulationRun) -> None: ...

        def _set_run_status(self, text: str, state: str) -> None: ...

        def _sync_scrub_after_run(self, run: SimulationRun) -> None: ...

        def _update_outcome_labels(self, run: SimulationRun) -> None: ...

        def _update_spatial_target_after_run(self, run: SimulationRun) -> None: ...

        def config(self) -> SimulationConfig: ...

    def run_now(self) -> SimulationRun | None:
        """Run the simulation and populate the scene + inspector."""
        try:
            run = run_simulation(self.config())
        except Exception as exc:  # noqa: BLE001 — surface physics failures
            logger.warning("simulation failed: %s", exc)
            retained = self._run is not None
            suffix = (
                "Prior accepted scene remains displayed."
                if retained
                else "No accepted simulation is available."
            )
            message = str(exc)[:512]
            self._set_run_status(
                f"Error — Simulation failed: {message}. {suffix}", "error"
            )
            return None
        self._run = run
        self._tau = run.impact_time_s
        self._sync_scrub_after_run(run)
        self._view.set_run(run)
        self._strike_view.set_run(run)
        self._kinetics_panel.set_run(run)
        self._flight_view.set_run(run)
        self._inspector.set_run(run)
        self._refresh_launch_rows()
        self._update_spatial_target_after_run(run)
        self._update_outcome_labels(run)
        self._set_completed_status(run)
        if run.config.swing_run_config.prescribed_profile_id is not None:
            self._torque_profile_panel.set_execution_status(
                "Prescribed profile executed in the double-pendulum dynamics kernel; "
                f"{self._torque_profile_panel.joint_lock_summary()}."
            )
        self.runCompleted.emit(run)
        return run

    def _refresh_launch_rows(self) -> None:
        """Format launch rows in the current distance display unit."""
        run = self._run
        if run is None:
            return
        if run.launch is None:
            for row in self._rows.values():
                row.value_label.setText("N/A — No Impact")
                row.setToolTip(
                    "No launch value exists because fixed-ball contact was not "
                    "detected."
                )
            return
        for field, _label, unit in LAUNCH_ROWS:
            value = run.launch[field]
            if not math.isfinite(value):
                text = "—"
            elif field == "carry_m":
                text = f"+{format_distance_m(value)}"
            else:
                text = f"{value:+.1f}{unit}"
            self._rows[field].value_label.setText(text)
            self._rows[field].setToolTip(
                "Click for the explanation and derivation trace"
            )

    def refresh_units(self) -> None:
        """Re-render distance surfaces after a display-unit change."""
        self._refresh_launch_rows()
        self._solver_panel.target_panel().refresh_units()
        self._flight_view.set_run(self._run)
