"""Synchronized compositor facade for the simulation session."""

from __future__ import annotations

from PyQt6.QtWidgets import QTabWidget

from rate_of_closure.simulation import SimulationRun
from rate_of_closure.ui.pyqt6.flight_view import FlightView
from rate_of_closure.ui.pyqt6.simulation_view import SimulationView
from rate_of_closure.ui.pyqt6.strike_view import StrikeView
from rate_of_closure.ui.pyqt6.view_compositor import ViewCompositor
from rate_of_closure.view_workspace import ViewKind


class SimulationTabCompositorMixin:
    """Expose compositor commands and synchronize its flight clock."""

    _compositor: ViewCompositor
    _compositor_flight_view: FlightView
    _display_tabs: QTabWidget
    _flight_view: FlightView
    _run: SimulationRun | None
    _strike_view: StrikeView
    _view: SimulationView

    def view(self) -> SimulationView:
        """Return the swing-scale scene with playback controls."""
        return self._view

    def strike_view(self) -> StrikeView:
        """Return the face-scale impact-zone viewer."""
        return self._strike_view

    def flight_view(self) -> FlightView:
        """Return the flight-scale trajectory viewer."""
        return self._flight_view

    def compositor(self) -> ViewCompositor:
        """Return the synchronized Impact/Swing/Flight compositor."""
        return self._compositor

    def show_compositor_view(self, kind: ViewKind) -> None:
        """Select the compositor and expose one stable real view host."""
        self._compositor.show_single_view(kind)
        self._display_tabs.setCurrentWidget(self._compositor)

    def _sync_compositor_playback(self, time_s: float) -> None:
        """Map the shared run clock onto solver-relative flight time."""
        run = self._run
        if run is None or run.impact_time_s is None:
            return
        self._compositor_flight_view.set_playback_time(
            max(0.0, time_s - run.impact_time_s)
        )


__all__ = ["SimulationTabCompositorMixin"]
