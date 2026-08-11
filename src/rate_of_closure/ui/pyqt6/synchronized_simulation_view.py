"""Simulation view adapter that publishes its shared playback clock."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal

from rate_of_closure.simulation import SimulationRun
from rate_of_closure.ui.pyqt6.simulation_view import SimulationView


class SynchronizedSimulationView(SimulationView):
    """Emit time changes while retaining the base view's camera and overlays."""

    playbackTimeChanged = pyqtSignal(float)  # noqa: N815

    def set_run(self, run: SimulationRun | None) -> None:
        """Adopt a shared run and publish the reset timeline position."""
        super().set_run(run)
        self.playbackTimeChanged.emit(self.playback_time())

    def set_playback_time(self, time_s: float) -> None:
        """Apply a shared time and publish the clamped result."""
        super().set_playback_time(time_s)
        self.playbackTimeChanged.emit(self.playback_time())

    def _on_slider_moved(self, value: int) -> None:
        previous = self.playback_time()
        super()._on_slider_moved(value)
        if self.playback_time() != previous:
            self.playbackTimeChanged.emit(self.playback_time())

    def _advance(self) -> None:
        super()._advance()
        self.playbackTimeChanged.emit(self.playback_time())


__all__ = ["SynchronizedSimulationView"]
