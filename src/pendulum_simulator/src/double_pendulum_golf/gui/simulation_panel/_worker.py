"""
Background simulation worker and viewer protocol used by SimulationPanel.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

_SCROLL_STYLE = "QScrollArea { border: none; background: transparent; }"


# ---------------------------------------------------------------------------
# Background simulation worker
# ---------------------------------------------------------------------------


class _SimWorker(QObject):
    """Runs the ODE integration on a background thread.

    Emits ``finished`` with the result object on success,
    or ``error`` with an error message string on failure.
    """

    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(
        self,
        run_fn: Any,
        run_kwargs: dict,
    ) -> None:
        assert run_kwargs is not None, "run_kwargs must be provided"
        super().__init__()
        self._run_fn = run_fn
        self._run_kwargs = run_kwargs

    def run(self) -> None:
        """Called by QThread.started — executes the ODE integration."""
        try:
            self.progress.emit(0)  # Start at 0%
            result = self._run_fn(**self._run_kwargs)
            self.progress.emit(100)  # End at 100%
            self.finished.emit(result)
        except (RuntimeError, ValueError, AssertionError, OSError) as exc:
            logger.error("Simulation worker error: %s", exc)
            self.error.emit(str(exc))


class _SimViewer(Protocol):
    """Structural typing for pendulum/matrix/torque_history widgets."""

    def set_simulation(self, result: object) -> None: ...
    def set_frame(self, idx: int) -> None: ...
    def clear(self) -> None: ...
