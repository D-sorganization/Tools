"""QThread worker running the impact-parameter solver off the UI thread.

Epic #4103, #4109/#4110. The Solver panel never blocks the GUI: the
multi-start ``swing_sim.solver.solve`` call runs inside this
:class:`QThread`, progress reports cross back as queued signals, and
cancellation is cooperative via the solver's ``cancel_event`` seam
(``threading.Event``) — the same plumbing the movement_optimizer UIs
use.
"""

from __future__ import annotations

import logging
import threading

from PyQt6.QtCore import QThread, pyqtSignal

from shared.python.swing_sim.solver.goals import ImpactGoal, VariablePartition
from shared.python.swing_sim.solver.solve import CancelledError, solve
from shared.python.swing_sim.solver.tuning import DEFAULT_MAX_NFEV_PER_START

logger = logging.getLogger(__name__)

__all__ = ["SolverWorker"]


class SolverWorker(QThread):
    """One solver run: construct, ``start()``, listen, optionally ``cancel()``.

    Signals (all delivered on the GUI thread via queued connections):
        progressed(object): A ``ProgressReport`` snapshot.
        succeeded(object): The final ``SolverResult``.
        cancelled(): The run was cancelled before any start completed.
        failed(str): The solver raised; the message is user-presentable.
    """

    progressed = pyqtSignal(object)  # noqa: N815 — Qt signal convention
    succeeded = pyqtSignal(object)  # noqa: N815
    cancelled = pyqtSignal()  # noqa: N815
    failed = pyqtSignal(str)  # noqa: N815

    def __init__(
        self,
        goal: ImpactGoal,
        partition: VariablePartition,
        n_starts: int,
        max_nfev_per_start: int = DEFAULT_MAX_NFEV_PER_START,
    ) -> None:
        super().__init__()
        self._goal = goal
        self._partition = partition
        self._n_starts = int(n_starts)
        self._max_nfev = int(max_nfev_per_start)
        self._cancel_event = threading.Event()

    @property
    def cancel_event(self) -> threading.Event:
        """The cooperative cancellation event wired into the solver."""
        return self._cancel_event

    @property
    def max_evaluations(self) -> int:
        """Upper bound on residual evaluations (drives the progress bar)."""
        return self._n_starts * self._max_nfev

    def cancel(self) -> None:
        """Request cooperative cancellation (in-flight starts unwind)."""
        self._cancel_event.set()

    def run(self) -> None:  # pragma: no cover — exercised via signals in tests
        """Thread body: run the solver, translate outcomes into signals."""
        try:
            result = solve(
                self._goal,
                self._partition,
                n_starts=self._n_starts,
                max_nfev_per_start=self._max_nfev,
                progress_cb=self.progressed.emit,
                cancel_event=self._cancel_event,
            )
        except CancelledError:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 — surface solver failures
            logger.warning("solver run failed: %s", exc)
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result)
