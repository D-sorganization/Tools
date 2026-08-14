"""Killable process boundary for CPU-bound PyQt plot computation."""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from PyQt6.QtCore import (
    QObject,
    QProcess,
    QProcessEnvironment,
    QTimer,
    pyqtSignal,
)

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.plotting import PlotData, PlotSpec, compute_plot_data
from rate_of_closure.simulation import SimulationConfig, SimulationRun, run_simulation

_DEFAULT_CLUB = "Driver 10.5°"
_MAX_COMPUTE_SECONDS = 120.0
_THREAD_LIMIT_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class PlotDataPayload:
    """Pickle-safe complete scientific plot data for one pane."""

    row: int
    spec: PlotSpec
    x_bytes: bytes
    series: tuple[tuple[str, bytes], ...]
    x_label: str
    y_label: str

    @classmethod
    def from_data(cls, row: int, data: PlotData) -> PlotDataPayload:
        return cls(
            row=row,
            spec=data.spec,
            x_bytes=np.asarray(data.x, dtype=np.float64).tobytes(),
            series=tuple(
                (label, np.asarray(values, dtype=np.float64).tobytes())
                for label, values in data.series.items()
            ),
            x_label=data.x_label,
            y_label=data.y_label,
        )

    def restore(self) -> PlotData:
        """Rebuild the strict immutable PlotData publication object."""
        return PlotData(
            spec=self.spec,
            x=np.frombuffer(self.x_bytes, dtype=np.float64),
            series={
                label: np.frombuffer(values, dtype=np.float64)
                for label, values in self.series
            },
            x_label=self.x_label,
            y_label=self.y_label,
        )


@dataclass(frozen=True)
class ProcessPlotOutcome:
    """Pickle-safe independent pane result."""

    row: int
    data: PlotDataPayload | None
    error: str | None


@dataclass(frozen=True)
class PlotProcessRequest:
    """Exact generation-bound inputs passed to the child process."""

    generation: int
    requests: tuple[tuple[int, PlotSpec], ...]
    reference_run: SimulationRun | None
    scenario: ImpactScenario


def _compute_in_process(
    request: PlotProcessRequest, authority_identity: str
) -> tuple[object, ...]:
    """Build a complete pickle-safe result without importing GUI ownership."""
    try:
        run = request.reference_run or run_simulation(
            SimulationConfig(
                scenario=request.scenario,
                club=get_club(_DEFAULT_CLUB),
            )
        )
        outcomes: list[ProcessPlotOutcome] = []
        for row, spec in request.requests:
            try:
                data = compute_plot_data(spec, run)
                outcomes.append(
                    ProcessPlotOutcome(row, PlotDataPayload.from_data(row, data), None)
                )
            except Exception as exc:  # noqa: BLE001 - pane-scoped status
                outcomes.append(ProcessPlotOutcome(row, None, str(exc)[:512]))
        return (
            "success",
            authority_identity,
            request.generation,
            run,
            tuple(outcomes),
        )
    except Exception as exc:  # noqa: BLE001 - retained-state status
        return (
            "failure",
            authority_identity,
            request.generation,
            str(exc)[:512],
        )


def _worker_main() -> int:
    """Read one internal request and write one all-or-nothing result."""
    try:
        request_bytes = sys.stdin.buffer.read()
        authority_identity = sha256(request_bytes).hexdigest()
        request = pickle.loads(request_bytes)  # noqa: S301 - internal IPC
        if not isinstance(request, PlotProcessRequest):
            raise TypeError("Malformed plot process request")
        sys.stdout.buffer.write(
            pickle.dumps(_compute_in_process(request, authority_identity))
        )
        sys.stdout.buffer.flush()
    except Exception as exc:  # noqa: BLE001 - bounded child failure envelope
        sys.stderr.write(f"Plot worker failed: {str(exc)[:512]}\n")
        return 1
    return 0


class PlotComputeProcess(QObject):
    """Asynchronous Qt subprocess with QThread-compatible lifecycle methods."""

    succeeded = pyqtSignal(int, object, object)
    failed = pyqtSignal(int, str)
    finished = pyqtSignal()

    def __init__(self, request: PlotProcessRequest) -> None:
        super().__init__()
        self.generation = request.generation
        self.requests = request.requests
        self._request_bytes = pickle.dumps(request)
        self.authority_identity = sha256(self._request_bytes).hexdigest()
        self._process = QProcess(self)
        self._process.setProcessChannelMode(
            QProcess.ProcessChannelMode.SeparateChannels
        )
        self._process.setProgram(sys.executable)
        self._process.setArguments(
            [
                "-m",
                "rate_of_closure.ui.pyqt6.plot_worker_bootstrap",
            ]
        )
        environment = QProcessEnvironment.systemEnvironment()
        for name in _THREAD_LIMIT_VARIABLES:
            environment.insert(name, "1")
        module_paths = [path for path in sys.path if path]
        inherited = environment.value("PYTHONPATH")
        if inherited:
            module_paths.append(inherited)
        environment.insert("PYTHONPATH", os.pathsep.join(module_paths))
        self._process.setProcessEnvironment(environment)
        self._process.started.connect(self._write_request)
        self._process.finished.connect(self._on_process_finished)
        self._process.errorOccurred.connect(self._on_process_error)
        self._process.readyReadStandardOutput.connect(self._read_stdout)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)
        self._stdout = bytearray()
        self._finished = False

    def start(self) -> None:
        """Start the process without importing the parent GUI module in it."""
        self._process.start()
        self._timer.start(round(_MAX_COMPUTE_SECONDS * 1000))

    def isRunning(self) -> bool:  # noqa: N802
        """Mirror QThread's runtime query for owning lifecycle code."""
        return (
            not self._finished
            and self._process.state() != QProcess.ProcessState.NotRunning
        )

    def requestInterruption(self) -> None:  # noqa: N802
        """Terminate stale CPU work; no partial payload can be published."""
        if self._finished:
            return
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()
            self._process.waitForFinished(1_000)
        self._finish()

    def wait(self, timeout_ms: int) -> bool:
        """Mirror QThread.wait for bounded close handling."""
        if self._process.state() == QProcess.ProcessState.NotRunning:
            return True
        return bool(self._process.waitForFinished(max(0, timeout_ms)))

    def _write_request(self) -> None:
        if self._finished:
            return
        self._process.write(self._request_bytes)
        self._process.closeWriteChannel()

    def _read_stdout(self) -> None:
        self._stdout.extend(bytes(self._process.readAllStandardOutput()))

    def _on_process_finished(self, exit_code: int, _status: object) -> None:
        if self._finished:
            return
        self._read_stdout()
        if exit_code == 0 and self._stdout:
            try:
                message = pickle.loads(self._stdout)  # noqa: S301 - internal IPC
            except Exception as exc:  # noqa: BLE001 - bounded IPC failure
                self.failed.emit(
                    self.generation, f"Malformed plot process result: {exc}"
                )
            else:
                self._publish(message)
        else:
            detail = bytes(self._process.readAllStandardError()).decode(
                "utf-8", errors="replace"
            )
            self.failed.emit(
                self.generation,
                (detail.strip() or "Plot worker exited without a result")[:512],
            )
        self._finish()

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        if self._finished or error is QProcess.ProcessError.Crashed:
            return
        self.failed.emit(self.generation, f"Plot worker process error: {error.name}")
        self._finish()

    def _on_timeout(self) -> None:
        if self._finished:
            return
        self.failed.emit(self.generation, "Plot computation exceeded 120 seconds")
        self._process.kill()
        self._process.waitForFinished(1_000)
        self._finish()

    def _publish(self, message: object) -> None:
        if not isinstance(message, tuple) or len(message) not in {4, 5}:
            self.failed.emit(self.generation, "Malformed plot process result")
            return
        if (
            message[0] == "failure"
            and len(message) == 4
            and message[1] == self.authority_identity
            and type(message[2]) is int
            and isinstance(message[3], str)
            and 0 < len(message[3]) <= 512
        ):
            self.failed.emit(message[2], message[3])
            return
        if (
            message[0] != "success"
            or len(message) != 5
            or message[1] != self.authority_identity
        ):
            self.failed.emit(self.generation, "Malformed plot process result")
            return
        generation, run, raw_outcomes = message[2:]
        if type(generation) is not int or not isinstance(raw_outcomes, tuple):
            self.failed.emit(self.generation, "Malformed plot process outcomes")
            return
        from rate_of_closure.ui.pyqt6.plots_tab_computation import (
            PlotComputationOutcome,
        )

        try:
            outcomes = tuple(
                PlotComputationOutcome(
                    item.row,
                    item.data.restore() if item.data is not None else None,
                    item.error,
                )
                for item in raw_outcomes
                if isinstance(item, ProcessPlotOutcome)
            )
        except Exception as exc:  # noqa: BLE001 - bounded IPC validation
            self.failed.emit(self.generation, f"Malformed plot process outcomes: {exc}")
            return
        if len(outcomes) != len(raw_outcomes):
            self.failed.emit(self.generation, "Malformed plot process outcomes")
            return
        self.succeeded.emit(generation, run, outcomes)

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._timer.stop()
        self.finished.emit()


__all__ = ["PlotComputeProcess", "PlotProcessRequest"]


if __name__ == "__main__" and "--worker" in sys.argv:
    raise SystemExit(_worker_main())
