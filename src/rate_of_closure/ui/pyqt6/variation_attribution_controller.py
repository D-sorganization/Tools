"""Generation-safe controller for explicit localized paired studies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt6.QtWidgets import QDialog, QFileDialog, QWidget

from rate_of_closure.ui.pyqt6.localized_attribution_archive import (
    ARCHIVED_AUTHORITY_DISCLAIMER,
    read_authority_json,
    write_authority_json,
)
from rate_of_closure.ui.pyqt6.localized_attribution_dialog import (
    LocalizedAttributionRunDialog,
)
from rate_of_closure.ui.pyqt6.localized_attribution_worker import (
    LocalizedAttributionWorker,
)
from rate_of_closure.variation.localized_attribution_producer import (
    LocalizedAttributionDesign,
    LocalizedAttributionProduction,
)
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import (
    LOCALIZED_TORQUE_VARIABLE_JOINTS,
    VariationPlan,
)

AttributionWorkerFactory = Callable[
    [LocalizedAttributionDesign], LocalizedAttributionWorker
]


class VariationAttributionControllerMixin:
    """Keep paired-study execution independent of ordinary Monte Carlo state."""

    _attribution_worker: LocalizedAttributionWorker | None
    _attribution_generation: int
    _attribution_production: LocalizedAttributionProduction | None
    _attribution_worker_factory: AttributionWorkerFactory
    _localized_attribution: Any
    _simulation_config_valid: bool
    _base_simulation_config: Any

    def build_plan(self) -> VariationPlan:
        """Return the concrete variation plan."""
        raise NotImplementedError

    def _initialize_attribution_controller(self) -> None:
        self._attribution_worker = None
        self._attribution_generation = 0
        self._attribution_production = None
        self._attribution_worker_factory = LocalizedAttributionWorker
        view = self._localized_attribution
        view._configure_run.clicked.connect(self._on_configure_attribution)
        view._cancel_study.clicked.connect(self._on_cancel_attribution)
        view._save_authority.clicked.connect(self._on_save_attribution_authority)
        view._load_authority.clicked.connect(self._on_load_attribution_authority)
        self._refresh_attribution_capability()

    def attribution_production(self) -> LocalizedAttributionProduction | None:
        """Return live verified production; archives deliberately return None."""
        return self._attribution_production

    def _attribution_capability(self) -> tuple[bool, str]:
        if not self._simulation_config_valid:
            return False, "Current Simulation inputs are incomplete or invalid."
        if self._base_simulation_config.source_kind != "double_pendulum":
            return False, "Localized paired studies require the double-pendulum source."
        try:
            plan = self.build_plan()
        except (ContractViolationError, TypeError, ValueError) as error:
            return False, f"Cannot configure paired study: {error}"
        if plan.mode != "swing":
            return False, "Localized paired studies require Swing pipeline mode."
        if not any(
            spec.variable_key in LOCALIZED_TORQUE_VARIABLE_JOINTS for spec in plan.noise
        ):
            return False, "Add at least one localized shoulder or wrist torque source."
        return True, ""

    def _refresh_attribution_capability(self) -> None:
        enabled, reason = self._attribution_capability()
        self._localized_attribution.set_configure_enabled(enabled, reason)

    def _on_configure_attribution(self) -> None:
        enabled, reason = self._attribution_capability()
        if not enabled:
            self._localized_attribution.set_study_status(reason)
            return
        dialog = LocalizedAttributionRunDialog(
            self.build_plan(), self._base_simulation_config, self._as_widget()
        )
        if dialog.exec() != int(QDialog.DialogCode.Accepted):
            return
        try:
            design = dialog.build_design()
        except (ContractViolationError, TypeError, ValueError) as error:
            self._localized_attribution.set_study_status(
                f"Cannot run paired study: {error}"
            )
            return
        self._start_attribution_worker(design)

    def _start_attribution_worker(self, design: LocalizedAttributionDesign) -> None:
        if (
            self._attribution_worker is not None
            and self._attribution_worker.isRunning()
        ):
            return
        self._attribution_generation += 1
        generation = self._attribution_generation
        worker = self._attribution_worker_factory(design)
        worker.progressed.connect(
            lambda report, current=generation: self._accept_attribution_progress(
                current, report
            )
        )
        worker.succeeded.connect(
            lambda result, current=generation: self._accept_attribution_succeeded(
                current, result
            )
        )
        worker.cancelled.connect(
            lambda current=generation: self._accept_attribution_cancelled(current)
        )
        worker.failed.connect(
            lambda message, current=generation: self._accept_attribution_failed(
                current, message
            )
        )
        worker.finished.connect(
            lambda current=generation, owner=worker: self._accept_attribution_finished(
                current, owner
            )
        )
        self._attribution_worker = worker
        self._localized_attribution.set_study_running(True, worker.total_runs)
        self._localized_attribution.set_study_status(
            "Running separate paired study: 0/"
            f"{worker.total_runs} trials. The displayed authority remains unchanged "
            "until successful completion."
        )
        worker.start()

    def _accept_attribution_progress(self, generation: int, report: object) -> None:
        if generation != self._attribution_generation:
            return
        completed = min(
            int(getattr(report, "iteration", 0)),
            self._localized_attribution._study_progress.maximum(),
        )
        self._localized_attribution._study_progress.setValue(completed)
        total = self._localized_attribution._study_progress.maximum()
        self._localized_attribution.set_study_status(
            f"Running separate paired study: {completed}/{total} trials. "
            "The displayed authority remains unchanged until successful completion."
        )

    def _accept_attribution_succeeded(self, generation: int, result: object) -> None:
        if generation != self._attribution_generation or not isinstance(
            result, LocalizedAttributionProduction
        ):
            return
        self._attribution_production = result
        self._localized_attribution.set_authority(result.authority)
        sources = len(result.authority.sources)
        trials = result.request.plan.n_runs
        self._localized_attribution.set_study_status(
            f"Paired study complete: {sources} sources, {trials} explicit trials. "
            "Loaded planted-intervention response authority; no causal inference."
        )

    def _accept_attribution_cancelled(self, generation: int) -> None:
        if generation == self._attribution_generation:
            self._localized_attribution.set_study_status(
                "Separate paired study cancelled. Prior paired authority was not "
                "replaced."
            )

    def _accept_attribution_failed(self, generation: int, message: str) -> None:
        if generation == self._attribution_generation:
            self._localized_attribution.set_study_status(
                f"Paired study failed before authority replacement: {message}."
            )

    def _accept_attribution_finished(
        self, generation: int, worker: LocalizedAttributionWorker
    ) -> None:
        owns_slot = worker is self._attribution_worker
        if owns_slot:
            self._attribution_worker = None
        if generation == self._attribution_generation:
            self._localized_attribution.set_study_running(False)
            self._refresh_attribution_capability()

    def _on_cancel_attribution(self) -> None:
        if self._attribution_worker is not None:
            self._attribution_worker.cancel()
            self._localized_attribution.set_study_status(
                "Cancelling separate paired study; prior authority remains displayed."
            )

    def _invalidate_attribution(self, reason: str) -> None:
        self._attribution_generation += 1
        if (
            self._attribution_worker is not None
            and self._attribution_worker.isRunning()
        ):
            self._attribution_worker.cancel()
        self._attribution_production = None
        self._localized_attribution.set_authority(None, reason)
        self._localized_attribution.set_study_status(reason)
        self._localized_attribution.set_study_running(False)
        self._refresh_attribution_capability()

    def _on_save_attribution_authority(self) -> None:
        authority = self._localized_attribution.authority()
        if authority is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self._as_widget(),
            "Save Paired Authority JSON",
            "localized_attribution_authority.json",
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            write_authority_json(path, authority)
        except (ContractViolationError, OSError, TypeError, ValueError) as error:
            self._localized_attribution.set_study_status(
                f"Cannot save paired authority: {error}"
            )

    def _on_load_attribution_authority(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self._as_widget(), "Load Paired Authority JSON", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            authority = read_authority_json(path)
        except (ContractViolationError, OSError, TypeError, ValueError) as error:
            self._localized_attribution.set_study_status(
                f"Cannot load paired authority: {error}"
            )
            return
        self._attribution_production = None
        self._localized_attribution.set_authority(authority)
        self._localized_attribution.set_study_status(ARCHIVED_AUTHORITY_DISCLAIMER)

    def _stop_attribution_worker(self) -> None:
        if self._attribution_worker is not None:
            self._attribution_worker.cancel()
            self._attribution_worker.wait(10_000)

    def _as_widget(self) -> QWidget:
        if not isinstance(self, QWidget):
            raise TypeError("attribution controller must be mixed into QWidget")
        return self


__all__ = ["VariationAttributionControllerMixin"]
