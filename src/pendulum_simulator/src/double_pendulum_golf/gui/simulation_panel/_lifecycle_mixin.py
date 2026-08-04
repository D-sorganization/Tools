# ruff: noqa: E501
"""
Simulation lifecycle helpers for SimulationPanel.

Contains:
- ``_on_run`` — validate params and launch the background worker
- ``_on_sim_done`` / ``_on_sim_error`` — main-thread completion callbacks
- ``_show_busy`` / ``_show_progress`` — busy indicator UI
- ``_apply_optimized_coefficients`` — apply optimizer results to controls

Factored out of the original ``simulation_panel.py`` to keep the main
``SimulationPanel`` class focused on widget construction and playback.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtWidgets import QLabel, QMessageBox, QProgressBar, QWidget

from ._worker import _SimWorker

logger = logging.getLogger(__name__)


class _SimulationLifecycleMixin:
    """Mixin providing the simulation lifecycle (run / done / error).

    Expected host attributes (provided by ``SimulationPanel``):
      - ``self.controls``              — user-facing control widget
      - ``self._params_builder`` / ``self._torque_builder`` / ``self._state_builder``
      - ``self._run_simulation`` callable
      - ``self._limits_builder`` / ``self._clamp_builder`` (optional)
      - ``self._progress_bar`` (``QProgressBar``)
      - ``self._display_frame`` / ``self.sim_started`` / ``self.sim_finished``
      - ``self.pendulum`` / ``self.matrix`` / ``self.torque_history``
    """

    # Declared attributes for type hints only
    controls: Any
    pendulum: Any
    matrix: Any
    torque_history: Any
    optimizer: Any
    _params_builder: Any
    _torque_builder: Any
    _state_builder: Any
    _run_simulation: Any
    _limits_builder: Any
    _clamp_builder: Any
    _progress_bar: QProgressBar
    _result: Any
    _anim_idx: int
    _sim_dt: float
    sim_started: Any
    sim_finished: Any

    def _on_run(self) -> None:
        from ..diagnostics import get_tracker

        try:
            p = self.controls.get_params()
        except ValueError as e:
            logger.warning("Parameter validation failed: %s", e)
            get_tracker().record_exception(
                "simulation", e, context="Parameter validation"
            )
            QMessageBox.warning(self, "Input Error", str(e))  # type: ignore[arg-type]
            return

        try:
            params = self._params_builder(p)
        except (ValueError, TypeError, KeyError) as e:
            logger.warning("Parameter build failed: %s", e, exc_info=True)
            get_tracker().record_exception("simulation", e, context="Parameter build")
            QMessageBox.warning(self, "Parameter Error", str(e))  # type: ignore[arg-type]
            return

        if p["t_end"] <= 0:
            QMessageBox.warning(self, "Input Error", "Duration must be positive")  # type: ignore[arg-type]
            return

        try:
            initial_state = self._state_builder(p)
            torque_func = self._torque_builder(p)
        except (ValueError, TypeError, KeyError) as e:
            logger.warning("State/torque build failed: %s", e, exc_info=True)
            get_tracker().record_exception(
                "simulation", e, context="State/torque build"
            )
            QMessageBox.warning(self, "Build Error", str(e))  # type: ignore[arg-type]
            return

        self.controls.btn_run.setEnabled(False)
        self.controls.btn_reset.setEnabled(False)
        self._show_busy(True)
        self._show_progress(True)
        self.sim_started.emit()
        logger.info(
            "Simulation started: t_end=%.3f, dt=%.4f",
            p["t_end"],
            float(p.get("dt", 0.005)),
        )

        # Build optional joint limits and torque clamp
        limits = self._limits_builder(p) if self._limits_builder else None
        clamp = self._clamp_builder(p) if self._clamp_builder else None

        # Build kwargs for the runner function
        run_kwargs: dict = dict(
            params=params,
            initial_state=initial_state,
            t_end=p["t_end"],
            torque_func=torque_func,
            dt=float(p.get("dt", 0.005)),
        )
        if limits is not None:
            run_kwargs["limits"] = limits
        if clamp is not None:
            run_kwargs["clamp"] = clamp

        # Spin up background thread
        self._sim_thread = QThread()
        self._sim_worker = _SimWorker(self._run_simulation, run_kwargs)
        self._sim_worker.moveToThread(self._sim_thread)
        self._sim_thread.started.connect(self._sim_worker.run)
        self._sim_worker.progress.connect(self._progress_bar.setValue)
        self._sim_worker.finished.connect(self._on_sim_done)
        self._sim_worker.error.connect(self._on_sim_error)
        self._sim_worker.finished.connect(self._sim_thread.quit)
        self._sim_worker.error.connect(self._sim_thread.quit)
        self._sim_thread.finished.connect(self._sim_thread.deleteLater)
        self._sim_thread.start()

    def _on_sim_done(self, result: object) -> None:
        """Called on the main thread when simulation completes.

        Pre: result has n_steps, t, states attributes (TrajectoryResultMixin).
        """
        if result is None:
            raise ValueError("Simulation result must not be None")
        assert hasattr(result, "n_steps"), "Result must have n_steps attribute"
        assert hasattr(result, "t"), "Result must have t attribute"

        res: Any = result  # pyqtSignal emits object; cast for attribute access

        self._show_busy(False)
        self._show_progress(False)
        self.controls.btn_run.setEnabled(True)
        self.controls.btn_reset.setEnabled(True)
        self.sim_finished.emit()

        self._result = res
        self._anim_idx = 0

        # Compute simulation dt for real-time playback (#1115)
        if res.n_steps > 1:
            self._sim_dt = float(res.t[-1] - res.t[0]) / (res.n_steps - 1)
        else:
            self._sim_dt = 0.005

        logger.info(
            "Simulation finished: %d steps, t=[0, %.3f]s, dt=%.4fms",
            res.n_steps,
            float(res.t[-1]),
            self._sim_dt * 1000,
        )

        self.pendulum.set_simulation(res)
        self.matrix.set_simulation(res)
        if self.torque_history is not None:
            self.torque_history.set_simulation(res)
        self.controls.set_slider_range(res.n_steps - 1)
        self.controls.set_slider_value(0)
        self._display_frame(0)

        # Auto-play
        self.controls.btn_play.setChecked(True)

    def _on_sim_error(self, msg: str) -> None:
        """Called on the main thread when simulation fails."""
        if msg is None:
            raise ValueError("msg must be provided")
        from ..diagnostics import get_tracker

        logger.error("Simulation failed: %s", msg)
        get_tracker().record(
            "simulation",
            f"Simulation failed: {msg}",
            severity="error",
            details=msg,
        )
        self._show_busy(False)
        self._show_progress(False)
        self.controls.btn_run.setEnabled(True)
        self.controls.btn_reset.setEnabled(True)
        self.sim_finished.emit()
        QMessageBox.critical(self, "Simulation Error", msg)  # type: ignore[arg-type]

    def _show_busy(self, busy: bool) -> None:
        """Show / hide a 'Simulating…' indicator in the top-right."""
        if busy is None:
            raise ValueError("busy must be provided")
        host = cast(QWidget, self)  # Mixin used only on QWidget subclasses
        if not hasattr(self, "_busy_label"):
            self._busy_label = QLabel("…  Simulating", host)
            self._busy_label.setStyleSheet(
                "background: #202040; color: #b0b0e8; border: 1px solid #404070;"
                "border-radius: 4px; padding: 4px 10px; font-size: 12px;"
            )
            self._busy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if busy:
            # Position top-centre of this panel
            self._busy_label.setFixedWidth(160)
            self._busy_label.move(
                (host.width() - 160) // 2,
                8,
            )
            self._busy_label.raise_()
            self._busy_label.show()
        else:
            self._busy_label.hide()

    def _show_progress(self, show: bool) -> None:
        """Show / hide the progress bar during simulation."""
        host = cast(QWidget, self)  # Mixin used only on QWidget subclasses
        if show:
            # Position below busy indicator
            self._progress_bar.move(
                (host.width() - 200) // 2,
                38,
            )
            self._progress_bar.raise_()
            self._progress_bar.show()
        else:
            self._progress_bar.hide()

    def _apply_optimized_coefficients(self, result: object) -> None:
        """Apply optimizer results back to the control widget's torque fields.

        Splits the flat coefficient array into per-joint polynomial strings.
        Supports double (2), triple (3), and golfer (7) torque-param models.

        Pre: result has 'coeffs' key with a numpy array.
        Closes #1151.
        """
        if result is None:
            raise ValueError("result must be provided")
        _log = logging.getLogger(__name__)

        if not isinstance(result, dict):
            _log.warning("Optimizer result is not a dict, skipping apply")
            return
        coeffs = result.get("coeffs")
        if coeffs is None:
            _log.warning("No coefficients in optimizer result")
            return

        coeffs = np.asarray(coeffs, dtype=float)

        # Determine model type by which control widget is active
        from ..controls_widget import ControlsWidget
        from ..controls_widget_golfer import ControlsWidgetGolfer
        from ..controls_widget_triple import ControlsWidgetTriple

        def _fmt_coeffs(arr: np.ndarray) -> str:
            """Format coefficient array as comma-separated string."""
            return ", ".join(f"{v:.4f}" for v in arr)

        if isinstance(self.controls, ControlsWidget):
            # Double: split into 2 groups (shoulder, wrist)
            n_half = len(coeffs) // 2
            self.controls.inp_tau_shoulder.set_value(_fmt_coeffs(coeffs[:n_half]))
            self.controls.inp_tau_wrist.set_value(_fmt_coeffs(coeffs[n_half:]))
            _log.info("Applied double pendulum optimizer coefficients")

        elif isinstance(self.controls, ControlsWidgetTriple):
            # Triple: split into 3 groups (shoulder, elbow, wrist)
            n_third = len(coeffs) // 3
            self.controls.inp_tau_shoulder.set_value(_fmt_coeffs(coeffs[:n_third]))
            self.controls.inp_tau_elbow.set_value(
                _fmt_coeffs(coeffs[n_third : 2 * n_third])
            )
            self.controls.inp_tau_wrist.set_value(_fmt_coeffs(coeffs[2 * n_third :]))
            _log.info("Applied triple pendulum optimizer coefficients")

        elif isinstance(self.controls, ControlsWidgetGolfer):
            # Golfer: split into 7 groups
            n_seventh = max(1, len(coeffs) // 7)
            field_names = [
                "inp_tau_hub",
                "inp_tau_rs",
                "inp_tau_re",
                "inp_tau_rh",
                "inp_tau_ls",
                "inp_tau_le",
                "inp_tau_lh",
            ]
            for i, field_name in enumerate(field_names):
                field = getattr(self.controls, field_name, None)
                if field is not None:
                    start = i * n_seventh
                    end = (i + 1) * n_seventh if i < 6 else len(coeffs)
                    field.set_value(_fmt_coeffs(coeffs[start:end]))
            _log.info("Applied golfer optimizer coefficients")

    def _display_frame(self, idx: int) -> None:  # pragma: no cover - overridden
        """Abstract mixin stub — overridden by SimulationPanel (the concrete class).

        This raise is intentional: the mixin cannot render frames standalone.
        Do not implement here; the concrete class provides the implementation.
        """
        raise NotImplementedError

    # QWidget-compat stubs so mypy/mixin checks don't trip over method calls
    def width(self) -> int:  # pragma: no cover
        """QWidget-compat stub — provided by the QWidget base class at runtime.

        Raises NotImplementedError in mixin context where QWidget is not yet
        in the MRO.  This is intentional for static analysis compatibility.
        """
        raise NotImplementedError
