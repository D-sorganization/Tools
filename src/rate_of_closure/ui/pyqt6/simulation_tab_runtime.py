"""Runtime and scrub behavior for the PyQt simulation-session tab."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from PyQt6.QtWidgets import QComboBox, QGroupBox, QLabel, QPushButton, QSlider

from rate_of_closure.club import get_club
from rate_of_closure.model import MPH_PER_MPS, ImpactScenario
from rate_of_closure.simulation import (
    SOURCE_KINDS,
    ContactMode,
    SimulationConfig,
    SimulationRun,
    delivery_at,
    make_source,
)
from rate_of_closure.ui.pyqt6.simulation_tab_controls import SCRUB_STEPS
from rate_of_closure.ui.pyqt6.torque_profile_controller import RunMode
from rate_of_closure.ui.pyqt6.torque_profile_panel import TorqueProfilePanel
from shared.python.swing_sim.types import PlaneOrientation

logger = logging.getLogger(__name__)


class SimulationTabRuntimeMixin:
    """Manage cached dynamics, contact policy, and live swing scrubbing."""

    _auto_tau_button: QPushButton
    _club_combo: QComboBox
    _contact_combo: QComboBox
    _contact_description: QLabel
    _delivery_label: QLabel
    _run: SimulationRun | None
    _run_status: QLabel
    _scenario: ImpactScenario
    _scrub_box: QGroupBox
    _scrub_label: QLabel
    _scrub_slider: QSlider
    _source: Any
    _source_combo: QComboBox
    _tau: float | None
    _torque_profile_panel: TorqueProfilePanel

    if TYPE_CHECKING:

        def _emit_config(self, *_args: object) -> None: ...

        def config(self) -> SimulationConfig: ...

        def contact_mode(self) -> ContactMode: ...

        def plane(self) -> PlaneOrientation: ...

        def run_now(self) -> SimulationRun | None: ...

        def source_kind(self) -> str: ...

    def _ensure_source(self):  # type: ignore[no-untyped-def]
        if self._source is None:
            config = self.config()
            self._source = make_source(
                config.source_kind,
                self._scenario,
                plane=self.plane(),
                duration=config.swing_duration_s,
                run_config=config.swing_run_config,
                torque_library=config.torque_library,
                pendulum_parameters=config.pendulum_parameters,
                manual_delivery=config.manual_delivery,
            )
        return self._source

    def _on_torque_selection_changed(self, *_args: object) -> None:
        """Keep the visible source and cached dynamics aligned with run mode."""
        selection = self._torque_profile_panel.selection()
        if (
            selection.mode is RunMode.PRESCRIBED_TORQUE
            and selection.profile is not None
            and selection.profile.model_id == "model.double_pendulum.v1"
        ):
            self._source_combo.setCurrentIndex(SOURCE_KINDS.index("double_pendulum"))
        self._emit_config()

    def _on_joint_locks_changed(self, *_args: object) -> None:
        """Select the compatible kernel whenever an ideal lock is enabled."""
        if self._torque_profile_panel.joint_locks().has_locks:
            self._source_combo.setCurrentIndex(SOURCE_KINDS.index("double_pendulum"))
        self._emit_config()

    def _reconcile_joint_locks_for_source(self, *_args: object) -> None:
        """Clear constraints when the user explicitly leaves the supported source."""
        if self.source_kind() != "double_pendulum":
            self._torque_profile_panel.clear_joint_locks(emit=False)

    def _fit_current_run(self, degree: int) -> None:
        """Fit the current retained double-pendulum torque history non-modally."""
        if self._run is None:
            self._torque_profile_panel.set_fit_error(
                "run a double-pendulum simulation first."
            )
            return
        self._torque_profile_panel.fit_current_run(self._run, degree)

    def _on_contact_mode_changed(self, *_args: object) -> None:
        """Reset incompatible impact-time state and explain the active policy."""
        self._tau = None
        self._update_contact_controls()
        self._mark_stale()
        self._emit_config()

    def _update_contact_controls(self) -> None:
        fixed_ball = self.contact_mode() is ContactMode.FIXED_BALL_CONTACT
        if fixed_ball:
            description = (
                "Retains the swing in its original frame and detects sampled "
                "clubhead-reference-point proximity to the fixed ball. A miss is "
                "a valid completed result; mesh contact and swept collision are "
                "not modeled."
            )
        else:
            description = (
                "Forced alignment translates the swing onto the ball at the "
                "selected inspection time. Use this to inspect delivery; it is "
                "not geometric contact detection."
            )
        self._contact_description.setText(description)
        if not hasattr(self, "_scrub_slider"):
            return
        self._scrub_slider.setEnabled(not fixed_ball)
        self._auto_tau_button.setEnabled(not fixed_ball)
        if fixed_ball:
            self._scrub_box.setTitle("Contact Detection (Fixed Ball)")
            self._scrub_label.setText("fixed-ball")
            self._delivery_label.setText(
                "Impact time is detected from sampled fixed-ball proximity; "
                "manual scrubbing is unavailable."
            )
        else:
            self._scrub_box.setTitle("Impact Time (Scrub the Swing Onto the Ball)")
            self._scrub_label.setText("auto")
            self._delivery_label.setText("Awaiting updated simulation")

    def _invalidate_source(self, *_args: object) -> None:
        self._source = None
        self._tau = None
        if hasattr(self, "_scrub_label"):
            self._update_contact_controls()
            self._mark_stale()

    def _scrub_time(self, value: int) -> float:
        source = self._ensure_source()
        duration: float = source.duration
        return float(value / SCRUB_STEPS * duration)

    def _sync_scrub_slider(self, tau: float) -> None:
        source = self._ensure_source()
        value = (
            round(tau / source.duration * SCRUB_STEPS) if source.duration > 0.0 else 0
        )
        self._scrub_slider.blockSignals(True)
        self._scrub_slider.setValue(value)
        self._scrub_slider.blockSignals(False)
        self._scrub_label.setText(f"{tau * 1000.0:.1f} ms")

    def _sync_scrub_after_run(self, run: SimulationRun) -> None:
        """Reflect a detected impact or closest-approach sample without fabrication."""
        if run.impact_time_s is not None:
            self._sync_scrub_slider(run.impact_time_s)
            return
        source = self._ensure_source()
        candidate = run.impact_outcome.candidate_time_s
        value = (
            round(candidate / source.duration * SCRUB_STEPS)
            if source.duration > 0.0
            else 0
        )
        self._scrub_slider.blockSignals(True)
        self._scrub_slider.setValue(value)
        self._scrub_slider.blockSignals(False)
        self._scrub_label.setText(f"closest {candidate * 1000.0:.1f} ms")

    def _update_outcome_labels(self, run: SimulationRun) -> None:
        """Show delivery for hits and proximity diagnostics for misses."""
        if run.impact_time_s is not None:
            self._update_delivery_label(run.impact_time_s)
            return
        outcome = run.impact_outcome
        miss_distance_mm = outcome.closest_approach_m * 1000.0
        threshold_mm = outcome.contact_threshold_m * 1000.0
        self._delivery_label.setText(
            f"No impact detected — closest sampled approach {miss_distance_mm:.1f} "
            f"mm; contact threshold {threshold_mm:.1f} mm."
        )

    def _update_delivery_label(self, tau: float) -> None:
        try:
            source = self._ensure_source()
            delivery = delivery_at(
                source, tau, self._scenario, get_club(self._club_combo.currentText())
            )
        except Exception as exc:  # noqa: BLE001 — zero-speed instants etc.
            self._delivery_label.setText(f"No delivery at this instant ({exc})")
            return
        velocity = delivery.clubhead_velocity
        vx, vy, vz = (float(component) for component in velocity)
        speed_mph = float(np.linalg.norm(velocity)) * MPH_PER_MPS
        path = math.degrees(math.atan2(vz, vx))
        aoa = math.degrees(math.atan2(vy, math.hypot(vx, vz)))
        self._delivery_label.setText(
            f"Delivery at τ: {speed_mph:.1f} mph, path {path:+.1f}°, "
            f"AoA {aoa:+.1f}°, spin loft {delivery.spin_loft_deg:.1f}°"
        )

    def _on_scrub_moved(self, value: int) -> None:
        if self.contact_mode() is ContactMode.FIXED_BALL_CONTACT:
            return
        tau = self._scrub_time(value)
        self._tau = tau
        self._scrub_label.setText(f"{tau * 1000.0:.1f} ms")
        self._update_delivery_label(tau)
        if not self._scrub_slider.isSliderDown() and self._run is not None:
            self.run_now()

    def _on_scrub_released(self) -> None:
        if self.contact_mode() is ContactMode.FIXED_BALL_CONTACT:
            return
        if self._run is not None:
            self.run_now()

    def _on_auto_tau(self) -> None:
        if self.contact_mode() is ContactMode.FIXED_BALL_CONTACT:
            return
        self._tau = None
        self.run_now()

    def _set_completed_status(self, run: SimulationRun) -> None:
        outcome = run.impact_outcome
        lock_summary = self._torque_profile_panel.joint_lock_summary()
        if outcome.is_hit:
            assert run.impact_time_s is not None
            text = (
                f"Completed — Hit at {run.impact_time_s * 1000.0:.1f} ms. "
                "Swing, impact, launch, and flight results are current. "
                f"Joint constraints: {lock_summary}."
            )
            self._set_run_status(text, "hit")
            return
        clearance_mm = -outcome.contact_margin_m * 1000.0
        text = (
            "Completed — No Impact. The closest approach remained "
            f"{clearance_mm:.1f} mm outside the sampled contact threshold. "
            "Swing playback and pendulum kinetics remain available; impact, "
            "launch, and flight values are unavailable. "
            f"Joint constraints: {lock_summary}."
        )
        self._set_run_status(text, "miss")

    def _mark_stale(self) -> None:
        if not hasattr(self, "_run_status"):
            return
        self._set_run_status(
            "Stale — Configuration changed. Run Simulation to refresh results.",
            "stale",
        )

    def _set_run_status(self, text: str, state: str) -> None:
        self._run_status.setText(text)
        self._run_status.setProperty("runState", state)
        self._run_status.setAccessibleDescription(text)
