"""Non-visual orchestration helpers extracted from the Simulation tab."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from rate_of_closure.club import get_club
from rate_of_closure.derivation_models import DerivationConfig
from rate_of_closure.model import MPH_PER_MPS
from rate_of_closure.simulation import (
    SOURCE_KINDS,
    SimulationConfig,
    SimulationRun,
    delivery_at,
)
from rate_of_closure.ui.pyqt6.result_row import explanation_html


def derivation_config(tab: Any) -> DerivationConfig:
    """Build calculation-description configuration from direct UI controls."""
    plane = tab.plane()
    return DerivationConfig(
        flight_model=tab._flight_combo.currentText(),
        swing_source=tab.source_kind(),
        gear_effect=True,
        plane_tilts_deg=(plane.yaw_deg, plane.side_tilt_deg, plane.forward_tilt_deg),
    )


def simulation_config(
    tab: Any, impact_model_labels: dict[str, str]
) -> SimulationConfig:
    """Build the validated simulation request represented by the controls."""
    selected_label = tab._impact_model_combo.currentText()
    impact_model = next(
        key for key, label in impact_model_labels.items() if label == selected_label
    )
    return SimulationConfig(
        scenario=tab._scenario,
        club=get_club(tab._club_combo.currentText()),
        source_kind=tab.source_kind(),
        plane=tab.plane(),
        impact_time_s=tab._tau,
        flight_model=tab._flight_combo.currentText(),
        impact_model=impact_model,
    )


def apply_solver_solution(
    tab: Any, result: object, use_swing_source: bool
) -> SimulationRun | None:
    """Apply solver variables through the tab's direct public/control seams."""
    variables: dict[str, float] = result.variables  # type: ignore[attr-defined]
    updates = {
        "impact_offset_toe_mm": variables["impact_offset_toe_mm"],
        "impact_offset_high_mm": variables["impact_offset_high_mm"],
    }
    if use_swing_source:
        tab._source_combo.setCurrentIndex(SOURCE_KINDS.index("double_pendulum"))
        for attr, variable in (
            ("yaw_deg", "swing_yaw_deg"),
            ("side_tilt_deg", "swing_side_tilt_deg"),
            ("forward_tilt_deg", "swing_forward_tilt_deg"),
        ):
            spin = tab._tilt_spins[attr]
            spin.blockSignals(True)
            spin.setValue(variables[variable])
            spin.blockSignals(False)
    else:
        tab._source_combo.setCurrentIndex(SOURCE_KINDS.index("manual"))
        updates["clubhead_speed_mph"] = variables["clubhead_speed_mps"] * MPH_PER_MPS
    tab._scenario = dataclasses.replace(tab._scenario, **updates)
    tab._invalidate_source()
    tab._tau = None
    run = tab.run_now()
    offset = variables.get("swing_impact_time_offset_s", 0.0)
    if run is not None and use_swing_source and abs(offset) > 1.0e-9:
        source = tab._ensure_source()
        tab._tau = min(max(run.impact_time_s + offset, 0.0), source.duration)
        run = tab.run_now()
    return run


def sync_scrub_slider(tab: Any, tau: float, scrub_steps: int) -> None:
    """Synchronize the impact-time slider and label without signal recursion."""
    source = tab._ensure_source()
    value = round(tau / source.duration * scrub_steps) if source.duration > 0.0 else 0
    tab._scrub_slider.blockSignals(True)
    tab._scrub_slider.setValue(value)
    tab._scrub_slider.blockSignals(False)
    tab._scrub_label.setText(f"{tau * 1000.0:.1f} ms")


def update_delivery_label(tab: Any, tau: float) -> None:
    """Render the live delivery summary for one scrubbed instant."""
    try:
        source = tab._ensure_source()
        delivery = delivery_at(
            source, tau, tab._scenario, get_club(tab._club_combo.currentText())
        )
    except Exception as exc:  # noqa: BLE001
        tab._delivery_label.setText(f"No delivery at this instant ({exc})")
        return
    vx, vy, vz = (float(component) for component in delivery.clubhead_velocity)
    speed_mph = math.sqrt(vx * vx + vy * vy + vz * vz) * MPH_PER_MPS
    path = math.degrees(math.atan2(vz, vx))
    aoa = math.degrees(math.atan2(vy, math.hypot(vx, vz)))
    tab._delivery_label.setText(
        f"Delivery at τ: {speed_mph:.1f} mph, path {path:+.1f}°, "
        f"AoA {aoa:+.1f}°, spin loft {delivery.spin_loft_deg:.1f}°"
    )


def show_explanation(
    tab: Any, field: str, rows: tuple[tuple[str, str, str], ...]
) -> None:
    """Apply one persistent launch-row selection and its explanation."""
    labels = {key: label for key, label, _unit in rows}
    text = tab._launch_explanations.get(field, "")
    for row_field, row in tab._rows.items():
        row.set_selected(row_field == field)
    tab._explanation.setHtml(explanation_html(labels.get(field, field), text, field))


__all__ = [
    "apply_solver_solution",
    "derivation_config",
    "show_explanation",
    "simulation_config",
    "sync_scrub_slider",
    "update_delivery_label",
]
