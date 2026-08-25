"""GUI contracts for the drift-transfer diagnostics panel."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import double_pendulum_golf.gui.transfer_strategy_panel as panel_module
from double_pendulum_golf.gui.transfer_strategy_panel import TransferStrategyPanel
from double_pendulum_golf.gui.analysis_tab import AnalysisTab
from double_pendulum_golf.transfer_strategy import TransferSignals


def _signals() -> TransferSignals:
    time = np.array([0.0, 0.1, 0.2, 0.3])
    velocity = np.column_stack((np.ones(4), np.zeros(4)))
    drift = np.column_stack((np.array([1.0, 2.0, 3.0, 4.0]), np.zeros(4)))
    control = np.column_stack((np.array([0.0, -1.0, -2.0, -1.0]), np.zeros(4)))
    return TransferSignals(
        time_s=time,
        proximal_angular_velocity_rad_s=np.array([4.0, 6.0, 8.0, 10.0]),
        distal_speed_m_s=np.array([8.0, 12.0, 18.0, 25.0]),
        distal_kinetic_energy_j=np.array([2.0, 4.0, 8.0, 14.0]),
        grip_velocity_m_s=velocity,
        grip_force_total_n=drift + control,
        grip_force_drift_n=drift,
        grip_force_control_n=control,
        wrist_control_couple_nm=np.array([0.0, -1.0, -1.0, 0.0]),
        club_angular_velocity_rad_s=np.array([2.0, 3.0, 4.0, 5.0]),
        model_tier="exact_planar_double_pendulum",
    )


def _attribution():
    def metric(signed: float, absolute: float, work: float):
        return SimpleNamespace(
            signed_tangent_impulse_n_s=signed,
            absolute_tangent_impulse_n_s=absolute,
            generalized_work_j=work,
        )

    return SimpleNamespace(
        metrics={
            "coriolis": metric(-2.0, 5.0, -12.0),
            "squared_speed": metric(-3.0, 4.0, -8.0),
            "gravity": metric(1.0, 2.0, 3.0),
            "damping": metric(-0.5, 0.5, -1.0),
            "applied": metric(8.0, 8.0, 22.0),
        }
    )


def test_panel_reports_phase_summary_and_model_boundary(qapp, monkeypatch) -> None:
    monkeypatch.setattr(panel_module, "double_pendulum_transfer_signals", lambda _: _signals())
    monkeypatch.setattr(
        panel_module, "double_pendulum_force_attribution", lambda _: _attribution()
    )
    panel = TransferStrategyPanel()

    panel.set_result(MagicMock(), "double")

    assert panel._status.text() == "Exact Planar Double Pendulum"
    assert "12.000 J" in panel._metric_labels["distal_energy_gain_j"].text()
    assert "-0.700 J" in panel._metric_labels["wrist_control_work_j"].text()
    assert panel._end_spin.value() == pytest.approx(0.3)
    assert panel._source_labels["coriolis"][0].text() == "-2.000 N s"
    assert panel._source_labels["coriolis"][1].text() == "5.000 N s"
    assert panel._source_labels["coriolis"][2].text() == "-12.000 J"

    panel.set_result(MagicMock(), "golfer")
    assert "not yet qualified" in panel._status.text().lower()


def test_panel_updates_declared_window(qapp, monkeypatch) -> None:
    monkeypatch.setattr(panel_module, "double_pendulum_transfer_signals", lambda _: _signals())
    monkeypatch.setattr(
        panel_module, "double_pendulum_force_attribution", lambda _: _attribution()
    )
    panel = TransferStrategyPanel()
    panel.set_result(MagicMock(), "double")

    panel._start_spin.setValue(0.1)
    panel._end_spin.setValue(0.2)
    panel.refresh()

    assert "4.000 J" in panel._metric_labels["distal_energy_gain_j"].text()
    assert panel._last_summary is not None
    assert panel._last_summary.start_s == pytest.approx(0.1)
    assert panel._last_summary.end_s == pytest.approx(0.2)


def test_panel_preserves_fail_closed_source_warning(qapp, monkeypatch) -> None:
    monkeypatch.setattr(panel_module, "double_pendulum_transfer_signals", lambda _: _signals())

    def reject_unsupported_terms(_):
        raise ValueError("Coulomb friction is unsupported")

    monkeypatch.setattr(
        panel_module, "double_pendulum_force_attribution", reject_unsupported_terms
    )
    panel = TransferStrategyPanel()

    panel.set_result(MagicMock(), "double")
    panel.refresh()

    assert panel._attribution is None
    assert "coordinate sources unavailable" in panel._status.text().lower()
    assert "coulomb friction is unsupported" in panel._status.text().lower()
    assert panel._source_labels["coriolis"][0].text() == "-- N s"


def test_panel_rejects_inverted_window(qapp, monkeypatch) -> None:
    monkeypatch.setattr(panel_module, "double_pendulum_transfer_signals", lambda _: _signals())
    panel = TransferStrategyPanel()
    panel.set_result(MagicMock(), "double")
    panel._start_spin.setValue(0.2)
    panel._end_spin.setValue(0.1)

    panel.refresh()

    assert "start must precede end" in panel._status.text().lower()


def test_analysis_tab_exposes_drift_transfer_panel(qapp) -> None:
    tab = AnalysisTab()

    labels = [tab._plot_tabs.tabText(index) for index in range(tab._plot_tabs.count())]

    assert "Drift Transfer" in labels
