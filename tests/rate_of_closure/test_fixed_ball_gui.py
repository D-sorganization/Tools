"""PyQt hit/miss presentation contracts for fixed-ball contact."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.simulation import ContactMode, ImpactStatus  # noqa: E402
from rate_of_closure.ui.pyqt6.simulation_tab import (  # noqa: E402
    LAUNCH_ROWS,
    SimulationTab,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def tab(qtbot):  # type: ignore[no-untyped-def]
    widget = SimulationTab()
    qtbot.addWidget(widget)
    yield widget
    widget.stop()


def _select_fixed_ball_miss(tab: SimulationTab) -> None:
    tab._source_combo.setCurrentIndex(1)  # double pendulum
    tab._contact_combo.setCurrentIndex(1)  # fixed-ball contact


def test_contact_policy_defaults_to_delivery_inspection(tab) -> None:  # type: ignore[no-untyped-def]
    assert tab._contact_combo.currentData() is ContactMode.DELIVERY_INSPECTION
    assert tab.config().contact_mode is ContactMode.DELIVERY_INSPECTION
    assert tab._scrub_slider.isEnabled()
    assert tab._auto_tau_button.isEnabled()
    assert "forced alignment" in tab._contact_description.text().lower()


def test_fixed_ball_policy_disables_and_explains_impact_scrubbing(tab) -> None:  # type: ignore[no-untyped-def]
    tab._contact_combo.setCurrentIndex(1)

    assert tab.config().contact_mode is ContactMode.FIXED_BALL_CONTACT
    assert tab.config().impact_time_s is None
    assert not tab._scrub_slider.isEnabled()
    assert not tab._auto_tau_button.isEnabled()
    assert "fixed-ball" in tab._scrub_label.text().lower()
    assert "sampled" in tab._delivery_label.text().lower()
    assert "Source:" in tab._contact_combo.toolTip()


def test_fixed_ball_miss_populates_all_null_safe_consumers(tab, qtbot) -> None:  # type: ignore[no-untyped-def]
    _select_fixed_ball_miss(tab)

    with qtbot.waitSignal(tab.runCompleted, timeout=10_000):
        run = tab.run_now()

    assert run is not None
    assert run.impact_outcome.status is ImpactStatus.MISS
    assert "Completed — No Impact" in tab._run_status.text()
    assert "closest approach" in tab._run_status.text().lower()
    assert all(
        "No Impact" in tab._rows[field].value_label.text()
        for field, _label, _unit in LAUNCH_ROWS
    )

    view = tab.view()
    assert view.run() is run
    assert view._inspection_button.text() == "Jump to Closest Approach"
    assert "Closest Approach Kinematics" in view._impact_kinematics_readout.text()
    assert "no shaft-twist degree of freedom" in view._impact_kinematics_readout.text()
    view._inspection_button.click()
    assert view.playback_time() == pytest.approx(run.inspection_time_s)
    view.set_playback_time(run.total_duration_s)
    assert view.playback_time() == pytest.approx(run.swing_times[-1])
    assert "no impact" in view._axes.get_title().lower()

    assert tab.strike_view().run() is run
    assert tab.strike_view().strike_history() == []
    assert "no impact" in tab.strike_view()._axes.get_title().lower()

    assert tab.flight_view().trajectory().shape == (0, 3)
    assert "no flight" in tab.flight_view()._figure.axes[0].get_title().lower()

    inspector = tab.inspector()
    assert inspector.run() is run
    assert inspector._table.rowCount() == len(run.swing_times)
    assert "no impact" in inspector._summary_label.text().lower()
    assert inspector._export_csv_button.isEnabled()
    assert inspector._export_json_button.isEnabled()

    kinetics = tab.kinetics_panel()
    assert kinetics.table().rowCount() > 0
    assert "closest approach" in kinetics._status.text().lower()
    assert len(kinetics._figure.axes[0].lines[0].get_xdata()) == len(run.swing_times)


def test_default_hit_remains_fully_inspectable(tab, qtbot) -> None:  # type: ignore[no-untyped-def]
    with qtbot.waitSignal(tab.runCompleted, timeout=10_000):
        run = tab.run_now()

    assert run is not None and run.impact_outcome.status is ImpactStatus.HIT
    assert "Completed — Hit" in tab._run_status.text()
    assert tab._scrub_slider.isEnabled()
    assert all(
        tab._rows[field].value_label.text() != "—"
        for field, _label, _unit in LAUNCH_ROWS
    )
    assert "Carry" in tab.inspector()._summary_label.text()


def test_changed_configuration_is_stale_and_errors_are_inline(tab, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    tab.run_now()
    next_index = (tab._club_combo.currentIndex() + 1) % tab._club_combo.count()
    tab._club_combo.setCurrentIndex(next_index)
    assert "Stale" in tab._run_status.text()

    def fail_run(_config):  # type: ignore[no-untyped-def]
        raise RuntimeError("intentional physics failure")

    monkeypatch.setattr(
        "rate_of_closure.ui.pyqt6.simulation_tab_publication.run_simulation", fail_run
    )
    assert tab.run_now() is None
    assert "Error" in tab._run_status.text()
    assert "intentional physics failure" in tab._run_status.text()
