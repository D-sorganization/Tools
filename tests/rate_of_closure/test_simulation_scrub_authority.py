"""Exact PyQt Simulation scrub and retained-result execution authority."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402

from rate_of_closure.model import ImpactScenario  # noqa: E402
from rate_of_closure.ui.pyqt6.simulation_tab import SimulationTab  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def ran_tab(qtbot):  # type: ignore[no-untyped-def]
    widget = SimulationTab()
    qtbot.addWidget(widget)
    widget.set_scenario(ImpactScenario(clubhead_speed_mph=113.0))
    widget.run_now()
    yield widget
    widget.stop()


def test_keyboard_scrub_commits_the_exact_new_time(ran_tab) -> None:  # type: ignore[no-untyped-def]
    previous = ran_tab.last_run()
    previous_value = ran_tab._scrub_slider.value()

    QTest.keyClick(ran_tab._scrub_slider, Qt.Key.Key_Right)

    current = ran_tab.last_run()
    assert current is not previous
    assert ran_tab._scrub_slider.value() == previous_value + 1
    assert current.config.impact_time_s == pytest.approx(ran_tab._tau)


def test_failure_retains_the_prior_run_and_scene(ran_tab, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import rate_of_closure.ui.pyqt6.simulation_tab_publication as publication

    prior = ran_tab.last_run()
    prior_view = ran_tab.view().run()
    monkeypatch.setattr(
        publication,
        "run_simulation",
        lambda _config: (_ for _ in ()).throw(RuntimeError("planted failure")),
    )

    assert ran_tab.run_now() is None
    assert ran_tab.last_run() is prior
    assert ran_tab.view().run() is prior_view
    assert (
        "prior accepted scene remains displayed" in ran_tab._run_status.text().lower()
    )


def test_first_failure_stays_empty_and_bounds_its_status(qtbot, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import rate_of_closure.ui.pyqt6.simulation_tab_publication as publication

    tab = SimulationTab()
    qtbot.addWidget(tab)
    monkeypatch.setattr(
        publication,
        "run_simulation",
        lambda _config: (_ for _ in ()).throw(RuntimeError("x" * 600)),
    )

    assert tab.run_now() is None
    assert tab.last_run() is None
    assert tab._run_status.text().count("x") == 512
    assert "no accepted simulation is available" in tab._run_status.text().lower()
    tab.stop()
