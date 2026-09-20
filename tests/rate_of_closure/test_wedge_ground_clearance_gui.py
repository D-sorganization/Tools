"""PyQt wedge ground-clearance presentation contracts."""

from __future__ import annotations

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import SimulationConfig, run_simulation
from rate_of_closure.ui.pyqt6.simulation_view import SimulationView


def _run(club_name: str):
    return run_simulation(
        SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=30.0),
            club=get_club(club_name),
            impact_time_s=0.03,
        )
    )


def test_wedge_run_adds_ground_clearance_to_engineering_readout(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = SimulationView()
    qtbot.addWidget(view)

    view.set_run(_run("Sand Wedge"))

    text = view._impact_kinematics_readout.text()
    assert "Wedge Ground-Clearance Sequence" in text
    assert "Leading-Edge Clearance at Ball" in text
    assert "Sole-Entry Margin" in text
    assert "Bounce-Utilization Angle Margin" in text
    assert "Illustrative 10-degree mid-bounce sole" in text
    labels = [artist.get_label() for artist in view._axes.lines]
    labels.extend(artist.get_label() for artist in view._axes.collections)
    assert "Wedge Sole Envelope" in labels
    assert "Ball Contact" in labels
    assert "Ground Contact" in labels
    assert "Swept Low Point" in labels


def test_non_wedge_run_does_not_claim_wedge_ground_clearance(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = SimulationView()
    qtbot.addWidget(view)

    view.set_run(_run("Driver 10.5°"))

    assert "Wedge Ground-Clearance" not in view._impact_kinematics_readout.text()
    labels = [artist.get_label() for artist in view._axes.lines]
    assert "Wedge Sole Envelope" not in labels
