"""Headless probe: Impact Explorer 3D playback replays imported records.

ADR-0047 H4 (UD #9353): the Flight Explorer tab's "Import Trajectory
Record…" action loads a ``swing_sim.ball_flight_trajectory/1`` record
and replays it through the *existing* P8 transport
(:class:`~rate_of_closure.ui.pyqt6.flight_playback_controls.FlightPlaybackPanel`)
— this file gates the wiring only; the frame-conversion math itself is
gated in ``tests/rate_of_closure/test_flight_record_playback.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import rate_of_closure.ui.pyqt6.flight_explorer_tab as flight_explorer_tab_module
from rate_of_closure.ui.pyqt6.flight_explorer_tab import FlightExplorerTab
from shared.python.swing_sim.flight_interchange import (
    FLIGHT_FRAME_ID,
    TOOLS_FLIGHT_FAMILY,
    TrajectoryProvenance,
    ball_flight_trajectory_to_json,
    from_samples,
    parameter_digest,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_DIGEST = parameter_digest({"cd": 0.22, "cl": 0.24})


def _record_json(*, model_family: str = TOOLS_FLIGHT_FAMILY) -> str:
    record = from_samples(
        source_id=f"{model_family}:test-model",
        frame_id=FLIGHT_FRAME_ID,
        provenance=TrajectoryProvenance(
            model_family=model_family,
            model_name="test-model",
            parameter_digest=_DIGEST,
        ),
        times_s=[0.0, 0.5, 1.0],
        positions_m=[[0.0, 0.0, 0.0], [10.0, 2.0, 4.0], [20.0, 0.0, 0.0]],
    )
    return ball_flight_trajectory_to_json(record)


def _patch_open_dialog(monkeypatch: pytest.MonkeyPatch, path: Path | str) -> None:
    monkeypatch.setattr(
        flight_explorer_tab_module.QFileDialog,
        "getOpenFileName",
        lambda *_args: (str(path), "Ball Flight Trajectory (*.json)"),
    )


def test_import_replays_a_flight_frame_record_through_the_existing_transport(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "imported.json"
    path.write_text(_record_json(model_family="ud.flight_models"), encoding="utf-8")
    _patch_open_dialog(monkeypatch, path)

    tab = FlightExplorerTab()
    qtbot.addWidget(tab)
    tab.show()

    tab._import_button.click()

    # Frame conversion: app = (flight_x, flight_z, -flight_y).
    expected = np.array([[0.0, 0.0, 0.0], [10.0, 4.0, -2.0], [20.0, 0.0, 0.0]])
    np.testing.assert_allclose(tab.flight_view().trajectory(), expected)
    assert tab.flight_view().playback_duration_s() == pytest.approx(1.0)
    assert tab.accepted_study() is None
    assert "ud.flight_models" in tab._context_status.text()
    assert "test-model" in tab._context_status.text()
    assert "Imported trajectory" in tab._context_status.text()
    assert tab._flight_panel.controls.current_time_s() == pytest.approx(0.0)
    # No new transport was invented: it is the same reusable panel/controls
    # class every solver-produced flight already drives.
    assert tab._flight_panel.controls.play_button.isEnabled()
    assert tab._flight_panel.controls.landing_button.isEnabled()


def test_import_does_not_disturb_a_prior_run_when_the_dialog_is_cancelled(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        flight_explorer_tab_module.QFileDialog,
        "getOpenFileName",
        lambda *_args: ("", ""),
    )
    tab = FlightExplorerTab()
    qtbot.addWidget(tab)
    tab.show()
    assert tab.run_now() is not None
    prior_trajectory = tab.flight_view().trajectory().copy()
    prior_context = tab._context_status.text()

    tab._import_button.click()

    np.testing.assert_allclose(tab.flight_view().trajectory(), prior_trajectory)
    assert tab._context_status.text() == prior_context
    assert tab.accepted_study() is not None


def test_import_refusal_is_named_and_preserves_the_prior_accepted_display(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    tab = FlightExplorerTab()
    qtbot.addWidget(tab)
    tab.show()
    assert tab.run_now() is not None
    prior_trajectory = tab.flight_view().trajectory().copy()
    prior_accepted = tab.accepted_study()

    bad_path = tmp_path / "malformed.json"
    bad_path.write_text('{"format": "not-a-real-format"}', encoding="utf-8")
    _patch_open_dialog(monkeypatch, bad_path)

    tab._import_button.click()

    np.testing.assert_allclose(tab.flight_view().trajectory(), prior_trajectory)
    assert tab.accepted_study() is prior_accepted
    assert tab._error_origin == "import"
    assert tab._error_status.text() != ""
    assert "\x00" not in tab._error_status.text()


def test_import_refuses_malformed_json_text_without_crashing(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    bad_path = tmp_path / "not-json.json"
    bad_path.write_text("not actually json", encoding="utf-8")
    _patch_open_dialog(monkeypatch, bad_path)

    tab = FlightExplorerTab()
    qtbot.addWidget(tab)
    tab.show()

    tab._import_button.click()

    assert tab._error_origin == "import"
    assert len(tab.flight_view().trajectory()) == 0
