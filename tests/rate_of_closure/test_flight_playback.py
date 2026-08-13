"""Deterministic ball-flight playback contract and PyQt controls (#4200)."""

from __future__ import annotations

import numpy as np
import pytest

from rate_of_closure.simulation.flight_playback import TimedTrajectory

pytestmark = pytest.mark.unit


class TestTimedTrajectory:
    def test_interpolates_in_physical_time_and_clamps_endpoints(self) -> None:
        trajectory = TimedTrajectory(
            times_s=np.array([0.0, 1.0, 3.0]),
            positions_m=np.array([[0.0, 0.0, 0.0], [4.0, 6.0, 0.0], [8.0, 0.0, 2.0]]),
        )

        np.testing.assert_allclose(
            trajectory.frame_at(-1.0).position_m, [0.0, 0.0, 0.0]
        )
        frame = trajectory.frame_at(2.0)
        np.testing.assert_allclose(frame.position_m, [6.0, 3.0, 1.0])
        assert frame.time_s == pytest.approx(2.0)
        assert frame.lower_index == 1
        assert frame.fraction == pytest.approx(0.5)
        assert trajectory.apex_time_s == pytest.approx(1.0)
        assert trajectory.frame_at(99.0).is_landing

    @pytest.mark.parametrize(
        ("times", "positions", "message"),
        [
            ([0.0], [[0.0, 0.0]], "shape"),
            ([0.0, 1.0], [[0.0, 0.0, 0.0]], "same sample count"),
            ([0.0, 0.0], [[0.0, 0.0, 0.0]] * 2, "strictly increasing"),
            ([0.0, float("nan")], [[0.0, 0.0, 0.0]] * 2, "finite"),
        ],
    )
    def test_rejects_invalid_timeline(self, times, positions, message: str) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match=message):
            TimedTrajectory(np.asarray(times), np.asarray(positions))


def test_pyqt_controls_own_one_timer_and_expose_accessible_transport(qtbot) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("PyQt6")
    pytest.importorskip("pytestqt")
    from rate_of_closure.ui.pyqt6.flight_playback_controls import (
        FlightPlaybackControls,
    )

    controls = FlightPlaybackControls()
    qtbot.addWidget(controls)
    controls.set_timeline(4.0, 1.5)

    timer = controls.timer()
    controls.play()
    controls.play()
    assert controls.timer() is timer
    assert timer.isActive()
    assert controls.play_button.accessibleName() == "Play or Pause Ball Flight"
    assert controls.scrubber.accessibleName() == "Ball Flight Time"
    assert controls.loop_check.accessibleName() == "Loop Ball Flight Playback"
    controls.set_looping(True)
    assert controls.loop_check.isChecked()
    controls.jump_to_apex()
    assert controls.current_time_s() == pytest.approx(1.5)
    controls.jump_to_landing()
    assert controls.current_time_s() == pytest.approx(4.0)
    assert not timer.isActive()


def test_pyqt_controls_loop_at_landing_without_stopping(qtbot, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("PyQt6")
    pytest.importorskip("pytestqt")
    from rate_of_closure.ui.pyqt6.flight_playback_controls import (
        FlightPlaybackControls,
    )

    controls = FlightPlaybackControls()
    qtbot.addWidget(controls)
    controls.set_timeline(2.0, 1.0)
    controls.set_looping(True)
    controls._set_time(1.99)  # noqa: SLF001 - deterministic transport boundary test
    monkeypatch.setattr(controls, "_elapsed_seconds", lambda: 0.02)

    controls.timer().start()
    controls._advance()  # noqa: SLF001 - exercise the timer callback directly

    assert controls.timer().isActive()
    assert controls.current_time_s() == pytest.approx(0.01)
