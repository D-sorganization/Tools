"""Shared playback transport model and cross-runtime golden parity (#4800 P8).

The TypeScript twin (`web/src/model/playbackTransport.ts` +
`web/src/model/flightPlayback.ts`) consumes the same golden fixture in
`playbackTransport.test.ts`; changing the fixture is a contract change on
both sides and must land together.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rate_of_closure.simulation.flight_playback import TimedTrajectory
from rate_of_closure.simulation.playback_transport import (
    DEFAULT_SPEED,
    PLAYBACK_SPEEDS,
    SCRUB_STEPS,
    advance_playback,
    clamp_time,
    scrub_value,
    time_at_scrub,
)

pytestmark = pytest.mark.unit

FIXTURE_PATH = (
    Path(__file__).parents[2]
    / "src/rate_of_closure/web/src/model/__fixtures__"
    / "playback_transport_golden_v1.json"
)


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    fixture: dict[str, Any] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert fixture["schema"] == "rate-of-closure-playback-transport/v1"
    return fixture


@pytest.fixture(scope="module")
def golden_trajectory(golden: dict[str, Any]) -> TimedTrajectory:
    return TimedTrajectory(
        times_s=np.asarray(golden["trajectory"]["times_s"], dtype=float),
        positions_m=np.asarray(golden["trajectory"]["positions_m"], dtype=float),
    )


class TestGoldenParity:
    def test_shared_constants_match_the_typescript_twin(
        self, golden: dict[str, Any]
    ) -> None:
        assert SCRUB_STEPS == golden["scrub_steps"]
        assert list(PLAYBACK_SPEEDS) == golden["speeds"]
        assert DEFAULT_SPEED == golden["default_speed"]
        assert DEFAULT_SPEED in PLAYBACK_SPEEDS

    def test_sample_to_frame_mapping_reproduces_every_golden_frame(
        self, golden: dict[str, Any], golden_trajectory: TimedTrajectory
    ) -> None:
        assert golden_trajectory.duration_s == golden["trajectory"]["duration_s"]
        assert golden_trajectory.apex_time_s == golden["trajectory"]["apex_time_s"]
        for case in golden["frames"]:
            frame = golden_trajectory.frame_at(float(case["requested_time_s"]))
            assert frame.time_s == pytest.approx(case["time_s"], abs=1e-12)
            assert frame.lower_index == case["lower_index"]
            assert frame.fraction == pytest.approx(case["fraction"], abs=1e-12)
            assert frame.is_landing == case["is_landing"]
            np.testing.assert_allclose(
                frame.position_m, case["position_m"], atol=1e-12
            )

    def test_adjacent_sample_steps_reproduce_every_golden_step(
        self, golden: dict[str, Any], golden_trajectory: TimedTrajectory
    ) -> None:
        for case in golden["steps"]:
            stepped = golden_trajectory.step_time(
                float(case["time_s"]), int(case["direction"])
            )
            assert stepped == pytest.approx(case["stepped_time_s"], abs=1e-12)

    def test_scrub_quantization_reproduces_the_golden_mapping_both_ways(
        self, golden: dict[str, Any]
    ) -> None:
        for case in golden["scrub_values"]:
            assert (
                scrub_value(float(case["time_s"]), float(case["duration_s"]))
                == case["value"]
            )
        for case in golden["scrub_times"]:
            assert time_at_scrub(
                int(case["value"]), float(case["duration_s"])
            ) == pytest.approx(case["time_s"], abs=1e-12)

    def test_wall_clock_advances_reproduce_the_golden_finish_flags(
        self, golden: dict[str, Any]
    ) -> None:
        for case in golden["advances"]:
            step = advance_playback(
                float(case["time_s"]),
                float(case["elapsed_s"]),
                float(case["speed"]),
                float(case["duration_s"]),
            )
            assert step.time_s == pytest.approx(case["next_time_s"], abs=1e-12)
            assert step.finished == case["finished"]


class TestTransportContract:
    def test_clamp_normalizes_onto_the_timeline_and_rejects_nonfinite(self) -> None:
        assert clamp_time(-1.0, 3.0) == 0.0
        assert clamp_time(9.0, 3.0) == 3.0
        with pytest.raises(ValueError, match="finite"):
            clamp_time(float("nan"), 3.0)
        with pytest.raises(ValueError, match="duration"):
            clamp_time(0.0, -1.0)

    def test_scrub_rejects_malformed_positions_and_step_counts(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            scrub_value(1.0, 3.0, steps=0)
        with pytest.raises(ValueError, match="within"):
            time_at_scrub(-1, 3.0)
        with pytest.raises(ValueError, match="within"):
            time_at_scrub(SCRUB_STEPS + 1, 3.0)

    def test_advance_rejects_nonphysical_requests(self) -> None:
        with pytest.raises(ValueError, match="elapsed"):
            advance_playback(0.0, -0.1, 1.0, 3.0)
        with pytest.raises(ValueError, match="speed"):
            advance_playback(0.0, 0.1, 0.0, 3.0)
        with pytest.raises(ValueError, match="speed"):
            advance_playback(0.0, 0.1, float("inf"), 3.0)

    def test_step_time_rejects_invalid_directions(
        self, golden_trajectory: TimedTrajectory
    ) -> None:
        with pytest.raises(ValueError, match="direction"):
            golden_trajectory.step_time(0.0, 0)


def test_transport_controls_serve_a_non_flight_subject_unchanged(qtbot) -> None:  # type: ignore[no-untyped-def]
    """The putting seam: the generic Qt widget needs no flight vocabulary."""
    pytest.importorskip("PyQt6")
    pytest.importorskip("pytestqt")
    from rate_of_closure.ui.pyqt6.playback_transport_controls import (
        PlaybackTransportControls,
    )

    controls = PlaybackTransportControls(
        subject_label="Putt",
        subject_phrase="putt",
        event_labels=("Strike", "Holed"),
        scrub_tooltip="Scrub physical putt time [s] from strike to rest.",
        help_text="Drag the 3D green to rotate; use the wheel to zoom.",
        help_tooltip="Physical metres on the green surface.",
    )
    qtbot.addWidget(controls)
    controls.set_transport_timeline(6.0, (0.0, 6.0))

    assert controls.scrubber.accessibleName() == "Putt Time"
    assert controls.play_button.accessibleName() == "Play or Pause Putt"
    assert controls.time_label.accessibleName() == "Putt Playback Time"
    assert [button.text() for button in controls.event_buttons] == [
        "Strike",
        "Holed",
    ]
    controls.jump_to_event(1)
    assert controls.current_time_s() == pytest.approx(6.0)
    controls.jump_to_event(0)
    assert controls.current_time_s() == pytest.approx(0.0)
    with pytest.raises(ValueError, match="event index"):
        controls.jump_to_event(2)
    with pytest.raises(ValueError, match="one time per event"):
        controls.set_transport_timeline(6.0, (0.0,))
    controls.play()
    assert controls.timer().isActive()
    controls.pause()
    assert not controls.timer().isActive()
