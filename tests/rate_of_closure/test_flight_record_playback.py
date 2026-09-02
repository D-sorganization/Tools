"""Loader gates for imported trajectory-record playback (ADR-0047 H4).

The loader lifts a validated ``swing_sim.ball_flight_trajectory/1``
record (:mod:`shared.python.swing_sim.flight_interchange`) onto the
shared P8 playback timeline without re-simulating or resampling a
single value; its one piece of real logic is the frame conversion,
which is why the conversion mapping is pinned against the same golden
fixture the TypeScript twin (``flightRecordPlayback.test.ts``)
consumes, following the precedent set by the ``putt`` block and
``TestPuttGoldenParity`` in ``test_playback_transport.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rate_of_closure.simulation.flight_record_playback import (
    UnsupportedTrajectoryFrameError,
    timed_trajectory_from_ball_flight_record,
)
from shared.python.swing_sim.flight_interchange import (
    APP_FRAME_ID,
    FLIGHT_FRAME_ID,
    TOOLS_FLIGHT_FAMILY,
    BallFlightTrajectory,
    TrajectoryProvenance,
    from_samples,
    parameter_digest,
)

pytestmark = pytest.mark.unit

FIXTURE_PATH = (
    Path(__file__).parents[2]
    / "src/rate_of_closure/web/src/model/__fixtures__"
    / "playback_transport_golden_v1.json"
)

_DIGEST = parameter_digest({"cd": 0.22, "cl": 0.24})


def _provenance(family: str = TOOLS_FLIGHT_FAMILY) -> TrajectoryProvenance:
    return TrajectoryProvenance(
        model_family=family, model_name="test-model", parameter_digest=_DIGEST
    )


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    fixture: dict[str, Any] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert fixture["schema"] == "rate-of-closure-playback-transport/v1"
    return fixture


def _record_from_golden(
    golden_data: dict[str, Any], *, frame_id: str | None = None
) -> BallFlightTrajectory:
    block = golden_data["imported_trajectory"]
    samples = block["samples"]
    return from_samples(
        source_id="test:imported",
        frame_id=frame_id or block["frame_id"],
        provenance=_provenance(),
        times_s=[sample["time_s"] for sample in samples],
        positions_m=[sample["position_m"] for sample in samples],
    )


class TestImportedTrajectoryGoldenParity:
    """The flight-frame -> app-frame conversion, pinned against the TS twin.

    The playback transport itself (interpolation, scrubbing, wall-clock
    advance) is already pinned by ``TestGoldenParity`` in
    ``test_playback_transport.py``; the only new logic this loader adds
    is the frame conversion, so that is the only thing this class pins.
    """

    def test_flight_frame_samples_convert_to_the_golden_app_positions(
        self, golden: dict[str, Any]
    ) -> None:
        record = _record_from_golden(golden)
        trajectory = timed_trajectory_from_ball_flight_record(record)
        block = golden["imported_trajectory"]
        assert list(trajectory.times_s) == pytest.approx(
            [sample["time_s"] for sample in block["samples"]]
        )
        np.testing.assert_allclose(
            trajectory.positions_m, block["app_positions_m"], atol=1e-12
        )
        assert trajectory.duration_s == pytest.approx(block["duration_s"])
        assert trajectory.apex_time_s == pytest.approx(block["apex_time_s"])

    def test_app_frame_samples_pass_through_unconverted(
        self, golden: dict[str, Any]
    ) -> None:
        """An already-app-frame record needs no conversion at all."""
        block = golden["imported_trajectory"]
        record = _record_from_golden(
            golden, frame_id=APP_FRAME_ID
        )  # samples reinterpreted as app-frame on purpose
        trajectory = timed_trajectory_from_ball_flight_record(record)
        np.testing.assert_allclose(
            trajectory.positions_m,
            [sample["position_m"] for sample in block["samples"]],
            atol=1e-12,
        )


class TestFlightRecordPlaybackGates:
    def test_refuses_a_non_record_object(self) -> None:
        with pytest.raises(TypeError, match="BallFlightTrajectory"):
            timed_trajectory_from_ball_flight_record(object())  # type: ignore[arg-type]

    def test_refuses_a_frame_id_this_loader_does_not_convert(self) -> None:
        """Defends the closed-enum contract documented in the module.

        ``BallFlightTrajectory`` itself only ever constructs with a
        ``frame_id`` from the wire's declared set, so the only way to
        exercise this loader's own refusal is to force an out-of-band
        value onto an already-validated (frozen) record — simulating a
        future wire frame this loader has not yet been taught to
        convert, exactly the scenario the module docstring calls out.
        """
        record = from_samples(
            source_id="test:unsupported-frame",
            frame_id=FLIGHT_FRAME_ID,
            provenance=_provenance(),
            times_s=[0.0, 1.0],
            positions_m=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        )
        object.__setattr__(record, "frame_id", "some_future_frame")
        with pytest.raises(UnsupportedTrajectoryFrameError, match="some_future_frame"):
            timed_trajectory_from_ball_flight_record(record)

    def test_replays_a_ud_family_record_identically_to_a_tools_family_one(self) -> None:
        """Cross-family records replay the same way — H4's ADR mandate."""
        shared_kwargs = {
            "source_id": "ud.flight_models:waterloo_penner",
            "frame_id": FLIGHT_FRAME_ID,
            "times_s": [0.0, 0.5, 1.0],
            "positions_m": [[0.0, 0.0, 0.0], [10.0, 2.0, 4.0], [20.0, 0.0, 0.0]],
        }
        ud_record = from_samples(
            provenance=_provenance(family="ud.flight_models"), **shared_kwargs
        )
        tools_record = from_samples(
            provenance=_provenance(family=TOOLS_FLIGHT_FAMILY), **shared_kwargs
        )
        ud_trajectory = timed_trajectory_from_ball_flight_record(ud_record)
        tools_trajectory = timed_trajectory_from_ball_flight_record(tools_record)
        np.testing.assert_allclose(
            ud_trajectory.positions_m, tools_trajectory.positions_m
        )
        assert list(ud_trajectory.times_s) == list(tools_trajectory.times_s)
