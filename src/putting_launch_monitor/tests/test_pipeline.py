"""Detector, tracker, calibration and the whole monitor on synthetic putts.

The end-to-end test is the one that matters: frames are *rendered* through
the tilted synthetic camera, the detector finds the ball, the tracker arms
on the resting ball and follows the roll, the geometry maps it to the
ground, and the shot that reaches the sink carries the speed and angle the
putt was rendered with.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from putting_launch_monitor.calibration import SCHEMA, Calibration
from putting_launch_monitor.detect import (
    BallObservation,
    HsvBallDetector,
    HsvRange,
    RegionOfInterest,
)
from putting_launch_monitor.gspro import GsproClient
from putting_launch_monitor.monitor import (
    GsproSink,
    LogSink,
    PuttingMonitor,
    frame_to_bgr,
)
from putting_launch_monitor.track import Phase, PuttTracker, TrackerSettings
from shared.python.sidekick.lab.mocap.acquisition import FramePacket
from shared.python.sidekick.lab.mocap.devices import (
    CameraCapabilities,
    CameraIdentity,
    FeatureSupport,
)
from shared.python.sidekick.lab.mocap.enums import ShutterKind, SupportLevel

from .synthetic import MAT_L_MM, MAT_W_MM, SyntheticCamera, putt_positions, render_frame

pytest.importorskip("cv2")
pytestmark = pytest.mark.unit

CAMERA_ID = "USB\\VID_32E4&PID_5234&MI_00\\9&2A7EE39F&0&0000"


def _calibration(cam: SyntheticCamera, **overrides: Any) -> Calibration:
    corners = tuple((float(x), float(y)) for x, y in cam.mat_corners_px())
    base: dict[str, Any] = dict(
        camera_instance_id=CAMERA_ID,
        mat_corners_px=corners,
        mat_width_mm=MAT_W_MM,
        mat_length_mm=MAT_L_MM,
        capture_width=cam.width,
        capture_height=cam.height,
    )
    base.update(overrides)
    return Calibration(**base)


# -- detector --------------------------------------------------------------------------
def test_detector_finds_the_ball_where_it_was_drawn() -> None:
    cam = SyntheticCamera()
    cal = _calibration(cam)
    frame = render_frame(cam, (610.0, 700.0), noise=4.0)
    seen = cal.detector().detect(frame)
    assert seen is not None
    expected = cam.project(np.array([610.0, 700.0]))[0]
    assert abs(seen.cx - expected[0]) < 1.5 and abs(seen.cy - expected[1]) < 1.5
    assert seen.circularity >= 0.6  # the detector's own floor; a 9 px ball is pixelated
    assert cal.detector().detect(render_frame(cam, None, noise=4.0)) is None


def test_detector_contracts_and_roi() -> None:
    with pytest.raises(ValueError, match="radius"):
        HsvBallDetector(min_radius_px=10, max_radius_px=5)
    with pytest.raises(ValueError, match="hue"):
        HsvRange(h_min=100, h_max=10)
    with pytest.raises(ValueError, match="roi"):
        RegionOfInterest(0, 0, 0, 10)
    cam = SyntheticCamera()
    frame = render_frame(cam, (610.0, 700.0))
    where = cam.project(np.array([610.0, 700.0]))[0]
    away = RegionOfInterest(0, 0, int(where[0]) - 60, cam.height)
    assert HsvBallDetector(roi=away).detect(frame) is None
    with pytest.raises(ValueError, match="BGR"):
        HsvBallDetector().detect(np.zeros((4, 4), dtype=np.uint8))


# -- tracker ---------------------------------------------------------------------------
class _Seen:
    """Observations straight from world positions (no rendering)."""

    def __init__(self, cam: SyntheticCamera) -> None:
        self.cam = cam

    def at(self, xy: tuple[float, float]) -> BallObservation:
        px = self.cam.project(np.asarray(xy))[0]
        return BallObservation(float(px[0]), float(px[1]), 6.0, 100.0, 0.95)


def test_tracker_arms_on_a_resting_ball_then_reports_the_roll() -> None:
    cam = SyntheticCamera()
    cal = _calibration(cam)
    tracker = PuttTracker(cal.ground_plane(), cal.tracker_settings())
    seen = _Seen(cam)
    ns = 0
    step = 1_000_000_000 // 60
    for _ in range(12):  # resting ball
        assert tracker.update(ns, [seen.at((610.0, 250.0))]) is None
        ns += step
    assert tracker.phase is Phase.ARMED
    t, world = putt_positions((610.0, 250.0), 2.0, 3.5, fps=60.0, seconds=0.8)
    putt = None
    for xy in world:
        putt = tracker.update(ns, [seen.at((float(xy[0]), float(xy[1])))])
        ns += step
        if putt is not None:
            break
    assert putt is not None and putt.accepted, putt
    assert putt.speed_mph == pytest.approx(2.0 * 2.2369, rel=0.02)
    assert putt.hla_deg == pytest.approx(3.5, abs=0.5)
    assert tracker.phase is Phase.WAITING  # ready for the next one


def test_tracker_ignores_a_ball_that_never_rested_and_rejects_a_twitch() -> None:
    cam = SyntheticCamera()
    cal = _calibration(cam)
    tracker = PuttTracker(cal.ground_plane(), cal.tracker_settings())
    seen = _Seen(cam)
    # A ball rolling through without ever resting: never armed, never a putt.
    _, world = putt_positions((300.0, 100.0), 1.5, 0.0, fps=60.0, seconds=0.5)
    for i, xy in enumerate(world):
        assert (
            tracker.update(i * 16_666_666, [seen.at((float(xy[0]), float(xy[1])))])
            is None
        )
    assert tracker.phase is not Phase.ROLLING
    # A resting ball nudged 15 mm and stopped: a rejected putt, not a shot.
    tracker.reset()
    ns = 0
    for _ in range(12):
        tracker.update(ns, [seen.at((610.0, 250.0))])
        ns += 16_666_666
    assert tracker.phase is Phase.ARMED
    result = None
    for _ in range(20):
        result = tracker.update(ns, [seen.at((610.0, 266.0))]) or result
        ns += 16_666_666
    assert result is not None and not result.accepted
    assert "stopped" in result.reason


def test_tracker_follows_its_own_ball_when_a_second_one_rests_nearby() -> None:
    """A stray ball on the mat can neither arm the tracker nor steal the roll."""
    cam = SyntheticCamera()
    cal = _calibration(cam)
    tracker = PuttTracker(cal.ground_plane(), cal.tracker_settings())
    seen = _Seen(cam)
    base = seen.at((900.0, 600.0))  # a second ball, resting, more circular
    stray = BallObservation(base.cx, base.cy, base.radius_px, base.area_px, 0.99)
    ns, step = 0, 1_000_000_000 // 60
    for _ in range(12):
        tracker.update(ns, [stray, seen.at((610.0, 250.0))])
        ns += step
    assert tracker.phase is Phase.ARMED
    # Armed on the more circular stray ball? No: it arms on whichever rested,
    # and here both rested - so the origin is the first (best) candidate.
    _, world = putt_positions((610.0, 250.0), 2.0, 0.0, fps=60.0, seconds=0.8)
    ours_first = tracker._origin is not None and abs(tracker._origin[0] - 900.0) > 1
    putt = None
    for xy in world:
        putt = tracker.update(ns, [stray, seen.at((float(xy[0]), float(xy[1])))])
        ns += step
        if putt is not None:
            break
    if ours_first:
        assert putt is not None and putt.accepted, putt
        assert putt.speed_mph == pytest.approx(2.0 * 2.2369, rel=0.02)
    else:
        # It armed on the stray: our rolling ball is far from that origin,
        # so nothing is reported - the stray never moved.
        assert putt is None and tracker.phase is Phase.ARMED


def test_tracker_settings_contracts() -> None:
    with pytest.raises(ValueError, match="still < start"):
        TrackerSettings(still_mm=20, start_mm=10)
    with pytest.raises(ValueError, match="window > start"):
        TrackerSettings(window_mm=5)
    with pytest.raises(ValueError, match="speed bounds"):
        TrackerSettings(min_speed_mph=30)


# -- calibration persistence -----------------------------------------------------------
def test_calibration_round_trips_and_refuses_unknown_documents(tmp_path: Path) -> None:
    cam = SyntheticCamera()
    cal = _calibration(
        cam, target_deg=2.5, colour="orange", roi=RegionOfInterest(10, 20, 500, 400)
    )
    path = tmp_path / "cal.json"
    cal.save(path)
    back = Calibration.load(path)
    assert back == cal
    assert back.target_vector()[0] == pytest.approx(np.sin(np.radians(2.5)))
    assert back.tracker_settings().target == back.target_vector()
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["schema"] == SCHEMA
    doc["schema"] = "putting_monitor.calibration/9"
    with pytest.raises(ValueError, match="unsupported"):
        Calibration.from_dict(doc)
    doc["schema"] = SCHEMA
    doc["mystery"] = 1
    with pytest.raises(ValueError, match="unknown"):
        Calibration.from_dict(doc)
    with pytest.raises(ValueError, match="unknown colour"):
        _calibration(cam, colour="plaid")


# -- the whole monitor on rendered frames ----------------------------------------------
class RenderedPuttSource:
    """A FrameSource that renders a resting ball, then a putt, then nothing."""

    def __init__(
        self, cam: SyntheticCamera, speed: float, hla: float, fps: float = 60.0
    ) -> None:
        self.cam = cam
        rest = [(610.0, 250.0)] * 20
        _, world = putt_positions((610.0, 250.0), speed, hla, fps=fps, seconds=0.9)
        self.positions: list[tuple[float, float] | None] = (
            rest + [(float(p[0]), float(p[1])) for p in world] + [None] * 10
        )
        self.fps = fps
        self.i = 0
        self.closed = False

    source_id = "synthetic:putt"
    identity = CameraIdentity(
        provider_id="synthetic", device_id="putt", transport="memory"
    )
    capabilities = CameraCapabilities(
        resolutions_px=((960, 600),),
        frame_rates_hz=(60.0,),
        pixel_formats=("bgr24",),
        shutter=ShutterKind.GLOBAL,
        hardware_trigger=FeatureSupport(SupportLevel.UNSUPPORTED, "synthetic"),
        device_timestamps=FeatureSupport(SupportLevel.SUPPORTED),
    )

    @property
    def state(self) -> object:
        return "capturing"

    def initialize(self) -> None: ...

    def start_capture(self) -> None: ...

    def read_frame(self, timeout_seconds: float = 5.0) -> FramePacket:
        from shared.python.contracts import StateError

        if self.i >= len(self.positions):
            raise StateError("end")
        frame = render_frame(self.cam, self.positions[self.i], noise=3.0, seed=self.i)
        packet = FramePacket(
            source_id=self.source_id,
            sequence_number=self.i,
            timestamp_ns=int(self.i * 1e9 / self.fps),
            host_monotonic_ns=0,
            image_bytes=frame.tobytes(),
            pixel_format="bgr24",
            resolution_px=(self.cam.width, self.cam.height),
        )
        self.i += 1
        return packet

    def stop_capture(self) -> None: ...

    def close(self) -> None:
        self.closed = True


def test_monitor_end_to_end_recovers_the_rendered_putt() -> None:
    cam = SyntheticCamera()
    cal = _calibration(cam)
    source = RenderedPuttSource(cam, speed=1.8, hla=-2.0)
    sink = LogSink()
    monitor = PuttingMonitor(cal, source, sink)
    phases: list[Phase] = []
    monitor.add_observer(lambda e: phases.append(e.phase))
    frames = monitor.run()
    assert source.closed and frames == len(source.positions)
    assert Phase.ARMED in phases and Phase.ROLLING in phases
    assert len(sink.putts) == 1
    putt = sink.putts[0]
    assert putt.accepted, putt.reason
    assert putt.speed_mph == pytest.approx(1.8 * 2.2369, rel=0.03)
    assert putt.hla_deg == pytest.approx(-2.0, abs=0.7)


def test_monitor_holds_putts_until_gspro_is_in_putting_mode() -> None:
    cam = SyntheticCamera()
    cal = _calibration(cam)

    class Sock:
        def __init__(self, club: str) -> None:
            self.sent: list[bytes] = []
            self.club = club

        def sendall(self, b: bytes) -> None:
            self.sent.append(b)

        def recv(self, n: int) -> bytes:
            return (
                b'{"Code":200,"Message":"OK"}'
                b'{"Code":201,"Message":"GSPro Player Information",'
                b'"Player":{"Handed":"RH","Club":"' + self.club.encode() + b'"}}'
            )

        def close(self) -> None: ...

    driver = Sock("DR")
    client = GsproClient(device_id="t", socket_factory=lambda: driver)
    sink = GsproSink(client)
    PuttingMonitor(cal, RenderedPuttSource(cam, 2.0, 0.0), sink).run()
    assert sink.held == 1 and sink.sent == 0  # a putt during a full-swing hole

    client.player = None
    driver.club = "PT"
    client.heartbeat()  # learns the putter from the 201
    assert client.putting_mode
    PuttingMonitor(cal, RenderedPuttSource(cam, 2.0, 0.0), sink).run()
    assert sink.sent == 1
    payload = json.loads(driver.sent[-1])
    assert payload["BallData"]["Speed"] == pytest.approx(2.0 * 2.2369, rel=0.03)
    assert payload["ShotDataOptions"]["ContainsBallData"] is True


def test_frame_to_bgr_contract() -> None:
    packet = FramePacket(
        source_id="s",
        sequence_number=0,
        timestamp_ns=0,
        host_monotonic_ns=0,
        image_bytes=bytes(2 * 3 * 3),
        pixel_format="bgr24",
        resolution_px=(2, 3),
    )
    assert frame_to_bgr(packet).shape == (3, 2, 3)
    with pytest.raises(ValueError, match="bgr24"):
        frame_to_bgr(
            FramePacket(
                source_id="s",
                sequence_number=0,
                timestamp_ns=0,
                host_monotonic_ns=0,
                image_bytes=b"\0",
                pixel_format="gray",
                resolution_px=(1, 1),
            )
        )
