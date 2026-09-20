"""Generate the synthetic replay corpus for the putting monitor regression suite.

Creates:
  - src/putting_launch_monitor/tests/data/calibration.json
  - src/putting_launch_monitor/tests/data/manifest.json
  - short MJPEG video clips (*.avi) under src/putting_launch_monitor/tests/data/

Usage:
  python scripts/generate_putting_replay_corpus.py
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from putting_launch_monitor.calibration import Calibration
from putting_launch_monitor.monitor import LogSink, PuttingMonitor
from putting_launch_monitor.tests.synthetic import (
    MAT_L_MM,
    MAT_W_MM,
    SyntheticCamera,
    putt_positions,
    render_frame,
)
from shared.python.camera.video_file_source import VideoFileSource

DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "putting_launch_monitor"
    / "tests"
    / "data"
)
FPS = 60.0


@dataclass(frozen=True)
class PuttClipSpec:
    file: str
    speed_mps: float
    hla_deg: float
    roll_s: float
    decel: float
    desc: str


@dataclass(frozen=True)
class NoPuttClipSpec:
    file: str
    generator: Callable[[SyntheticCamera], list[np.ndarray]]
    desc: str


def _build_calibration(cam: SyntheticCamera) -> Calibration:
    corners = tuple((float(x), float(y)) for x, y in cam.mat_corners_px())
    return Calibration(
        camera_instance_id="synthetic_lab_overhead_30deg",
        mat_corners_px=corners,
        mat_width_mm=MAT_W_MM,
        mat_length_mm=MAT_L_MM,
        capture_width=cam.width,
        capture_height=cam.height,
        fps=int(FPS),
        target_deg=0.0,
        colour="white",
        notes="Replay corpus calibration matching 30 deg overhead lab setup",
    )


def _write_clip(
    path: Path,
    cam: SyntheticCamera,
    frames: list[np.ndarray],
) -> None:
    fourcc = int(cv2.VideoWriter.fourcc(*"MJPG"))
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (cam.width, cam.height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _make_putt_clip(
    cam: SyntheticCamera,
    speed_mps: float,
    hla_deg: float,
    *,
    decel_mps2: float = 0.1,
    roll_seconds: float = 0.9,
    rest_frames: int = 20,
    trail_frames: int = 15,
) -> list[np.ndarray]:
    start_mm = (610.0, 250.0)
    _, world = putt_positions(
        start_mm,
        speed_mps,
        hla_deg,
        fps=FPS,
        seconds=roll_seconds,
        decel_mps2=decel_mps2,
    )
    positions: list[tuple[float, float] | None] = (
        [start_mm] * rest_frames
        + [(float(p[0]), float(p[1])) for p in world]
        + [None] * trail_frames
    )
    frames: list[np.ndarray] = []
    for i, pos in enumerate(positions):
        f = render_frame(cam, pos, noise=2.0, seed=i)
        frames.append(f)
    return frames


def _make_hand_place_clip(cam: SyntheticCamera) -> list[np.ndarray]:
    """Hand places a ball and retracts; ball remains at rest (no putt)."""
    ball_pt = (610.0, 250.0)
    target_px = cam.project(np.array(ball_pt))[0]
    frames: list[np.ndarray] = []
    total = 95
    for i in range(total):
        if i < 15:
            f = render_frame(cam, None, noise=2.0, seed=i)
            # Simulated hand entering from player side
            cv2.ellipse(
                f,
                (int(target_px[0] - (15 - i) * 14), int(target_px[1])),
                (45, 25),
                20,
                0,
                360,
                (140, 160, 210),
                -1,
            )
        elif i < 30:
            f = render_frame(cam, ball_pt, noise=2.0, seed=i)
            # Hand retracting
            cv2.ellipse(
                f,
                (int(target_px[0] - (i - 15) * 18), int(target_px[1])),
                (45, 25),
                20,
                0,
                360,
                (140, 160, 210),
                -1,
            )
        else:
            # Ball resting motionless
            f = render_frame(cam, ball_pt, noise=2.0, seed=i)
        frames.append(f)
    return frames


def _make_rolling_unarmed_clip(cam: SyntheticCamera) -> list[np.ndarray]:
    """Ball rolls across mat without having rested; tracker never arms (no putt)."""
    _, world = putt_positions(
        (610.0, 50.0), speed_mps=2.4, hla_deg=0.0, fps=FPS, seconds=1.0
    )
    positions: list[tuple[float, float] | None] = [
        (float(p[0]), float(p[1])) for p in world
    ] + [None] * 20
    frames: list[np.ndarray] = []
    for i, pos in enumerate(positions):
        frames.append(render_frame(cam, pos, noise=2.0, seed=i))
    return frames


def generate_all() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cam = SyntheticCamera()
    cal = _build_calibration(cam)
    cal_path = DATA_DIR / "calibration.json"
    cal.save(cal_path)

    putt_specs = [
        PuttClipSpec(
            file="putt_slow_straight.avi",
            speed_mps=1.5,
            hla_deg=0.0,
            roll_s=0.9,
            decel=0.1,
            desc="Slow straight putt ~3.4 mph, HLA 0.0 deg",
        ),
        PuttClipSpec(
            file="putt_medium_pull.avi",
            speed_mps=2.0,
            hla_deg=-2.5,
            roll_s=0.8,
            decel=0.15,
            desc="Medium pulled putt ~4.5 mph, HLA -2.5 deg",
        ),
        PuttClipSpec(
            file="putt_medium_push.avi",
            speed_mps=2.3,
            hla_deg=2.0,
            roll_s=0.8,
            decel=0.15,
            desc="Medium pushed putt ~5.1 mph, HLA +2.0 deg",
        ),
        PuttClipSpec(
            file="putt_fast_straight.avi",
            speed_mps=3.0,
            hla_deg=0.0,
            roll_s=0.7,
            decel=0.2,
            desc="Fast straight putt ~6.7 mph, HLA 0.0 deg",
        ),
        PuttClipSpec(
            file="putt_fast_slight_pull.avi",
            speed_mps=2.8,
            hla_deg=-1.2,
            roll_s=0.7,
            decel=0.2,
            desc="Fast slight pull ~6.3 mph, HLA -1.2 deg",
        ),
    ]

    no_putt_specs = [
        NoPuttClipSpec(
            file="no_putt_hand_place.avi",
            generator=_make_hand_place_clip,
            desc="Hand placing ball at rest without a launch",
        ),
        NoPuttClipSpec(
            file="no_putt_rolling_unarmed.avi",
            generator=_make_rolling_unarmed_clip,
            desc="Ball rolling through the mat without resting (unarmed)",
        ),
    ]

    manifest_entries: list[dict[str, Any]] = []

    for pspec in putt_specs:
        filename = pspec.file
        clip_path = DATA_DIR / filename
        frames = _make_putt_clip(
            cam,
            pspec.speed_mps,
            pspec.hla_deg,
            decel_mps2=pspec.decel,
            roll_seconds=pspec.roll_s,
        )
        _write_clip(clip_path, cam, frames)

        src = VideoFileSource(clip_path, fps=FPS)
        sink = LogSink()
        mon = PuttingMonitor(cal, src, sink)
        mon.run()

        putt_count = len(mon.putts)
        assert putt_count == 1, f"Expected 1 putt in {filename}, got {putt_count}"
        putt = mon.putts[0]
        assert putt.accepted, f"Putt rejected in {filename}: {putt.reason}"
        manifest_entries.append(
            {
                "file": filename,
                "kind": "putt",
                "expected_speed_mph": round(putt.speed_mph, 2),
                "expected_hla_deg": round(putt.hla_deg, 2),
                "tolerance_speed_mph": 0.25,
                "tolerance_hla_deg": 0.8,
                "description": pspec.desc,
            }
        )

    for nspec in no_putt_specs:
        filename = nspec.file
        clip_path = DATA_DIR / filename
        frames = nspec.generator(cam)
        _write_clip(clip_path, cam, frames)

        src = VideoFileSource(clip_path, fps=FPS)
        sink = LogSink()
        mon = PuttingMonitor(cal, src, sink)
        mon.run()

        accepted_putts = [p for p in mon.putts if p.accepted]
        acc_count = len(accepted_putts)
        assert acc_count == 0, f"Expected 0 putts in {filename}, got {acc_count}"
        manifest_entries.append(
            {
                "file": filename,
                "kind": "no_putt",
                "expected_putts": 0,
                "description": nspec.desc,
            }
        )

    manifest_doc = {
        "schema": "putting_monitor.replay_manifest/1",
        "calibration_file": "calibration.json",
        "description": (
            "Recorded-putt regression corpus captured/rendered at 960x600@60 fps "
            "matching overhead lab geometry."
        ),
        "reference_source": (
            "Simulated overhead camera tracker measurements pending physical "
            "rig validation #5221"
        ),
        "clips": manifest_entries,
    }

    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_doc, indent=2), encoding="utf-8")
    print(f"Generated {len(manifest_entries)} clips in {DATA_DIR}")


if __name__ == "__main__":
    generate_all()
