"""Regression tests replaying recorded putts through the PuttingMonitor pipeline.

Each clip under ``tests/data/`` is described by ``manifest.json``. Putts must be
accepted within speed and HLA tolerances; negative controls (hand placing ball,
unarmed roll through) must report no accepted putts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from putting_launch_monitor.calibration import Calibration
from putting_launch_monitor.monitor import LogSink, PuttingMonitor
from shared.python.camera.video_file_source import VideoFileSource
from shared.python.contracts import require

pytest.importorskip("cv2")

DATA_DIR = Path(__file__).resolve().parent / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"
MANIFEST_SCHEMA = "putting_monitor.replay_manifest/1"


def load_manifest() -> dict[str, Any]:
    require(MANIFEST_PATH.is_file(), "manifest.json must exist", str(MANIFEST_PATH))
    doc: dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    require(
        doc.get("schema") == MANIFEST_SCHEMA,
        f"unsupported manifest schema: {doc.get('schema')}",
    )
    require("clips" in doc and isinstance(doc["clips"], list), "clips list required")
    return doc


def _manifest_clips() -> list[dict[str, Any]]:
    if not MANIFEST_PATH.is_file():
        return []
    doc = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    clips: list[dict[str, Any]] = doc.get("clips", [])
    return clips


@pytest.mark.unit
def test_replay_manifest_integrity_and_clips_exist() -> None:
    """Verify manifest schema, calibration and presence of every referenced video."""
    manifest = load_manifest()
    cal_file = manifest.get("calibration_file")
    require(bool(cal_file), "manifest missing calibration_file")
    cal_path = DATA_DIR / str(cal_file)
    require(cal_path.is_file(), f"calibration file not found: {cal_path}")

    # Load calibration to verify validity
    cal = Calibration.load(cal_path)
    assert cal.capture_width > 0 and cal.capture_height > 0
    assert len(cal.mat_corners_px) == 4

    clips = manifest["clips"]
    assert len(clips) >= 5, f"expected at least 5 clips in corpus, found {len(clips)}"

    has_putt = False
    has_hand_place = False
    has_rolling_unarmed = False

    for clip in clips:
        clip_file = clip["file"]
        clip_path = DATA_DIR / clip_file
        assert clip_path.is_file(), f"clip file missing: {clip_path}"
        clip_bytes = clip_path.stat().st_size
        assert clip_bytes > 0, f"clip file is empty: {clip_path}"
        max_bytes = 5 * 1024 * 1024
        assert clip_bytes < max_bytes, f"{clip_file} exceeds 5 MB: {clip_bytes} bytes"

        kind = clip.get("kind")
        assert kind in ("putt", "no_putt"), f"invalid clip kind: {kind}"
        if kind == "putt":
            has_putt = True
            assert "expected_speed_mph" in clip
            assert "expected_hla_deg" in clip
            assert "tolerance_speed_mph" in clip
            assert "tolerance_hla_deg" in clip
        elif "hand" in clip_file:
            has_hand_place = True
        elif "unarmed" in clip_file:
            has_rolling_unarmed = True

    assert has_putt, "corpus must contain valid putt clips"
    assert has_hand_place, "corpus must contain a hand placing ball clip"
    assert has_rolling_unarmed, "corpus must contain a rolling unarmed clip"


@pytest.mark.slow
@pytest.mark.parametrize(
    "clip_spec",
    _manifest_clips(),
    ids=lambda c: str(c.get("file", "unknown")),
)
def test_replay_corpus_clip(clip_spec: dict[str, Any]) -> None:
    """Replay a recorded clip through PuttingMonitor and assert expected outcome."""
    manifest = load_manifest()
    cal_path = DATA_DIR / str(manifest["calibration_file"])
    cal = Calibration.load(cal_path)

    clip_file = clip_spec["file"]
    clip_path = DATA_DIR / clip_file
    require(clip_path.is_file(), f"clip file not found: {clip_path}")

    source = VideoFileSource(clip_path, fps=float(cal.fps))
    sink = LogSink()
    monitor = PuttingMonitor(cal, source, sink)
    frames = monitor.run()
    assert frames > 0, f"no frames were processed from {clip_file}"

    kind = clip_spec["kind"]
    if kind == "putt":
        accepted = [p for p in monitor.putts if p.accepted]
        assert len(accepted) == 1, (
            f"expected 1 accepted putt for {clip_file}, got {len(accepted)} "
            f"(total: {len(monitor.putts)}, "
            f"reasons: {[p.reason for p in monitor.putts]})"
        )
        putt = accepted[0]

        expected_speed = clip_spec["expected_speed_mph"]
        tol_speed = clip_spec["tolerance_speed_mph"]
        assert abs(putt.speed_mph - expected_speed) <= tol_speed, (
            f"speed mismatch for {clip_file}: got {putt.speed_mph:.2f} mph, "
            f"expected {expected_speed:.2f} ± {tol_speed:.2f} mph"
        )

        expected_hla = clip_spec["expected_hla_deg"]
        tol_hla = clip_spec["tolerance_hla_deg"]
        assert abs(putt.hla_deg - expected_hla) <= tol_hla, (
            f"HLA mismatch for {clip_file}: got {putt.hla_deg:+.2f} deg, "
            f"expected {expected_hla:+.2f} ± {tol_hla:.2f} deg"
        )
    elif kind == "no_putt":
        accepted = [p for p in monitor.putts if p.accepted]
        assert len(accepted) == 0, (
            f"expected 0 accepted putts for {clip_file}, got {len(accepted)} "
            f"(putts: {[p.speed_mph for p in accepted]})"
        )
