"""Rendered restart evidence for bounded PyQt visual preferences."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")


@pytest.mark.parametrize("scale", [1.0, 1.5])
@pytest.mark.timeout(150)
def test_visual_layout_restores_without_hiding_club_canvas(
    tmp_path: Path,
    scale: float,
) -> None:
    output = (
        Path(os.environ.get("RATE_PYQT_EVIDENCE_DIR", str(tmp_path)))
        / f"visual-layout-{scale:g}"
    )
    repository = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment.update(
        {
            "QT_QPA_PLATFORM": "offscreen",
            "QT_SCALE_FACTOR": str(scale),
            "MPLBACKEND": "qtagg",
            "PYTHONPATH": os.pathsep.join((str(repository / "src"), str(repository))),
        }
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("pyqt_visual_layout_persistence_probe.py")),
            "--output",
            str(output),
            "--scale",
            str(scale),
        ],
        check=True,
        env=environment,
        timeout=120,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["requested_scale"] == scale
    assert manifest["font"] == {
        "font_family": "DejaVu Sans",
        "font_ascii_supported": True,
    }
    before, restored = manifest["states"]
    assert [before["state"], restored["state"]] == ["before-restart", "restored"]
    for state in (before, restored):
        assert state["bytes"] > 10_000
        assert state["canvas_bytes"] > 10_000
        assert state["canvas"][2] >= 240 and state["canvas"][3] >= 240
        assert state["tab_width"] >= 640
        assert state["device_pixel_ratio"] == pytest.approx(scale, rel=0.05)
        assert state["status"]
    assert restored["camera"] == before["camera"] == [-40.0, 35.0, 2.25]
    assert restored["sidebar_fraction"] == pytest.approx(
        before["sidebar_fraction"], abs=0.01
    )
    comparison = manifest["pixel_comparison"]
    assert comparison["mean_absolute_channel_delta"] <= 0.001
    assert comparison["changed_pixel_fraction"] <= 0.05
