"""Rendered Simulation scrub/error evidence at supported DPI scales."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.mark.parametrize("scale", [1.0, 1.5])
def test_simulation_auto_and_error_prior_are_visible(
    tmp_path: Path, scale: float
) -> None:
    output_root = Path(os.environ.get("RATE_PYQT_EVIDENCE_DIR", str(tmp_path)))
    output = output_root / f"simulation-scrub-scale-{scale:g}"
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
            str(Path(__file__).with_name("pyqt_simulation_scrub_probe.py")),
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
    assert manifest["artifact_policy"].startswith("diagnostic PNG")
    assert manifest["requested_scale"] == scale
    assert manifest["font"]["font_id"] >= 0
    assert manifest["font"]["font_family"] == "DejaVu Sans"
    assert manifest["font"]["ascii"] is True
    result, error_prior = manifest["states"]
    assert [result["state"], error_prior["state"]] == ["result-auto", "error-prior"]
    for state in manifest["states"]:
        assert state["window_bytes"] > 10_000
        assert state["canvas_bytes"] > 10_000
        assert state["visible_visual"][2] >= 240
        assert state["visible_visual"][3] >= 240
        assert state["visible_visual"][2] <= state["tab_size"][0]
        assert state["visible_visual"][3] <= state["tab_size"][1]
        assert not state["control_overlap"]
        assert state["device_pixel_ratio"] == pytest.approx(scale, rel=0.02)
    assert result["requested_impact_time_s"] is None
    assert result["run_identity"] == error_prior["run_identity"]
    assert result["canvas_sha256"] == error_prior["canvas_sha256"]
    assert "completed" in result["status"].lower()
    assert "prior accepted scene remains displayed" in error_prior["status"].lower()
    assert error_prior["status_visible"][2] > 0
    assert error_prior["status_visible"][3] > 0
    assert error_prior["controls_scroll_y"] > 0
