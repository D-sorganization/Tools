"""Rendered Flight inspector evidence at both supported DPI scales."""

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
def test_flight_selected_and_error_prior_are_visible(
    tmp_path: Path, scale: float
) -> None:
    output_root = Path(os.environ.get("RATE_PYQT_EVIDENCE_DIR", str(tmp_path)))
    output = output_root / f"flight-inspector-scale-{scale:g}"
    environment = dict(os.environ)
    repository = Path(__file__).resolve().parents[2]
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
            str(Path(__file__).with_name("pyqt_flight_sample_inspector_probe.py")),
            "--output",
            str(output),
            "--scale",
            str(scale),
        ],
        check=True,
        env=environment,
        timeout=90,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_policy"].startswith("diagnostic PNG")
    assert manifest["requested_scale"] == scale
    assert manifest["font"]["font_id"] >= 0
    assert manifest["font"]["font_family"] == "DejaVu Sans"
    assert manifest["font"]["font_ascii_supported"]
    assert [state["state"] for state in manifest["states"]] == [
        "selected-result",
        "error-prior",
    ]
    selected, error_prior = manifest["states"]
    for state in manifest["states"]:
        assert state["window_bytes"] > 10_000
        assert state["canvas_bytes"] > 10_000
        assert state["visible_visual"][2] >= 240
        assert state["visible_visual"][3] >= 240
        assert state["visible_visual"][2] <= state["tab_size"][0]
        assert state["visible_visual"][3] <= state["tab_size"][1]
        assert not state["control_overlap"]
        assert state["canvas_has_focus"]
        assert state["device_pixel_ratio"] == pytest.approx(scale, rel=0.02)
        assert state["selected_raw_index"] == 0
        assert state["playback_time_s"] == 0.0
        assert "source sample 1/" in state["status"].lower()
        assert state["context"]
    assert selected["accepted_generation"] == error_prior["accepted_generation"]
    assert selected["accepted_context"] == error_prior["accepted_context"]
    assert selected["raw_sample_count"] == error_prior["raw_sample_count"]
    assert selected["canvas_sha256"] == error_prior["canvas_sha256"]
    assert selected["error"] == ""
    assert "prior accepted flight remains displayed" in error_prior["error"].lower()
