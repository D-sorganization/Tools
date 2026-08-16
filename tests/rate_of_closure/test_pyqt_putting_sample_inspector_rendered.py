"""Rendered Putting inspector evidence at both supported DPI scales."""

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
def test_putting_selected_and_error_prior_are_visible(
    tmp_path: Path, scale: float
) -> None:
    output_root = Path(os.environ.get("RATE_PYQT_EVIDENCE_DIR", str(tmp_path)))
    output = output_root / f"putting-inspector-scale-{scale:g}"
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
            str(Path(__file__).with_name("pyqt_putting_sample_inspector_probe.py")),
            "--output",
            str(output),
            "--scale",
            str(scale),
        ],
        check=True,
        env=environment,
        timeout=60,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["requested_scale"] == scale
    assert [state["state"] for state in manifest["states"]] == [
        "selected-result",
        "error-prior",
    ]
    for state in manifest["states"]:
        assert state["bytes"] > 10_000
        assert state["visible_visual"][2] >= 240
        assert state["visible_visual"][3] >= 240
        assert state["visible_visual"][2] <= state["tab_size"][0]
        assert state["visible_visual"][3] <= state["tab_size"][1]
        assert not state["control_overlap"]
        assert state["canvas_has_focus"]
        assert state["selected_raw_index"] == 0
        assert state["selected_marker_count"] == 2
        assert "Source sample 0" in state["status"]
        assert state["context"].startswith("Displayed result:")
        if state["state"] == "error-prior":
            assert "accepted context below remains displayed" in state["error"]
        else:
            assert state["error"] == ""
