"""Rendered PyQt Variation lifecycle evidence at both supported DPI scales."""

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
def test_variation_lifecycle_states_are_visible_without_occlusion(
    tmp_path: Path, scale: float
) -> None:
    output_root = Path(os.environ.get("RATE_PYQT_EVIDENCE_DIR", str(tmp_path)))
    output = output_root / f"variation-state-scale-{scale:g}"
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
            str(Path(__file__).with_name("pyqt_variation_visual_state_probe.py")),
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
    assert [state["state"] for state in manifest["states"]] == [
        "empty",
        "loading-no-prior",
        "error-empty",
        "result",
        "loading-prior",
        "error-prior",
    ]
    for state in manifest["states"]:
        assert state["bytes"] > 10_000
        assert state["visible_content"][2] >= 240
        assert state["visible_content"][3] >= 240
        assert state["visible_content"][2] <= state["tab_size"][0]
        assert state["visible_content"][3] <= state["tab_size"][1]
        assert not state["overlap"]
        assert not state["control_overlap"]
        assert state["status"]
        assert state["strip_visible"] == (state["phase"] in {"loading", "error"})
