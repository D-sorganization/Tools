from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")


@pytest.mark.parametrize("scale", [1.0, 1.5])
def test_club_camera_source_states_render_at_supported_dpi(
    tmp_path: Path,
    scale: float,
) -> None:
    output = (
        Path(os.environ.get("RATE_PYQT_EVIDENCE_DIR", str(tmp_path)))
        / f"club-{scale:g}"
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
            str(Path(__file__).with_name("pyqt_club_camera_probe.py")),
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
        "procedural",
        "imported-selected-camera",
        "error-prior",
    ]
    for state in manifest["states"]:
        assert state["bytes"] > 10_000
        assert state["canvas"][2] >= 240 and state["canvas"][3] >= 240
        assert state["status"]
        assert state["focus"]
    assert manifest["states"][1]["camera"] == manifest["states"][2]["camera"]
    assert manifest["states"][0]["source"]["kind"] == "procedural"
    assert manifest["states"][1]["source"] == manifest["states"][2]["source"]
    assert manifest["states"][1]["status"] == manifest["states"][2]["status"]
    assert (
        manifest["states"][1]["canvas_sha256"] == manifest["states"][2]["canvas_sha256"]
    )
    assert len(manifest["states"][1]["source"]["sha256"]) == 64
    for state in manifest["states"]:
        assert state["device_pixel_ratio"] == pytest.approx(scale, rel=0.05)
    assert "prior head and camera remain displayed" in manifest["states"][2]["error"]
