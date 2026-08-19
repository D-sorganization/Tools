"""DPI-isolated PyQt rendered interaction evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("PyQt6")

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _run_probe(tmp_path: Path, scale: float) -> dict[str, Any]:
    output = tmp_path / f"scale-{scale:g}"
    environment = dict(os.environ)
    repository = Path(__file__).resolve().parents[2]
    python_path = os.pathsep.join((str(repository / "src"), str(repository)))
    environment.update(
        {
            "QT_QPA_PLATFORM": "offscreen",
            "QT_SCALE_FACTOR": str(scale),
            "MPLBACKEND": "qtagg",
            "PYTHONPATH": python_path,
        }
    )
    helper = Path(__file__).with_name("pyqt_variation_render_probe.py")
    subprocess.run(
        [sys.executable, str(helper), "--output", str(output), "--scale", str(scale)],
        check=True,
        env=environment,
        timeout=90,
    )
    loaded = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise AssertionError("render manifest must be an object")
    return cast(dict[str, Any], loaded)


def test_rendered_controls_at_100_and_150_percent_dpi(tmp_path: Path) -> None:
    output_root = Path(os.environ.get("RATE_PYQT_EVIDENCE_DIR", str(tmp_path)))
    output_root.mkdir(parents=True, exist_ok=True)
    manifests = [_run_probe(output_root, scale) for scale in (1.0, 1.5)]
    for requested_scale, manifest in zip((1.0, 1.5), manifests, strict=True):
        assert (
            manifest["artifact_policy"]
            == "diagnostic PNG; semantic manifest is test authority"
        )
        assert manifest["requested_scale"] == requested_scale
        for key in ("arc", "plot"):
            evidence = manifest[key]
            assert evidence["overlaps"] == []
            assert evidence["bytes"] > 10_000
            assert evidence["device_pixel_ratio"] == pytest.approx(requested_scale)
        assert manifest["arc"]["ellipsoid_toggle"] is True
        assert manifest["arc"]["metric"] == "confidence-ellipsoid-volume"
        assert manifest["arc"]["camera"] == pytest.approx(
            {"azimuth_deg": -37.0, "elevation_deg": 22.0}
        )
        assert manifest["plot"]["zoom_percent"] == 125
        assert manifest["plot"]["legend"] == "outside_right"

    for key in ("arc", "plot"):
        base_pixels = manifests[0][key]["pixel_size"]
        scaled_pixels = manifests[1][key]["pixel_size"]
        assert scaled_pixels[0] / base_pixels[0] == pytest.approx(1.5, rel=0.03)
        assert scaled_pixels[1] / base_pixels[1] == pytest.approx(1.5, rel=0.03)
