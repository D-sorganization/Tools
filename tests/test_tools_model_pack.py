"""Tests for the Tools provider manifest consumed by UpstreamDrift."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tools_model_pack_exposes_shared_video_and_data_surfaces() -> None:
    manifest_path = REPO_ROOT / "model_pack.yaml"
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert raw["provider"] == "tools"
    models = {entry["id"]: entry for entry in raw["models"]}

    for tool_id in (
        "video_analyzer",
        "video_processor",
        "data_explorer",
        "data_processor",
    ):
        assert tool_id in models, f"Missing shared Tools model: {tool_id}"
        assert (REPO_ROOT / models[tool_id]["path"]).exists()
        assert models[tool_id]["launcher"]["status"] == "ready"

    assert models["video_processor"]["working_dir"] == (
        "src/media_processing/video_processor/apps/web"
    )
    assert (
        "src/data_processing/data_processor/python"
        in models["data_processor"]["python_paths"]
    )


def test_tools_model_pack_exposes_one_canonical_movement_optimizer() -> None:
    """UpstreamDrift should discover the Tools-resident canonical optimizer only."""
    manifest_path = REPO_ROOT / "model_pack.yaml"
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    models = {entry["id"]: entry for entry in raw["models"]}

    movement = models["tools_movement_optimizer"]
    assert movement["path"] == "src/movement_optimizer/launch_pyqt6.py"
    assert movement["launcher"]["web_route"] == "/tools/movement-optimizer"
    assert movement["capabilities"] == [
        "optimization",
        "biomechanics",
        "trajectory",
        "cli",
        "pyqt6",
        "swingset",
        "chain_dynamics",
    ]
    assert movement["supported_exercises"] == [
        "squat",
        "full_squat",
        "deadlift",
        "bench_press",
        "snatch",
        "clean",
        "jerk",
    ]

    advertised_paths = {entry["path"] for entry in raw["models"]}
    assert "src/optimizer_gui/launch_pyqt6.py" not in advertised_paths
