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
        assert models[tool_id]["launcher"]["status"] == "external"

    assert models["video_processor"]["working_dir"] == (
        "src/media_processing/video_processor/apps/web"
    )
    assert (
        "src/data_processing/data_processor/python"
        in models["data_processor"]["python_paths"]
    )
