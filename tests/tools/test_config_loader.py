"""Tests for config_loader."""

import json

from tools.config_loader import load_tools_config, validate_tools_config


def test_validate_tools_config(tmp_path):
    repo_root = tmp_path
    repo_root.mkdir(exist_ok=True)

    config = {
        "Media Processing": [
            {"name": "Valid Tool", "path": "src/media/tool.py"},
            {"name": "Escape Repo", "path": "../outside.py"},
            {"path": "no_name.py"},
            "invalid_string",
        ]
    }

    validated = validate_tools_config(config, repo_root=repo_root)
    assert "Media Processing" in validated
    tools = validated["Media Processing"]
    assert len(tools) == 1
    assert tools[0]["name"] == "Valid Tool"


def test_load_tools_config(tmp_path):
    repo_root = tmp_path
    tools_json = repo_root / "tools.json"

    config = {
        "Media Processing": [
            {"name": "Valid Tool", "path": "src/media/tool.py"},
        ]
    }
    tools_json.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_tools_config(repo_root)
    assert "Media Processing" in loaded
    assert len(loaded["Media Processing"]) == 1


def test_load_tools_empty(tmp_path):
    loaded = load_tools_config(tmp_path)
    assert loaded == {}
