"""Tests for config_loader.

Expanded test suite covering validation, sanitization, path-security,
categories, and edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.config_loader import (
    CATEGORY_ORDER,
    load_tools_config,
    validate_tools_config,
)

# ─── CATEGORY_ORDER Tests ────────────────────────────────────


class TestCategoryOrder:
    """Test the CATEGORY_ORDER constant."""

    def test_is_list(self) -> None:
        assert isinstance(CATEGORY_ORDER, list)

    def test_not_empty(self) -> None:
        assert len(CATEGORY_ORDER) > 0

    def test_all_strings(self) -> None:
        for cat in CATEGORY_ORDER:
            assert isinstance(cat, str), f"{cat} not a string"

    def test_contains_media_processing(self) -> None:
        assert "Media Processing" in CATEGORY_ORDER

    def test_no_duplicates(self) -> None:
        assert len(CATEGORY_ORDER) == len(set(CATEGORY_ORDER))


# ─── validate_tools_config Tests ─────────────────────────────


class TestValidateToolsConfig:
    """Test the validate_tools_config function."""

    def test_valid_tool_passes(self, tmp_path: Path) -> None:
        config: dict[str, Any] = {
            "Media Processing": [
                {"name": "My Tool", "path": "src/media/tool.py"},
            ]
        }
        result = validate_tools_config(config, repo_root=tmp_path)
        assert "Media Processing" in result
        assert len(result["Media Processing"]) == 1
        assert result["Media Processing"][0]["name"] == "My Tool"

    def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        config: dict[str, Any] = {"Cat": [{"name": "Escape", "path": "../../../etc/passwd"}]}
        result = validate_tools_config(config, repo_root=tmp_path)
        assert "Cat" not in result or len(result.get("Cat", [])) == 0

    def test_rejects_missing_name(self, tmp_path: Path) -> None:
        config: dict[str, Any] = {"Cat": [{"path": "src/tool.py"}]}
        result = validate_tools_config(config, repo_root=tmp_path)
        assert "Cat" not in result

    def test_rejects_missing_path(self, tmp_path: Path) -> None:
        config: dict[str, Any] = {"Cat": [{"name": "Tool"}]}
        result = validate_tools_config(config, repo_root=tmp_path)
        assert "Cat" not in result

    def test_rejects_string_entries(self, tmp_path: Path) -> None:
        config: dict[str, Any] = {"Cat": ["just_a_string"]}
        result = validate_tools_config(config, repo_root=tmp_path)
        assert "Cat" not in result

    def test_skips_non_list_categories(self, tmp_path: Path) -> None:
        config: dict[str, Any] = {"Cat": "not a list"}
        result = validate_tools_config(config, repo_root=tmp_path)
        assert result == {}

    def test_multiple_categories(self, tmp_path: Path) -> None:
        config: dict[str, Any] = {
            "Media Processing": [{"name": "A", "path": "src/a.py"}],
            "Data Processing": [{"name": "B", "path": "src/b.py"}],
        }
        result = validate_tools_config(config, repo_root=tmp_path)
        assert len(result) == 2

    def test_mixed_valid_invalid(self, tmp_path: Path) -> None:
        """Valid tools survive alongside invalid entries."""
        config: dict[str, Any] = {
            "Cat": [
                {"name": "Good", "path": "src/good.py"},
                "bad_entry",
                {"path": "missing_name.py"},
            ]
        }
        result = validate_tools_config(config, repo_root=tmp_path)
        assert len(result["Cat"]) == 1
        assert result["Cat"][0]["name"] == "Good"

    def test_no_repo_root_rejects_dotdot(self) -> None:
        """Without repo_root, paths with .. are rejected."""
        config: dict[str, Any] = {"Cat": [{"name": "Bad", "path": "../outside.py"}]}
        result = validate_tools_config(config, repo_root=None)
        assert "Cat" not in result

    def test_no_repo_root_allows_clean_paths(self) -> None:
        config: dict[str, Any] = {"Cat": [{"name": "Good", "path": "src/tool.py"}]}
        result = validate_tools_config(config, repo_root=None)
        assert len(result["Cat"]) == 1

    def test_empty_config(self, tmp_path: Path) -> None:
        result = validate_tools_config({}, repo_root=tmp_path)
        assert result == {}

    def test_empty_tools_list(self, tmp_path: Path) -> None:
        config: dict[str, Any] = {"Cat": []}
        result = validate_tools_config(config, repo_root=tmp_path)
        assert "Cat" not in result  # Empty categories are excluded

    def test_returns_dict(self, tmp_path: Path) -> None:
        result = validate_tools_config({}, repo_root=tmp_path)
        assert isinstance(result, dict)


# ─── load_tools_config Tests ─────────────────────────────────


class TestLoadToolsConfig:
    """Test the load_tools_config function."""

    def test_loads_valid_json(self, tmp_path: Path) -> None:
        config = {
            "Media Processing": [
                {"name": "Tool", "path": "src/media/tool.py"},
            ]
        }
        tools_json = tmp_path / "tools.json"
        tools_json.write_text(json.dumps(config), encoding="utf-8")

        loaded = load_tools_config(tmp_path)
        assert "Media Processing" in loaded
        assert len(loaded["Media Processing"]) == 1

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        loaded = load_tools_config(tmp_path)
        assert loaded == {}

    def test_empty_json_returns_empty(self, tmp_path: Path) -> None:
        tools_json = tmp_path / "tools.json"
        tools_json.write_text("{}", encoding="utf-8")
        loaded = load_tools_config(tmp_path)
        assert loaded == {}

    def test_multi_category_load(self, tmp_path: Path) -> None:
        config = {
            "Media Processing": [{"name": "A", "path": "src/a.py"}],
            "Scientific Modeling": [{"name": "B", "path": "src/b.py"}],
        }
        tools_json = tmp_path / "tools.json"
        tools_json.write_text(json.dumps(config), encoding="utf-8")

        loaded = load_tools_config(tmp_path)
        assert len(loaded) == 2

    def test_returns_dict_type(self, tmp_path: Path) -> None:
        loaded = load_tools_config(tmp_path)
        assert isinstance(loaded, dict)
