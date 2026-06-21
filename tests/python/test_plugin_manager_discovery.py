"""Real-file coverage for plugin manager loading and discovery."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PYTHON_SRC = Path(__file__).resolve().parents[2] / "src" / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from src.core.plugin_manager import PluginManager  # noqa: E402


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_tools_keeps_valid_entries_and_skips_bad_tools_json_data(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """load_tools tolerates malformed categories and entries from real JSON."""
    (tmp_path / "valid.py").write_text("print('valid')\n", encoding="utf-8")
    outside_tool = tmp_path.parent / f"{tmp_path.name}_outside.py"
    outside_tool.write_text("print('outside')\n", encoding="utf-8")
    _write_json(
        tmp_path / "tools.json",
        {
            "Development Tools": [
                {
                    "name": "Valid",
                    "path": "valid.py",
                    "type": "python",
                    "desc": "loadable tool",
                },
                {
                    "name": "Traversal",
                    "path": f"../{outside_tool.name}",
                    "type": "python",
                    "desc": "outside repo",
                },
                {
                    "name": "MissingKey",
                    "path": "valid.py",
                    "type": "python",
                },
                "not-a-tool-entry",
            ],
            "Broken Category": {
                "name": "Scalar",
                "path": "valid.py",
                "type": "python",
                "desc": "category must be a list",
            },
        },
    )

    caplog.set_level("WARNING")
    loaded = PluginManager(tmp_path).load_tools()

    assert list(loaded) == ["Development Tools"]
    assert [tool.name for tool in loaded["Development Tools"]] == ["Valid"]
    assert "Security Alert" in caplog.text
    assert "Skipping invalid tool entry" in caplog.text
    assert "Skipping invalid tool category Broken Category" in caplog.text


def test_validate_tool_path_rejects_existing_file_outside_repo(tmp_path: Path) -> None:
    """validate_tool_path rejects traversal even when the target exists."""
    outside_tool = tmp_path.parent / f"{tmp_path.name}_outside.py"
    outside_tool.write_text("print('outside')\n", encoding="utf-8")

    valid, message = PluginManager(tmp_path).validate_tool_path(
        f"../{outside_tool.name}"
    )

    assert valid is False
    assert message is not None
    assert "Security Alert" in message


def test_scan_for_tools_discovers_real_manifest_with_default_fields(
    tmp_path: Path,
) -> None:
    """scan_for_tools reads a manifest and infers the entry point."""
    tool_dir = tmp_path / "tools" / "demo"
    tool_dir.mkdir(parents=True)
    (tool_dir / "demo.py").write_text("print('demo')\n", encoding="utf-8")
    _write_json(
        tool_dir / "tool_manifest.json",
        {
            "name": "Demo",
            "description": "manifest demo",
        },
    )

    discovered = PluginManager(tmp_path).scan_for_tools()

    assert list(discovered) == ["Development Tools"]
    tool = discovered["Development Tools"][0]
    assert tool.name == "Demo"
    assert Path(tool.path).as_posix() == "tools/demo/demo.py"
    assert tool.type == "python"
    assert tool.desc == "manifest demo"


def test_load_tools_with_discovery_replaces_duplicate_tools_json_entry(
    tmp_path: Path,
) -> None:
    """load_tools_with_discovery merges real files and lets manifests win."""
    (tmp_path / "stale.py").write_text("print('stale')\n", encoding="utf-8")
    _write_json(
        tmp_path / "tools.json",
        {
            "Development Tools": [
                {
                    "name": "Demo",
                    "path": "stale.py",
                    "type": "python",
                    "desc": "from tools.json",
                }
            ]
        },
    )
    tool_dir = tmp_path / "tools" / "demo"
    tool_dir.mkdir(parents=True)
    (tool_dir / "fresh.py").write_text("print('fresh')\n", encoding="utf-8")
    _write_json(
        tool_dir / "tool_manifest.json",
        {
            "name": "Demo",
            "path": "fresh.py",
            "type": "python",
            "desc": "from manifest",
            "category": "Development Tools",
        },
    )

    merged = PluginManager(tmp_path).load_tools_with_discovery()

    demo_tools = [tool for tool in merged["Development Tools"] if tool.name == "Demo"]
    assert len(demo_tools) == 1
    assert Path(demo_tools[0].path).as_posix() == "tools/demo/fresh.py"
    assert demo_tools[0].desc == "from manifest"
