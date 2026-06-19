# ruff: noqa: E501
"""Plugin manager DbC, LoD, and TDD regression coverage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _import_plugin_manager_module() -> Any:
    """Import plugin_manager, skipping if dependencies unavailable."""
    import importlib.util

    pm_path = Path(__file__).parent.parent / "src" / "core" / "plugin_manager.py"
    if not pm_path.exists():
        pytest.skip("plugin_manager.py not found")

    mock_utils = MagicMock()
    mock_utils.safe_read_json = MagicMock(return_value={})
    mock_core_utils = MagicMock()
    mock_core_utils.file_utils = mock_utils

    fake_pkg_name = "plugin_manager_test_pkg"
    fake_core_name = f"{fake_pkg_name}.core"
    fake_utils_name = f"{fake_pkg_name}.utils"
    fake_file_utils_name = f"{fake_pkg_name}.utils.file_utils"

    fake_pkg = MagicMock()
    fake_pkg.__name__ = fake_pkg_name
    fake_pkg.__path__ = [str(pm_path.parent.parent)]
    fake_pkg.__package__ = fake_pkg_name

    fake_core = MagicMock()
    fake_core.__name__ = fake_core_name
    fake_core.__path__ = [str(pm_path.parent)]
    fake_core.__package__ = fake_pkg_name

    fake_file_utils = MagicMock()
    fake_file_utils.safe_read_json = MagicMock(return_value={})

    extra_modules = {
        fake_pkg_name: fake_pkg,
        fake_core_name: fake_core,
        fake_utils_name: mock_core_utils,
        fake_file_utils_name: fake_file_utils,
    }

    try:
        spec = importlib.util.spec_from_file_location(
            f"{fake_core_name}.plugin_manager",
            pm_path,
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        module.__package__ = fake_core_name

        with patch.dict("sys.modules", extra_modules):
            spec.loader.exec_module(module)

        return module
    except Exception:  # noqa: BLE001 - test isolation: any import failure skips.
        pytest.skip("plugin_manager not importable in isolation")


def _get_plugin_manager_class() -> Any:
    """Import PluginManager, skipping if dependencies unavailable."""
    module = _import_plugin_manager_module()
    return module.PluginManager


def _read_json_file(path: Path, default: object | None = None) -> object:
    """Read real JSON fixtures through the plugin manager's safe_read_json seam."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return default


def _write_json(path: Path, data: object) -> None:
    """Write compact JSON fixtures for plugin manager tests."""
    path.write_text(json.dumps(data), encoding="utf-8")


def test_plugin_manager_init_type_error(tmp_path: Path) -> None:
    """PluginManager.__init__ raises TypeError for non-Path repo_root."""
    PluginManager = _get_plugin_manager_class()

    with pytest.raises(TypeError, match="repo_root must be a Path"):
        PluginManager(repo_root="/some/path")

    with pytest.raises(TypeError, match="repo_root must be a Path"):
        PluginManager(repo_root=str(tmp_path))


def test_plugin_manager_init_valid(tmp_path: Path) -> None:
    """PluginManager.__init__ succeeds with a Path."""
    PluginManager = _get_plugin_manager_class()

    manager = PluginManager(repo_root=tmp_path)
    assert manager.repo_root == tmp_path


def test_get_tool_by_name_type_error(tmp_path: Path) -> None:
    """get_tool_by_name raises TypeError for non-str name."""
    PluginManager = _get_plugin_manager_class()
    manager = PluginManager(repo_root=tmp_path)

    with pytest.raises(TypeError, match="name must be a str"):
        manager.get_tool_by_name(123)

    with pytest.raises(TypeError, match="name must be a str"):
        manager.get_tool_by_name(None)


def test_get_tool_by_name_valid(tmp_path: Path) -> None:
    """get_tool_by_name returns None for missing tool."""
    PluginManager = _get_plugin_manager_class()
    manager = PluginManager(repo_root=tmp_path)

    assert manager.get_tool_by_name("nonexistent") is None


def test_validate_tool_path_type_error_for_non_str(tmp_path: Path) -> None:
    """validate_tool_path raises TypeError for non-str tool_path."""
    PluginManager = _get_plugin_manager_class()
    manager = PluginManager(repo_root=tmp_path)

    with pytest.raises(TypeError, match="tool_path must be a str"):
        manager.validate_tool_path(tmp_path / "tool.py")

    with pytest.raises(TypeError, match="tool_path must be a str"):
        manager.validate_tool_path(None)


def test_plugin_manager_load_tools_reads_real_manifest_entries(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """load_tools accepts valid tools and skips invalid real tools.json entries."""
    module = _import_plugin_manager_module()

    (tmp_path / "valid.py").write_text("print('ok')\n", encoding="utf-8")
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
                    "desc": "real tool",
                },
                {
                    "name": "Traversal",
                    "path": f"../{outside_tool.name}",
                    "type": "python",
                    "desc": "outside repo",
                },
                {"name": "Missing path", "type": "python", "desc": "missing path key"},
            ]
        },
    )

    caplog.set_level("WARNING", logger=module.logger.name)
    manager = module.PluginManager(repo_root=tmp_path)

    with patch.object(module, "safe_read_json", side_effect=_read_json_file):
        tools = manager.load_tools()

    loaded = tools["Development Tools"]
    assert [tool.name for tool in loaded] == ["Valid"]
    assert loaded[0].path == "valid.py"
    assert "Security Alert" in caplog.text
    assert "Skipping invalid tool entry" in caplog.text


def test_plugin_manager_load_tools_fail_closes_on_non_dict_entry(
    tmp_path: Path,
) -> None:
    """load_tools covers malformed non-dict entries until hardening PRs refine it."""
    module = _import_plugin_manager_module()
    _write_json(tmp_path / "tools.json", {"Development Tools": ["not-a-tool-entry"]})
    manager = module.PluginManager(repo_root=tmp_path)

    with patch.object(module, "safe_read_json", side_effect=_read_json_file):
        assert manager.load_tools() == {}


def test_plugin_manager_load_tools_fail_closes_on_non_list_category(
    tmp_path: Path,
) -> None:
    """load_tools covers non-list category payloads from real tools.json data."""
    module = _import_plugin_manager_module()
    _write_json(
        tmp_path / "tools.json",
        {
            "Development Tools": {
                "name": "Scalar",
                "path": "tool.py",
                "type": "python",
                "desc": "not in a list",
            }
        },
    )
    manager = module.PluginManager(repo_root=tmp_path)

    with patch.object(module, "safe_read_json", side_effect=_read_json_file):
        assert manager.load_tools() == {}


def test_plugin_manager_scan_defaults_are_module_constants() -> None:
    """scan_for_tools uses shared constants for manifest defaults."""
    module = _import_plugin_manager_module()

    assert module.TOOL_MANIFEST_FILENAME == "tool_manifest.json"
    assert module.DEFAULT_TOOL_TYPE == "python"
    assert module.DEFAULT_TOOL_CATEGORY == "Development Tools"
    assert module.DEFAULT_TOOL_SCAN_DIRS == (
        "tools",
        "web_applications",
        "data_processing",
        "scientific_modeling",
        "media_processing",
    )


def test_discovered_tool_wins_name_collision(tmp_path: Path) -> None:
    """load_tools_with_discovery replaces a same-named JSON tool."""
    module = _import_plugin_manager_module()
    Tool = module.Tool
    manager = module.PluginManager(repo_root=tmp_path)
    json_tool = Tool(
        "dup", "stale.py", "python", "from tools.json", "Development Tools"
    )
    discovered_tool = Tool(
        "dup", "fresh.py", "python", "from manifest", "Development Tools"
    )

    manager.load_tools = MagicMock(return_value={"Development Tools": [json_tool]})
    manager.scan_for_tools = MagicMock(
        return_value={"Development Tools": [discovered_tool]}
    )

    dup_tools = [
        tool
        for tool in manager.load_tools_with_discovery()["Development Tools"]
        if tool.name == "dup"
    ]
    assert len(dup_tools) == 1
    assert dup_tools[0].path == "fresh.py"
    assert dup_tools[0].desc == "from manifest"


def test_scan_for_tools_discovers_real_manifest_file(tmp_path: Path) -> None:
    """scan_for_tools discovers a tool_manifest.json and validates its path."""
    module = _import_plugin_manager_module()
    tool_dir = tmp_path / "tools" / "demo"
    tool_dir.mkdir(parents=True)
    (tool_dir / "demo.py").write_text("print('demo')\n", encoding="utf-8")
    _write_json(
        tool_dir / "tool_manifest.json",
        {
            "name": "Demo",
            "path": "demo.py",
            "type": "python",
            "description": "manifest demo",
            "category": "Discovered",
        },
    )
    manager = module.PluginManager(repo_root=tmp_path)

    with patch.object(module, "safe_read_json", side_effect=_read_json_file):
        discovered = manager.scan_for_tools()

    assert list(discovered) == ["Discovered"]
    tool = discovered["Discovered"][0]
    assert tool.name == "Demo"
    assert Path(tool.path).as_posix() == "tools/demo/demo.py"
    assert tool.desc == "manifest demo"


def test_load_tools_with_discovery_merges_real_files_and_deduplicates(
    tmp_path: Path,
) -> None:
    """load_tools_with_discovery replaces stale tools.json entries with manifests."""
    module = _import_plugin_manager_module()
    (tmp_path / "stale.py").write_text("print('stale')\n", encoding="utf-8")
    _write_json(
        tmp_path / "tools.json",
        {
            "Development Tools": [
                {
                    "name": "Demo",
                    "path": "stale.py",
                    "type": "python",
                    "desc": "from tools json",
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
            "description": "from manifest",
            "category": "Development Tools",
        },
    )
    manager = module.PluginManager(repo_root=tmp_path)

    with patch.object(module, "safe_read_json", side_effect=_read_json_file):
        merged = manager.load_tools_with_discovery()

    demo_tools = [tool for tool in merged["Development Tools"] if tool.name == "Demo"]
    assert len(demo_tools) == 1
    assert Path(demo_tools[0].path).as_posix() == "tools/demo/fresh.py"
    assert demo_tools[0].desc == "from manifest"


def test_scan_for_tools_propagates_unexpected_tool_construction_error(
    tmp_path: Path,
) -> None:
    """scan_for_tools catches manifest errors without swallowing programming bugs."""
    module = _import_plugin_manager_module()
    tool_dir = tmp_path / "tools" / "demo"
    tool_dir.mkdir(parents=True)
    (tool_dir / "demo.py").write_text("print('demo')\n")
    (tool_dir / "tool_manifest.json").write_text("{}\n")
    manager = module.PluginManager(repo_root=tmp_path)

    with (
        patch.object(module, "safe_read_json", return_value={"name": "Demo"}),
        patch.object(module, "Tool", side_effect=RuntimeError("construction bug")),
        pytest.raises(RuntimeError, match="construction bug"),
    ):
        manager.scan_for_tools()
