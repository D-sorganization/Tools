# ruff: noqa: E501
"""Tests for DbC and LoD fixes in src/python.

Covers:
- performance_utils.OptimizedFileScanner: DbC preconditions + LoD cache fix
- plugin_manager.PluginManager: DbC preconditions + LoD manifest_path.parent extraction
- help_system: DbC preconditions on markdown handlers and public API
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# performance_utils tests
# ---------------------------------------------------------------------------


def test_optimized_file_scanner_init_valid() -> None:
    """OptimizedFileScanner initializes with valid max_workers values."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
    from performance_utils import OptimizedFileScanner

    scanner = OptimizedFileScanner(max_workers=4)
    assert scanner.max_workers == 4

    scanner_auto = OptimizedFileScanner(max_workers=-1)
    assert scanner_auto.max_workers > 0  # auto-detected

    scanner_default = OptimizedFileScanner()
    assert scanner_default.max_workers > 0


def test_optimized_file_scanner_init_type_error() -> None:
    """OptimizedFileScanner raises TypeError for non-int max_workers."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
    from performance_utils import OptimizedFileScanner

    with pytest.raises(TypeError, match="max_workers must be an int"):
        OptimizedFileScanner(max_workers="4")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="max_workers must be an int"):
        OptimizedFileScanner(max_workers=4.0)  # type: ignore[arg-type]


def test_optimized_file_scanner_init_value_error() -> None:
    """OptimizedFileScanner raises ValueError for max_workers < -1."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
    from performance_utils import OptimizedFileScanner

    with pytest.raises(ValueError, match="max_workers must be >= -1"):
        OptimizedFileScanner(max_workers=-2)


def test_scan_directory_parallel_type_error() -> None:
    """scan_directory_parallel raises TypeError for non-Path root_path."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
    from performance_utils import OptimizedFileScanner

    scanner = OptimizedFileScanner()

    with pytest.raises(TypeError, match="root_path must be a Path"):
        list(scanner.scan_directory_parallel("/some/path"))  # type: ignore[arg-type]


def test_scan_directory_parallel_with_temp_dir(tmp_path: Path) -> None:
    """scan_directory_parallel works on a real temporary directory."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
    from performance_utils import OptimizedFileScanner

    # Create test files
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.py").write_text("print('hi')")
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "c.txt").write_text("world")

    scanner = OptimizedFileScanner(max_workers=1)
    found = list(scanner.scan_directory_parallel(tmp_path, pattern="*.txt"))
    names = {f.name for f in found}
    assert "a.txt" in names
    assert "c.txt" in names
    assert "b.py" not in names


def test_scan_directory_parallel_cache_uses_stat_mtime(tmp_path: Path) -> None:
    """Cache uses Path.stat().st_mtime — no os.path.getmtime chain (LoD fix)."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
    import performance_utils

    # Verify os.path.getmtime is NOT called during scanning
    with patch("os.path.getmtime") as mock_getmtime:
        scanner = performance_utils.OptimizedFileScanner(max_workers=1)
        (tmp_path / "test.txt").write_text("data")
        list(scanner.scan_directory_parallel(tmp_path, pattern="*.txt"))
        mock_getmtime.assert_not_called()


# ---------------------------------------------------------------------------
# plugin_manager tests
# ---------------------------------------------------------------------------


def _import_plugin_manager_module():
    """Import plugin_manager, skipping if dependencies unavailable."""
    import importlib.util

    pm_path = Path(__file__).parent.parent / "src" / "core" / "plugin_manager.py"
    if not pm_path.exists():
        return None

    # Build a fake package hierarchy to resolve relative imports
    mock_utils = MagicMock()
    mock_utils.safe_read_json = MagicMock(return_value={})
    mock_core_utils = MagicMock()
    mock_core_utils.file_utils = mock_utils

    fake_pkg_name = "plugin_manager_test_pkg"
    fake_core_name = f"{fake_pkg_name}.core"
    fake_utils_name = f"{fake_pkg_name}.utils"
    fake_file_utils_name = f"{fake_pkg_name}.utils.file_utils"

    # Register stubs so relative imports resolve
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
        module.__package__ = fake_core_name  # type: ignore[assignment]

        with patch.dict("sys.modules", extra_modules):
            spec.loader.exec_module(module)  # type: ignore[union-attr]

        return module
    except Exception:  # noqa: BLE001 — test isolation: any import failure returns None to skip
        return None


def _get_plugin_manager_class():
    """Import PluginManager, skipping if dependencies unavailable."""
    module = _import_plugin_manager_module()
    if module is None:
        return None
    return module.PluginManager


def test_plugin_manager_init_type_error(tmp_path: Path) -> None:
    """PluginManager.__init__ raises TypeError for non-Path repo_root."""
    PluginManager = _get_plugin_manager_class()
    if PluginManager is None:
        pytest.skip("PluginManager not importable in isolation")

    with pytest.raises(TypeError, match="repo_root must be a Path"):
        PluginManager(repo_root="/some/path")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="repo_root must be a Path"):
        PluginManager(repo_root=str(tmp_path))  # type: ignore[arg-type]


def test_plugin_manager_init_valid(tmp_path: Path) -> None:
    """PluginManager.__init__ succeeds with a Path."""
    PluginManager = _get_plugin_manager_class()
    if PluginManager is None:
        pytest.skip("PluginManager not importable in isolation")

    manager = PluginManager(repo_root=tmp_path)
    assert manager.repo_root == tmp_path


def test_get_tool_by_name_type_error(tmp_path: Path) -> None:
    """get_tool_by_name raises TypeError for non-str name."""
    PluginManager = _get_plugin_manager_class()
    if PluginManager is None:
        pytest.skip("PluginManager not importable in isolation")

    manager = PluginManager(repo_root=tmp_path)

    with pytest.raises(TypeError, match="name must be a str"):
        manager.get_tool_by_name(123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="name must be a str"):
        manager.get_tool_by_name(None)  # type: ignore[arg-type]


def test_get_tool_by_name_valid(tmp_path: Path) -> None:
    """get_tool_by_name returns None for missing tool (valid str)."""
    PluginManager = _get_plugin_manager_class()
    if PluginManager is None:
        pytest.skip("PluginManager not importable in isolation")

    manager = PluginManager(repo_root=tmp_path)
    result = manager.get_tool_by_name("nonexistent")
    assert result is None


def test_validate_tool_path_type_error_for_non_str(tmp_path: Path) -> None:
    """validate_tool_path raises TypeError for non-str tool_path."""
    PluginManager = _get_plugin_manager_class()
    if PluginManager is None:
        pytest.skip("PluginManager not importable in isolation")

    manager = PluginManager(repo_root=tmp_path)

    with pytest.raises(TypeError, match="tool_path must be a str"):
        manager.validate_tool_path(tmp_path / "tool.py")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="tool_path must be a str"):
        manager.validate_tool_path(None)  # type: ignore[arg-type]


def test_plugin_manager_scan_defaults_are_module_constants() -> None:
    """scan_for_tools uses shared constants for manifest defaults."""
    module = _import_plugin_manager_module()
    if module is None:
        pytest.skip("plugin_manager not importable in isolation")

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
    """load_tools_with_discovery: a discovered tool overrides a same-named JSON tool.

    Regression for the precedence contract: the merge comment promises discovered
    tools take precedence for duplicates, so a discovered manifest entry must
    replace a stale tools.json entry of the same name in the same category.
    """
    module = _import_plugin_manager_module()
    if module is None:
        pytest.skip("plugin_manager not importable in isolation")

    Tool = module.Tool
    PluginManager = module.PluginManager

    manager = PluginManager(repo_root=tmp_path)

    json_tool = Tool(
        name="dup",
        path="stale.py",
        type="python",
        desc="from tools.json",
        category="Development Tools",
    )
    discovered_tool = Tool(
        name="dup",
        path="fresh.py",
        type="python",
        desc="from manifest",
        category="Development Tools",
    )

    manager.load_tools = MagicMock(return_value={"Development Tools": [json_tool]})
    manager.scan_for_tools = MagicMock(
        return_value={"Development Tools": [discovered_tool]}
    )

    merged = manager.load_tools_with_discovery()
    category = merged["Development Tools"]

    # Exactly one tool named "dup" survives, and it is the discovered one.
    dup_tools = [t for t in category if t.name == "dup"]
    assert len(dup_tools) == 1
    surviving = dup_tools[0]
    assert surviving.path == "fresh.py"
    assert surviving.desc == "from manifest"


def test_scan_for_tools_propagates_unexpected_tool_construction_error(
    tmp_path: Path,
) -> None:
    """scan_for_tools catches manifest errors without swallowing programming bugs."""
    module = _import_plugin_manager_module()
    if module is None:
        pytest.skip("plugin_manager not importable in isolation")

    tool_dir = tmp_path / "tools" / "demo"
    tool_dir.mkdir(parents=True)
    tool_file = tool_dir / "demo.py"
    tool_file.write_text("print('demo')\n")
    manifest_path = tool_dir / "tool_manifest.json"
    manifest_path.write_text("{}\n")

    manager = module.PluginManager(repo_root=tmp_path)

    with (
        patch.object(module, "safe_read_json", return_value={"name": "Demo"}),
        patch.object(module, "Tool", side_effect=RuntimeError("construction bug")),
        pytest.raises(RuntimeError, match="construction bug"),
    ):
        manager.scan_for_tools()


# ---------------------------------------------------------------------------
# help_system tests
# ---------------------------------------------------------------------------


def _import_help_handlers():
    """Import help_system handler functions, skipping if PyQt6 unavailable."""
    pytest.importorskip("PyQt6", reason="PyQt6 not available")
    try:
        import importlib.util

        hs_path = Path(__file__).parent.parent / "src" / "help" / "help_system.py"
        spec = importlib.util.spec_from_file_location("help_system_test", hs_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module
    except Exception as e:  # noqa: BLE001 — test isolation: any import failure skips suite
        pytest.skip(f"help_system not importable: {e}")


def test_handle_code_block_type_error() -> None:
    """_handle_code_block raises TypeError for non-str line."""
    hs = _import_help_handlers()
    state = hs._MarkdownState()

    with pytest.raises(TypeError, match="line must be a str"):
        hs._handle_code_block(None, state)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="line must be a str"):
        hs._handle_code_block(42, state)  # type: ignore[arg-type]


def test_handle_horizontal_rule_type_error() -> None:
    """_handle_horizontal_rule raises TypeError for non-str line."""
    hs = _import_help_handlers()
    state = hs._MarkdownState()

    with pytest.raises(TypeError, match="line must be a str"):
        hs._handle_horizontal_rule(None, state)  # type: ignore[arg-type]


def test_handle_table_line_type_error() -> None:
    """_handle_table_line raises TypeError for non-str line."""
    hs = _import_help_handlers()
    state = hs._MarkdownState()

    with pytest.raises(TypeError, match="line must be a str"):
        hs._handle_table_line(None, state)  # type: ignore[arg-type]


def test_handle_header_type_error() -> None:
    """_handle_header raises TypeError for non-str line."""
    hs = _import_help_handlers()
    state = hs._MarkdownState()

    with pytest.raises(TypeError, match="line must be a str"):
        hs._handle_header(None, state)  # type: ignore[arg-type]


def test_handle_list_item_type_error() -> None:
    """_handle_list_item raises TypeError for non-str line."""
    hs = _import_help_handlers()
    state = hs._MarkdownState()

    with pytest.raises(TypeError, match="line must be a str"):
        hs._handle_list_item(None, state)  # type: ignore[arg-type]


def test_handle_paragraph_type_error() -> None:
    """_handle_paragraph raises TypeError for non-str line."""
    hs = _import_help_handlers()
    state = hs._MarkdownState()

    with pytest.raises(TypeError, match="line must be a str"):
        hs._handle_paragraph(None, state)  # type: ignore[arg-type]


def test_markdown_to_html_basic() -> None:
    """_markdown_to_html converts basic markdown correctly."""
    hs = _import_help_handlers()

    result = hs._markdown_to_html("# Hello World")
    assert "<h1" in result
    assert "Hello World" in result

    result2 = hs._markdown_to_html("- item one\n- item two")
    assert "<li>" in result2
    assert "item one" in result2
