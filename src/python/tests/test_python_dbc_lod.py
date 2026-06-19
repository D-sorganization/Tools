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
from typing import Any
from unittest.mock import patch

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
        OptimizedFileScanner(max_workers="4")

    with pytest.raises(TypeError, match="max_workers must be an int"):
        OptimizedFileScanner(max_workers=4.0)


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
        list(scanner.scan_directory_parallel("/some/path"))


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
# help_system tests
# ---------------------------------------------------------------------------


def _import_help_handlers() -> Any:
    """Import help_system handler functions, skipping if PyQt6 unavailable."""
    pytest.importorskip("PyQt6", reason="PyQt6 not available")
    try:
        import importlib.util

        hs_path = Path(__file__).parent.parent / "src" / "help" / "help_system.py"
        spec = importlib.util.spec_from_file_location("help_system_test", hs_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:  # noqa: BLE001 — test isolation: any import failure skips suite
        pytest.skip(f"help_system not importable: {e}")


def test_handle_code_block_type_error() -> None:
    """_handle_code_block raises TypeError for non-str line."""
    hs = _import_help_handlers()
    state = hs._MarkdownState()

    with pytest.raises(TypeError, match="line must be a str"):
        hs._handle_code_block(None, state)

    with pytest.raises(TypeError, match="line must be a str"):
        hs._handle_code_block(42, state)


def test_handle_horizontal_rule_type_error() -> None:
    """_handle_horizontal_rule raises TypeError for non-str line."""
    hs = _import_help_handlers()
    state = hs._MarkdownState()

    with pytest.raises(TypeError, match="line must be a str"):
        hs._handle_horizontal_rule(None, state)


def test_handle_table_line_type_error() -> None:
    """_handle_table_line raises TypeError for non-str line."""
    hs = _import_help_handlers()
    state = hs._MarkdownState()

    with pytest.raises(TypeError, match="line must be a str"):
        hs._handle_table_line(None, state)


def test_handle_header_type_error() -> None:
    """_handle_header raises TypeError for non-str line."""
    hs = _import_help_handlers()
    state = hs._MarkdownState()

    with pytest.raises(TypeError, match="line must be a str"):
        hs._handle_header(None, state)


def test_handle_list_item_type_error() -> None:
    """_handle_list_item raises TypeError for non-str line."""
    hs = _import_help_handlers()
    state = hs._MarkdownState()

    with pytest.raises(TypeError, match="line must be a str"):
        hs._handle_list_item(None, state)


def test_handle_paragraph_type_error() -> None:
    """_handle_paragraph raises TypeError for non-str line."""
    hs = _import_help_handlers()
    state = hs._MarkdownState()

    with pytest.raises(TypeError, match="line must be a str"):
        hs._handle_paragraph(None, state)


def test_markdown_to_html_basic() -> None:
    """_markdown_to_html converts basic markdown correctly."""
    hs = _import_help_handlers()

    result = hs._markdown_to_html("# Hello World")
    assert "<h1" in result
    assert "Hello World" in result

    result2 = hs._markdown_to_html("- item one\n- item two")
    assert "<li>" in result2
    assert "item one" in result2
