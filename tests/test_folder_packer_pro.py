"""Smoke tests for folder_packer_pro non-GUI modules.

These tests cover the pack engine utilities, file operations, and
constants that can be tested without instantiating tkinter/GUI components.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_constants_compression_levels_present() -> None:
    """COMPRESSION_LEVELS must include the four named presets."""
    # Import the module directly to avoid pulling in the GUI __init__
    import importlib

    spec = importlib.util.spec_from_file_location(
        "folder_packer_pro.constants",
        Path(__file__).resolve().parents[1]
        / "src"
        / "folder_packer_pro"
        / "constants.py",
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    levels = mod.COMPRESSION_LEVELS
    assert "none" in levels
    assert "fast" in levels
    assert "balanced" in levels
    assert "best" in levels
    assert levels["none"] == 0
    assert levels["best"] == 9


# ---------------------------------------------------------------------------
# file_ops — should_exclude
# ---------------------------------------------------------------------------


@pytest.fixture()
def _import_file_ops():  # type: ignore[return]
    """Import file_ops module directly (no GUI dependencies)."""
    import importlib

    spec = importlib.util.spec_from_file_location(
        "folder_packer_pro.constants",
        Path(__file__).resolve().parents[1]
        / "src"
        / "folder_packer_pro"
        / "constants.py",
    )
    assert spec is not None
    constants_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(constants_mod)  # type: ignore[union-attr]
    sys.modules["folder_packer_pro.constants"] = constants_mod

    spec2 = importlib.util.spec_from_file_location(
        "folder_packer_pro.file_ops",
        Path(__file__).resolve().parents[1]
        / "src"
        / "folder_packer_pro"
        / "file_ops.py",
    )
    assert spec2 is not None
    file_ops_mod = importlib.util.module_from_spec(spec2)
    assert spec2.loader is not None
    spec2.loader.exec_module(file_ops_mod)  # type: ignore[union-attr]
    return file_ops_mod


def test_should_exclude_git_directory(_import_file_ops) -> None:  # type: ignore[no-untyped-def]
    """should_exclude returns True for .git paths when include_git=False."""
    file_ops = _import_file_ops
    git_path = Path(".git") / "config"
    assert file_ops.should_exclude(git_path, set(), include_git=False) is True


def test_should_exclude_git_included(_import_file_ops) -> None:  # type: ignore[no-untyped-def]
    """should_exclude returns False for .git path when include_git=True."""
    file_ops = _import_file_ops
    git_path = Path(".git") / "config"
    assert file_ops.should_exclude(git_path, set(), include_git=True) is False


def test_should_exclude_pattern_suffix(_import_file_ops) -> None:  # type: ignore[no-untyped-def]
    """should_exclude matches wildcard suffix exclusion patterns."""
    file_ops = _import_file_ops
    pyc_path = Path("src") / "module.pyc"
    assert file_ops.should_exclude(pyc_path, {"*.pyc"}) is True


def test_should_exclude_no_match(_import_file_ops) -> None:  # type: ignore[no-untyped-def]
    """should_exclude returns False when no patterns match."""
    file_ops = _import_file_ops
    py_path = Path("src") / "module.py"
    assert file_ops.should_exclude(py_path, {"*.pyc", "__pycache__"}) is False


def test_format_size_bytes(_import_file_ops) -> None:  # type: ignore[no-untyped-def]
    """format_size formats bytes, KB, and MB correctly."""
    file_ops = _import_file_ops
    assert "B" in file_ops.format_size(512)
    assert "KB" in file_ops.format_size(2048)
    assert "MB" in file_ops.format_size(2 * 1024 * 1024)
