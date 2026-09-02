"""Real-file coverage for plugin manager loading and discovery."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

PYTHON_SRC = Path(__file__).resolve().parents[2] / "src" / "python"


def _import_plugin_manager() -> Any:
    """Import ``PluginManager`` without leaking ``sys.path``/``sys.modules``.

    This sub-app's own package is unfortunately also named ``src``
    (``src/python/src/``), which collides with the repo-root ``src``
    implicit namespace package that dozens of other test modules import
    (there is no repo-root ``src/__init__.py``, so ``import src.shared...``
    binds ``sys.modules["src"]`` to a namespace package rooted at the repo
    root). Whichever one is imported *first* in a given pytest-xdist worker
    permanently wins that binding for the rest of the process: if the
    repo-root binding wins, ``from src.core.plugin_manager import ...``
    resolves against the repo root's own (unrelated) ``src/core/`` package
    instead, which has no ``plugin_manager`` module and raises
    ``ModuleNotFoundError`` -- a real, order-dependent CI failure this
    reproduces.

    Saving and restoring both ``sys.path`` and every ``src``/``src.*``
    entry in ``sys.modules`` around the import makes this module's result
    independent of what ran earlier in its worker, without disturbing
    whichever ``src`` package (this one or the repo-root one) any other
    test in the same worker is relying on.
    """
    saved_path = list(sys.path)
    saved_src_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "src" or name.startswith("src.")
    }
    for name in saved_src_modules:
        del sys.modules[name]

    sys.path.insert(0, str(PYTHON_SRC))
    try:
        from src.core.plugin_manager import PluginManager

        return PluginManager
    finally:
        sys.path[:] = saved_path
        for name in list(sys.modules):
            if name == "src" or name.startswith("src."):
                del sys.modules[name]
        sys.modules.update(saved_src_modules)


PluginManager = _import_plugin_manager()


def test_import_plugin_manager_is_resilient_to_a_stale_src_namespace_binding() -> None:
    """A different cached ``src`` module must not break this import.

    Regression test for the collision documented on ``_import_plugin_manager``:
    simulate the repo-root ``src`` namespace package having already claimed
    ``sys.modules["src"]`` (as would happen if a test importing it collected
    first in the same xdist worker) and confirm the helper still resolves
    the real ``PluginManager`` -- and restores the pre-existing binding
    afterwards rather than leaking its own.
    """
    fake_repo_root_src = types.ModuleType("src")
    fake_repo_root_src.__path__ = []  # namespace-package-like: no __file__
    sys.modules["src"] = fake_repo_root_src
    try:
        imported = _import_plugin_manager()
        assert imported.__name__ == "PluginManager"
        assert sys.modules["src"] is fake_repo_root_src
    finally:
        del sys.modules["src"]


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
