"""Tests for launcher-backed tool manifest layout checks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_script_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "check_tools_manifest_layout.py"

    spec = importlib.util.spec_from_file_location(
        "check_tools_manifest_layout_module", script_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load script {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_tools_manifest_layout_module"] = module
    spec.loader.exec_module(module)
    return module


def test_launcher_without_gui_registration_is_reported(tmp_path):
    module = load_script_module()
    tool_dir = tmp_path / "src" / "orphan_tool"
    tool_dir.mkdir(parents=True)
    (tool_dir / "launch_pyqt6.py").write_text("", encoding="utf-8")

    issues = module.check_manifest_layout(tmp_path)

    assert len(issues) == 1
    assert issues[0].path == "src/orphan_tool"
    assert "gui_registration.py" in issues[0].message


def test_registered_launcher_directory_passes(tmp_path):
    module = load_script_module()
    tool_dir = tmp_path / "src" / "registered_tool"
    tool_dir.mkdir(parents=True)
    (tool_dir / "launch_web.py").write_text("", encoding="utf-8")
    (tool_dir / "gui_registration.py").write_text("GUI_INFO = {}\n", encoding="utf-8")

    assert module.check_manifest_layout(tmp_path) == []


def test_repository_has_no_launcher_backed_orphans():
    module = load_script_module()
    repo_root = Path(__file__).resolve().parents[2]

    assert module.check_manifest_layout(repo_root) == []
