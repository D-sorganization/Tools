"""Tests for tools.json generation from gui_registration.py sources."""

import json
import importlib.util
import sys
from pathlib import Path

import pytest

def load_script_module():
    """Load the generate_tools_json script module."""
    # Find the repo root by navigating up from this test file
    # tests/scripts/test_generate_tools_json.py -> tests/scripts -> tests -> repo_root
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "generate_tools_json.py"
    
    spec = importlib.util.spec_from_file_location("generate_tools_json_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load script {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_tools_json_module"] = module
    spec.loader.exec_module(module)
    return module

@pytest.fixture
def manifest_gen_module():
    return load_script_module()

@pytest.fixture
def mock_repo_root(tmp_path):
    """Create a mock repository structure with gui_registration.py files."""
    src = tmp_path / "src"
    src.mkdir()

    # Tool 1: electrode_advisor (Dual Surface)
    t1 = src / "electrode_advisor"
    t1.mkdir()
    (t1 / "gui_registration.py").write_text(
        'GUI_INFO = {\n'
        '    "name": "Electrode Advisor",\n'
        '    "tool_name": "electrode_advisor",\n'
        '    "description": "AC Electrode Analysis",\n'
        '    "category": "Process Simulation",\n'
        '    "pyqt6": {\n'
        '        "module": "electrode.ui",\n'
        '        "class": "Widget"\n'
        '    },\n'
        '    "web": {"port": 3000}\n'
        '}\n'
        'def get_gui_info(): return GUI_INFO\n',
        encoding="utf-8"
    )
    # create dummy launch scripts relative to what generate_tools_json expects
    # The generator usually looks relative to repo root, e.g. src/electrode_advisor/launch_pyqt6.py
    (t1 / "launch_pyqt6.py").touch()
    (t1 / "launch_web.py").touch()

    # Tool 2: pressure_drop (PyQt only)
    t2 = src / "pressure_drop"
    t2.mkdir()
    (t2 / "gui_registration.py").write_text(
        'GUI_INFO = {\n'
        '    "name": "Pressure Drop",\n'
        '    "tool_name": "pressure_drop",\n'
        '    "description": "Pipe Analysis",\n'
        '    "category": "Engineering Tools",\n'
        '    "pyqt6": {\n'
        '        "module": "pressure.ui",\n'
        '        "class": "Widget"\n'
        '    }\n'
        '}\n'
        'def get_gui_info(): return GUI_INFO\n',
        encoding="utf-8"
    )
    (t2 / "launch_pyqt6.py").touch()
    
    return tmp_path

def test_generate_manifest_structure(manifest_gen_module, mock_repo_root):
    """Test standard manifest generation."""
    manifest = manifest_gen_module.generate_manifest_data(mock_repo_root)
    
    # Assert top-level categories exist
    assert "Process Simulation" in manifest
    assert "Engineering Tools" in manifest
    
    # Check Engineering Tools (Pressure Drop - PyQt only)
    eng_tools = manifest["Engineering Tools"]
    assert len(eng_tools) == 1
    tool = eng_tools[0]
    assert tool["name"] == "Pressure Drop" # Single surface name, usually just the name
    assert tool["type"] == "python"
    assert "src/pressure_drop/launch_pyqt6.py" in tool["path"].replace("\\", "/")

def test_dual_surface_expansion(manifest_gen_module, mock_repo_root):
    """Test that dual-surface tools expand into two entries (PyQt and Web)."""
    manifest = manifest_gen_module.generate_manifest_data(mock_repo_root)
    
    sim_tools = manifest["Process Simulation"]
    # Electrode Advisor has both pyqt6 and web keys -> expecting 2 entries
    assert len(sim_tools) == 2
    
    names = {t["name"] for t in sim_tools}
    assert "Electrode Advisor (PyQt6)" in names
    assert "Electrode Advisor (Web)" in names
    
    web_tool = next(t for t in sim_tools if t["name"] == "Electrode Advisor (Web)")
    assert "launch_web.py" in web_tool["path"]

def test_idempotency(manifest_gen_module, mock_repo_root):
    """Test that generation is deterministic (sorted keys/lists)."""
    run1 = manifest_gen_module.generate_manifest_data(mock_repo_root)
    run2 = manifest_gen_module.generate_manifest_data(mock_repo_root)
    
    assert json.dumps(run1, sort_keys=True) == json.dumps(run2, sort_keys=True)
