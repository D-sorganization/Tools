"""Tests for tools.json and tool_surface_contract.json generation from gui_registration.py sources."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_script_module():
    """Load the generate_tools_json script module."""
    # Find the repo root by navigating up from this test file
    # tests/scripts/test_generate_tools_json.py -> tests/scripts -> tests -> repo_root
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "generate_tools_json.py"

    spec = importlib.util.spec_from_file_location(
        "generate_tools_json_module", script_path
    )
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
        "GUI_INFO = {\n"
        '    "name": "Electrode Advisor",\n'
        '    "tool_name": "electrode_advisor",\n'
        '    "description": "AC Electrode Analysis",\n'
        '    "category": "Process Simulation",\n'
        '    "pyqt6": {\n'
        '        "module": "electrode.ui",\n'
        '        "class": "Widget"\n'
        "    },\n"
        '    "web": {"port": 3000}\n'
        "}\n"
        "def get_gui_info(): return GUI_INFO\n",
        encoding="utf-8",
    )
    # create dummy launch scripts relative to what generate_tools_json expects
    # The generator usually looks relative to repo root, e.g. src/electrode_advisor/launch_pyqt6.py
    (t1 / "launch_pyqt6.py").touch()
    (t1 / "launch_web.py").touch()

    # Tool 2: pressure_drop (PyQt only)
    t2 = src / "pressure_drop"
    t2.mkdir()
    (t2 / "gui_registration.py").write_text(
        "GUI_INFO = {\n"
        '    "name": "Pressure Drop",\n'
        '    "tool_name": "pressure_drop",\n'
        '    "description": "Pipe Analysis",\n'
        '    "category": "Engineering Tools",\n'
        '    "pyqt6": {\n'
        '        "module": "pressure.ui",\n'
        '        "class": "Widget"\n'
        "    }\n"
        "}\n"
        "def get_gui_info(): return GUI_INFO\n",
        encoding="utf-8",
    )
    (t2 / "launch_pyqt6.py").touch()

    return tmp_path


@pytest.fixture
def mock_repo_with_web_only(tmp_path):
    """Create a mock repo with a tool that only has web surface (no pyqt6)."""
    src = tmp_path / "src"
    src.mkdir()

    t1 = src / "web_only_tool"
    t1.mkdir()
    (t1 / "gui_registration.py").write_text(
        "GUI_INFO = {\n"
        '    "name": "Web Only Tool",\n'
        '    "tool_name": "web_only_tool",\n'
        '    "description": "A web-only tool",\n'
        '    "category": "Testing",\n'
        '    "web": {"port": 5000}\n'
        "}\n"
        "def get_gui_info(): return GUI_INFO\n",
        encoding="utf-8",
    )
    (t1 / "launch_web.py").touch()

    return tmp_path


@pytest.fixture
def mock_repo_no_tool_name(tmp_path):
    """Create a mock repo with a tool that has no tool_name (tests fallback)."""
    src = tmp_path / "src"
    src.mkdir()

    t1 = src / "legacy_tool"
    t1.mkdir()
    (t1 / "gui_registration.py").write_text(
        "GUI_INFO = {\n"
        '    "name": "Legacy Tool",\n'
        '    "description": "Tool without tool_name field",\n'
        '    "category": "Legacy",\n'
        '    "pyqt6": {\n'
        '        "module": "legacy.ui",\n'
        '        "class": "Widget"\n'
        "    }\n"
        "}\n"
        "def get_gui_info(): return GUI_INFO\n",
        encoding="utf-8",
    )
    (t1 / "launch_pyqt6.py").touch()

    return tmp_path


# ============================================================================
# Phase 1 Tests: Manifest Generation (existing)
# ============================================================================


class TestManifestGeneration:
    """Tests for tools.json manifest generation."""

    def test_generate_manifest_structure(self, manifest_gen_module, mock_repo_root):
        """Test standard manifest generation."""
        manifest = manifest_gen_module.generate_manifest_data(mock_repo_root)

        # Assert top-level categories exist
        assert "Process Simulation" in manifest
        assert "Engineering Tools" in manifest

        # Check Engineering Tools (Pressure Drop - PyQt only)
        eng_tools = manifest["Engineering Tools"]
        assert len(eng_tools) == 1
        tool = eng_tools[0]
        assert (
            tool["name"] == "Pressure Drop"
        )  # Single surface name, usually just the name
        assert tool["type"] == "python"
        assert "src/pressure_drop/launch_pyqt6.py" in tool["path"].replace("\\", "/")

    def test_dual_surface_expansion(self, manifest_gen_module, mock_repo_root):
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

    def test_idempotency(self, manifest_gen_module, mock_repo_root):
        """Test that generation is deterministic (sorted keys/lists)."""
        run1 = manifest_gen_module.generate_manifest_data(mock_repo_root)
        run2 = manifest_gen_module.generate_manifest_data(mock_repo_root)

        assert json.dumps(run1, sort_keys=True) == json.dumps(run2, sort_keys=True)

    def test_catalog_hidden_registration_is_not_exported(
        self, manifest_gen_module, tmp_path
    ):
        """Compatibility launchers can keep metadata without entering catalogs."""
        src = tmp_path / "src"
        src.mkdir()
        hidden = src / "legacy_movement_optimizer"
        hidden.mkdir()
        (hidden / "gui_registration.py").write_text(
            "GUI_INFO = {\n"
            '    "name": "Movement Optimizer",\n'
            '    "tool_name": "optimizer_gui",\n'
            '    "catalog_visible": False,\n'
            '    "description": "Legacy compatibility launcher",\n'
            '    "category": "Optimization",\n'
            '    "pyqt6": {"module": "legacy.ui", "class": "Window"}\n'
            "}\n",
            encoding="utf-8",
        )
        (hidden / "launch_pyqt6.py").touch()

        assert manifest_gen_module.generate_manifest_data(tmp_path) == {}
        assert manifest_gen_module.generate_contract_data(tmp_path)["tools"] == []


# ============================================================================
# Phase 1, Task 1.4 Tests: Contract Generation
# ============================================================================


class TestContractGeneration:
    """Tests for tool_surface_contract.json generation."""

    def test_contract_has_version(self, manifest_gen_module, mock_repo_root):
        """Contract must contain a semver version string."""
        contract = manifest_gen_module.generate_contract_data(mock_repo_root)
        assert "version" in contract
        assert isinstance(contract["version"], str)
        # Semver format
        parts = contract["version"].split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()

    def test_contract_has_tools_list(self, manifest_gen_module, mock_repo_root):
        """Contract must contain a 'tools' list."""
        contract = manifest_gen_module.generate_contract_data(mock_repo_root)
        assert "tools" in contract
        assert isinstance(contract["tools"], list)
        assert len(contract["tools"]) > 0

    def test_contract_tool_entry_structure(self, manifest_gen_module, mock_repo_root):
        """Each contract tool entry must have id, name, description, category, and surfaces."""
        contract = manifest_gen_module.generate_contract_data(mock_repo_root)

        for tool in contract["tools"]:
            assert "id" in tool, f"Missing 'id' in tool entry: {tool}"
            assert "name" in tool, f"Missing 'name' in tool entry: {tool}"
            assert "description" in tool, f"Missing 'description' in tool entry: {tool}"
            assert "category" in tool, f"Missing 'category' in tool entry: {tool}"
            assert "surfaces" in tool, f"Missing 'surfaces' in tool entry: {tool}"

    def test_contract_tool_id_format(self, manifest_gen_module, mock_repo_root):
        """Tool IDs must be snake_case."""
        import re

        contract = manifest_gen_module.generate_contract_data(mock_repo_root)
        pattern = re.compile(r"^[a-z0-9_]+$")

        for tool in contract["tools"]:
            assert pattern.match(
                tool["id"]
            ), f"Tool ID '{tool['id']}' is not snake_case"

    def test_contract_surfaces_structure(self, manifest_gen_module, mock_repo_root):
        """Each tool's surfaces dict must have exactly pyqt6 and web booleans."""
        contract = manifest_gen_module.generate_contract_data(mock_repo_root)

        for tool in contract["tools"]:
            surfaces = tool["surfaces"]
            assert "pyqt6" in surfaces, f"Missing 'pyqt6' in surfaces for {tool['id']}"
            assert "web" in surfaces, f"Missing 'web' in surfaces for {tool['id']}"
            assert isinstance(surfaces["pyqt6"], bool)
            assert isinstance(surfaces["web"], bool)

    def test_contract_dual_surface_tool(self, manifest_gen_module, mock_repo_root):
        """Electrode advisor with both surfaces should show pyqt6=True, web=True."""
        contract = manifest_gen_module.generate_contract_data(mock_repo_root)
        electrode = next(t for t in contract["tools"] if t["id"] == "electrode_advisor")

        assert electrode["surfaces"]["pyqt6"] is True
        assert electrode["surfaces"]["web"] is True
        assert electrode["name"] == "Electrode Advisor"
        assert electrode["description"] == "AC Electrode Analysis"
        assert electrode["category"] == "Process Simulation"

    def test_contract_pyqt_only_tool(self, manifest_gen_module, mock_repo_root):
        """Pressure drop (pyqt only) should show pyqt6=True, web=False."""
        contract = manifest_gen_module.generate_contract_data(mock_repo_root)
        pd = next(t for t in contract["tools"] if t["id"] == "pressure_drop")

        assert pd["surfaces"]["pyqt6"] is True
        assert pd["surfaces"]["web"] is False

    def test_contract_web_only_tool(self, manifest_gen_module, mock_repo_with_web_only):
        """Web-only tool should show pyqt6=False, web=True."""
        contract = manifest_gen_module.generate_contract_data(mock_repo_with_web_only)
        web_tool = next(t for t in contract["tools"] if t["id"] == "web_only_tool")

        assert web_tool["surfaces"]["pyqt6"] is False
        assert web_tool["surfaces"]["web"] is True

    def test_contract_tool_name_fallback(
        self, manifest_gen_module, mock_repo_no_tool_name
    ):
        """Tools without tool_name should use directory name as ID."""
        contract = manifest_gen_module.generate_contract_data(mock_repo_no_tool_name)

        assert len(contract["tools"]) == 1
        tool = contract["tools"][0]
        # Should fall back to using directory name
        assert tool["id"] == "legacy_tool"

    def test_contract_is_sorted_by_id(self, manifest_gen_module, mock_repo_root):
        """Contract tools must be sorted by ID for deterministic output."""
        contract = manifest_gen_module.generate_contract_data(mock_repo_root)
        ids = [t["id"] for t in contract["tools"]]
        assert ids == sorted(ids)

    def test_contract_idempotency(self, manifest_gen_module, mock_repo_root):
        """Contract generation must be deterministic."""
        run1 = manifest_gen_module.generate_contract_data(mock_repo_root)
        run2 = manifest_gen_module.generate_contract_data(mock_repo_root)

        assert json.dumps(run1, sort_keys=True) == json.dumps(run2, sort_keys=True)

    def test_contract_no_duplicate_ids(self, manifest_gen_module, mock_repo_root):
        """Contract must not contain duplicate tool IDs."""
        contract = manifest_gen_module.generate_contract_data(mock_repo_root)
        ids = [t["id"] for t in contract["tools"]]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"

    def test_contract_schema_compliance(self, manifest_gen_module, mock_repo_root):
        """Contract must match the expected schema structure."""
        contract = manifest_gen_module.generate_contract_data(mock_repo_root)

        # Top-level keys
        assert set(contract.keys()) == {"version", "tools"}

        # Each tool entry
        expected_tool_keys = {"id", "name", "description", "category", "surfaces"}
        expected_surface_keys = {"pyqt6", "web", "legacy_gui"}

        for tool in contract["tools"]:
            assert (
                set(tool.keys()) == expected_tool_keys
            ), f"Unexpected keys in tool entry: {set(tool.keys()) - expected_tool_keys}"
            assert set(tool["surfaces"].keys()) == expected_surface_keys


def test_main_emits_cli_summary_and_writes_outputs(
    manifest_gen_module,
    mock_repo_root,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """The CLI entrypoint should emit a stable stdout summary and write both files."""
    script_path = mock_repo_root / "scripts" / "generate_tools_json.py"
    script_path.parent.mkdir()
    script_path.write_text("# stub entrypoint\n", encoding="utf-8")
    monkeypatch.setattr(manifest_gen_module, "__file__", str(script_path))

    result = manifest_gen_module.main()

    output = capsys.readouterr().out
    assert result == 0
    assert "Generated tools.json with 3 tools." in output
    assert "Generated tool_surface_contract.json with 2 tools." in output
    assert (mock_repo_root / "tools.json").exists()
    assert (mock_repo_root / "tool_surface_contract.json").exists()
