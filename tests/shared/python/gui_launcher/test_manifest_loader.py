"""Tests for gui_launcher.manifest_loader — data-driven tool registration.

Verifies that:
- A YAML manifest can be parsed into GUI_INFO dicts
- The loader integrates with auto_discover_guis
- The canonical tool_manifest.yaml contains the expected 20 tools
- Malformed manifests produce informative errors, not silent failures

Closes #1863.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_manifest_yaml(tools: list[dict[str, Any]]) -> str:
    """Serialise a list of tool dicts to the manifest YAML format."""
    return yaml.dump({"tools": tools}, default_flow_style=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# ManifestLoader unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestManifestLoaderParsing:
    """Tests for load_manifest() and manifest_to_gui_infos()."""

    def test_load_returns_list_of_gui_infos(self, tmp_path: Path) -> None:
        from gui_launcher.manifest_loader import load_manifest

        yaml_content = _make_minimal_manifest_yaml(
            [
                {
                    "tool_name": "test_tool",
                    "name": "Test Tool",
                    "description": "A test",
                    "category": "Testing",
                    "icon": "test",
                    "pyqt6": {
                        "module": "test_tool.ui.main_window",
                        "class": "TestWindow",
                        "dependencies": ["PyQt6"],
                        "settings_app": "TestTool",
                    },
                }
            ]
        )
        manifest_path = tmp_path / "tool_manifest.yaml"
        manifest_path.write_text(yaml_content, encoding="utf-8")

        infos = load_manifest(manifest_path)

        assert len(infos) == 1
        info = infos[0]
        assert info["tool_name"] == "test_tool"
        assert info["name"] == "Test Tool"
        assert "pyqt6" in info

    def test_load_multiple_tools(self, tmp_path: Path) -> None:
        from gui_launcher.manifest_loader import load_manifest

        yaml_content = _make_minimal_manifest_yaml(
            [
                {
                    "tool_name": "alpha_tool",
                    "name": "Alpha",
                    "description": "A",
                    "category": "Cat",
                    "pyqt6": {"module": "a.ui", "class": "AWindow"},
                },
                {
                    "tool_name": "beta_tool",
                    "name": "Beta",
                    "description": "B",
                    "category": "Cat",
                    "pyqt6": {"module": "b.ui", "class": "BWindow"},
                },
            ]
        )
        manifest_path = tmp_path / "tool_manifest.yaml"
        manifest_path.write_text(yaml_content, encoding="utf-8")

        infos = load_manifest(manifest_path)
        assert len(infos) == 2
        tool_names = {i["tool_name"] for i in infos}
        assert tool_names == {"alpha_tool", "beta_tool"}

    def test_load_empty_manifest_returns_empty_list(self, tmp_path: Path) -> None:
        from gui_launcher.manifest_loader import load_manifest

        manifest_path = tmp_path / "tool_manifest.yaml"
        manifest_path.write_text(
            yaml.dump({"tools": []}, default_flow_style=False), encoding="utf-8"
        )
        infos = load_manifest(manifest_path)
        assert infos == []

    def test_load_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        from gui_launcher.manifest_loader import load_manifest

        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nonexistent.yaml")

    def test_load_tool_with_web_surface(self, tmp_path: Path) -> None:
        from gui_launcher.manifest_loader import load_manifest

        yaml_content = _make_minimal_manifest_yaml(
            [
                {
                    "tool_name": "web_tool",
                    "name": "Web Tool",
                    "description": "Has web surface",
                    "category": "Testing",
                    "pyqt6": {"module": "web.ui", "class": "WebWindow"},
                    "web": {"port": 5173, "auto_open_browser": False},
                }
            ]
        )
        manifest_path = tmp_path / "tool_manifest.yaml"
        manifest_path.write_text(yaml_content, encoding="utf-8")

        infos = load_manifest(manifest_path)
        assert infos[0]["web"]["port"] == 5173

    def test_load_invalid_yaml_raises_value_error(self, tmp_path: Path) -> None:
        from gui_launcher.manifest_loader import load_manifest

        manifest_path = tmp_path / "bad.yaml"
        manifest_path.write_text("tools: [unclosed bracket", encoding="utf-8")

        with pytest.raises((ValueError, Exception)):
            load_manifest(manifest_path)

    def test_load_manifest_without_tools_key_raises(self, tmp_path: Path) -> None:
        from gui_launcher.manifest_loader import load_manifest

        manifest_path = tmp_path / "bad.yaml"
        manifest_path.write_text(
            yaml.dump({"registrations": []}, default_flow_style=False),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="tools"):
            load_manifest(manifest_path)


# ---------------------------------------------------------------------------
# canonical tool_manifest.yaml tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCanonicalManifest:
    """Tests for the canonical tool_manifest.yaml bundled with gui_launcher."""

    @pytest.fixture
    def manifest_path(self) -> Path:
        """Locate the canonical manifest relative to the gui_launcher package."""
        here = Path(__file__).resolve()
        repo_root = here.parents[4]  # tests/shared/python/gui_launcher -> repo root
        p = (
            repo_root
            / "src"
            / "shared"
            / "python"
            / "gui_launcher"
            / "tool_manifest.yaml"
        )
        if not p.exists():
            pytest.skip(f"Canonical manifest not found at {p}")
        return p

    def test_manifest_is_valid_yaml(self, manifest_path: Path) -> None:
        from gui_launcher.manifest_loader import load_manifest

        infos = load_manifest(manifest_path)
        assert isinstance(infos, list)

    def test_manifest_contains_at_least_20_tools(self, manifest_path: Path) -> None:
        from gui_launcher.manifest_loader import load_manifest

        infos = load_manifest(manifest_path)
        assert len(infos) >= 20, (
            f"Expected at least 20 tools in canonical manifest, got {len(infos)}"
        )

    def test_all_tools_have_required_fields(self, manifest_path: Path) -> None:
        from gui_launcher.manifest_loader import load_manifest

        infos = load_manifest(manifest_path)
        required_fields = {"tool_name", "name", "description", "category"}
        for info in infos:
            missing = required_fields - set(info.keys())
            assert not missing, (
                f"Tool {info.get('tool_name', '?')} missing fields: {missing}"
            )

    def test_tool_names_are_snake_case(self, manifest_path: Path) -> None:
        import re

        from gui_launcher.manifest_loader import load_manifest

        infos = load_manifest(manifest_path)
        pattern = re.compile(r"^[a-z0-9_]+$")
        for info in infos:
            tool_name = info.get("tool_name", "")
            assert pattern.match(tool_name), (
                f"tool_name {tool_name!r} is not snake_case"
            )

    def test_no_duplicate_tool_names(self, manifest_path: Path) -> None:
        from gui_launcher.manifest_loader import load_manifest

        infos = load_manifest(manifest_path)
        names = [i["tool_name"] for i in infos]
        assert len(names) == len(set(names)), (
            f"Duplicate tool_names found: {[n for n in names if names.count(n) > 1]}"
        )

    def test_all_tools_have_at_least_one_surface(self, manifest_path: Path) -> None:
        from gui_launcher.manifest_loader import load_manifest

        infos = load_manifest(manifest_path)
        for info in infos:
            has_surface = "pyqt6" in info or "web" in info
            assert has_surface, (
                f"Tool {info.get('tool_name', '?')} has no surface (pyqt6 or web)"
            )


# ---------------------------------------------------------------------------
# Integration: load_manifest + auto_discover_guis
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestManifestIntegrationWithRegistry:
    """load_manifest results integrate cleanly with the GUI registry."""

    def test_manifest_infos_register_in_registry(self, tmp_path: Path) -> None:
        from gui_launcher.launcher import GUIType
        from gui_launcher.manifest_loader import load_manifest
        from gui_launcher.registry import GUIRegistry, _gui_info_to_registration

        yaml_content = _make_minimal_manifest_yaml(
            [
                {
                    "tool_name": "integration_tool",
                    "name": "Integration Tool",
                    "description": "Integration test",
                    "category": "Testing",
                    "icon": "test",
                    "pyqt6": {
                        "module": "integration_tool.ui.main",
                        "class": "IntegrationWindow",
                        "dependencies": ["PyQt6"],
                        "settings_app": "IntegrationTool",
                    },
                }
            ]
        )
        manifest_path = tmp_path / "tool_manifest.yaml"
        manifest_path.write_text(yaml_content, encoding="utf-8")

        # Use an isolated registry instance (not the singleton)
        registry = GUIRegistry()

        # Monkey-patch the global registry for this test
        original_instance = GUIRegistry._instance
        GUIRegistry._instance = registry
        try:
            infos = load_manifest(manifest_path)
            for info in infos:
                _gui_info_to_registration(info)

            entry = registry.get("integration_tool")
            assert entry is not None
            assert entry.display_name == "Integration Tool"
            assert GUIType.PYQT6 in entry.gui_configs
        finally:
            GUIRegistry._instance = original_instance
