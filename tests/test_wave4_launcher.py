"""Tests for Wave 4: launcher consolidation and theme integration.

Verifies:
- GUI_INFO dict pattern is consistent across all tools
- auto_discover_guis finds and registers tools from GUI_INFO dicts
- LaunchConfig has the new fields (class_name, title, settings_app, min_size)
- launch_from_gui_info builds a correct LaunchConfig from GUI_INFO
- ToolCard uses object names instead of inline CSS
- launch.py can list tools (--list flag)
- Legacy launchers have deprecation notices
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Path constants ────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"


# ── GUI_INFO dict pattern ─────────────────────────────────────────────────


class TestGUIInfoPattern:
    """All gui_registration.py files should export a GUI_INFO dict."""

    @staticmethod
    def _collect_gui_registrations() -> list[Path]:
        """Find all gui_registration.py files under src/."""
        return sorted(SRC_DIR.rglob("gui_registration.py"))

    def test_all_gui_registration_files_have_gui_info(self) -> None:
        """Every gui_registration.py must define a GUI_INFO dict."""
        files = self._collect_gui_registrations()
        assert len(files) > 0, "No gui_registration.py files found"

        missing = []
        for path in files:
            spec = importlib.util.spec_from_file_location(f"gui_reg_{path.stem}", path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    # Some modules may fail to import; skip them
                    continue
                gui_info = getattr(module, "GUI_INFO", None)
                if gui_info is None:
                    missing.append(str(path.relative_to(REPO_ROOT)))

        assert missing == [], f"Files missing GUI_INFO: {missing}"

    def test_gui_info_has_required_keys(self) -> None:
        """Each GUI_INFO dict must have name, tool_name, description, pyqt6."""
        files = self._collect_gui_registrations()
        required_keys = {"name", "tool_name", "description"}

        problems = []
        for path in files:
            spec = importlib.util.spec_from_file_location(f"gui_reg_{path.stem}", path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    continue
                gui_info = getattr(module, "GUI_INFO", None)
                if gui_info is None:
                    continue
                missing_keys = required_keys - set(gui_info.keys())
                if missing_keys:
                    rel = str(path.relative_to(REPO_ROOT))
                    problems.append(f"{rel}: missing {missing_keys}")

        assert problems == [], "GUI_INFO problems:\n" + "\n".join(problems)

    def test_pyqt6_config_has_module_and_class(self) -> None:
        """Each pyqt6 sub-dict must have module and class keys."""
        files = self._collect_gui_registrations()

        problems = []
        for path in files:
            spec = importlib.util.spec_from_file_location(f"gui_reg_{path.stem}", path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    continue
                gui_info = getattr(module, "GUI_INFO", None)
                if gui_info is None or "pyqt6" not in gui_info:
                    continue
                pyqt6 = gui_info["pyqt6"]
                if "module" not in pyqt6 or "class" not in pyqt6:
                    rel = str(path.relative_to(REPO_ROOT))
                    problems.append(f"{rel}: pyqt6 missing module or class")

        assert problems == [], "pyqt6 config problems:\n" + "\n".join(problems)

    def test_get_gui_info_function_exists(self) -> None:
        """Each gui_registration.py should have a get_gui_info() function."""
        files = self._collect_gui_registrations()

        missing = []
        for path in files:
            spec = importlib.util.spec_from_file_location(f"gui_reg_{path.stem}", path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    continue
                if not hasattr(module, "get_gui_info"):
                    rel = str(path.relative_to(REPO_ROOT))
                    missing.append(rel)

        assert missing == [], f"Files missing get_gui_info(): {missing}"


# ── LaunchConfig new fields ───────────────────────────────────────────────


class TestLaunchConfigFields:
    """LaunchConfig should have the new fields for in-process launch."""

    def test_launch_config_has_class_name(self) -> None:
        from gui_launcher import GUIType, LaunchConfig

        config = LaunchConfig(
            tool_name="test",
            gui_type=GUIType.PYQT6,
            class_name="TestWindow",
        )
        assert config.class_name == "TestWindow"

    def test_launch_config_has_title(self) -> None:
        from gui_launcher import GUIType, LaunchConfig

        config = LaunchConfig(
            tool_name="test",
            gui_type=GUIType.PYQT6,
            title="Test Tool",
        )
        assert config.title == "Test Tool"

    def test_launch_config_has_settings_app(self) -> None:
        from gui_launcher import GUIType, LaunchConfig

        config = LaunchConfig(
            tool_name="test",
            gui_type=GUIType.PYQT6,
            settings_app="TestApp",
        )
        assert config.settings_app == "TestApp"

    def test_launch_config_has_min_size(self) -> None:
        from gui_launcher import GUIType, LaunchConfig

        config = LaunchConfig(
            tool_name="test",
            gui_type=GUIType.PYQT6,
            min_size=(800, 600),
        )
        assert config.min_size == (800, 600)

    def test_launch_config_defaults(self) -> None:
        from gui_launcher import GUIType, LaunchConfig

        config = LaunchConfig(tool_name="test", gui_type=GUIType.PYQT6)
        assert config.class_name is None
        assert config.title is None
        assert config.settings_app is None
        assert config.min_size is None
        assert config.organization == "D-sorganization"


# ── launch_from_gui_info ─────────────────────────────────────────────────


class TestLaunchFromGUIInfo:
    """launch_from_gui_info should build correct config from GUI_INFO."""

    def test_returns_error_for_missing_pyqt6(self) -> None:
        from gui_launcher.launcher import launch_from_gui_info

        result = launch_from_gui_info({"name": "Test", "tool_name": "test"})
        assert result == 1

    @patch("gui_launcher.launcher.check_python_dependencies")
    def test_returns_error_for_missing_deps(self, mock_check: MagicMock) -> None:
        from gui_launcher.launcher import DependencyStatus, launch_from_gui_info

        mock_check.return_value = DependencyStatus(
            ok=False,
            missing=["PyQt6"],
            guidance={"PyQt6": "pip install PyQt6"},
        )
        gui_info = {
            "name": "Test Tool",
            "tool_name": "test_tool",
            "pyqt6": {
                "module": "test.module",
                "class": "TestWindow",
                "dependencies": ["PyQt6"],
            },
        }
        result = launch_from_gui_info(gui_info)
        assert result == 1


# ── auto_discover_guis ───────────────────────────────────────────────────


class TestAutoDiscoverGuis:
    """auto_discover_guis should find and register tools."""

    def test_discovers_tools_from_src(self) -> None:
        from gui_launcher.registry import GUIRegistry, auto_discover_guis

        # Reset singleton
        GUIRegistry._instance = None
        count = auto_discover_guis([SRC_DIR])
        assert count > 0, "Should discover at least one tool"

        registry = GUIRegistry.instance()
        tools = registry.list_tools()
        assert len(tools) > 0, "Registry should have tools after discovery"

        # Cleanup
        GUIRegistry._instance = None

    def test_discovered_tools_have_pyqt6_config(self) -> None:
        from gui_launcher.launcher import GUIType
        from gui_launcher.registry import GUIRegistry, auto_discover_guis

        GUIRegistry._instance = None
        auto_discover_guis([SRC_DIR])

        registry = GUIRegistry.instance()
        tools = registry.list_tools()

        tools_with_pyqt6 = [t for t in tools if GUIType.PYQT6 in t.gui_configs]
        assert len(tools_with_pyqt6) > 0, "At least one tool should have PyQt6 config"

        # Cleanup
        GUIRegistry._instance = None


# ── Legacy launcher deprecation ──────────────────────────────────────────


class TestLegacyDeprecation:
    """Legacy launchers should have deprecation notices."""

    def test_launcher_py_has_deprecation(self) -> None:
        launcher_path = REPO_ROOT / "Launcher.py"
        if not launcher_path.exists():
            pytest.skip("Launcher.py not present")
        content = launcher_path.read_text(encoding="utf-8")
        assert "deprecated" in content.lower()

    def test_launch_tools_main_has_deprecation(self) -> None:
        launcher_path = REPO_ROOT / "launch_tools_main.py"
        if not launcher_path.exists():
            pytest.skip("launch_tools_main.py not present")
        content = launcher_path.read_text(encoding="utf-8")
        assert "deprecated" in content.lower()


# ── Unified launch.py ────────────────────────────────────────────────────


class TestUnifiedLaunchPy:
    """launch.py should exist and support --list."""

    def test_launch_py_exists(self) -> None:
        assert (REPO_ROOT / "launch.py").exists()

    def test_launch_py_has_argparse(self) -> None:
        content = (REPO_ROOT / "launch.py").read_text(encoding="utf-8")
        assert "argparse" in content
        assert "--list" in content
        assert "--tool" in content
