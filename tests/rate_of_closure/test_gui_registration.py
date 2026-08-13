"""Registration metadata tests for rate_of_closure."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def gui_info() -> dict[str, Any]:
    """Load GUI registration info from the module."""
    from rate_of_closure.gui_registration import get_gui_info

    return cast(dict[str, Any], get_gui_info())


class TestGuiRegistration:
    """Tests for rate_of_closure GUI registration metadata."""

    def test_get_gui_info_returns_dict(self, gui_info: dict) -> None:
        assert isinstance(gui_info, dict)

    def test_gui_info_has_required_keys(self, gui_info: dict) -> None:
        required = {"name", "tool_name", "description", "category", "icon"}
        assert required.issubset(gui_info.keys())

    def test_tool_name_matches_package(self, gui_info: dict) -> None:
        assert gui_info["tool_name"] == "rate_of_closure"

    def test_pyqt6_block_is_complete(self, gui_info: dict) -> None:
        block = gui_info["pyqt6"]
        assert block["class"] == "RateOfClosureStandaloneMainWindow"
        assert "PyQt6" in block["dependencies"]
        assert block["settings_app"] == "RateOfClosure"

    def test_declared_module_imports_and_exposes_class(self, gui_info: dict) -> None:
        """The registration must point at a real, importable window class."""
        pytest.importorskip("PyQt6")
        module = importlib.import_module(gui_info["pyqt6"]["module"])
        assert hasattr(module, gui_info["pyqt6"]["class"])

    def test_web_port_is_declared(self, gui_info: dict) -> None:
        assert gui_info["web"]["port"] == 5193

    def test_documented_web_launcher_direct_path_smoke(self) -> None:
        """The documented direct script must import and delegate successfully."""
        repository = Path(__file__).resolve().parents[2]
        launcher = repository / "src" / "rate_of_closure" / "launch_web.py"
        probe = "\n".join(
            (
                "import runpy",
                f"launcher_path = {str(launcher)!r}",
                "import importlib.util",
                f"bootstrap_path = {str(repository / '_bootstrap.py')!r}",
                "factory = importlib.util.spec_from_file_location",
                "spec = factory('_test_bootstrap', bootstrap_path)",
                "assert spec is not None and spec.loader is not None",
                "module = importlib.util.module_from_spec(spec)",
                "spec.loader.exec_module(module)",
                "module.bootstrap(launcher_path)",
                "import shared.python.gui_launcher as gui_launcher",
                "def successful_launch(*_args, **_kwargs): return 0",
                "gui_launcher.launch_web_from_gui_info = successful_launch",
                "runpy.run_path(launcher_path, run_name='__main__')",
            )
        )

        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr

    def test_web_package_declares_only_supported_release_surfaces(self) -> None:
        """Do not advertise a desktop wrapper without its source project."""
        repository = Path(__file__).resolve().parents[2]
        web_directory = repository / "src" / "rate_of_closure" / "web"
        package_text = (web_directory / "package.json").read_text(encoding="utf-8")
        package = json.loads(package_text)

        assert not (web_directory / "src-tauri").exists()
        assert all(not name.startswith("tauri") for name in package["scripts"])
        assert "@tauri-apps/cli" not in package["devDependencies"]

    def test_optional_rust_accelerator_does_not_warn_during_import(self) -> None:
        """A normal Python-backend launch must not resemble a crash."""
        repository = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                (str(repository / "src"), environment.get("PYTHONPATH", "")),
            )
        )
        completed = subprocess.run(
            [sys.executable, "-c", "import shared.python.swing_sim"],
            cwd=repository,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

        assert "swing_core wheel not available" not in completed.stderr
