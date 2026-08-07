"""Registration metadata tests for rate_of_closure."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def gui_info() -> dict:
    """Load GUI registration info from the module."""
    from rate_of_closure.gui_registration import get_gui_info

    return get_gui_info()


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
        assert block["class"] == "RateOfClosureMainWindow"
        assert "PyQt6" in block["dependencies"]
        assert block["settings_app"] == "RateOfClosure"

    def test_declared_module_imports_and_exposes_class(self, gui_info: dict) -> None:
        """The registration must point at a real, importable window class."""
        pytest.importorskip("PyQt6")
        module = importlib.import_module(gui_info["pyqt6"]["module"])
        assert hasattr(module, gui_info["pyqt6"]["class"])

    def test_web_port_is_declared(self, gui_info: dict) -> None:
        assert gui_info["web"]["port"] == 5193

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
