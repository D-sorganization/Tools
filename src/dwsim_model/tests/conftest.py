import importlib.util
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

dwsim_path_env = os.environ.get("DWSIM_PATH", r"C:\Users\diete\AppData\Local\DWSIM")
automation_dll = os.path.join(dwsim_path_env, "DWSIM.Automation.dll")
dwsim_available = os.path.exists(automation_dll)

clr_available = importlib.util.find_spec("clr") is not None and dwsim_available

if not clr_available:
    sys.modules["clr"] = MagicMock()

    # Mock get_automation
    def mock_get_automation(dwsim_path=None):
        mock_interf = MagicMock()
        mock_interf.AvailablePropertyPackages = {"Peng-Robinson (PR)": MagicMock()}

        mock_obj_type = MagicMock()
        mock_obj_type.MaterialStream = MagicMock()

        return mock_interf, mock_obj_type

    # Needs to patch early before test collection starts instantiating FlowsheetBuilder
    from dwsim_model import core

    core.get_automation = mock_get_automation

    # Patch FlowsheetBuilder to handle compound addition/counting in mocks
    original_init = core.FlowsheetBuilder.__init__
    original_add_pp = core.FlowsheetBuilder.add_property_package

    def patched_init(self, dwsim_path=None):
        original_init(self, dwsim_path)

        self._mock_compounds = []

        def mock_add_compound(name):
            self._mock_compounds.append(MagicMock(Name=name))

        self.sim.SelectedCompounds.Values = self._mock_compounds
        self.add_compound = mock_add_compound

        self._mock_packages = []
        mock_pp = MagicMock()
        type(mock_pp).Values = property(lambda self_mock: self._mock_packages)
        mock_pp.__iter__ = lambda x: iter(self._mock_packages)
        mock_pp.__len__ = lambda x: len(self._mock_packages)

        self.sim.PropertyPackages = mock_pp

    def patched_add_pp(self, package_name="Peng-Robinson (PR)"):
        pkg = original_add_pp(self, package_name)
        self._mock_packages.append(pkg)
        return pkg

    core.FlowsheetBuilder.__init__ = patched_init
    core.FlowsheetBuilder.add_property_package = patched_add_pp


FILE_MARKERS = {
    "test_biomass_decomposer.py": ("unit",),
    "test_metrics.py": ("unit",),
    "test_schema.py": ("unit",),
    "test_builder.py": ("contract",),
    "test_config_loader.py": ("contract",),
    "test_gasification_build.py": ("contract",),
    "test_reactions.py": ("contract",),
    "test_sweep.py": ("contract",),
    "test_topology.py": ("contract",),
    "test_acceptance_baseline.py": ("acceptance", "integration"),
    "test_gasification_module.py": ("dwsim", "integration"),
    "test_standalone.py": ("dwsim", "integration"),
}


def pytest_collection_modifyitems(config, items):
    dwsim_path = os.environ.get("DWSIM_PATH", r"C:\Users\diete\AppData\Local\DWSIM")
    automation_dll = os.path.join(dwsim_path, "DWSIM.Automation.dll")
    dwsim_available = os.path.exists(automation_dll)
    run_dwsim_tests = os.environ.get("RUN_DWSIM_TESTS") == "1"

    # Only skip tests that require the DWSIM runtime.
    #
    # Tests in the DWSIM_TEST_FILES set (which call FlowsheetBuilder.build/solve
    # against the real DWSIM COM object) are skipped when DWSIM is absent.
    #
    # Any test marked @pytest.mark.dwsim is also skipped without DWSIM.
    #
    # All other tests — pure-Python tests for schema validation,
    # biomass decomposer, metrics, parameter sweep, etc. — run freely
    # regardless of whether DWSIM is installed.
    DWSIM_TEST_FILES = {"test_standalone.py", "test_gasification_module.py"}

    for item in items:
        for marker_name in FILE_MARKERS.get(item.fspath.basename, ()):
            item.add_marker(getattr(pytest.mark, marker_name))

        in_dwsim_file = item.fspath.basename in DWSIM_TEST_FILES
        has_dwsim_marker = item.get_closest_marker("dwsim") is not None

        if not dwsim_available and (in_dwsim_file or has_dwsim_marker):
            skip_dwsim = pytest.mark.skip(
                reason=(
                    f"DWSIM not installed — DLL not found at {automation_dll}. "
                    "Only DWSIM-dependent tests are skipped."
                )
            )
            item.add_marker(skip_dwsim)
        elif (in_dwsim_file or has_dwsim_marker) and not run_dwsim_tests:
            skip_dwsim = pytest.mark.skip(
                reason=(
                    "DWSIM-backed tests are opt-in. Set RUN_DWSIM_TESTS=1 to execute "
                    "live runtime integration checks."
                )
            )
            item.add_marker(skip_dwsim)
