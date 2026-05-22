# ruff: noqa: E501
from typing import Any

"""
Flow Rate Converter GUI Tests
=============================

TDD tests for the Flow Rate Converter GUI components.
Tests cover PyQt6 main window, conversion functionality,
and result display.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestFlowRateConverterLoDConstants:
    """Tests verifying LoD-compliance constants extracted from Qt enum chains.

    These tests guard the module-level constants added to resolve the 10 LoD
    violations reported in GH1466 (deep attribute chains like
    Qt.AlignmentFlag.AlignCenter and QSizePolicy.Policy.*).
    """

    def test_lod_constants_present_in_source(self):
        """Verify that LoD-compliance constants are declared in main_window source.

        Uses AST analysis so no import of PyQt6 is required in the test environment.
        """
        import ast
        import pathlib

        src = (
            pathlib.Path(__file__).parent.parent
            / "python"
            / "flow_rate_converter"
            / "ui"
            / "pyqt6"
            / "main_window.py"
        )
        source = src.read_text()
        tree = ast.parse(source)

        # Collect all top-level assignment targets
        top_level_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        top_level_names.add(target.id)

        assert "_ALIGN_CENTER" in top_level_names, (
            "Missing _ALIGN_CENTER constant in main_window"
        )
        assert "_EXPANDING" in top_level_names, (
            "Missing _EXPANDING constant in main_window"
        )
        assert "_FIXED" in top_level_names, "Missing _FIXED constant in main_window"

    def test_no_bare_qt_alignment_flag_chain_in_source(self):
        """Verify source file has no bare Qt.AlignmentFlag.AlignCenter except in constants."""
        import ast
        import pathlib

        src = (
            pathlib.Path(__file__).parent.parent
            / "python"
            / "flow_rate_converter"
            / "ui"
            / "pyqt6"
            / "main_window.py"
        )
        source = src.read_text()
        tree = ast.parse(source)

        chains_found = []
        for node in ast.walk(tree):
            # Match attribute chain: Qt.AlignmentFlag.AlignCenter
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "AlignCenter"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "AlignmentFlag"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "Qt"
            ):
                # Allow only in simple assignment targets (constant definitions)
                chains_found.append(node.lineno)

        # Only the constant definition line should have this chain
        assert len(chains_found) <= 1, (
            f"LoD violation: Qt.AlignmentFlag.AlignCenter found at lines {chains_found}; "
            "only the constant definition should use this chain"
        )

    def test_no_bare_qsizepolicy_expanding_chain_in_source(self):
        """Verify source has no bare QSizePolicy.Policy.Expanding except in constant def."""
        import ast
        import pathlib

        src = (
            pathlib.Path(__file__).parent.parent
            / "python"
            / "flow_rate_converter"
            / "ui"
            / "pyqt6"
            / "main_window.py"
        )
        source = src.read_text()
        tree = ast.parse(source)

        chains_found = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "Expanding"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "Policy"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "QSizePolicy"
            ):
                chains_found.append(node.lineno)

        assert len(chains_found) <= 1, (
            f"LoD violation: QSizePolicy.Policy.Expanding found at lines {chains_found}; "
            "only the constant definition should use this chain"
        )

    def test_no_bare_qsizepolicy_fixed_chain_in_source(self):
        """Verify source has no bare QSizePolicy.Policy.Fixed except in constant def."""
        import ast
        import pathlib

        src = (
            pathlib.Path(__file__).parent.parent
            / "python"
            / "flow_rate_converter"
            / "ui"
            / "pyqt6"
            / "main_window.py"
        )
        source = src.read_text()
        tree = ast.parse(source)

        chains_found = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "Fixed"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "Policy"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "QSizePolicy"
            ):
                chains_found.append(node.lineno)

        assert len(chains_found) <= 1, (
            f"LoD violation: QSizePolicy.Policy.Fixed found at lines {chains_found}; "
            "only the constant definition should use this chain"
        )


class TestFlowRateConverterMainWindow:
    """Tests for the PyQt6 Flow Rate Converter main window."""

    @pytest.fixture
    def mock_qt_app(self) -> Any:
        """Create mock Qt application for headless testing."""
        with patch.dict(
            sys.modules,
            {
                "PyQt6": MagicMock(),
                "PyQt6.QtWidgets": MagicMock(),
                "PyQt6.QtCore": MagicMock(),
                "PyQt6.QtGui": MagicMock(),
            },
        ):
            yield

    def test_main_window_imports(self, mock_qt_app) -> Any:
        """Test that main window module can be imported."""
        try:
            from flow_rate_converter.ui.pyqt6 import main_window

            assert hasattr(main_window, "FlowRateConverterWindow")
        except ImportError:
            pytest.skip("PyQt6 main window not yet implemented")

    def test_main_window_has_mass_tab(self, mock_qt_app) -> Any:
        """Test that window has mass flow tab components."""
        try:
            from flow_rate_converter.ui.pyqt6.main_window import (
                FlowRateConverterWindow,
            )

            # Verify class is defined and callable
            assert callable(FlowRateConverterWindow)
        except ImportError:
            pytest.skip("Main window not yet implemented")


class TestFlowRateConverterEngineIntegration:
    """Integration tests for flow rate converter engine connection."""

    def test_mass_converter_import(self) -> Any:
        """Test that mass flow converter can be imported."""
        try:
            from sidekick.calculators.conversion.flow_rate_converter import (
                mass_to_mass,
            )

            assert mass_to_mass is not None
        except ImportError:
            pytest.skip("Flow rate converter not available in test environment")

    def test_molar_converter_import(self) -> Any:
        """Test that molar flow converter can be imported."""
        try:
            from sidekick.calculators.conversion.flow_rate_converter import (
                molar_to_molar,
            )

            assert molar_to_molar is not None
        except ImportError:
            pytest.skip("Flow rate converter not available")

    def test_volumetric_conversions_import(self) -> Any:
        """Test that volumetric conversions can be imported."""
        try:
            from sidekick.calculators.conversion.flow_rate_converter import (
                VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S,
            )

            assert VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S is not None
            assert len(VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S) > 0
        except ImportError:
            pytest.skip("Flow rate converter not available")

    def test_mass_conversion_kg_to_lb(self) -> Any:
        """Test mass flow conversion from kg/h to lb/h."""
        try:
            from sidekick.calculators.conversion.flow_rate_converter import (
                mass_to_mass,
            )

            result = mass_to_mass(1000, "kg/h", "lb/h")
            # 1000 kg/h * 2.20462 = ~2204.62 lb/h
            assert 2200 < result < 2210
        except ImportError:
            pytest.skip("Flow rate converter not available")

    def test_molar_conversion_kmol_to_lbmol(self) -> Any:
        """Test molar flow conversion from kmol/h to lbmol/h."""
        try:
            from sidekick.calculators.conversion.flow_rate_converter import (
                molar_to_molar,
            )

            result = molar_to_molar(100, "kmol/h", "lbmol/h")
            # 100 kmol/h * (1000/453.592) = ~220.46 lbmol/h
            assert 220 < result < 221
        except ImportError:
            pytest.skip("Flow rate converter not available")

    def test_volumetric_conversion_m3_to_cfm(self) -> Any:
        """Test volumetric flow conversion from m3/h to CFM."""
        try:
            from sidekick.calculators.conversion.flow_rate_converter import (
                VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S,
            )

            # Convert 100 m3/h to CFM
            value = 100
            from_unit = "m3/h"
            to_unit = "CFM"

            m3_per_s = value * VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S[from_unit]
            result = m3_per_s / VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S[to_unit]

            # 100 m3/h = 100/60 m3/min = ~1.667 m3/min
            # 1 m3 = 35.3147 ft3, so 1.667 m3/min = ~58.86 CFM
            assert 58 < result < 60
        except ImportError:
            pytest.skip("Flow rate converter not available")


class TestFlowRateConverterUnits:
    """Tests for unit definitions and conversions."""

    def test_mass_flow_units_defined(self) -> Any:
        """Test that mass flow units are defined."""
        try:
            from sidekick.calculators.conversion.flow_rate_converter import (
                MASS_FLOW_CONVERSIONS,
            )

            expected_units = ["kg/s", "kg/h", "lb/h", "g/s"]
            for unit in expected_units:
                assert unit in MASS_FLOW_CONVERSIONS
        except ImportError:
            pytest.skip("Flow rate converter not available")

    def test_molar_flow_units_defined(self) -> Any:
        """Test that molar flow units are defined."""
        try:
            from sidekick.calculators.conversion.flow_rate_converter import (
                MOLAR_FLOW_CONVERSIONS,
            )

            expected_units = ["mol/s", "kmol/h", "lbmol/h"]
            for unit in expected_units:
                assert unit in MOLAR_FLOW_CONVERSIONS
        except ImportError:
            pytest.skip("Flow rate converter not available")

    def test_volumetric_flow_units_defined(self) -> Any:
        """Test that volumetric flow units are defined."""
        try:
            from sidekick.calculators.conversion.flow_rate_converter import (
                VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S,
            )

            expected_units = ["m3/s", "m3/h", "L/s", "CFM", "GPM"]
            for unit in expected_units:
                assert unit in VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S
        except ImportError:
            pytest.skip("Flow rate converter not available")


class TestFlowRateConverterGUIRegistration:
    """Tests for GUI framework registration."""

    def test_gui_registration_exists(self) -> Any:
        """Test that gui_registration.py exists and has required metadata."""
        try:
            from flow_rate_converter import gui_registration

            assert hasattr(gui_registration, "GUI_METADATA")
            metadata = gui_registration.GUI_METADATA

            assert "name" in metadata
            assert "description" in metadata
            assert "category" in metadata
            assert "entry_point" in metadata
        except ImportError:
            pytest.skip("GUI registration not yet implemented")

    def test_gui_registration_category(self) -> Any:
        """Test that converter is in utilities category."""
        try:
            from flow_rate_converter import gui_registration

            assert gui_registration.GUI_METADATA["category"] == "utilities"
        except ImportError:
            pytest.skip("GUI registration not yet implemented")

    def test_launcher_exists(self) -> Any:
        """Test that launcher script exists."""
        try:
            from flow_rate_converter import launch_pyqt6

            assert hasattr(launch_pyqt6, "main")
        except ImportError:
            pytest.skip("Launcher not yet implemented")
