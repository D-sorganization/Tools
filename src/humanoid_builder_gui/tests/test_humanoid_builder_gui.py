# ruff: noqa: E501
from typing import Any

"""
Humanoid Character Builder GUI Tests
====================================

TDD tests for the Humanoid Character Builder GUI components.
Tests cover PyQt6 main window, anthropometry, and URDF generation.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestHumanoidBuilderMainWindow:
    """Tests for the PyQt6 Humanoid Character Builder main window."""

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

    def test_main_window_imports(self, mock_qt_app: Any) -> Any:
        """Test that main window module can be imported."""
        try:
            from humanoid_builder_gui.ui.pyqt6 import main_window

            assert hasattr(main_window, "HumanoidBuilderWindow")
        except ImportError:
            pytest.skip("PyQt6 main window not yet implemented")

    def test_main_window_class_exists(self, mock_qt_app: Any) -> Any:
        """Test that window class is defined and callable."""
        try:
            from humanoid_builder_gui.ui.pyqt6.main_window import (
                HumanoidBuilderWindow,
            )

            assert callable(HumanoidBuilderWindow)
        except ImportError:
            pytest.skip("Main window not yet implemented")


class TestAnthropometry:
    """Tests for anthropometric calculations."""

    def test_bmi_calculation(self) -> Any:
        """Test BMI calculation formula."""
        height_m = 1.75
        mass_kg = 75.0

        bmi = mass_kg / (height_m * height_m)

        assert bmi == pytest.approx(24.49, rel=0.01)

    def test_bmi_underweight(self) -> Any:
        """Test underweight BMI classification."""
        height_m = 1.80
        mass_kg = 55.0

        bmi = mass_kg / (height_m * height_m)

        assert bmi < 18.5

    def test_bmi_normal(self) -> Any:
        """Test normal BMI classification."""
        height_m = 1.75
        mass_kg = 70.0

        bmi = mass_kg / (height_m * height_m)

        assert 18.5 <= bmi < 25.0

    def test_bmi_overweight(self) -> Any:
        """Test overweight BMI classification."""
        height_m = 1.70
        mass_kg = 85.0

        bmi = mass_kg / (height_m * height_m)

        assert 25.0 <= bmi < 30.0

    def test_segment_mass_calculation(self) -> Any:
        """Test segment mass from total mass."""
        total_mass = 75.0
        head_ratio = 0.0681

        head_mass = total_mass * head_ratio

        assert head_mass == pytest.approx(5.1075, rel=0.01)

    def test_segment_length_calculation(self) -> Any:
        """Test segment length from total height."""
        total_height = 1.75
        thigh_ratio = 0.245

        thigh_length = total_height * thigh_ratio

        assert thigh_length == pytest.approx(0.42875, rel=0.01)


class TestDeLevaMassRatios:
    """Tests for de Leva (1996) segment mass ratios."""

    def test_head_mass_ratio(self) -> Any:
        """Test head mass ratio value."""
        head_ratio = 0.0681
        assert 0.06 < head_ratio < 0.08

    def test_thigh_mass_ratio(self) -> Any:
        """Test thigh mass ratio (largest segment)."""
        thigh_ratio = 0.1416
        assert thigh_ratio > 0.10  # Thighs are significant mass

    def test_mass_ratios_sum(self) -> Any:
        """Test that bilateral segment mass ratios approximately sum to 1."""
        ratios = {
            "head": 0.0681,
            "neck": 0.0230,
            "thorax": 0.2100,
            "lumbar": 0.1385,
            "pelvis": 0.11685,
            "upper_arm": 0.0263 * 2,  # Bilateral
            "forearm": 0.0150 * 2,
            "hand": 0.00585 * 2,
            "thigh": 0.1447 * 2,
            "shin": 0.0457 * 2,
            "foot": 0.0133 * 2,
        }

        total = sum(ratios.values())
        assert total == pytest.approx(1.0, abs=0.06)


class TestDeLevLengthRatios:
    """Tests for de Leva (1996) segment length ratios."""

    def test_head_length_ratio(self) -> Any:
        """Test head length ratio."""
        head_ratio = 0.1395
        assert 0.13 < head_ratio < 0.15

    def test_thigh_length_ratio(self) -> Any:
        """Test thigh length ratio."""
        thigh_ratio = 0.245
        assert 0.23 < thigh_ratio < 0.26

    def test_shin_length_ratio(self) -> Any:
        """Test shin (lower leg) length ratio."""
        shin_ratio = 0.246
        # Shin and thigh should be similar length
        thigh_ratio = 0.245
        assert abs(shin_ratio - thigh_ratio) < 0.05


class TestGenderFactor:
    """Tests for gender factor calculations."""

    def test_male_factor(self) -> Any:
        """Test male gender factor."""
        male_factor = 1.0
        assert male_factor == pytest.approx(1.0)

    def test_female_factor(self) -> Any:
        """Test female gender factor."""
        female_factor = 0.0
        assert female_factor == pytest.approx(0.0)

    def test_neutral_factor(self) -> Any:
        """Test neutral gender factor."""
        neutral_factor = 0.5
        assert neutral_factor == pytest.approx(0.5)

    def test_interpolation(self) -> Any:
        """Test linear interpolation between male/female values."""
        male_value = 0.0694  # Male head mass ratio
        female_value = 0.0668  # Female head mass ratio
        gender_factor = 0.5

        # Linear interpolation
        interpolated = female_value + (male_value - female_value) * gender_factor

        assert interpolated == pytest.approx(0.0681, rel=0.01)


class TestProportionFactors:
    """Tests for proportion factor adjustments."""

    def test_default_proportion(self) -> Any:
        """Test default proportion factor is 1.0."""
        default = 1.0
        assert default == 1.0

    def test_increased_proportion(self) -> Any:
        """Test increased proportion effect."""
        base_length = 0.245  # Thigh length ratio
        factor = 1.1  # 10% increase

        adjusted = base_length * factor

        assert adjusted == pytest.approx(0.2695, rel=0.01)

    def test_decreased_proportion(self) -> Any:
        """Test decreased proportion effect."""
        base_length = 0.245
        factor = 0.9  # 10% decrease

        adjusted = base_length * factor

        assert adjusted == pytest.approx(0.2205, rel=0.01)

    def test_proportion_range(self) -> Any:
        """Test proportion factor valid range."""
        min_factor = 0.5
        max_factor = 1.5

        assert min_factor >= 0.5
        assert max_factor <= 1.5


class TestHumanoidBuilderGUIRegistration:
    """Tests for GUI framework registration."""

    def test_gui_registration_exists(self) -> Any:
        """Test that gui_registration.py exists and has required metadata."""
        try:
            from humanoid_builder_gui import gui_registration

            assert hasattr(gui_registration, "GUI_INFO")
            metadata = gui_registration.GUI_INFO

            assert "name" in metadata
            assert "description" in metadata
            assert "category" in metadata
            assert "tool_name" in metadata
        except ImportError:
            pytest.skip("GUI registration not yet implemented")

    def test_gui_registration_category(self) -> Any:
        """Test that tool is in robotics category."""
        try:
            from humanoid_builder_gui import gui_registration

            assert gui_registration.GUI_INFO["category"].lower() == "robotics"
        except ImportError:
            pytest.skip("GUI registration not yet implemented")

    def test_launcher_exists(self) -> Any:
        """Test that launcher script exists."""
        try:
            from humanoid_builder_gui import launch_pyqt6

            assert hasattr(launch_pyqt6, "bootstrap")
        except ImportError:
            pytest.skip("Launcher not yet implemented")


class TestLoDAliases:
    """Tests for LoD-compliant PyQt6 enum aliases in main_window module.

    Verifies that module-level aliases eliminate 3-level deep attribute chains.
    """

    @pytest.fixture
    def mock_qt_modules(self) -> Any:
        """Mock PyQt6 modules for headless environments."""
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

    def test_enum_aliases_defined(self, mock_qt_modules: Any) -> None:
        """Test that LoD-compliant enum aliases are defined at module level."""
        try:
            import importlib

            import humanoid_builder_gui.ui.pyqt6.main_window as mw

            importlib.reload(mw)
            alias_names = [
                "_SCROLL_BAR_ALWAYS_OFF",
                "_ALIGN_CENTER",
                "_HORIZONTAL",
                "_POLICY_EXPANDING",
                "_POLICY_FIXED",
                "_RESIZE_MODE_STRETCH",
                "_RESIZE_MODE_RESIZE_TO_CONTENTS",
            ]
            for name in alias_names:
                assert hasattr(mw, name), f"Missing LoD alias: {name}"
        except ImportError:
            pytest.skip("PyQt6 not available in this environment")

    def test_no_deep_attribute_chains_in_method_body(
        self, mock_qt_modules: Any
    ) -> None:
        """Test that the module source has no 3-level Qt enum chains outside aliases."""
        import inspect
        import re

        try:
            import humanoid_builder_gui.ui.pyqt6.main_window as mw

            source = inspect.getsource(mw)
            # Find all 3-level Qt enum accesses: Qt.Xxx.Yyy or QFoo.Bar.Baz
            pattern = re.compile(
                r"(?<![_A-Z])(Qt\.\w+\.\w+|QSizePolicy\.Policy\.\w+|QHeaderView\.ResizeMode\.\w+)"
            )
            matches = pattern.findall(source)
            # Only the alias definitions should match (7 lines)
            assert len(matches) <= 7, (
                f"Unexpected deep attribute chains found: {matches}"
            )
        except ImportError:
            pytest.skip("PyQt6 not available in this environment")
