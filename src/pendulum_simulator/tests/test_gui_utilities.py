"""Tests for the pure-math GUI utility modules (no Qt dependency).

This file covers:
- catmull_rom.catmull_rom_smooth: spline interpolation
- torque_history_constants: color palette management, joint labels
- unit_converter: UnitPreferences, UnitConverter, to_si/from_si functions

All modules import without Qt being present.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from double_pendulum_golf.gui.catmull_rom import catmull_rom_smooth
import double_pendulum_golf.gui.torque_history_constants as thc
from double_pendulum_golf.gui.unit_converter import (
    UnitCategory,
    UnitConverter,
    UnitPreferences,
    UnitSystem,
    _get_factor,
    from_si,
    get_available_units,
    get_preset_names,
    get_unit_label,
    to_si,
)


class TestCatmullRomSmooth:
    """Tests for Catmull-Rom spline smoothing."""

    # --- Boundary cases ---

    def test_fewer_than_4_points_returns_unchanged(self) -> None:
        pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
        result = catmull_rom_smooth(pts)
        assert result is pts  # returned same object

    def test_exactly_4_points_produces_output(self) -> None:
        pts = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        result = catmull_rom_smooth(pts)
        assert len(result) >= len(pts)

    def test_empty_returns_empty(self) -> None:
        result = catmull_rom_smooth([])
        assert result == []

    def test_one_point_returns_unchanged(self) -> None:
        pts = [(5.0, 3.0)]
        result = catmull_rom_smooth(pts)
        assert result == pts

    # --- Contract: endpoint preservation ---

    def test_endpoint_preserved(self) -> None:
        pts = [(0.0, 0.0), (1.0, 2.0), (2.0, 1.0), (3.0, 3.0), (4.0, 0.0)]
        result = catmull_rom_smooth(pts)
        assert result[-1] == pts[-1]

    def test_result_longer_than_input(self) -> None:
        pts = [(float(i), float(i) ** 2) for i in range(6)]
        result = catmull_rom_smooth(pts, n_sub=4)
        assert len(result) >= len(pts)

    # --- n_sub parameter ---

    def test_n_sub_1_gives_minimal_output(self) -> None:
        pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]
        result = catmull_rom_smooth(pts, n_sub=1)
        # With n_sub=1, each segment contributes 1 point + final endpoint
        assert len(result) >= len(pts)

    def test_n_sub_increases_resolution(self) -> None:
        pts = [(float(i), 0.0) for i in range(5)]
        r1 = catmull_rom_smooth(pts, n_sub=2)
        r4 = catmull_rom_smooth(pts, n_sub=8)
        assert len(r4) > len(r1)

    def test_n_sub_zero_raises(self) -> None:
        pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        with pytest.raises((ValueError, TypeError)):
            catmull_rom_smooth(pts, n_sub=0)

    def test_negative_n_sub_raises(self) -> None:
        pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        with pytest.raises((ValueError, TypeError)):
            catmull_rom_smooth(pts, n_sub=-1)

    # --- Mathematical: straight line should be smooth ---

    def test_collinear_points_stay_collinear(self) -> None:
        """Points on a straight line: interpolated points should also be on the line."""
        pts = [(float(i), float(i)) for i in range(6)]
        result = catmull_rom_smooth(pts, n_sub=4)
        for x, y in result:
            # On line y = x
            assert abs(x - y) < 1e-6, f"Point ({x},{y}) not on diagonal"

    # --- Return type ---

    def test_returns_list_of_tuples(self) -> None:
        pts = [(0.0, 0.0), (1.0, 0.5), (2.0, 1.0), (3.0, 0.5)]
        result = catmull_rom_smooth(pts)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2


# ===========================================================================
# torque_history_constants tests
# ===========================================================================


class TestTorqueHistoryConstants:
    """Tests for color palette management and joint label helpers."""

    # --- Color palette basics ---

    def test_drive_colors_count(self) -> None:
        """Default drive color palette has 7 entries."""
        colors = thc.get_drive_colors()
        assert len(colors) == 7

    def test_friction_colors_count(self) -> None:
        colors = thc.get_friction_colors()
        assert len(colors) == 7

    def test_total_colors_count(self) -> None:
        colors = thc.get_total_colors()
        assert len(colors) == 7

    def test_colors_are_rgb_tuples(self) -> None:
        for colors in [
            thc.get_drive_colors(),
            thc.get_friction_colors(),
            thc.get_total_colors(),
        ]:
            for r, g, b in colors:
                assert 0 <= r <= 255
                assert 0 <= g <= 255
                assert 0 <= b <= 255

    # --- Colorblind mode toggling ---

    def test_default_mode_is_non_colorblind(self) -> None:
        thc.set_colorblind_mode(False)
        default_drive = thc.get_drive_colors()
        assert default_drive == list(thc._DRIVE_COLORS)

    def test_colorblind_mode_on_changes_palette(self) -> None:
        thc.set_colorblind_mode(True)
        cb_drive = thc.get_drive_colors()
        assert cb_drive == list(thc._CB_DRIVE_COLORS)
        thc.set_colorblind_mode(False)  # restore

    def test_colorblind_friction_palette(self) -> None:
        thc.set_colorblind_mode(True)
        assert thc.get_friction_colors() == list(thc._CB_FRICTION_COLORS)
        thc.set_colorblind_mode(False)

    def test_colorblind_total_palette(self) -> None:
        thc.set_colorblind_mode(True)
        assert thc.get_total_colors() == list(thc._CB_TOTAL_COLORS)
        thc.set_colorblind_mode(False)

    def test_set_colorblind_mode_with_truthy_value(self) -> None:
        thc.set_colorblind_mode(1)  # truthy but not bool
        assert thc.get_drive_colors() == list(thc._CB_DRIVE_COLORS)
        thc.set_colorblind_mode(0)

    # --- Joint label helpers ---

    def test_labels_for_2_dof(self) -> None:
        labels = thc._joint_labels_for_ndof(2)
        assert labels == ["Shoulder", "Wrist"]
        assert len(labels) == 2

    def test_labels_for_3_dof(self) -> None:
        labels = thc._joint_labels_for_ndof(3)
        assert labels == ["Shoulder", "Elbow", "Wrist"]
        assert len(labels) == 3

    def test_labels_for_7_dof(self) -> None:
        labels = thc._joint_labels_for_ndof(7)
        assert len(labels) == 7
        assert "Hub" in labels

    def test_labels_for_generic_ndof(self) -> None:
        labels = thc._joint_labels_for_ndof(5)
        assert len(labels) == 5
        for i, lbl in enumerate(labels):
            assert f"Joint {i + 1}" == lbl

    def test_labels_for_1_dof(self) -> None:
        labels = thc._joint_labels_for_ndof(1)
        assert len(labels) == 1

    def test_zero_ndof_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            thc._joint_labels_for_ndof(0)

    def test_colors_are_copies(self) -> None:
        """get_*_colors() should return a new list each time."""
        c1 = thc.get_drive_colors()
        c2 = thc.get_drive_colors()
        assert c1 is not c2  # different list objects


# ===========================================================================
# unit_converter tests
# ===========================================================================


class TestUnitPreferences:
    """Tests for UnitPreferences dataclass."""

    def test_default_is_si(self) -> None:
        prefs = UnitPreferences()
        assert prefs.get_unit(UnitCategory.LENGTH) == "m"
        assert prefs.get_unit(UnitCategory.MASS) == "kg"
        assert prefs.get_unit(UnitCategory.TORQUE) == "N·m"

    def test_apply_preset_si(self) -> None:
        prefs = UnitPreferences()
        prefs.apply_preset("SI")
        assert prefs.get_unit(UnitCategory.LENGTH) == "m"

    def test_apply_preset_imperial(self) -> None:
        prefs = UnitPreferences()
        prefs.apply_preset("Imperial")
        assert prefs.get_unit(UnitCategory.LENGTH) == "in"
        assert prefs.get_unit(UnitCategory.MASS) == "lb"
        assert prefs.get_unit(UnitCategory.TORQUE) == "lbf·in"

    def test_apply_preset_engineering(self) -> None:
        prefs = UnitPreferences()
        prefs.apply_preset("Engineering")
        assert prefs.get_unit(UnitCategory.LENGTH) == "cm"
        assert prefs.get_unit(UnitCategory.ANGLE) == "deg"

    def test_invalid_preset_raises(self) -> None:
        prefs = UnitPreferences()
        with pytest.raises((ValueError, TypeError), match="Unknown preset"):
            prefs.apply_preset("ByteImperial")

    def test_set_unit_valid(self) -> None:
        prefs = UnitPreferences()
        prefs.set_unit(UnitCategory.LENGTH, "cm")
        assert prefs.get_unit(UnitCategory.LENGTH) == "cm"

    def test_set_unit_invalid_raises(self) -> None:
        prefs = UnitPreferences()
        with pytest.raises((ValueError, TypeError), match="Invalid unit"):
            prefs.set_unit(UnitCategory.LENGTH, "parsec")

    def test_get_unit_default_for_missing_category(self) -> None:
        """get_unit returns first option if category not in selections."""
        prefs = UnitPreferences()
        prefs.selections = {}
        # Should return the SI default (first option)
        label = prefs.get_unit(UnitCategory.LENGTH)
        assert label == "m"

    def test_save_to_qsettings(self) -> None:
        prefs = UnitPreferences()
        mock_settings = MagicMock()
        prefs.save_to_qsettings(mock_settings)
        assert mock_settings.setValue.called

    def test_load_from_qsettings_valid(self) -> None:
        prefs = UnitPreferences()
        mock_settings = MagicMock()
        mock_settings.value.side_effect = lambda key: "cm" if "length" in key else None
        prefs.load_from_qsettings(mock_settings)
        assert prefs.get_unit(UnitCategory.LENGTH) == "cm"

    def test_load_from_qsettings_invalid_unit_ignored(self) -> None:
        """Invalid saved unit should be silently ignored."""
        prefs = UnitPreferences()
        mock_settings = MagicMock()
        mock_settings.value.return_value = "fathoms"  # not a valid unit
        prefs.load_from_qsettings(mock_settings)
        # Should remain at default SI
        assert prefs.get_unit(UnitCategory.LENGTH) == "m"

    def test_load_from_qsettings_non_string_ignored(self) -> None:
        prefs = UnitPreferences()
        mock_settings = MagicMock()
        mock_settings.value.return_value = 42  # not a string
        prefs.load_from_qsettings(mock_settings)
        assert prefs.get_unit(UnitCategory.LENGTH) == "m"


class TestGetFactor:
    """Tests for internal _get_factor function."""

    def test_si_length_factor_is_1(self) -> None:
        factor = _get_factor(UnitCategory.LENGTH, "m")
        assert factor == pytest.approx(1.0)

    def test_inch_factor(self) -> None:
        factor = _get_factor(UnitCategory.LENGTH, "in")
        assert factor == pytest.approx(0.0254, abs=1e-10)

    def test_invalid_unit_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown unit"):
            _get_factor(UnitCategory.LENGTH, "furlongs")

    def test_all_factors_positive(self) -> None:
        """Every registered unit must have a positive conversion factor."""
        from double_pendulum_golf.gui.unit_converter import _UNIT_OPTIONS

        for cat, options in _UNIT_OPTIONS.items():
            for label, factor in options:
                assert (
                    factor > 0
                ), f"Non-positive factor for {cat.value}/{label}: {factor}"


class TestToSiFromSi:
    """Tests for to_si and from_si conversion functions."""

    @pytest.fixture
    def si_prefs(self) -> UnitPreferences:
        return UnitPreferences()

    @pytest.fixture
    def imperial_prefs(self) -> UnitPreferences:
        p = UnitPreferences()
        p.apply_preset("Imperial")
        return p

    def test_to_si_noop_in_si(self, si_prefs: UnitPreferences) -> None:
        """In SI mode, to_si is a no-op."""
        assert to_si(1.0, UnitCategory.LENGTH, si_prefs) == pytest.approx(1.0)

    def test_from_si_noop_in_si(self, si_prefs: UnitPreferences) -> None:
        assert from_si(1.0, UnitCategory.LENGTH, si_prefs) == pytest.approx(1.0)

    def test_to_si_inches_to_meters(self, imperial_prefs: UnitPreferences) -> None:
        result = to_si(1.0, UnitCategory.LENGTH, imperial_prefs)
        assert result == pytest.approx(0.0254, abs=1e-10)

    def test_from_si_meters_to_inches(self, imperial_prefs: UnitPreferences) -> None:
        result = from_si(0.0254, UnitCategory.LENGTH, imperial_prefs)
        assert result == pytest.approx(1.0, abs=1e-8)

    def test_roundtrip_si(self, si_prefs: UnitPreferences) -> None:
        """to_si → from_si should recover original value."""
        for cat in UnitCategory:
            val = 5.0
            assert from_si(to_si(val, cat, si_prefs), cat, si_prefs) == pytest.approx(
                val, rel=1e-9
            )

    def test_roundtrip_imperial(self, imperial_prefs: UnitPreferences) -> None:
        """to_si → from_si should recover original value in imperial mode."""
        for cat in UnitCategory:
            val = 3.14
            result = from_si(to_si(val, cat, imperial_prefs), cat, imperial_prefs)
            assert result == pytest.approx(val, rel=1e-9)

    def test_mass_kg_to_lb(self, imperial_prefs: UnitPreferences) -> None:
        result_kg = to_si(1.0, UnitCategory.MASS, imperial_prefs)
        assert result_kg == pytest.approx(0.45359237, abs=1e-6)

    def test_torque_nm_in_si(self, si_prefs: UnitPreferences) -> None:
        assert to_si(10.0, UnitCategory.TORQUE, si_prefs) == pytest.approx(10.0)


class TestGetAvailableUnitsAndPresets:
    def test_length_has_m_in_it(self) -> None:
        units = get_available_units(UnitCategory.LENGTH)
        assert "m" in units

    def test_length_has_multiple_options(self) -> None:
        units = get_available_units(UnitCategory.LENGTH)
        assert len(units) >= 2

    def test_preset_names_includes_si_and_imperial(self) -> None:
        names = get_preset_names()
        assert "SI" in names
        assert "Imperial" in names

    def test_get_unit_label_si(self) -> None:
        prefs = UnitPreferences()
        label = get_unit_label(UnitCategory.LENGTH, prefs)
        assert label == "m"

    def test_all_categories_have_at_least_one_unit(self) -> None:
        for cat in UnitCategory:
            units = get_available_units(cat)
            assert len(units) >= 1


class TestUnitConverter:
    """Tests for the legacy UnitConverter class."""

    def test_si_length_noop(self) -> None:
        uc = UnitConverter(UnitSystem.SI)
        assert uc.to_si_length(1.0) == pytest.approx(1.0)
        assert uc.from_si_length(1.0) == pytest.approx(1.0)

    def test_imperial_to_si_length(self) -> None:
        uc = UnitConverter(UnitSystem.IMPERIAL)
        assert uc.to_si_length(1.0) == pytest.approx(0.0254, abs=1e-10)

    def test_imperial_from_si_length(self) -> None:
        uc = UnitConverter(UnitSystem.IMPERIAL)
        result = uc.from_si_length(0.0254)
        assert result == pytest.approx(1.0, abs=1e-8)

    def test_si_mass_noop(self) -> None:
        uc = UnitConverter(UnitSystem.SI)
        assert uc.to_si_mass(5.0) == pytest.approx(5.0)
        assert uc.from_si_mass(5.0) == pytest.approx(5.0)

    def test_imperial_mass_lb_to_kg(self) -> None:
        uc = UnitConverter(UnitSystem.IMPERIAL)
        assert uc.to_si_mass(1.0) == pytest.approx(0.45359237, abs=1e-6)

    def test_si_torque_noop(self) -> None:
        uc = UnitConverter(UnitSystem.SI)
        assert uc.to_si_torque(1.0) == pytest.approx(1.0)
        assert uc.from_si_torque(1.0) == pytest.approx(1.0)

    def test_length_unit_si(self) -> None:
        uc = UnitConverter(UnitSystem.SI)
        assert uc.length_unit == "m"

    def test_length_unit_imperial(self) -> None:
        uc = UnitConverter(UnitSystem.IMPERIAL)
        assert uc.length_unit == "in"

    def test_mass_unit_si(self) -> None:
        uc = UnitConverter(UnitSystem.SI)
        assert uc.mass_unit == "kg"

    def test_mass_unit_imperial(self) -> None:
        uc = UnitConverter(UnitSystem.IMPERIAL)
        assert uc.mass_unit == "lb"

    def test_torque_unit_si(self) -> None:
        uc = UnitConverter(UnitSystem.SI)
        assert uc.torque_unit == "N·m"

    def test_velocity_unit_always_rad_s(self) -> None:
        for system in UnitSystem:
            uc = UnitConverter(system)
            assert uc.velocity_unit == "rad/s"

    def test_angle_unit(self) -> None:
        uc = UnitConverter(UnitSystem.SI)
        assert uc.angle_unit == "°"

    def test_roundtrip_torque_imperial(self) -> None:
        uc = UnitConverter(UnitSystem.IMPERIAL)
        original = 50.0
        result = uc.from_si_torque(uc.to_si_torque(original))
        assert result == pytest.approx(original, rel=1e-9)
