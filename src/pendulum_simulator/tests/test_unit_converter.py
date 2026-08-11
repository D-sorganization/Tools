from typing import Any
import pytest
from unittest.mock import MagicMock

from double_pendulum_golf.gui.unit_converter import (
    UnitCategory,
    UnitPreferences,
    to_si,
    from_si,
    get_unit_label,
    get_available_units,
    get_preset_names,
    UnitSystem,
    UnitConverter,
)
from sidekick.utils.unit_constants import (
    FOOT_POUND_PER_SECOND_TO_WATT,
    FOOT_POUND_TO_JOULE,
    FOOT_POUND_TO_NEWTON_METER,
    POUND_FORCE_INCH_TO_NEWTON_METER,
)


def test_unit_preferences_init() -> Any:
    prefs = UnitPreferences()
    assert prefs.selections[UnitCategory.LENGTH] == "m"
    assert prefs.selections[UnitCategory.MASS] == "kg"


def test_unit_preferences_apply_preset() -> Any:
    prefs = UnitPreferences()
    prefs.apply_preset("Imperial")
    assert prefs.selections[UnitCategory.LENGTH] == "in"
    assert prefs.selections[UnitCategory.MASS] == "lb"

    with pytest.raises((ValueError, TypeError)):
        prefs.apply_preset("Unknown")


def test_unit_preferences_get_set() -> Any:
    prefs = UnitPreferences()
    prefs.set_unit(UnitCategory.LENGTH, "cm")
    assert prefs.get_unit(UnitCategory.LENGTH) == "cm"

    with pytest.raises((ValueError, TypeError)):
        prefs.set_unit(UnitCategory.LENGTH, "not_a_unit")


def test_qsettings_persistence() -> Any:
    prefs = UnitPreferences()
    prefs.set_unit(UnitCategory.LENGTH, "cm")

    mock_settings = MagicMock()
    mock_settings.value.return_value = "in"

    prefs.save_to_qsettings(mock_settings)
    mock_settings.setValue.assert_any_call("units/length", "cm")

    prefs.load_from_qsettings(mock_settings)
    assert prefs.get_unit(UnitCategory.LENGTH) == "in"

    # Test invalid string from settings
    mock_settings.value.return_value = "invalid_unit"
    prefs.load_from_qsettings(mock_settings)
    # Should ignore and keep the old value
    assert prefs.get_unit(UnitCategory.LENGTH) == "in"


def test_to_from_si() -> Any:
    prefs = UnitPreferences()
    prefs.apply_preset("SI")

    assert to_si(10.0, UnitCategory.LENGTH, prefs) == 10.0
    assert from_si(10.0, UnitCategory.LENGTH, prefs) == 10.0

    prefs.apply_preset("Engineering")  # length is cm
    assert to_si(100.0, UnitCategory.LENGTH, prefs) == 1.0
    assert from_si(1.0, UnitCategory.LENGTH, prefs) == 100.0


def test_imperial_foot_pound_units_use_shared_constants() -> None:
    prefs = UnitPreferences()

    prefs.set_unit(UnitCategory.TORQUE, "lbf·ft")
    assert to_si(1.0, UnitCategory.TORQUE, prefs) == pytest.approx(
        FOOT_POUND_TO_NEWTON_METER
    )
    assert from_si(
        to_si(1.0, UnitCategory.TORQUE, prefs), UnitCategory.TORQUE, prefs
    ) == pytest.approx(1.0, rel=1e-12)

    prefs.set_unit(UnitCategory.TORQUE, "lbf·in")
    assert to_si(1.0, UnitCategory.TORQUE, prefs) == pytest.approx(
        POUND_FORCE_INCH_TO_NEWTON_METER
    )

    prefs.set_unit(UnitCategory.ENERGY, "ft·lbf")
    assert to_si(1.0, UnitCategory.ENERGY, prefs) == pytest.approx(FOOT_POUND_TO_JOULE)
    assert from_si(
        to_si(1.0, UnitCategory.ENERGY, prefs), UnitCategory.ENERGY, prefs
    ) == pytest.approx(1.0, rel=1e-12)

    prefs.set_unit(UnitCategory.POWER, "ft·lbf/s")
    assert to_si(1.0, UnitCategory.POWER, prefs) == pytest.approx(
        FOOT_POUND_PER_SECOND_TO_WATT
    )
    assert from_si(
        to_si(1.0, UnitCategory.POWER, prefs), UnitCategory.POWER, prefs
    ) == pytest.approx(1.0, rel=1e-12)


def test_unknown_unit_factor() -> Any:
    prefs = UnitPreferences()
    prefs.selections[UnitCategory.LENGTH] = "foo"  # Bypass setter
    with pytest.raises(KeyError):
        to_si(10.0, UnitCategory.LENGTH, prefs)


def test_getters() -> Any:
    prefs = UnitPreferences()
    assert get_unit_label(UnitCategory.LENGTH, prefs) == "m"
    assert "cm" in get_available_units(UnitCategory.LENGTH)
    assert "SI" in get_preset_names()


def test_legacy_converter() -> Any:
    conv_si = UnitConverter(system=UnitSystem.SI)
    assert conv_si.to_si_length(10.0) == 10.0
    assert conv_si.from_si_length(10.0) == 10.0
    assert conv_si.to_si_mass(5.0) == 5.0
    assert conv_si.from_si_mass(5.0) == 5.0
    assert conv_si.to_si_torque(2.0) == 2.0
    assert conv_si.from_si_torque(2.0) == 2.0
    assert conv_si.length_unit == "m"
    assert conv_si.mass_unit == "kg"
    assert conv_si.torque_unit == "N·m"
    assert conv_si.velocity_unit == "rad/s"
    assert conv_si.angle_unit == "°"

    conv_imp = UnitConverter(system=UnitSystem.IMPERIAL)
    assert conv_imp.length_unit == "in"
    assert conv_imp.mass_unit == "lb"
    assert conv_imp.torque_unit == "lbf·in"

    assert conv_imp.to_si_length(1.0) == 0.0254
    assert conv_imp.from_si_length(0.0254) == 1.0
    assert conv_imp.to_si_mass(1.0) == 0.45359237
    assert conv_imp.to_si_torque(1.0) == POUND_FORCE_INCH_TO_NEWTON_METER
