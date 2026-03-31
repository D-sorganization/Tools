"""Unit conversion utilities for the pressure drop interface.

Internal submodule extracted from pressure_drop_interface.py to keep file
size within the line budget.  Import these symbols via
``pressure_drop_calculator.pressure_drop_interface`` (the public module)
rather than directly from this private module.
"""


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert temperature between units.

    Args:
        value: Temperature value to convert
        from_unit: Source unit ('K', 'C', 'F')
        to_unit: Target unit ('K', 'C', 'F')

    Returns:
        Converted temperature value

    Raises:
        ValueError: If unit is not recognised
    """
    if not (value is not None):
        raise ValueError("value must be provided")
    from_unit = from_unit.upper()
    to_unit = to_unit.upper()

    if from_unit == "K":
        temp_k = value
    elif from_unit == "C":
        temp_k = value + 273.15
    elif from_unit == "F":
        temp_k = (value - 32) * 5 / 9 + 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit}")

    if to_unit == "K":
        return temp_k
    elif to_unit == "C":
        return temp_k - 273.15
    elif to_unit == "F":
        return (temp_k - 273.15) * 9 / 5 + 32
    else:
        raise ValueError(f"Unknown temperature unit: {to_unit}")


def _convert_pressure(value: float, from_unit: str, to_unit: str) -> float:
    """Convert pressure between units.

    Args:
        value: Pressure value to convert
        from_unit: Source unit (Pa, kPa, MPa, bar, mbar, atm, psi, psia, psig)
        to_unit: Target unit (same options)

    Returns:
        Converted pressure value

    Raises:
        ValueError: If unit is not recognised
    """
    to_pa = {
        "Pa": 1.0,
        "kPa": 1000.0,
        "MPa": 1e6,
        "bar": 1e5,
        "mbar": 100.0,
        "atm": 101325.0,
        "psi": 6894.76,
        "psia": 6894.76,
        "psig": 6894.76,  # Note: gauge pressure, caller should handle offset
    }

    if from_unit not in to_pa:
        raise ValueError(f"Unknown pressure unit: {from_unit}")
    if to_unit not in to_pa:
        raise ValueError(f"Unknown pressure unit: {to_unit}")

    pa = value * to_pa[from_unit]
    return pa / to_pa[to_unit]
