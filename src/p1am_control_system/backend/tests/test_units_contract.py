"""Cross-boundary unit contracts between the firmware and the backend.

The firmware publishes every thermocouple as *percent of full scale*, not
degrees C. Full scale is therefore a two-sided contract: the firmware scales
degC -> % on the way out and the backend scales % -> degC on the way back. If
the two halves disagree, every temperature in the system is wrong by the ratio
and the high-high heater cutoff -- which is expressed in degC -- trips late.

Until now that contract was a comment. These tests make it a gate.
"""

from __future__ import annotations

import re
from pathlib import Path

import hardware
import pytest

_FIRMWARE_DIR = Path(__file__).resolve().parents[2] / "firmware"
_FULL_SCALE_PATTERN = re.compile(
    r"kThermocoupleFullScaleC\s*=\s*([0-9]+(?:\.[0-9]*)?)f?\s*;"
)


def _firmware_full_scale_c() -> float:
    """Parse the firmware's thermocouple full scale out of its own source.

    Searches the whole firmware directory rather than one file so the test
    keeps working when the constant moves between the .cpp and the header.
    """
    matches: list[str] = []
    for path in sorted(_FIRMWARE_DIR.glob("*.[ch]*")):
        matches.extend(_FULL_SCALE_PATTERN.findall(path.read_text(encoding="utf-8")))
    if not matches:
        pytest.fail(
            "kThermocoupleFullScaleC not found in "
            f"{_FIRMWARE_DIR} -- the firmware/backend scaling contract cannot "
            "be verified. If the constant was renamed, update this test."
        )
    unique = {float(value) for value in matches}
    assert len(unique) == 1, (
        f"firmware declares conflicting thermocouple full scales: {sorted(unique)}"
    )
    return unique.pop()


def test_backend_full_scale_matches_firmware() -> None:
    """The backend constant must equal the firmware's compile-time constant.

    A mismatch under-reads every temperature and delays the HH heater cutoff by
    the same ratio -- e.g. a backend at 1000 against a firmware at 1400 shows
    900 degC while the process is actually at 1260 degC.
    """
    assert hardware.THERMOCOUPLE_FULL_SCALE_C == _firmware_full_scale_c()


def test_full_scale_is_a_positive_finite_temperature() -> None:
    """DbC: the scale divides measurements, so zero or negative is nonsense."""
    assert hardware.THERMOCOUPLE_FULL_SCALE_C > 0.0


def test_percent_to_celsius_round_trips() -> None:
    """The conversion the whole system depends on, pinned in both directions."""
    full_scale = hardware.THERMOCOUPLE_FULL_SCALE_C
    assert hardware.percent_to_celsius(0.0) == 0.0
    assert hardware.percent_to_celsius(100.0) == pytest.approx(full_scale)
    assert hardware.percent_to_celsius(50.0) == pytest.approx(full_scale / 2.0)
    assert hardware.celsius_to_percent(full_scale) == pytest.approx(100.0)
    assert hardware.celsius_to_percent(0.0) == 0.0


@pytest.mark.parametrize("bad", ["50", None, True])
def test_percent_to_celsius_rejects_non_numeric(bad: object) -> None:
    """DbC: wrong type is a TypeError, not a silent coercion."""
    with pytest.raises(TypeError):
        hardware.percent_to_celsius(bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_percent_to_celsius_rejects_non_finite(bad: float) -> None:
    """DbC: a non-finite reading is a fault, not a temperature."""
    with pytest.raises(ValueError):
        hardware.percent_to_celsius(bad)


class TestDeglitchFilterRange:
    """#4035 -- both filters were pinned to the module default full scale."""

    def _service(self) -> object:
        import logging

        from temperature_integration import TemperatureService

        return TemperatureService(plc_client=None, logger=logging.getLogger("test"))

    def test_filter_is_built_for_its_channel_range(self) -> None:
        """A shorter-range channel must get a correspondingly lower rail.

        Pinned to 1400 C, the high-side burnout rail sat above any reading a
        700 C channel could produce, so an open thermocouple reading full scale
        was accepted as a genuine measurement.
        """
        from temperature_integration import TemperatureService
        from temperature_models import ThermocoupleChannel

        channel = ThermocoupleChannel(tag="TAG_0", full_scale_c=700.0, label="short")
        built = TemperatureService._build_filter(channel)

        assert built._full_scale_c == pytest.approx(700.0)
        # The non-physical-step threshold scales with the span rather than
        # staying at the 1400 C default.
        assert built._max_step_c < 250.0

    def test_full_scale_default_comes_from_the_hardware_contract(self) -> None:
        """The filter must not keep its own copy of the full-scale constant."""
        import thermocouple_filter

        assert thermocouple_filter._FULL_SCALE_C == hardware.THERMOCOUPLE_FULL_SCALE_C
