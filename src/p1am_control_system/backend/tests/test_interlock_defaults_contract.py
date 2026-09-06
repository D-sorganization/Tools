"""Contract tests for the default interlock configuration (issues #4001, #4911).

The failure these pin: ``defaults.py`` used to push ``low_limit=5.0`` to all 32
tags. The firmware's ``SafetyInterlock::Evaluate`` walks every tag, so the
unrouted ones (holding 0.0) -- and a routed type-K thermocouple at room
temperature (25 C = 1.8 % of the 1400 C full scale) -- tripped the plant the
moment the default config was deployed, and nothing could reset it.

Three contracts:

1. Deploying ``default_routing_config()`` to a freshly booted PLC does not trip.
   Simulated here with a Python model of the firmware's low/high evaluation
   over the register image the backend would actually write.
2. Every tag that is not a routed input is fully disabled (all limits ``None``)
   and encodes to the firmware's disabled sentinels.
3. The backend sentinels equal the firmware's ``kDisabledLowLimit`` /
   ``kDisabledHighLimit`` (parsed out of ``SafetyInterlock.h``), the same way
   ``test_units_contract.py`` pins the thermocouple full scale.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import hardware
import pytest

pytest.importorskip("sqlmodel")

from defaults import default_routing_config  # noqa: E402
from modbus_codec import (  # noqa: E402
    DEFAULT_INTERLOCK,
    INTERLOCK_REGISTER_WIDTH,
    decode_interlocks,
    encode_interlocks,
    registers_to_float,
)
from models import InterlockConfig  # noqa: E402

_FIRMWARE_DIR = Path(__file__).resolve().parents[2] / "firmware"
_SENTINEL_RE = re.compile(
    r"kDisabled(Low|High)Limit\s*=\s*(-?[0-9]+(?:\.[0-9]*)?)f?\s*;"
)

# The value an unrouted broker tag holds after boot, and what a routed
# thermocouple channel reads at room temperature (percent of full scale).
_UNROUTED_VALUE = 0.0
_ROOM_TEMP_PERCENT = hardware.celsius_to_percent(25.0)


def _firmware_sentinels() -> dict[str, float]:
    found: dict[str, float] = {}
    for path in _FIRMWARE_DIR.glob("*.h"):
        for side, literal in _SENTINEL_RE.findall(path.read_text(encoding="utf-8")):
            found[side] = float(literal)
    assert set(found) == {"Low", "High"}, (
        f"kDisabledLowLimit/kDisabledHighLimit not both found under {_FIRMWARE_DIR}"
    )
    return found


def _firmware_trip_model(register_image: list[int], tag_values: dict[int, float]):
    """Python model of ``SafetyInterlock::FindTripCause`` over a register image.

    Mirrors the firmware: a tag is interlocked when its low limit is above the
    low sentinel or its high limit is below the high sentinel; only interlocked
    tags are compared; NaN on an interlocked tag trips.
    """
    for tag_index in range(hardware.TAG_COUNT):
        base = tag_index * INTERLOCK_REGISTER_WIDTH
        low = registers_to_float(register_image[base + 2], register_image[base + 3])
        high = registers_to_float(register_image[base + 4], register_image[base + 5])
        interlocked = (
            low > hardware.INTERLOCK_DISABLED_LOW
            or high < hardware.INTERLOCK_DISABLED_HIGH
        )
        if not interlocked:
            continue
        value = tag_values.get(tag_index, _UNROUTED_VALUE)
        if not math.isfinite(value) or value > high or value < low:
            return tag_index
    return None


def test_backend_sentinels_match_firmware() -> None:
    sentinels = _firmware_sentinels()
    assert hardware.INTERLOCK_DISABLED_LOW == sentinels["Low"]
    assert hardware.INTERLOCK_DISABLED_HIGH == sentinels["High"]


def test_default_config_does_not_trip_a_fresh_plc() -> None:
    """Simulated boot: unrouted tags at 0.0, routed TCs at room temperature."""
    config = default_routing_config()
    image = encode_interlocks(config.interlocks)

    tag_values = {
        hardware.tag_index(name): _ROOM_TEMP_PERCENT
        for name in config.input_routing[:4]  # the four thermocouple channels
    }
    assert _firmware_trip_model(image, tag_values) is None

    # ...and the same image with every tag at exactly 0.0 (a cold boot before
    # the first hardware scan).
    assert _firmware_trip_model(image, {}) is None


def test_unrouted_tags_are_fully_disabled() -> None:
    config = default_routing_config()
    routed = set(config.input_routing)
    for name, interlock in config.interlocks.items():
        if name in routed:
            # Routed inputs keep a high-side band only.
            assert interlock.low_limit is None
            assert interlock.lolo_limit is None
            assert interlock.high_limit is not None
            assert interlock.hihi_limit is not None
        else:
            assert interlock.is_disabled(), f"{name} must not be interlocked by default"


def test_no_default_limit_is_violated_by_zero_or_room_temperature() -> None:
    """Belt and braces on the model above: check the config values directly."""
    for name, interlock in default_routing_config().interlocks.items():
        for value in (_UNROUTED_VALUE, _ROOM_TEMP_PERCENT):
            if interlock.low_limit is not None:
                assert value >= interlock.low_limit, name
            if interlock.high_limit is not None:
                assert value <= interlock.high_limit, name


def test_codec_default_for_absent_tag_is_disabled() -> None:
    assert DEFAULT_INTERLOCK.is_disabled()
    image = encode_interlocks({})
    decoded = decode_interlocks(image)
    assert all(cfg.is_disabled() for cfg in decoded.values())


def test_none_round_trips_through_the_register_contract() -> None:
    config = {
        "TAG_3": InterlockConfig(high_limit=95.0, hihi_limit=100.0),
        "TAG_9": InterlockConfig(lolo_limit=0.5, low_limit=1.0),
    }
    decoded = decode_interlocks(encode_interlocks(config))
    assert decoded["TAG_3"].lolo_limit is None
    assert decoded["TAG_3"].low_limit is None
    assert decoded["TAG_3"].high_limit == pytest.approx(95.0)
    assert decoded["TAG_3"].hihi_limit == pytest.approx(100.0)
    assert decoded["TAG_9"].lolo_limit == pytest.approx(0.5)
    assert decoded["TAG_9"].low_limit == pytest.approx(1.0)
    assert decoded["TAG_9"].high_limit is None
    assert decoded["TAG_9"].hihi_limit is None
    assert decoded["TAG_0"].is_disabled()


def test_engine_limits_fold_none_to_infinity() -> None:
    limits = InterlockConfig(high_limit=95.0, hihi_limit=100.0).engine_limits()
    assert limits["lolo"] == float("-inf")
    assert limits["low"] == float("-inf")
    assert limits["high"] == 95.0
    assert limits["hihi"] == 100.0
    # Engine contract lolo <= low <= high <= hihi still holds.
    assert limits["lolo"] <= limits["low"] <= limits["high"] <= limits["hihi"]


def test_interlock_config_rejects_non_finite_limits() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            InterlockConfig(low_limit=bad)


def test_interlock_status_registers_sit_after_the_heartbeat() -> None:
    """The firmware widened its holding-register window for the read-back."""
    assert hardware.INTERLOCK_TRIPPED_REGISTER == hardware.HOST_HEARTBEAT_REGISTER + 1
    assert (
        hardware.INTERLOCK_TRIP_TAG_REGISTER == hardware.INTERLOCK_TRIPPED_REGISTER + 1
    )
    ino = (_FIRMWARE_DIR / "firmware.ino").read_text(encoding="utf-8")
    assert f"kInterlockTrippedReg = {hardware.INTERLOCK_TRIPPED_REGISTER};" in ino
    assert f"kInterlockTripTagReg = {hardware.INTERLOCK_TRIP_TAG_REGISTER};" in ino
    assert f"kInterlockResetCoil = {hardware.ESTOP_RESET_COIL};" in ino
