"""Single source of truth for the P1AM hardware contract.

The register map and tag-naming scheme are dictated by the firmware. Encoding
them in one module (rather than as scattered literals in modbus_client,
power_supply_integration, simulator_client, and main) means a firmware re-layout
or a different I/O count is a one-file change — and removes the divergent,
sometimes-unsafe copies of the TAG_<n> parser.

Mirrors firmware/README.md "Modbus register map".
"""

from __future__ import annotations

import math

# ---- Tag broker -------------------------------------------------------
TAG_COUNT = 32  # tags exposed by the firmware broker
TAG_PREFIX = "TAG_"
UNMAPPED_TAG_INDEX = 255  # firmware kUnmappedTag sentinel for routing/PID fields
UNMAPPED_TAG_NAME = f"{TAG_PREFIX}{UNMAPPED_TAG_INDEX}"

# ---- Signal scaling ---------------------------------------------------
# The firmware publishes thermocouples as PERCENT OF FULL SCALE, not degrees C
# (SignalBroker::ReadHardwareInputs scales degC -> % on the way out). Full
# scale is therefore a two-sided contract, and it lives here for the same
# reason the register map does: one definition, not a literal copied into
# temperature_integration, power_supply_integration and the HMI.
#
# Must equal the firmware's kThermocoupleFullScaleC. Enforced by
# tests/test_units_contract.py, which parses the firmware source -- a
# divergence under-reads every temperature and delays the high-high heater
# cutoff by the same ratio (issue #3998).
THERMOCOUPLE_FULL_SCALE_C = 1400.0

# ---- Register map (holding registers; see firmware/README.md) ---------
TAG_VALUE_BASE = 0  # tag values: TAG_i at (i*2, i*2+1) little-endian float
INPUT_ROUTING_BASE = 100  # channel -> tag id (slots 0-3 TC, 4-5 AI)
OUTPUT_ROUTING_BASE = 110  # channel -> tag id (slots 0-1 AO)
PID_CONFIG_BASE = 200  # 4 PIDs x 10 regs
PID_STRIDE = 10  # registers per PID block
PID_SETPOINT_OFFSET = 2  # setpoint is the 3rd field (regs +2, +3)
INTERLOCK_BASE = 300  # 32 tags x 8 regs (lolo/low/high/hihi)
PID_COUNT = 4

# ---- Interlock "disabled" sentinels ------------------------------------
# A limit of ``None`` in InterlockConfig means "not interlocked on this side".
# The register contract has no way to say that, so the backend encodes None as
# the exact sentinel the firmware's SafetyInterlock treats as disabled
# (kDisabledLowLimit / kDisabledHighLimit in SafetyInterlock.h) and decodes
# the sentinel back to None. A tag left at both sentinels is skipped by the
# firmware's Evaluate() entirely, which is what makes the default config safe
# to deploy: an unrouted tag reading 0.0 can no longer trip the plant (#4001).
# Enforced against the firmware source by
# src/p1am_control_system/backend/tests/test_interlock_defaults_contract.py.
INTERLOCK_DISABLED_LOW = -99999.0
INTERLOCK_DISABLED_HIGH = 99999.0
# Host-liveness watchdog. The firmware proves the host is alive from a CHANGE
# to this register (the value itself is meaningless), not from its content. If
# it sees neither a Modbus TCP connection nor a heartbeat change for
# HEARTBEAT_TIMEOUT_S it drives all analog outputs to 0 %, opens the heater
# relay, asserts Inhibit and holds the PID loops. The backend must therefore
# bump it once per successful scan — see AsyncModbusManager.write_heartbeat.
HOST_HEARTBEAT_REGISTER = 560
HEARTBEAT_TIMEOUT_S = 2.0  # firmware-side watchdog window
# Interlock status read-back, written by the firmware every scan (read-only
# for the host): 561 = 1 while the trip latch is set; 562 = broker tag index
# that latched the trip, or UNMAPPED_TAG_INDEX when clear. Lets the host
# confirm that a coil-1 reset actually took (issue #4001).
INTERLOCK_TRIPPED_REGISTER = 561
INTERLOCK_TRIP_TAG_REGISTER = 562

# Plant wiring: the DC power supply's analog command rides PID loop 0. Named
# here so the shutdown safe-state and the power-supply service agree on which
# loop must be zeroed to de-energize the supply.
POWER_SUPPLY_PID_INDEX = 0

# ---- Coils ------------------------------------------------------------
SAVE_TO_FLASH_COIL = 0
# Interlock reset request. The host writes 1; the firmware consumes the pulse
# (writes it back to 0) and clears its trip latch ONLY if no tag is still
# outside its band -- a reset while the cause persists is refused. See
# firmware/README.md "Interlock latch and reset".
ESTOP_RESET_COIL = 1
HEATER_RELAY_COIL = 2  # 24 V DO -> relay -> 110 V resistive heater (temp ctrl)
# Selects the P1-04THM open-circuit (burnout) fail direction: 1 = HIGH-side
# (an open thermocouple reads full scale -> heater shuts off, fail-safe),
# 0 = LOW-side (an open reads 0 C -> looks cold). The firmware reconfigures the
# module on change; the backend re-asserts this each scan so it survives a PLC
# reboot. See temperature_integration.TemperatureService.set_burnout_high_side.
THM_BURNOUT_COIL = 3


class NonFiniteValueError(ValueError):
    """A command value (tag force, setpoint, limit) is NaN or infinite.

    Distinct from transport failures on purpose (issue #3974): a NaN in a
    request body is a *precondition* violation by the caller, not an I/O
    fault, and must never be mistaken for a lost PLC link. The Modbus client
    raises this before touching the socket, so ``_connected`` is untouched.
    """


def require_finite_value(value: object, name: str = "value") -> float:
    """Return ``value`` as a finite float or raise :class:`NonFiniteValueError`.

    Raises:
        TypeError: If ``value`` is not a real number (``bool`` is rejected).
        NonFiniteValueError: If ``value`` is NaN or infinite.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise NonFiniteValueError(f"{name} must be finite, got {numeric!r}")
    return numeric


def json_safe(value: object) -> object:
    """Recursively replace non-finite floats with their ``repr`` string.

    FastAPI's default 422 response embeds the offending ``input`` and renders
    with ``allow_nan=False``, so a body of ``{"value": NaN}`` used to turn a
    validation error into a 500 (#3974). Run error payloads through this
    before serialising them.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    return value


def tag_name(index: int) -> str:
    """Return the canonical tag name for a broker index (e.g. 5 -> 'TAG_5').

    Raises:
        TypeError: If ``index`` is not an int.
        ValueError: If ``index`` is outside [0, TAG_COUNT).
    """
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError(f"index must be an int, got {type(index).__name__}")
    if not 0 <= index < TAG_COUNT:
        raise ValueError(f"tag index {index} out of range [0, {TAG_COUNT})")
    return f"{TAG_PREFIX}{index}"


def tag_index(name: str) -> int:
    """Parse a tag name to its broker index (e.g. 'TAG_5' -> 5).

    The single, strict parser: a malformed or out-of-range name raises rather
    than silently coercing to 0 (which would route a bad tag to TAG_0).

    Raises:
        TypeError: If ``name`` is not a str.
        ValueError: If ``name`` is not 'TAG_<n>' with n in [0, TAG_COUNT).
    """
    if not isinstance(name, str):
        raise TypeError(f"name must be a str, got {type(name).__name__}")
    if not name.startswith(TAG_PREFIX):
        raise ValueError(f"tag name {name!r} must start with {TAG_PREFIX!r}")
    suffix = name[len(TAG_PREFIX) :]
    if not suffix.isdigit():
        raise ValueError(f"tag name {name!r} has non-numeric index")
    index = int(suffix)
    if not 0 <= index < TAG_COUNT:
        raise ValueError(f"tag index {index} out of range [0, {TAG_COUNT})")
    return index


def pid_setpoint_address(pid_index: int) -> int:
    """Holding-register address of a PID loop's setpoint.

    Raises:
        ValueError: If ``pid_index`` is outside [0, PID_COUNT).
    """
    if not 0 <= pid_index < PID_COUNT:
        raise ValueError(f"pid index {pid_index} out of range [0, {PID_COUNT})")
    return PID_CONFIG_BASE + pid_index * PID_STRIDE + PID_SETPOINT_OFFSET


def percent_to_celsius(percent: float, full_scale_c: float | None = None) -> float:
    """Convert a broker tag percentage to degrees Celsius.

    The firmware scales degC -> % using ``THERMOCOUPLE_FULL_SCALE_C``; this is
    the inverse. Every consumer of a thermocouple tag must go through here so
    the conversion cannot drift between subsystems (issues #3998, #4003).

    Args:
        percent: Tag value in [0, 100] as published by the firmware broker.
        full_scale_c: Override for a channel with a different range. Defaults
            to the firmware contract value.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is not finite, or ``full_scale_c`` <= 0.
    """
    scale = THERMOCOUPLE_FULL_SCALE_C if full_scale_c is None else full_scale_c
    _require_finite_number(percent, "percent")
    _require_finite_number(scale, "full_scale_c")
    if scale <= 0.0:
        raise ValueError(f"full_scale_c must be positive, got {scale}")
    return float(percent) * float(scale) / 100.0


def celsius_to_percent(celsius: float, full_scale_c: float | None = None) -> float:
    """Convert degrees Celsius to a broker tag percentage.

    Inverse of :func:`percent_to_celsius`. Used to express a degC threshold in
    the tag domain the firmware interlock actually compares against.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is not finite, or ``full_scale_c`` <= 0.
    """
    scale = THERMOCOUPLE_FULL_SCALE_C if full_scale_c is None else full_scale_c
    _require_finite_number(celsius, "celsius")
    _require_finite_number(scale, "full_scale_c")
    if scale <= 0.0:
        raise ValueError(f"full_scale_c must be positive, got {scale}")
    return float(celsius) * 100.0 / float(scale)


def _require_finite_number(value: object, name: str) -> None:
    """DbC helper: reject non-numeric and non-finite inputs.

    A NaN temperature is a sensor fault, not a measurement, and must not be
    allowed to propagate into a trip comparison where it compares False
    against every threshold.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value}")
