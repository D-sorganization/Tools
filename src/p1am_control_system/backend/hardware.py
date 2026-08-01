"""Single source of truth for the P1AM hardware contract.

The register map and tag-naming scheme are dictated by the firmware. Encoding
them in one module (rather than as scattered literals in modbus_client,
power_supply_integration, simulator_client, and main) means a firmware re-layout
or a different I/O count is a one-file change — and removes the divergent,
sometimes-unsafe copies of the TAG_<n> parser.

Mirrors firmware/README.md "Modbus register map".
"""

from __future__ import annotations

# ---- Tag broker -------------------------------------------------------
TAG_COUNT = 32  # tags exposed by the firmware broker
TAG_PREFIX = "TAG_"
UNMAPPED_TAG_INDEX = 255  # firmware kUnmappedTag sentinel for routing/PID fields
UNMAPPED_TAG_NAME = f"{TAG_PREFIX}{UNMAPPED_TAG_INDEX}"

# ---- Register map (holding registers; see firmware/README.md) ---------
TAG_VALUE_BASE = 0  # tag values: TAG_i at (i*2, i*2+1) little-endian float
INPUT_ROUTING_BASE = 100  # channel -> tag id (slots 0-3 TC, 4-5 AI)
OUTPUT_ROUTING_BASE = 110  # channel -> tag id (slots 0-1 AO)
PID_CONFIG_BASE = 200  # 4 PIDs x 10 regs
PID_STRIDE = 10  # registers per PID block
PID_SETPOINT_OFFSET = 2  # setpoint is the 3rd field (regs +2, +3)
INTERLOCK_BASE = 300  # 32 tags x 8 regs (lolo/low/high/hihi)
PID_COUNT = 4
# Host-liveness watchdog. The firmware proves the host is alive from a CHANGE
# to this register (the value itself is meaningless), not from its content. If
# it sees neither a Modbus TCP connection nor a heartbeat change for
# HEARTBEAT_TIMEOUT_S it drives all analog outputs to 0 %, opens the heater
# relay, asserts Inhibit and holds the PID loops. The backend must therefore
# bump it once per successful scan — see AsyncModbusManager.write_heartbeat.
HOST_HEARTBEAT_REGISTER = 560
HEARTBEAT_TIMEOUT_S = 2.0  # firmware-side watchdog window

# Plant wiring: the DC power supply's analog command rides PID loop 0. Named
# here so the shutdown safe-state and the power-supply service agree on which
# loop must be zeroed to de-energize the supply.
POWER_SUPPLY_PID_INDEX = 0

# ---- Coils ------------------------------------------------------------
SAVE_TO_FLASH_COIL = 0
ESTOP_RESET_COIL = 1
HEATER_RELAY_COIL = 2  # 24 V DO -> relay -> 110 V resistive heater (temp ctrl)
# Selects the P1-04THM open-circuit (burnout) fail direction: 1 = HIGH-side
# (an open thermocouple reads full scale -> heater shuts off, fail-safe),
# 0 = LOW-side (an open reads 0 C -> looks cold). The firmware reconfigures the
# module on change; the backend re-asserts this each scan so it survives a PLC
# reboot. See temperature_integration.TemperatureService.set_burnout_high_side.
THM_BURNOUT_COIL = 3


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
