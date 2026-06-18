"""Pure Modbus register encoding helpers for the P1AM backend.

The transport client owns locks, retries, and request sequencing. This module
owns the deterministic register contract so it can be tested without a PLC.
"""

from __future__ import annotations

import math
import struct
from typing import Any

import hardware
from models import InterlockConfig, PIDConfig

# Single source of truth lives in ``hardware`` (the firmware contract). These
# names are re-exported so existing ``modbus_codec`` importers keep working
# without a second, drift-prone literal (issue #3531).
TAG_COUNT = hardware.TAG_COUNT
PID_COUNT = hardware.PID_COUNT
PID_REGISTER_WIDTH = 10
INTERLOCK_REGISTER_WIDTH = 8
INTERLOCK_CHUNK_OFFSETS = (0, 64, 128, 192)
DEFAULT_INTERLOCK = InterlockConfig(
    lolo_limit=0.0,
    low_limit=5.0,
    high_limit=95.0,
    hihi_limit=100.0,
)


def registers_to_float(low: int, high: int) -> float:
    """Convert two 16-bit registers to a 32-bit float (IEEE-754).

    Raises:
        TypeError: If low/high are not ints.
        ValueError: If low/high are outside [0, 65535].
    """
    if not isinstance(low, int) or not isinstance(high, int):
        raise TypeError("registers must be ints")
    if not (0 <= low <= 0xFFFF and 0 <= high <= 0xFFFF):
        raise ValueError(f"registers out of 16-bit range: {low}, {high}")
    return float(struct.unpack("<f", struct.pack("<HH", low, high))[0])


def float_to_registers(val: float) -> list[int]:
    """Convert a finite numeric value to two 16-bit float registers."""
    if not isinstance(val, int | float) or isinstance(val, bool):
        raise TypeError(f"val must be numeric, got {type(val).__name__}")
    if not math.isfinite(val):
        raise ValueError(f"val must be finite, got {val}")
    return list(struct.unpack("<HH", struct.pack("<f", float(val))))


def tag_to_index(tag_name: str) -> int:
    """Return the numeric index from ``TAG_n``.

    Delegates to the single strict parser (``hardware.tag_index``). A malformed
    or out-of-range name raises ``ValueError`` rather than silently coercing to
    ``0`` — on a control system, quietly routing a bad tag onto ``TAG_0`` is
    unsafe (issue #3531). The encoders below run on validated ``RoutingConfig``
    tags, and ``write_routing`` treats any raised error as a refused write.

    Raises:
        TypeError: If ``tag_name`` is not a str.
        ValueError: If ``tag_name`` is not a well-formed in-range ``TAG_<n>``.
    """
    index = hardware.tag_index(tag_name)
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError(f"tag index must be an int, got {type(index).__name__}")
    return index


def encode_tag_indices(tag_names: list[str]) -> list[int]:
    """Encode routing tag names into PLC tag indices.

    Raises:
        ValueError: If any name is not a well-formed in-range ``TAG_<n>`` —
            a bad routing config must be refused, not silently mapped to TAG_0.
    """
    return [tag_to_index(tag_name) for tag_name in tag_names]


def decode_pid_configs(registers: list[int]) -> list[PIDConfig]:
    """Decode four PID configurations from the 40-register PLC block."""
    pids: list[PIDConfig] = []
    for pid_index in range(PID_COUNT):
        base = pid_index * PID_REGISTER_WIDTH
        pv = registers[base]
        cv = registers[base + 1]
        pids.append(
            PIDConfig(
                pv_tag=f"TAG_{pv}",
                cv_tag=f"TAG_{cv}",
                setpoint=registers_to_float(registers[base + 2], registers[base + 3]),
                kp=registers_to_float(registers[base + 4], registers[base + 5]),
                ki=registers_to_float(registers[base + 6], registers[base + 7]),
                kd=registers_to_float(registers[base + 8], registers[base + 9]),
            )
        )
    return pids


def encode_pid_configs(pids: list[PIDConfig]) -> list[int]:
    """Encode PID configurations into the PLC's 10-register-per-loop layout."""
    registers: list[int] = []
    for pid in pids:
        registers.append(tag_to_index(pid.pv_tag))
        registers.append(tag_to_index(pid.cv_tag))
        registers.extend(float_to_registers(pid.setpoint))
        registers.extend(float_to_registers(pid.kp))
        registers.extend(float_to_registers(pid.ki))
        registers.extend(float_to_registers(pid.kd))
    return registers


def decode_interlocks(registers: list[int]) -> dict[str, InterlockConfig]:
    """Decode the 256-register interlock block into per-tag limits."""
    interlocks: dict[str, InterlockConfig] = {}
    for tag_index in range(TAG_COUNT):
        base = tag_index * INTERLOCK_REGISTER_WIDTH
        interlocks[f"TAG_{tag_index}"] = InterlockConfig(
            lolo_limit=registers_to_float(registers[base], registers[base + 1]),
            low_limit=registers_to_float(registers[base + 2], registers[base + 3]),
            high_limit=registers_to_float(registers[base + 4], registers[base + 5]),
            hihi_limit=registers_to_float(registers[base + 6], registers[base + 7]),
        )
    return interlocks


def encode_interlocks(interlocks: dict[str, InterlockConfig]) -> list[int]:
    """Encode interlocks into the PLC's four-float-per-tag layout."""
    registers: list[int] = []
    for tag_index in range(TAG_COUNT):
        interlock = interlocks.get(f"TAG_{tag_index}", DEFAULT_INTERLOCK)
        registers.extend(float_to_registers(interlock.lolo_limit))
        registers.extend(float_to_registers(interlock.low_limit))
        registers.extend(float_to_registers(interlock.high_limit))
        registers.extend(float_to_registers(interlock.hihi_limit))
    return registers


def zero_float_registers(count: int) -> list[int]:
    """Return ``count`` zero-valued float register pairs."""
    if count < 0:
        raise ValueError("count must be non-negative")
    registers: list[int] = []
    for _ in range(count):
        registers.extend(float_to_registers(0.0))
    return registers


def direct_tag_address(
    tag_name: str, tag_map: dict[str, Any] | None = None
) -> int | None:
    """Resolve a direct tag write address from TAG_n or a dynamic V-register map."""
    if tag_name.startswith("TAG_"):
        try:
            tag_idx = int(tag_name.split("_")[1])
        except (ValueError, IndexError):
            tag_idx = None
        if tag_idx is not None and 0 <= tag_idx < TAG_COUNT:
            return tag_idx * 2

    if tag_map and tag_name in tag_map:
        tag_def = tag_map[tag_name]
        if tag_def.register_type == "V" and tag_def.register_num is not None:
            return int(tag_def.register_num)
    return None
