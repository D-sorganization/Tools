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
TAG_COUNT: int = hardware.TAG_COUNT
UNMAPPED_TAG_INDEX: int = hardware.UNMAPPED_TAG_INDEX
UNMAPPED_TAG_NAME: str = hardware.UNMAPPED_TAG_NAME
PID_COUNT: int = hardware.PID_COUNT
PID_REGISTER_WIDTH = 10
INTERLOCK_REGISTER_WIDTH = 8
INTERLOCK_CHUNK_OFFSETS = (0, 64, 128, 192)
# A tag absent from the config is NOT interlocked. Encoding it as a live band
# (the old 0/5/95/100) tripped the firmware on every unrouted tag (#4001).
DEFAULT_INTERLOCK = InterlockConfig()


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
    """Return the PLC routing/PID index from ``TAG_n``.

    The firmware uses ``TAG_255`` as the kUnmappedTag sentinel in routing and
    PID pv/cv fields after an all-default boot. That exact sentinel is a valid
    register value here even though it is not a broker tag. All other names
    delegate to the strict parser (``hardware.tag_index``): malformed or
    out-of-range names raise ``ValueError`` rather than silently coercing to
    ``0`` — on a control system, quietly routing a bad tag onto ``TAG_0`` is
    unsafe (issue #3531). ``write_routing`` treats any raised error as a
    refused write.

    Raises:
        TypeError: If ``tag_name`` is not a str.
        ValueError: If ``tag_name`` is neither ``TAG_255`` nor a well-formed
            in-range ``TAG_<n>``.
    """
    if tag_name == UNMAPPED_TAG_NAME:
        return UNMAPPED_TAG_INDEX
    index = hardware.tag_index(tag_name)
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError(f"tag index must be an int, got {type(index).__name__}")
    return index


def encode_tag_indices(tag_names: list[str]) -> list[int]:
    """Encode routing tag names into PLC tag indices.

    Raises:
        ValueError: If any name is neither the unmapped sentinel nor a
            well-formed in-range ``TAG_<n>`` — a bad routing config must be
            refused, not silently mapped to TAG_0.
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


def encode_low_limit(limit: float | None) -> list[int]:
    """Encode a low-side limit; ``None`` becomes the firmware's disabled sentinel."""
    value = hardware.INTERLOCK_DISABLED_LOW if limit is None else limit
    return float_to_registers(value)


def encode_high_limit(limit: float | None) -> list[int]:
    """Encode a high-side limit; ``None`` becomes the firmware's disabled sentinel."""
    value = hardware.INTERLOCK_DISABLED_HIGH if limit is None else limit
    return float_to_registers(value)


def decode_low_limit(low: int, high: int) -> float | None:
    """Decode a low-side limit; at/below the disabled sentinel reads as ``None``."""
    value = registers_to_float(low, high)
    if not math.isfinite(value) or value <= hardware.INTERLOCK_DISABLED_LOW:
        return None
    return value


def decode_high_limit(low: int, high: int) -> float | None:
    """Decode a high-side limit; at/above the disabled sentinel reads as ``None``."""
    value = registers_to_float(low, high)
    if not math.isfinite(value) or value >= hardware.INTERLOCK_DISABLED_HIGH:
        return None
    return value


def decode_interlocks(registers: list[int]) -> dict[str, InterlockConfig]:
    """Decode the 256-register interlock block into per-tag limits.

    The firmware's disabled sentinels (and any non-finite register garbage)
    decode to ``None`` rather than a number the alarm engine would compare
    against (#3973).
    """
    interlocks: dict[str, InterlockConfig] = {}
    for tag_index in range(TAG_COUNT):
        base = tag_index * INTERLOCK_REGISTER_WIDTH
        interlocks[f"TAG_{tag_index}"] = InterlockConfig(
            lolo_limit=decode_low_limit(registers[base], registers[base + 1]),
            low_limit=decode_low_limit(registers[base + 2], registers[base + 3]),
            high_limit=decode_high_limit(registers[base + 4], registers[base + 5]),
            hihi_limit=decode_high_limit(registers[base + 6], registers[base + 7]),
        )
    return interlocks


def encode_interlocks(interlocks: dict[str, InterlockConfig]) -> list[int]:
    """Encode interlocks into the PLC's four-float-per-tag layout."""
    registers: list[int] = []
    for tag_index in range(TAG_COUNT):
        interlock = interlocks.get(f"TAG_{tag_index}", DEFAULT_INTERLOCK)
        registers.extend(encode_low_limit(interlock.lolo_limit))
        registers.extend(encode_low_limit(interlock.low_limit))
        registers.extend(encode_high_limit(interlock.high_limit))
        registers.extend(encode_high_limit(interlock.hihi_limit))
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
