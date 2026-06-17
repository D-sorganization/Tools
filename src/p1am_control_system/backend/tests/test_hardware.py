"""Unit tests for the hardware-contract constants and helpers.

hardware.py has no DB/PLC imports, so these run in CI without importorskip —
real, gating coverage of the tag/register contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import hardware  # noqa: E402


class TestTagName:
    def test_roundtrip(self) -> None:
        for i in (0, 1, 5, hardware.TAG_COUNT - 1):
            assert hardware.tag_index(hardware.tag_name(i)) == i

    def test_format(self) -> None:
        assert hardware.tag_name(12) == "TAG_12"

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            hardware.tag_name(hardware.TAG_COUNT)
        with pytest.raises(ValueError):
            hardware.tag_name(-1)

    def test_rejects_non_int(self) -> None:
        with pytest.raises(TypeError):
            hardware.tag_name("3")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            hardware.tag_name(True)  # bool is not a valid index


class TestTagIndex:
    def test_parses(self) -> None:
        assert hardware.tag_index("TAG_0") == 0
        assert hardware.tag_index("TAG_31") == 31

    def test_rejects_bad_prefix(self) -> None:
        with pytest.raises(ValueError):
            hardware.tag_index("FOO_1")

    def test_rejects_non_numeric(self) -> None:
        with pytest.raises(ValueError):
            hardware.tag_index("TAG_x")

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            hardware.tag_index(f"TAG_{hardware.TAG_COUNT}")

    def test_rejects_non_str(self) -> None:
        with pytest.raises(TypeError):
            hardware.tag_index(5)  # type: ignore[arg-type]


class TestPidSetpointAddress:
    def test_matches_contract(self) -> None:
        # PID 0 setpoint at base 200 + 0 + offset 2 = 202; PID 1 at 212.
        assert hardware.pid_setpoint_address(0) == 202
        assert hardware.pid_setpoint_address(1) == 212

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            hardware.pid_setpoint_address(hardware.PID_COUNT)
