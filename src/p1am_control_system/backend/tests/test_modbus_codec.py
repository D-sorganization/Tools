from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

import modbus_codec  # noqa: E402
from modbus_codec import (  # noqa: E402
    TAG_COUNT,
    decode_interlocks,
    decode_pid_configs,
    direct_tag_address,
    encode_interlocks,
    encode_pid_configs,
    encode_tag_indices,
    float_to_registers,
    registers_to_float,
    zero_float_registers,
)
from models import InterlockConfig, PIDConfig  # noqa: E402


class TestFloatRegisters:
    def test_round_trip_finite_float(self) -> None:
        low, high = float_to_registers(12.5)
        assert registers_to_float(low, high) == pytest.approx(12.5)

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_writes(self, value: float) -> None:
        with pytest.raises(ValueError):
            float_to_registers(value)

    def test_rejects_out_of_range_read_registers(self) -> None:
        with pytest.raises(ValueError):
            registers_to_float(-1, 0)


class TestRoutingCodec:
    def test_pid_configs_round_trip(self) -> None:
        pids = [
            PIDConfig(
                pv_tag=f"TAG_{idx}",
                cv_tag=f"TAG_{idx + 10}",
                setpoint=idx + 0.25,
                kp=1.0,
                ki=0.1,
                kd=0.01,
            )
            for idx in range(4)
        ]
        decoded = decode_pid_configs(encode_pid_configs(pids))
        assert [pid.pv_tag for pid in decoded] == [pid.pv_tag for pid in pids]
        assert [pid.cv_tag for pid in decoded] == [pid.cv_tag for pid in pids]
        assert [pid.setpoint for pid in decoded] == pytest.approx(
            [pid.setpoint for pid in pids]
        )

    def test_interlocks_round_trip_and_default_fill(self) -> None:
        interlocks = {
            "TAG_3": InterlockConfig(
                lolo_limit=0.0,
                low_limit=2.0,
                high_limit=90.0,
                hihi_limit=100.0,
            )
        }
        decoded = decode_interlocks(encode_interlocks(interlocks))
        assert len(decoded) == TAG_COUNT
        assert decoded["TAG_3"].hihi_limit == pytest.approx(100.0)
        assert decoded["TAG_0"].low_limit is None
        assert decoded["TAG_0"].is_disabled()

    def test_tag_index_encoding_valid_names(self) -> None:
        assert encode_tag_indices(["TAG_1", "TAG_0", "TAG_7"]) == [1, 0, 7]

    def test_tag_index_encoding_passes_unmapped_sentinel(self) -> None:
        assert encode_tag_indices(["TAG_255", "TAG_0", "TAG_31"]) == [255, 0, 31]

    def test_pid_config_encoding_passes_unmapped_sentinel(self) -> None:
        registers = encode_pid_configs(
            [
                PIDConfig(
                    pv_tag="TAG_255",
                    cv_tag="TAG_255",
                    setpoint=0.0,
                    kp=0.0,
                    ki=0.0,
                    kd=0.0,
                )
            ]
        )

        assert registers[:2] == [255, 255]
        decoded = decode_pid_configs(registers + [0] * 30)
        assert decoded[0].pv_tag == "TAG_255"
        assert decoded[0].cv_tag == "TAG_255"

    def test_tag_index_encoding_rejects_non_sentinel_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            encode_tag_indices(["TAG_254"])

    def test_tag_index_encoding_rejects_malformed_name(self) -> None:
        # A malformed tag must raise, not be silently coerced to TAG_0 (#3531).
        with pytest.raises(ValueError):
            encode_tag_indices(["TAG_1", "bad", "TAG_7"])

    def test_tag_index_encoding_rejects_non_integer_parser_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(modbus_codec.hardware, "tag_index", lambda _name: True)
        with pytest.raises(TypeError, match="tag index must be an int"):
            modbus_codec.tag_to_index("TAG_1")

    def test_zero_float_registers_returns_register_pair_per_tag(self) -> None:
        assert zero_float_registers(3) == [0, 0, 0, 0, 0, 0]


class TestDirectTagAddress:
    def test_direct_tag_address_uses_tag_index(self) -> None:
        assert direct_tag_address("TAG_5") == 10

    def test_direct_tag_address_uses_dynamic_v_register(self) -> None:
        tag_map = {"PumpSpeed": SimpleNamespace(register_type="V", register_num=42)}
        assert direct_tag_address("PumpSpeed", tag_map) == 42

    def test_direct_tag_address_rejects_unknown_tags(self) -> None:
        assert direct_tag_address("TAG_99") is None
