"""Config-level validation tests for TemperatureConfig + TemperatureStatus.

Companion (`test_temperature_controller.py`) covers the state machine, setpoint
clamping, hysteresis, HH cutoff, acknowledge, and E-stop behavior.

Covers here:
    - Pydantic config validation (field constraints, cross-field invariants)
    - heater_label trimming / blank / overlong rejection
    - TemperatureStatus construction + StrEnum serialization
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import ValidationError  # noqa: E402
from temperature_models import (  # noqa: E402  (path setup above must run first)
    TcPath,
    TcType,
    TemperatureConfig,
    TemperatureState,
    TemperatureStatus,
    ThermocoupleChannel,
)

# --------------------------------------------------------------------------
# TemperatureConfig validation
# --------------------------------------------------------------------------


class TestTemperatureConfigValidation:
    def test_defaults_construct(self) -> None:
        config = TemperatureConfig()
        assert config.temp_tag == "TAG_0"
        assert config.temp_full_scale_c == 1400.0
        assert config.setpoint_min_c == 0.0
        assert config.setpoint_max_c == 1400.0
        assert config.deadband_c == 5.0
        assert config.hh_limit_c == 1400.0
        assert config.heater_label == "Heater"

    def test_full_scale_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            ThermocoupleChannel(full_scale_c=0.0)
        with pytest.raises(ValidationError):
            ThermocoupleChannel(full_scale_c=-1.0)

    def test_setpoint_min_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TemperatureConfig(setpoint_min_c=-1.0)

    def test_setpoint_max_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            TemperatureConfig(setpoint_max_c=0.0)

    def test_deadband_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            TemperatureConfig(deadband_c=0.0)
        with pytest.raises(ValidationError):
            TemperatureConfig(deadband_c=-5.0)

    def test_hh_limit_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            TemperatureConfig(hh_limit_c=0.0)

    def test_setpoint_min_cannot_exceed_max(self) -> None:
        with pytest.raises(ValidationError):
            TemperatureConfig(setpoint_min_c=600.0, setpoint_max_c=500.0)

    def test_setpoint_min_equal_to_max_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TemperatureConfig(setpoint_min_c=500.0, setpoint_max_c=500.0)

    def test_setpoint_max_cannot_exceed_active_full_scale(self) -> None:
        with pytest.raises(ValidationError):
            TemperatureConfig(
                type_k=ThermocoupleChannel(full_scale_c=1000.0, label="K"),
                setpoint_max_c=1200.0,
            )

    def test_hh_limit_cannot_exceed_active_full_scale(self) -> None:
        with pytest.raises(ValidationError):
            TemperatureConfig(
                type_k=ThermocoupleChannel(full_scale_c=1000.0, label="K"),
                hh_limit_c=1200.0,
            )

    def test_in_range_invariants_construct(self) -> None:
        cfg = TemperatureConfig(
            setpoint_min_c=50.0,
            setpoint_max_c=900.0,
            hh_limit_c=1000.0,
        )
        assert cfg.setpoint_max_c == 900.0
        assert cfg.hh_limit_c == 1000.0

    def test_heater_label_is_trimmed(self) -> None:
        assert TemperatureConfig(heater_label="  Furnace  ").heater_label == "Furnace"

    def test_blank_heater_label_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TemperatureConfig(heater_label="   ")
        with pytest.raises(ValidationError):
            TemperatureConfig(heater_label="")

    def test_overlong_heater_label_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TemperatureConfig(heater_label="x" * 41)


# --------------------------------------------------------------------------
# TemperatureStatus snapshot model
# --------------------------------------------------------------------------


class TestTemperatureStatus:
    def test_status_constructs_and_serializes(self) -> None:
        status = TemperatureStatus(
            state=TemperatureState.RUNNING,
            permissive=True,
            setpoint_c=500.0,
            measured_temp_c=480.0,
            relay_on=True,
            trips=[],
            hh_limit_c=1400.0,
            deadband_c=5.0,
        )
        assert status.state == TemperatureState.RUNNING
        assert status.estopped is False  # default
        dumped = status.model_dump()
        # StrEnum members serialize as plain strings.
        assert dumped["state"] == "running"

    def test_state_enum_values(self) -> None:
        assert TemperatureState.IDLE == "idle"
        assert TemperatureState.ARMED == "armed"
        assert TemperatureState.RUNNING == "running"
        assert TemperatureState.TRIPPED == "tripped"


# --------------------------------------------------------------------------
# Dual thermocouple (type K / type R) + active selection
# --------------------------------------------------------------------------


class TestThermocoupleChannel:
    def test_defaults(self) -> None:
        ch = ThermocoupleChannel()
        assert ch.tag == "TAG_0"
        assert ch.full_scale_c == 1400.0
        assert ch.label

    def test_full_scale_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            ThermocoupleChannel(full_scale_c=0.0)

    def test_label_trimmed_and_blank_rejected(self) -> None:
        assert ThermocoupleChannel(label="  Type R  ").label == "Type R"
        with pytest.raises(ValidationError):
            ThermocoupleChannel(label="   ")


class TestDualThermocouple:
    def test_defaults_have_k_active_on_tag0_and_r_on_tag1(self) -> None:
        cfg = TemperatureConfig()
        assert cfg.active_tc_type == TcType.TYPE_K
        assert cfg.type_k.tag == "TAG_0"
        assert cfg.type_r.tag == "TAG_1"
        # The derived accessors point at the active (K) channel.
        assert cfg.temp_tag == "TAG_0"
        assert cfg.temp_full_scale_c == 1400.0
        assert cfg.active_tc_label == cfg.type_k.label

    def test_active_accessors_follow_selection(self) -> None:
        cfg = TemperatureConfig(
            type_r=ThermocoupleChannel(tag="TAG_2", full_scale_c=1400.0, label="R"),
            active_tc_type=TcType.TYPE_R,
        )
        assert cfg.temp_tag == "TAG_2"
        assert cfg.active_tc_label == "R"
        assert cfg.active_channel is cfg.type_r

    def test_invariant_checks_the_active_channel(self) -> None:
        # hh_limit 1300 is fine against the active R channel (1400)...
        cfg = TemperatureConfig(
            type_r=ThermocoupleChannel(full_scale_c=1400.0, label="R"),
            active_tc_type=TcType.TYPE_R,
            hh_limit_c=1300.0,
        )
        assert cfg.hh_limit_c == 1300.0
        # ...but exceeding the active channel's full scale is rejected.
        with pytest.raises(ValidationError):
            TemperatureConfig(
                type_r=ThermocoupleChannel(full_scale_c=1000.0, label="R"),
                active_tc_type=TcType.TYPE_R,
                hh_limit_c=1200.0,
            )

    def test_computed_fields_serialize(self) -> None:
        dumped = TemperatureConfig(active_tc_type=TcType.TYPE_R).model_dump()
        assert dumped["active_tc_type"] == "R"
        assert dumped["temp_tag"] == "TAG_1"  # active R channel's tag
        assert dumped["active_tc_label"] == "Type R (Ch 2)"

    def test_tc_type_enum_values(self) -> None:
        assert TcType.TYPE_K == "K"
        assert TcType.TYPE_R == "R"

    def test_status_reports_active_tc(self) -> None:
        status = TemperatureStatus(
            state=TemperatureState.IDLE,
            permissive=False,
            setpoint_c=0.0,
            measured_temp_c=20.0,
            relay_on=False,
            trips=[],
            hh_limit_c=1400.0,
            deadband_c=5.0,
            active_tc_type=TcType.TYPE_R,
            active_tc_label="Type R",
        )
        assert status.active_tc_type == TcType.TYPE_R
        assert status.model_dump()["active_tc_type"] == "R"


# --------------------------------------------------------------------------
# Acquisition path (TC card vs analog conditioner) + 2x2 source matrix
# --------------------------------------------------------------------------


class TestAcquisitionPath:
    def test_tc_path_enum_values(self) -> None:
        assert TcPath.TC_CARD == "thm"
        assert TcPath.ANALOG == "analog"

    def test_defaults_use_the_tc_card_path(self) -> None:
        cfg = TemperatureConfig()
        # Backward compatible: an unspecified path keeps the original behavior.
        assert cfg.active_tc_path == TcPath.TC_CARD
        assert cfg.temp_tag == "TAG_0"  # TC-card K
        assert cfg.active_tc_label == "Type K (Ch 1)"

    def test_analog_channels_default_to_last_two_analog_inputs(self) -> None:
        cfg = TemperatureConfig()
        assert cfg.analog_k.tag == "TAG_14"  # AI2
        assert cfg.analog_r.tag == "TAG_15"  # AI3
        assert cfg.analog_k.label == "Type K (Analog)"
        assert cfg.analog_r.label == "Type R (Analog)"

    def test_channel_for_maps_all_four_sources(self) -> None:
        cfg = TemperatureConfig()
        assert cfg.channel_for(TcType.TYPE_K, TcPath.TC_CARD) is cfg.type_k
        assert cfg.channel_for(TcType.TYPE_R, TcPath.TC_CARD) is cfg.type_r
        assert cfg.channel_for(TcType.TYPE_K, TcPath.ANALOG) is cfg.analog_k
        assert cfg.channel_for(TcType.TYPE_R, TcPath.ANALOG) is cfg.analog_r

    def test_active_channel_follows_type_and_path(self) -> None:
        cfg = TemperatureConfig(
            active_tc_type=TcType.TYPE_R,
            active_tc_path=TcPath.ANALOG,
        )
        assert cfg.active_channel is cfg.analog_r
        assert cfg.temp_tag == "TAG_15"
        assert cfg.temp_full_scale_c == 1400.0
        assert cfg.active_tc_label == "Type R (Analog)"

    def test_analog_path_can_use_a_conditioner_specific_full_scale(self) -> None:
        # A conditioner scaled 0-1200 C -> 4-20 mA sets the analog channel's span.
        cfg = TemperatureConfig(
            analog_k=ThermocoupleChannel(
                tag="TAG_14", full_scale_c=1200.0, label="Type K (Analog)"
            ),
            active_tc_type=TcType.TYPE_K,
            active_tc_path=TcPath.ANALOG,
            setpoint_max_c=1100.0,
            hh_limit_c=1150.0,
        )
        assert cfg.temp_full_scale_c == 1200.0

    def test_invariants_check_the_active_analog_channel(self) -> None:
        # hh_limit above the active analog channel's full scale is rejected.
        with pytest.raises(ValidationError):
            TemperatureConfig(
                analog_k=ThermocoupleChannel(
                    tag="TAG_14", full_scale_c=1000.0, label="Type K (Analog)"
                ),
                active_tc_type=TcType.TYPE_K,
                active_tc_path=TcPath.ANALOG,
                hh_limit_c=1200.0,
            )

    def test_computed_fields_and_path_serialize(self) -> None:
        dumped = TemperatureConfig(
            active_tc_type=TcType.TYPE_R, active_tc_path=TcPath.ANALOG
        ).model_dump()
        assert dumped["active_tc_path"] == "analog"
        assert dumped["temp_tag"] == "TAG_15"
        assert dumped["active_tc_label"] == "Type R (Analog)"

    def test_status_reports_active_path(self) -> None:
        status = TemperatureStatus(
            state=TemperatureState.RUNNING,
            permissive=True,
            setpoint_c=500.0,
            measured_temp_c=480.0,
            relay_on=True,
            trips=[],
            hh_limit_c=1400.0,
            deadband_c=5.0,
            active_tc_type=TcType.TYPE_K,
            active_tc_path=TcPath.ANALOG,
            active_tc_label="Type K (Analog)",
        )
        assert status.active_tc_path == TcPath.ANALOG
        assert status.model_dump()["active_tc_path"] == "analog"
