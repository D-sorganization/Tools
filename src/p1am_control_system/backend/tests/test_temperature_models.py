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
    TemperatureConfig,
    TemperatureState,
    TemperatureStatus,
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
            TemperatureConfig(temp_full_scale_c=0.0)
        with pytest.raises(ValidationError):
            TemperatureConfig(temp_full_scale_c=-1.0)

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

    def test_setpoint_max_cannot_exceed_full_scale(self) -> None:
        with pytest.raises(ValidationError):
            TemperatureConfig(temp_full_scale_c=1000.0, setpoint_max_c=1200.0)

    def test_hh_limit_cannot_exceed_full_scale(self) -> None:
        with pytest.raises(ValidationError):
            TemperatureConfig(temp_full_scale_c=1000.0, hh_limit_c=1200.0)

    def test_in_range_invariants_construct(self) -> None:
        cfg = TemperatureConfig(
            temp_full_scale_c=1400.0,
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
