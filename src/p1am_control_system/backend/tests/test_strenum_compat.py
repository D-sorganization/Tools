"""Regression tests for shared StrEnum compatibility in backend models."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import power_supply_models  # noqa: E402
import signal_stats  # noqa: E402
import temperature_models  # noqa: E402

from shared.python.compatibility import StrEnum  # noqa: E402


def test_backend_modules_share_canonical_strenum_helper() -> None:
    assert signal_stats.StrEnum is StrEnum
    assert power_supply_models.StrEnum is StrEnum
    assert temperature_models.StrEnum is StrEnum


def test_backend_strenum_models_still_behave_like_strings() -> None:
    assert signal_stats.NoiseMetric.STD == "std"
    assert power_supply_models.PowerSupplyMode.CURRENT == "current"
    assert temperature_models.TemperatureState.RUNNING == "running"

    status = temperature_models.TemperatureStatus(
        state=temperature_models.TemperatureState.RUNNING,
        permissive=True,
        setpoint_c=500.0,
        measured_temp_c=480.0,
        relay_on=True,
        trips=[],
        hh_limit_c=1400.0,
        deadband_c=5.0,
    )
    assert status.model_dump()["state"] == "running"
