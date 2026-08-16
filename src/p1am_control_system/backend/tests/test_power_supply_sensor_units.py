"""Power-supply sensor path: units, faults, and honest setpoint reporting.

These cover three defects that all share a root cause -- the power-supply
service treated a raw broker tag as though it were already in engineering
units, and treated a missing tag as though it were a real reading of zero.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import hardware  # noqa: E402
import pytest  # noqa: E402
from _power_supply_helpers import fresh_running_controller, test_config  # noqa: E402
from power_supply import PowerSupplyState  # noqa: E402
from power_supply_integration import PowerSupplyService  # noqa: E402


def _feedback(cfg: object, *, temp_pct: float) -> dict[str, float]:
    """A complete feedback frame.

    All three feedback tags must be present -- a partial frame is now a
    SENSOR_FAULT, which is the point of #4016. Tests that are about units or
    trips supply a full frame so they exercise only the behaviour under test.
    """
    return {
        cfg.current_feedback_tag: 0.0,  # type: ignore[attr-defined]
        cfg.voltage_feedback_tag: 0.0,  # type: ignore[attr-defined]
        cfg.temp_tag: temp_pct,  # type: ignore[attr-defined]
    }


def _service(**overrides: object) -> PowerSupplyService:
    """A service whose controller uses the stable test config.

    The service builds its own controller from production defaults, so swap in
    the test config afterwards -- these tests exercise the sensor path, not the
    bench-specific defaults.
    """
    svc = PowerSupplyService(plc_client=None, logger=logging.getLogger("test"))
    svc.controller.update_config(test_config(**overrides))
    return svc


class TestTemperatureUnits:
    """#4003 -- the HH_TEMP trip compared a percentage against a degC limit."""

    def test_temperature_tag_is_scaled_from_percent_to_celsius(self) -> None:
        svc = _service()
        cfg = svc.controller.config
        # The firmware publishes thermocouples as percent of full scale.
        # 85.714 % of 1400 degC is 1200 degC.
        pct = hardware.celsius_to_percent(1200.0)
        _, _, temp_c = svc._inputs_from_tags(_feedback(cfg, temp_pct=pct))
        assert temp_c == pytest.approx(1200.0, rel=1e-6)

    def test_hh_temp_trips_at_the_configured_celsius_threshold(self) -> None:
        """The trip must be reachable from a real tag value.

        Broker tags are bounded [0, 100], so before the fix a 1200 degC
        threshold could not be crossed by ANY physically possible reading --
        the supply's only over-temperature interlock was dead code.
        """
        svc = _service(temp_alarm_max_c=1200.0)
        cfg = svc.controller.config
        svc.controller.set_permissive(True)
        svc.controller.set_current_setpoint(10.0)

        just_under = hardware.celsius_to_percent(1199.0)
        svc.controller.tick(
            *svc._inputs_from_tags(_feedback(cfg, temp_pct=just_under)),
        )
        assert "HH_TEMP" not in svc.controller.status().trips

        # Deliberately above the threshold rather than exactly on it: the
        # degC -> % -> degC round trip lands a hair under (1200 -> 85.714... ->
        # 1199.9999999999998), so an exact-equality assertion here would be
        # testing float representation, not the trip.
        over_limit = hardware.celsius_to_percent(1205.0)
        svc.controller.tick(
            *svc._inputs_from_tags(_feedback(cfg, temp_pct=over_limit)),
        )
        assert "HH_TEMP" in svc.controller.status().trips
        assert svc.controller.status().state == PowerSupplyState.TRIPPED


class TestSensorFault:
    """#4016 -- a missing or non-finite tag was fabricated as a real 0.0."""

    def test_missing_temperature_tag_raises_a_sensor_fault(self) -> None:
        svc = _service()
        svc.controller.set_permissive(True)
        svc.controller.set_current_setpoint(10.0)

        # Tag absent entirely: an unmapped/renamed route, not a cold load.
        asyncio.run(svc.poll({}))

        trips = svc.controller.status().trips
        assert "SENSOR_FAULT" in trips
        assert svc.controller.status().state == PowerSupplyState.TRIPPED

    def test_non_finite_feedback_raises_a_sensor_fault(self) -> None:
        svc = _service()
        cfg = svc.controller.config
        svc.controller.set_permissive(True)
        svc.controller.set_current_setpoint(10.0)

        asyncio.run(svc.poll(_feedback(cfg, temp_pct=float("nan"))))

        assert "SENSOR_FAULT" in svc.controller.status().trips

    def test_a_genuine_zero_reading_is_not_a_fault(self) -> None:
        """Missing data and a real zero must be distinguishable."""
        svc = _service()
        cfg = svc.controller.config
        svc.controller.set_permissive(True)
        svc.controller.set_current_setpoint(10.0)

        asyncio.run(svc.poll(_feedback(cfg, temp_pct=0.0)))

        assert "SENSOR_FAULT" not in svc.controller.status().trips


class TestSetpointAcceptance:
    """#4017 -- rejected setpoints were reported to the operator as applied."""

    def test_setpoint_rejected_while_tripped_is_not_reported_as_applied(self) -> None:
        controller = fresh_running_controller(10.0)
        # Drive a genuine over-power trip rather than poking private state, so
        # the test exercises the same latch the plant would.
        controller.tick(
            measured_current_a=100.0,
            measured_voltage_v=50.0,
            measured_temp_c=25.0,
        )
        assert controller.status().state == PowerSupplyState.TRIPPED

        applied = controller.set_current_setpoint(40.0)

        # The controller is latched at 0 A; telling the operator 40 A was
        # applied sends them off troubleshooting the load instead of the trip.
        assert applied != 40.0
        assert applied == controller.status().setpoint_a

    def test_setpoint_rejected_while_idle_is_not_reported_as_applied(self) -> None:
        controller = fresh_running_controller(10.0)
        controller.set_permissive(False)
        assert controller.status().state == PowerSupplyState.IDLE

        applied = controller.set_current_setpoint(25.0)

        assert applied == controller.status().setpoint_a

    def test_accepted_setpoint_is_reported_and_clamped(self) -> None:
        controller = fresh_running_controller(10.0)
        applied = controller.set_current_setpoint(20.0)
        assert applied == controller.status().setpoint_a == pytest.approx(20.0)
