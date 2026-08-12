"""Tests for the selectable thermocouple SOURCE (type x acquisition path).

Each physical thermocouple (K/R) can be read two ways: straight into the
P1-04THM card, or through an external 4-20 mA signal conditioner into an analog
input. This module covers the integration + router behavior added for that 2x2
selection; the pure config/controller behavior lives in test_temperature_models
and test_temperature_controller.

Covers:
    - poll() scales the analog-path tags (TAG_14/TAG_15) into deg C and controls
      on them when the analog path is active.
    - The published type_k_temp_c / type_r_temp_c pair follows the ACTIVE path.
    - The HH-on-either backstop uses the active path's OTHER sensor.
    - The /tc_type endpoint accepts an optional active_tc_path (and defaults it
      to the TC-card path for backward compatibility).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from temperature_integration import (  # noqa: E402
    TemperatureService,
    create_temperature_router,
)
from temperature_models import (  # noqa: E402
    TcPath,
    TcType,
    TemperatureState,
)


class _FakePLC:
    """Minimal PLC double — only the coil seam TemperatureService touches."""

    def __init__(self) -> None:
        self.write_coil = AsyncMock(return_value=True)


def _service() -> TemperatureService:
    return TemperatureService(_FakePLC(), logging.getLogger("test"))


def _running_on(svc: TemperatureService, tc_type: TcType, tc_path: TcPath) -> None:
    """Select a source, then arm + run at a mid-range setpoint (deterministic)."""
    svc.controller.set_active_source(tc_type, tc_path)
    svc.controller.set_permissive(True)
    svc.controller.set_setpoint_c(700.0)  # ARMED -> RUNNING


# All four sources map to their AI/TC tags; poll() scales percent -> deg C.
FOUR_TAGS = {"TAG_0": 10.0, "TAG_1": 20.0, "TAG_14": 30.0, "TAG_15": 40.0}


# --------------------------------------------------------------------------
# Analog-path scaling + control
# --------------------------------------------------------------------------


class TestAnalogPathControl:
    def test_controls_on_analog_k_tag(self) -> None:
        async def _go() -> None:
            svc = _service()
            _running_on(svc, TcType.TYPE_K, TcPath.ANALOG)
            # Analog K (FC-T1 -150..1372 C) at 0 % -> -150 C, well below the band.
            status = await svc.poll({"TAG_14": 0.0, "TAG_15": 0.0})
            assert status.relay_on is True
            assert status.active_tc_path == TcPath.ANALOG
            assert status.measured_temp_c == pytest.approx(-150.0)

        asyncio.run(_go())

    def test_analog_tag_scales_percent_to_celsius(self) -> None:
        async def _go() -> None:
            svc = _service()
            svc.controller.set_active_source(TcType.TYPE_R, TcPath.ANALOG)
            # Analog R (FC-T1 65..1768 C) at 50 % -> 65 + 0.5*1703 = 916.5 C.
            status = await svc.poll({"TAG_15": 50.0})
            assert status.measured_temp_c == pytest.approx(916.5)

        asyncio.run(_go())

    def test_ai0_ai1_are_never_read_as_thermocouples(self) -> None:
        # The analog TCs live on the LAST two AIs (TAG_14/15); the PSU V/I on
        # AI0/AI1 (TAG_12/13) must never be mistaken for a temperature.
        async def _go() -> None:
            svc = _service()
            svc.controller.set_active_source(TcType.TYPE_K, TcPath.ANALOG)
            status = await svc.poll({"TAG_12": 99.0, "TAG_13": 99.0, "TAG_14": 0.0})
            # Reads TAG_14 (0 % -> -150 C), NOT the 99 % on the PSU tags.
            assert status.measured_temp_c == pytest.approx(-150.0)

        asyncio.run(_go())


# --------------------------------------------------------------------------
# Published pair follows the active path
# --------------------------------------------------------------------------


class TestPublishedPairFollowsPath:
    def test_tc_card_path_publishes_thm_tags(self) -> None:
        async def _go() -> None:
            svc = _service()  # default path = TC card
            status = await svc.poll(dict(FOUR_TAGS))
            assert status.type_k_temp_c == pytest.approx(140.0)  # TAG_0 10%
            assert status.type_r_temp_c == pytest.approx(280.0)  # TAG_1 20%

        asyncio.run(_go())

    def test_analog_path_publishes_analog_tags(self) -> None:
        async def _go() -> None:
            svc = _service()
            svc.controller.set_active_source(TcType.TYPE_K, TcPath.ANALOG)
            status = await svc.poll(dict(FOUR_TAGS))
            # Analog K 30 %: -150 + 0.3*1522 = 306.6 ; R 40 %: 65 + 0.4*1703 = 746.2
            assert status.type_k_temp_c == pytest.approx(306.6)  # TAG_14 30%
            assert status.type_r_temp_c == pytest.approx(746.2)  # TAG_15 40%

        asyncio.run(_go())

    def test_switching_path_immediately_reads_the_new_tag(self) -> None:
        async def _go() -> None:
            svc = _service()
            await svc.poll(dict(FOUR_TAGS))  # warm all four filters (TC-card path)
            svc.controller.set_active_source(TcType.TYPE_K, TcPath.ANALOG)
            status = await svc.poll(dict(FOUR_TAGS))
            # No transient hold: the analog-K filter was already warm.
            assert status.measured_temp_c == pytest.approx(306.6)  # TAG_14 30%

        asyncio.run(_go())


# --------------------------------------------------------------------------
# Safety backstop uses the active path's OTHER sensor
# --------------------------------------------------------------------------


class TestCrossPathSafety:
    def test_hh_on_analog_other_sensor_trips(self) -> None:
        # Controlling analog K reads cold, but the analog R on the SAME path is
        # at/over HH -> the HH-on-either backstop must still latch.
        async def _go() -> None:
            svc = _service()
            _running_on(svc, TcType.TYPE_K, TcPath.ANALOG)
            status = await svc.poll({"TAG_14": 0.0, "TAG_15": 100.0})  # R -> 1400 C
            assert "HH_TEMP" in status.trips
            assert status.relay_on is False
            assert status.state == TemperatureState.TRIPPED

        asyncio.run(_go())

    def test_thm_other_sensor_not_used_on_analog_path(self) -> None:
        # A hot TC-card R (other path) must NOT trip HH while the analog path is
        # active — only the active path's sensors back each other up.
        async def _go() -> None:
            svc = _service()
            _running_on(svc, TcType.TYPE_K, TcPath.ANALOG)
            status = await svc.poll(
                {"TAG_14": 0.0, "TAG_15": 0.0, "TAG_1": 100.0}  # TC-card R hot
            )
            assert "HH_TEMP" not in status.trips
            assert status.relay_on is True

        asyncio.run(_go())


# --------------------------------------------------------------------------
# Router: /tc_type gains an optional path
# --------------------------------------------------------------------------


@pytest.fixture
def client_and_service(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, TemperatureService]:
    monkeypatch.setenv("P1AM_DEV_NO_AUTH", "1")
    svc = _service()
    app = FastAPI()
    app.include_router(create_temperature_router(svc))
    return TestClient(app), svc


class TestSourceEndpoint:
    def test_endpoint_selects_analog_r(
        self, client_and_service: tuple[TestClient, TemperatureService]
    ) -> None:
        client, svc = client_and_service
        resp = client.post(
            "/api/temperature/tc_type",
            json={"active_tc_type": "R", "active_tc_path": "analog"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["active_tc_type"] == "R"
        assert body["active_tc_path"] == "analog"
        assert svc.controller.config.temp_tag == "TAG_15"

    def test_endpoint_defaults_path_to_tc_card(
        self, client_and_service: tuple[TestClient, TemperatureService]
    ) -> None:
        # Backward compatible: a K/R-only client omits the path.
        client, svc = client_and_service
        resp = client.post("/api/temperature/tc_type", json={"active_tc_type": "R"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["active_tc_path"] == "thm"
        assert svc.controller.config.temp_tag == "TAG_1"  # TC-card R

    def test_endpoint_rejects_bad_path(
        self, client_and_service: tuple[TestClient, TemperatureService]
    ) -> None:
        client, _ = client_and_service
        resp = client.post(
            "/api/temperature/tc_type",
            json={"active_tc_type": "K", "active_tc_path": "nope"},
        )
        assert resp.status_code == 422  # pydantic enum validation
