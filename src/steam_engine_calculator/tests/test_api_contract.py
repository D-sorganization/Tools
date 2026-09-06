"""Contract tests for the Steam Engine Calculator FastAPI backend.

Covers issue #3980: the request schema must match the modal semantics of
``CalculationMode`` — ``sat_t`` is driven by temperature alone, ``sat_p``
by pressure alone and ``tp`` needs both — so clients stop inventing
placeholder values for fields a mode never reads.  Engine selection is
public engine API; the HTTP layer must not reach into private internals.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from steam_engine_calculator.api import CalculationMode, SteamRequest, app

pytestmark = pytest.mark.contract

ENDPOINT = "/api/steam/calculate"
_VALID_ENGINES = {"coolprop", "cantera", "simplified"}
_RESPONSE_KEYS = (
    "temperature",
    "pressure",
    "density",
    "specificVolume",
    "enthalpy",
    "entropy",
    "internalEnergy",
    "cp",
    "cv",
    "speedOfSound",
    "thermalConductivity",
    "dynamicViscosity",
    "kinematicViscosity",
    "quality",
    "phase",
    "compressibilityFactor",
    "prandtlNumber",
    "specificHeatRatio",
    "engine",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    """HTTP client bound to the FastAPI app under test."""
    return TestClient(app)


def _assert_full_payload(payload: dict[str, Any]) -> None:
    """Every successful response must carry the complete SteamResponse surface."""
    for key in _RESPONSE_KEYS:
        assert key in payload, f"missing response field: {key}"
    assert payload["engine"] in _VALID_ENGINES


class TestHappyPathPerMode:
    """Each mode works with exactly the fields it actually consumes."""

    def test_tp_mode(self, client: TestClient) -> None:
        response = client.post(
            ENDPOINT,
            json={"mode": "tp", "temperature": 400.0, "pressure": 101325.0},
        )
        assert response.status_code == 200
        payload = response.json()
        _assert_full_payload(payload)
        assert payload["temperature"] == pytest.approx(400.0, abs=1.0)
        assert payload["pressure"] == pytest.approx(101325.0, abs=1.0)

    def test_sat_t_needs_only_temperature(self, client: TestClient) -> None:
        response = client.post(ENDPOINT, json={"mode": "sat_t", "temperature": 373.15})
        assert response.status_code == 200
        payload = response.json()
        _assert_full_payload(payload)
        assert payload["temperature"] == pytest.approx(373.15, abs=1.0)
        # Saturation pressure at 100 °C is ~1 atm regardless of backend.
        assert 5.0e4 < payload["pressure"] < 2.0e5

    def test_sat_p_needs_only_pressure(self, client: TestClient) -> None:
        response = client.post(ENDPOINT, json={"mode": "sat_p", "pressure": 101325.0})
        assert response.status_code == 200
        payload = response.json()
        _assert_full_payload(payload)
        assert payload["pressure"] == pytest.approx(101325.0, abs=1.0)
        # Boiling temperature at 1 atm is ~100 °C regardless of backend.
        assert 350.0 < payload["temperature"] < 400.0


class TestBothPresentStillValidates:
    """Backward compatibility: clients that always send both fields keep working."""

    def test_sat_t_accepts_legacy_both_fields(self, client: TestClient) -> None:
        response = client.post(
            ENDPOINT,
            json={"mode": "sat_t", "temperature": 373.15, "pressure": 101325.0},
        )
        assert response.status_code == 200
        payload = response.json()
        _assert_full_payload(payload)
        assert payload["temperature"] == pytest.approx(373.15, abs=1.0)

    def test_sat_p_accepts_legacy_both_fields(self, client: TestClient) -> None:
        response = client.post(
            ENDPOINT,
            json={"mode": "sat_p", "temperature": 373.15, "pressure": 101325.0},
        )
        assert response.status_code == 200
        payload = response.json()
        _assert_full_payload(payload)
        assert payload["pressure"] == pytest.approx(101325.0, abs=1.0)


class TestInvalidCombosRejected:
    """Missing or wrong-field payloads produce a clear 422, never dummy echoes."""

    def test_sat_t_without_temperature_is_422(self, client: TestClient) -> None:
        response = client.post(ENDPOINT, json={"mode": "sat_t", "pressure": 101325.0})
        assert response.status_code == 422
        assert "temperature" in response.text

    def test_sat_p_without_pressure_is_422(self, client: TestClient) -> None:
        response = client.post(ENDPOINT, json={"mode": "sat_p", "temperature": 373.15})
        assert response.status_code == 422
        assert "pressure" in response.text

    def test_tp_with_no_fields_is_422(self, client: TestClient) -> None:
        response = client.post(ENDPOINT, json={"mode": "tp"})
        assert response.status_code == 422
        assert "temperature" in response.text
        assert "pressure" in response.text

    def test_tp_missing_pressure_is_422(self, client: TestClient) -> None:
        response = client.post(ENDPOINT, json={"mode": "tp", "temperature": 400.0})
        assert response.status_code == 422
        assert "pressure" in response.text

    def test_explicit_null_is_422(self, client: TestClient) -> None:
        response = client.post(
            ENDPOINT,
            json={"mode": "sat_t", "temperature": None, "pressure": 101325.0},
        )
        assert response.status_code == 422
        assert "temperature" in response.text

    def test_non_positive_temperature_is_422(self, client: TestClient) -> None:
        response = client.post(
            ENDPOINT, json={"mode": "tp", "temperature": 0.0, "pressure": 101325.0}
        )
        assert response.status_code == 422


class TestSteamRequestModel:
    """Unit-level validation contract for the request model (issue #3980)."""

    @pytest.mark.unit
    def test_sat_t_with_temperature_only_is_valid(self) -> None:
        request = SteamRequest(mode=CalculationMode.SAT_T, temperature=373.15)
        assert request.temperature == 373.15
        assert request.pressure is None

    @pytest.mark.unit
    def test_sat_p_with_pressure_only_is_valid(self) -> None:
        request = SteamRequest(mode=CalculationMode.SAT_P, pressure=101325.0)
        assert request.pressure == 101325.0
        assert request.temperature is None

    @pytest.mark.unit
    def test_tp_with_both_fields_is_valid(self) -> None:
        request = SteamRequest(
            mode=CalculationMode.TP, temperature=400.0, pressure=101325.0
        )
        assert request.temperature == 400.0
        assert request.pressure == 101325.0

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("mode", "fields", "missing"),
        [
            (CalculationMode.SAT_T, {"pressure": 101325.0}, "temperature"),
            (CalculationMode.SAT_P, {"temperature": 373.15}, "pressure"),
            (CalculationMode.TP, {}, "temperature"),
            (CalculationMode.TP, {"temperature": 400.0}, "pressure"),
            (CalculationMode.TP, {"pressure": 101325.0}, "temperature"),
        ],
    )
    def test_missing_required_field_raises(
        self,
        mode: CalculationMode,
        fields: dict[str, float],
        missing: str,
    ) -> None:
        with pytest.raises(ValidationError, match=missing):
            SteamRequest(mode=mode, **fields)
