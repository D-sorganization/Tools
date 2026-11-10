"""Oracle 6 tests: syngas non-finite typed failure (Issue #6003 Slice 5)."""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("calc_backend", reason="calc_backend not installed")

from calc_backend.app import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from shared.python.calc_backend.contracts.syngas_water import (  # noqa: E402
    SyngasCalculationStatus,
    SyngasNumericalFailureError,
    SyngasWaterRequest,
)
from shared.python.calc_backend.routers.syngas_water import (  # noqa: E402
    _fallback_calculate,
    _require_finite,
    _sanitize,
)

client = TestClient(app)
PAYLOAD = {"temperature_c": 40.0, "pressure_bar": 30.0}


def _check_422(resp: Any, expected_field: str) -> None:
    assert resp.status_code == 422
    d = resp.json()["detail"]
    assert d["status"] == "NUMERICAL_FAILURE" and d["code"] == "NON_FINITE_OUTPUT"
    assert d["field"] == expected_field and "non-finite" in d["message"].lower()


def test_sanitize_and_require_finite_valid() -> None:
    assert _require_finite(42.5, "temp") == 42.5
    assert _sanitize(0.0, "zero") == 0.0 and _sanitize(-15.2, "neg") == -15.2


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_sanitize_rejects_non_finite(bad_val: float) -> None:
    with pytest.raises(SyngasNumericalFailureError) as exc_info:
        _sanitize(bad_val, "mole_fraction_water")
    err = exc_info.value
    assert err.status == SyngasCalculationStatus.NUMERICAL_FAILURE
    assert err.field_name == "mole_fraction_water"
    assert math.isnan(err.value) if math.isnan(bad_val) else err.value == bad_val
    assert "mole_fraction_water" in str(err) and "non-finite" in str(err).lower()


def test_syngas_water_endpoint_success() -> None:
    resp = client.post("/api/calc/syngas-water", json=PAYLOAD)
    assert resp.status_code == 200 and resp.json()["status"] == "SUCCEEDED"


def test_syngas_water_endpoint_rejects_nan_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calc = MagicMock()
    calc.calculate_water_content.return_value = MagicMock(
        mole_fraction_water=float("nan"), dew_point_margin_c=10.0
    )
    monkeypatch.setattr(
        "shared.python.sidekick.process_calculators.syngas_water_calculator.SyngasWaterCalculator",
        lambda: calc,
    )
    _check_422(
        client.post("/api/calc/syngas-water", json=PAYLOAD), "mole_fraction_water"
    )


def test_syngas_water_endpoint_rejects_inf_risk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "shared.python.sidekick.process_calculators.syngas_water_calculator.estimate_condensation_risk",
        lambda t, p: {
            "temperature_margin_c": float("inf"),
            "condensation_risk": "High",
            "recommended_temperature_c": 50.0,
        },
    )
    _check_422(
        client.post("/api/calc/syngas-water", json=PAYLOAD), "temperature_margin_c"
    )


def test_fallback_calculate_rejects_non_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "shared.python.sidekick.process_calculators.water_vapor_pressure.antoine_pressure_pa",
        lambda a, b, c, t: float("nan"),
    )
    with pytest.raises(SyngasNumericalFailureError) as exc_info:
        _fallback_calculate(SyngasWaterRequest(**PAYLOAD))
    assert exc_info.value.status == SyngasCalculationStatus.NUMERICAL_FAILURE
    assert exc_info.value.field_name == "vapor_pressure_bar"
