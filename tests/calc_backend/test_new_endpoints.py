"""Tests for new calc backend endpoints: syngas-water, thermal-profile, ode-solver.

Uses FastAPI TestClient so no running server is needed.
See issues #608.
"""

from __future__ import annotations

import pytest
from calc_backend.app import app
from fastapi.testclient import TestClient

client = TestClient(app)


# ---------------------------------------------------------------------------
# Syngas Water
# ---------------------------------------------------------------------------


class TestSyngasWaterEndpoint:
    """POST /api/calc/syngas-water"""

    PAYLOAD = {
        "temperature_c": 40.0,
        "pressure_bar": 30.0,
        "composition_key": "typical_syngas",
        "method": "auto",
    }

    def test_syngas_water_success(self) -> None:
        resp = client.post("/api/calc/syngas-water", json=self.PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()

        wc = data["water_content"]
        assert wc["mole_fraction_water"] > 0
        assert wc["water_content_ppmv"] > 0
        assert wc["vapor_pressure_bar"] > 0
        assert wc["dew_point_c"] is not None

        risk = data["risk_assessment"]
        assert risk["condensation_risk"] in [
            "Low",
            "Medium",
            "High",
            "Critical - Condensation Occurring",
        ]

    def test_syngas_water_all_presets(self) -> None:
        for key in [
            "typical_syngas",
            "biomass_syngas",
            "coal_syngas",
            "natural_gas_reforming",
        ]:
            payload = {**self.PAYLOAD, "composition_key": key}
            resp = client.post("/api/calc/syngas-water", json=payload)
            assert resp.status_code == 200

    def test_syngas_water_all_methods(self) -> None:
        for method in ["auto", "antoine", "buck", "magnus"]:
            payload = {**self.PAYLOAD, "method": method}
            resp = client.post("/api/calc/syngas-water", json=payload)
            assert resp.status_code == 200

    def test_syngas_water_high_temp_no_error(self) -> None:
        """High temperature should not cause JSON serialization errors."""
        payload = {**self.PAYLOAD, "temperature_c": 200.0}
        resp = client.post("/api/calc/syngas-water", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        # Should succeed without NaN errors
        assert data["risk_assessment"]["condensation_risk"] in [
            "Low",
            "Medium",
            "High",
            "Critical - Condensation Occurring",
        ]


# ---------------------------------------------------------------------------
# Thermal Profile
# ---------------------------------------------------------------------------


class TestThermalProfileEndpoint:
    """POST /api/calc/thermal-profile"""

    PAYLOAD = {
        "initial_temp_c": 25.0,
        "ambient_temp_c": 25.0,
        "thermal_mass_j_per_k": 50000.0,
        "heat_loss_coeff_w_per_k": 50.0,
        "power_w": 5000.0,
        "power_profile": "constant",
        "t_start_s": 0.0,
        "t_end_s": 3600.0,
        "num_points": 50,
    }

    def test_thermal_profile_success(self) -> None:
        resp = client.post("/api/calc/thermal-profile", json=self.PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()

        assert len(data["data"]) == 50
        assert data["final_temp_c"] > 25.0  # Should heat up
        assert data["max_temp_c"] >= data["final_temp_c"]
        assert data["min_temp_c"] <= data["final_temp_c"]
        assert data["temp_change_c"] > 0

    def test_thermal_profile_constant_steady_state(self) -> None:
        """Constant power should approach P/h + T_amb."""
        resp = client.post("/api/calc/thermal-profile", json=self.PAYLOAD)
        data = resp.json()

        expected_ss = 5000.0 / 50.0 + 25.0  # = 125 C
        assert data["steady_state_temp_c"] == pytest.approx(expected_ss, rel=0.01)
        assert data["time_constant_s"] == pytest.approx(1000.0, rel=0.01)

    def test_thermal_profile_step_function(self) -> None:
        payload = {
            **self.PAYLOAD,
            "power_profile": "step",
            "step_time_s": 1800.0,
        }
        resp = client.post("/api/calc/thermal-profile", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        # After step, temp should start decreasing
        assert data["steady_state_temp_c"] is None  # No steady state for step

    def test_thermal_profile_ramp(self) -> None:
        payload = {
            **self.PAYLOAD,
            "power_profile": "linear_ramp",
            "ramp_rate_w_per_s": 1.0,
        }
        resp = client.post("/api/calc/thermal-profile", json=payload)
        assert resp.status_code == 200

    def test_thermal_profile_monotonic_heating(self) -> None:
        """With constant power and no initial overshoot, temps should increase monotonically."""
        resp = client.post("/api/calc/thermal-profile", json=self.PAYLOAD)
        data = resp.json()
        temps = [pt["temperature_c"] for pt in data["data"]]
        for i in range(1, len(temps)):
            assert temps[i] >= temps[i - 1] - 0.001  # Allow tiny float error


# ---------------------------------------------------------------------------
# ODE Solver
# ---------------------------------------------------------------------------


class TestODESolverEndpoint:
    """POST /api/calc/ode-solver"""

    def test_exponential_decay(self) -> None:
        payload = {
            "derivatives": {"y": "-k*y"},
            "parameters": {"k": 0.1},
            "initial_conditions": {"y": 100.0},
            "t_start": 0.0,
            "t_end": 50.0,
            "num_points": 100,
        }
        resp = client.post("/api/calc/ode-solver", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["success"] is True
        assert len(data["times"]) == 100
        assert "y" in data["solutions"]

        # y(50) = 100 * exp(-0.1*50) = 100 * exp(-5) ~= 0.674
        import math

        expected_final = 100 * math.exp(-5)
        assert data["solutions"]["y"][-1] == pytest.approx(expected_final, rel=0.01)

    def test_harmonic_oscillator(self) -> None:
        payload = {
            "derivatives": {"x": "v", "v": "-omega*omega*x"},
            "parameters": {"omega": 1.0},
            "initial_conditions": {"x": 1.0, "v": 0.0},
            "t_start": 0.0,
            "t_end": 6.28,
            "num_points": 200,
        }
        resp = client.post("/api/calc/ode-solver", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert len(data["variable_summaries"]) == 2
        # After one full period, x should return near 1.0
        assert data["solutions"]["x"][-1] == pytest.approx(1.0, abs=0.1)

    def test_lotka_volterra(self) -> None:
        payload = {
            "derivatives": {"x": "a*x - b*x*y", "y": "-c*y + d*x*y"},
            "parameters": {"a": 1.0, "b": 0.1, "c": 1.5, "d": 0.075},
            "initial_conditions": {"x": 10.0, "y": 5.0},
            "t_start": 0.0,
            "t_end": 30.0,
            "num_points": 300,
        }
        resp = client.post("/api/calc/ode-solver", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        # Both populations should remain positive
        assert all(v > 0 for v in data["solutions"]["x"])
        assert all(v > 0 for v in data["solutions"]["y"])

    def test_missing_initial_condition(self) -> None:
        payload = {
            "derivatives": {"y": "-k*y"},
            "parameters": {"k": 0.1},
            "initial_conditions": {},  # Missing!
            "t_start": 0.0,
            "t_end": 10.0,
            "num_points": 50,
        }
        resp = client.post("/api/calc/ode-solver", json=payload)
        assert resp.status_code == 422

    def test_variable_summaries(self) -> None:
        payload = {
            "derivatives": {"y": "-0.1*y"},
            "parameters": {},
            "initial_conditions": {"y": 100.0},
            "t_start": 0.0,
            "t_end": 10.0,
            "num_points": 50,
        }
        resp = client.post("/api/calc/ode-solver", json=payload)
        data = resp.json()
        summary = data["variable_summaries"][0]
        assert summary["name"] == "y"
        assert summary["initial_value"] == pytest.approx(100.0)
        assert summary["max_value"] == pytest.approx(100.0)
        assert summary["final_value"] < 100.0


# ---------------------------------------------------------------------------
# Endpoints list updated
# ---------------------------------------------------------------------------


class TestEndpointsListUpdated:
    """Verify the new endpoints appear in the list."""

    def test_endpoints_include_new_calculators(self) -> None:
        resp = client.get("/api/calc/endpoints")
        assert resp.status_code == 200
        endpoints = resp.json()["calculators"]
        assert any("syngas-water" in ep for ep in endpoints)
        assert any("thermal-profile" in ep for ep in endpoints)
        assert any("ode-solver" in ep for ep in endpoints)
        assert len(endpoints) >= 11
