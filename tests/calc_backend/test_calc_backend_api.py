"""Comprehensive tests for the shared calculation backend API.

Uses FastAPI TestClient so no running server is needed.
See issue #613.
"""

from __future__ import annotations

import pytest
from calc_backend.app import app
from fastapi.testclient import TestClient

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health / meta
# ---------------------------------------------------------------------------


class TestHealthAndMeta:
    """Health check and metadata endpoints."""

    def test_health(self) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_list_endpoints(self) -> None:
        resp = client.get("/api/calc/endpoints")
        assert resp.status_code == 200
        data = resp.json()
        assert "calculators" in data
        assert len(data["calculators"]) >= 7


# ---------------------------------------------------------------------------
# Flare
# ---------------------------------------------------------------------------


class TestFlareEndpoint:
    """POST /api/calc/flare"""

    PAYLOAD = {
        "total_flow_kg_hr": 5000.0,
        "gas_composition": {"H2": 30, "CO": 40, "CH4": 10, "N2": 20},
        "temperature_k": 400.0,
        "pressure_bar": 1.5,
    }

    def test_flare_success(self) -> None:
        resp = client.post("/api/calc/flare", json=self.PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()

        assert "design" in data
        assert data["design"]["height_m"] > 0
        assert data["design"]["diameter_m"] > 0
        assert data["design"]["heat_release_kw"] > 0

        assert "radiation_zones" in data
        assert data["radiation_zones"]["safe_m"] > 0

        assert 0.95 <= data["combustion_efficiency"] <= 1.0

    def test_flare_validation_error(self) -> None:
        bad = {**self.PAYLOAD, "total_flow_kg_hr": -1}
        resp = client.post("/api/calc/flare", json=bad)
        assert resp.status_code == 422

    def test_flare_radiation_zone_ordering(self) -> None:
        resp = client.post("/api/calc/flare", json=self.PAYLOAD)
        zones = resp.json()["radiation_zones"]
        # Lethal should be closest, comfort farthest
        assert zones["lethal_m"] < zones["damage_m"]
        assert zones["damage_m"] < zones["safe_m"]
        assert zones["safe_m"] < zones["comfort_m"]


# ---------------------------------------------------------------------------
# Baghouse
# ---------------------------------------------------------------------------


class TestBaghouseEndpoint:
    """POST /api/calc/baghouse"""

    PAYLOAD = {
        "gas_flow_kg_s": 5.0,
        "inlet_temp_k": 500.0,
        "pressure_pa": 101325.0,
        "composition": {"N2": 0.7, "CO2": 0.15, "H2O": 0.1, "CO": 0.05},
        "solid_carbon_in_kg_hr": 50.0,
        "ash_in_kg_hr": 20.0,
        "carbon_removal_efficiency": 0.99,
        "ash_removal_efficiency": 0.999,
        "heat_loss_w": 5000.0,
        "drum_volume_m3": 2.0,
        "solid_density_kg_m3": 500.0,
        "bag_area_ft2": 5000.0,
    }

    def test_baghouse_success(self) -> None:
        resp = client.post("/api/calc/baghouse", json=self.PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()

        assert data["carbon_removed_rate_kg_hr"] == pytest.approx(49.5, rel=0.01)
        assert data["ash_removed_rate_kg_hr"] == pytest.approx(19.98, rel=0.01)
        assert data["total_solids_removed_rate_kg_hr"] > 0
        assert data["drum_fill_time_hours"] > 0
        assert data["flow_acfm"] > 0

    def test_baghouse_mass_balance(self) -> None:
        resp = client.post("/api/calc/baghouse", json=self.PAYLOAD)
        data = resp.json()
        total = data["carbon_removed_rate_kg_hr"] + data["ash_removed_rate_kg_hr"]
        assert total == pytest.approx(data["total_solids_removed_rate_kg_hr"], rel=1e-6)


# ---------------------------------------------------------------------------
# Financial
# ---------------------------------------------------------------------------


class TestFinancialEndpoint:
    """POST /api/calc/financial"""

    PAYLOAD = {
        "plant_capacity_tpd": 100.0,
        "operating_days_per_year": 330,
        "capacity_utilization": 0.85,
        "product_price_per_ton": 500.0,
        "feedstock_cost_per_ton": 100.0,
        "total_capital_investment": 10_000_000.0,
        "depreciation_years": 20,
        "tax_rate": 0.25,
    }

    def test_financial_success(self) -> None:
        resp = client.post("/api/calc/financial", json=self.PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()

        r = data["results"]
        assert r["annual_feedstock_tons"] > 0
        assert r["total_revenue"] > 0

    def test_financial_projections(self) -> None:
        payload = {**self.PAYLOAD, "projection_years": 5}
        resp = client.post("/api/calc/financial", json=payload)
        data = resp.json()
        assert len(data["projections"]) == 5
        assert data["projections"][0]["year"] == 1

    def test_financial_no_projections_by_default(self) -> None:
        resp = client.post("/api/calc/financial", json=self.PAYLOAD)
        assert resp.json()["projections"] == []


# ---------------------------------------------------------------------------
# Acid Gas Dewpoint
# ---------------------------------------------------------------------------


class TestAcidGasDewpointEndpoint:
    """POST /api/calc/acid-gas-dewpoint"""

    PAYLOAD = {
        "temperature_c": 150.0,
        "pressure_bar": 30.0,
        "h2o_fraction": 0.15,
        "hf_fraction": 0.001,
        "hcl_fraction": 0.002,
        "h2s_fraction": 0.005,
    }

    def test_dewpoint_success(self) -> None:
        resp = client.post("/api/calc/acid-gas-dewpoint", json=self.PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()

        assert data["overall_dewpoint_c"] is not None
        assert data["limiting_component"] in ["H2O", "HF", "HCl", "H2S"]
        assert "condensation_risk" in data
        assert "components" in data
        assert len(data["components"]) == 4

    def test_dewpoint_zero_composition(self) -> None:
        payload = {
            "temperature_c": 100.0,
            "pressure_bar": 1.0,
            "h2o_fraction": 0.0,
            "hf_fraction": 0.0,
            "hcl_fraction": 0.0,
            "h2s_fraction": 0.0,
        }
        resp = client.post("/api/calc/acid-gas-dewpoint", json=payload)
        assert resp.status_code == 200
        # With zero composition, all dewpoints should be null
        data = resp.json()
        for comp in data["components"].values():
            assert comp["dewpoint_c"] is None

    def test_dewpoint_margin_sign(self) -> None:
        """Margin should be positive when T > dewpoint."""
        resp = client.post("/api/calc/acid-gas-dewpoint", json=self.PAYLOAD)
        data = resp.json()
        # 150 C is well above typical dewpoints for these small fractions
        assert data["dewpoint_margin_c"] is not None
        assert data["dewpoint_margin_c"] > 0


# ---------------------------------------------------------------------------
# Pressure Drop
# ---------------------------------------------------------------------------


class TestPressureDropEndpoint:
    """POST /api/calc/pressure-drop"""

    PAYLOAD = {
        "pipe_diameter_m": 0.1,
        "pipe_length_m": 100.0,
        "roughness_m": 0.000045,
        "flow_rate_kg_s": 1.0,
        "temperature_k": 400.0,
        "pressure_pa": 500000.0,
        "molecular_weight_kg_mol": 0.029,
    }

    def test_pressure_drop_success(self) -> None:
        resp = client.post("/api/calc/pressure-drop", json=self.PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()

        assert data["pressure_drop_pa"] > 0
        assert data["reynolds_number"] > 0
        assert data["velocity_m_s"] > 0
        assert data["flow_regime"] in ["Laminar", "Transitional", "Turbulent"]

    def test_pressure_drop_increases_with_length(self) -> None:
        short = {**self.PAYLOAD, "pipe_length_m": 10.0}
        long = {**self.PAYLOAD, "pipe_length_m": 100.0}

        r_short = client.post("/api/calc/pressure-drop", json=short).json()
        r_long = client.post("/api/calc/pressure-drop", json=long).json()

        assert r_long["pressure_drop_pa"] > r_short["pressure_drop_pa"]


# ---------------------------------------------------------------------------
# Scrubber
# ---------------------------------------------------------------------------


class TestScrubberEndpoint:
    """POST /api/calc/scrubber"""

    PAYLOAD = {
        "gas_flow_kg_hr": 10000.0,
        "gas_temperature_k": 400.0,
        "gas_pressure_pa": 200000.0,
        "gas_molecular_weight": 28.0,
        "liquid_flow_kg_hr": 50000.0,
        "packing_type": "Metal Pall Rings",
        "percent_of_flood": 70.0,
        "acid_gas_removed_kg_hr": {"HCl": 5.0, "SO2": 2.0},
        "caustic_concentration_pct": 10.0,
    }

    def test_scrubber_success(self) -> None:
        resp = client.post("/api/calc/scrubber", json=self.PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()

        assert data["gas_density_kg_m3"] > 0
        assert data["column_diameter_m"] > 0
        assert "caustic_requirement" in data

    def test_scrubber_invalid_packing(self) -> None:
        bad = {**self.PAYLOAD, "packing_type": "NonExistentPacking"}
        resp = client.post("/api/calc/scrubber", json=bad)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# WGS Reactor (requires scipy -- may skip if unavailable)
# ---------------------------------------------------------------------------


class TestWGSReactorEndpoint:
    """POST /api/calc/wgs-reactor"""

    PAYLOAD = {
        "inlet_composition": {"CO": 25.0, "H2": 20.0, "CO2": 10.0, "H2O": 5.0},
        "temperature_k": 673.15,
        "pressure_bar": 25.0,
        "steam_ratio": 2.0,
    }

    @staticmethod
    def _skip_if_unavailable(resp):  # type: ignore[no-untyped-def]
        """Skip test if WGS engine or species database is unavailable."""
        if resp.status_code in (503, 422):
            detail = resp.json().get("detail", "")
            if "not available" in detail or "get_species" in detail:
                pytest.skip(
                    "WGS engine not fully available (missing scipy or species DB)"
                )

    def test_wgs_equilibrium(self) -> None:
        resp = client.post("/api/calc/wgs-reactor", json=self.PAYLOAD)
        self._skip_if_unavailable(resp)
        assert resp.status_code == 200
        data = resp.json()

        eq = data["equilibrium"]
        assert 0 <= eq["conversion_pct"] <= 100
        assert eq["equilibrium_constant"] > 0
        assert eq["h2_co_ratio"] > 0

    def test_wgs_with_sizing(self) -> None:
        payload = {**self.PAYLOAD, "feed_rate_kmol_hr": 100.0}
        resp = client.post("/api/calc/wgs-reactor", json=payload)
        self._skip_if_unavailable(resp)
        assert resp.status_code == 200

        sizing = resp.json()["sizing"]
        assert sizing is not None
        assert sizing["reactor_volume_m3"] > 0
        assert sizing["diameter_m"] > 0

    def test_wgs_no_sizing_when_zero_feed(self) -> None:
        payload = {**self.PAYLOAD, "feed_rate_kmol_hr": 0.0}
        resp = client.post("/api/calc/wgs-reactor", json=payload)
        self._skip_if_unavailable(resp)
        assert resp.status_code == 200
        assert resp.json()["sizing"] is None
