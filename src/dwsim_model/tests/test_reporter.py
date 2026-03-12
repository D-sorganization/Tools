"""
test_reporter.py
================
Tests for results/reporter.py — HTML and JSON report generation.
Uses mock FlowsheetResults and GasificationMetrics objects so no DWSIM
runtime is required.
"""

from __future__ import annotations

import json

import pytest
from dwsim_model.results.extractor import (
    EnergyStreamResult,
    FlowsheetResults,
    StreamResult,
)
from dwsim_model.results.metrics import GasificationMetrics
from dwsim_model.results.reporter import generate_html_report, generate_json_report

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_results() -> FlowsheetResults:
    results = FlowsheetResults()
    results.converged = True
    results.streams = {
        "Syngas_Pre_PEM": StreamResult(
            name="Syngas_Pre_PEM",
            temperature_C=800.0,
            pressure_kPa=101.325,
            mass_flow_kg_s=0.5,
            mole_fractions={
                "Carbon monoxide": 0.35,
                "Hydrogen": 0.30,
                "Carbon dioxide": 0.15,
                "Methane": 0.05,
                "Nitrogen": 0.15,
            },
            volumetric_flow_Nm3_h=2500.0,
        )
    }
    results.energy_streams = {
        "E_Gasifier_HeatLoss": EnergyStreamResult(
            name="E_Gasifier_HeatLoss", energy_flow_kW=-50.0
        ),
        "E_PEM_AC_Power": EnergyStreamResult(
            name="E_PEM_AC_Power", energy_flow_kW=120.0
        ),
    }
    results.errors = []
    return results


@pytest.fixture()
def mock_metrics() -> GasificationMetrics:
    m = GasificationMetrics()
    m.cold_gas_efficiency = 0.72
    m.carbon_conversion_efficiency = 0.95
    m.h2_co_ratio = 0.86
    m.syngas_lhv_mj_nm3 = 5.4
    m.syngas_lhv_mj_kg = 10.2
    m.syngas_mass_flow_kg_s = 0.5
    m.syngas_volumetric_flow_Nm3_h = 2500.0
    m.syngas_temperature_C = 800.0
    m.feed_mass_flow_kg_s = 0.15
    m.mass_balance_closure = 0.999
    m.energy_balance_closure = 0.987
    return m


# ---------------------------------------------------------------------------
# HTML report tests
# ---------------------------------------------------------------------------


class TestGenerateHtmlReport:
    def test_creates_file(self, tmp_path, mock_results, mock_metrics):
        out = tmp_path / "report.html"
        result_path = generate_html_report(mock_results, mock_metrics, out)
        assert result_path == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_html_contains_scenario_name(self, tmp_path, mock_results, mock_metrics):
        out = tmp_path / "report.html"
        generate_html_report(
            mock_results, mock_metrics, out, scenario_name="TestScenario"
        )
        content = out.read_text(encoding="utf-8")
        assert "TestScenario" in content

    def test_html_contains_doctype(self, tmp_path, mock_results, mock_metrics):
        out = tmp_path / "report.html"
        generate_html_report(mock_results, mock_metrics, out)
        content = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content or "<html" in content

    def test_html_contains_kpi_values(self, tmp_path, mock_results, mock_metrics):
        out = tmp_path / "report.html"
        generate_html_report(mock_results, mock_metrics, out)
        content = out.read_text(encoding="utf-8")
        # Cold gas efficiency of 0.72 = 72%
        assert "72" in content

    def test_creates_parent_directory(self, tmp_path, mock_results, mock_metrics):
        out = tmp_path / "subdir" / "deep" / "report.html"
        generate_html_report(mock_results, mock_metrics, out)
        assert out.exists()

    def test_energy_table_uses_energy_flow_kw(
        self, tmp_path, mock_results, mock_metrics
    ):
        out = tmp_path / "report.html"
        generate_html_report(mock_results, mock_metrics, out)
        content = out.read_text(encoding="utf-8")
        # -50 kW heat loss and 120 kW AC power should appear
        assert "-50" in content or "50.0" in content
        assert "120" in content


# ---------------------------------------------------------------------------
# JSON report tests
# ---------------------------------------------------------------------------


class TestGenerateJsonReport:
    def test_creates_valid_json(self, tmp_path, mock_results, mock_metrics):
        out = tmp_path / "report.json"
        generate_json_report(mock_results, mock_metrics, out)
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_json_contains_scenario(self, tmp_path, mock_results, mock_metrics):
        out = tmp_path / "report.json"
        generate_json_report(mock_results, mock_metrics, out, scenario_name="Air_Blown")
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["scenario"] == "Air_Blown"

    def test_json_has_metrics(self, tmp_path, mock_results, mock_metrics):
        out = tmp_path / "report.json"
        generate_json_report(mock_results, mock_metrics, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "metrics" in data
        assert "cold_gas_efficiency" in data["metrics"]

    def test_json_has_streams(self, tmp_path, mock_results, mock_metrics):
        out = tmp_path / "report.json"
        generate_json_report(mock_results, mock_metrics, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "streams" in data
        assert "Syngas_Pre_PEM" in data["streams"]

    def test_json_converged_flag(self, tmp_path, mock_results, mock_metrics):
        out = tmp_path / "report.json"
        generate_json_report(mock_results, mock_metrics, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["converged"] is True
