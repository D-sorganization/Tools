"""Tests for the HTML/JSON report generation in reporter.py (#1179).

Uses lightweight mock objects — no DWSIM runtime required.

Design by Contract
------------------
- generate_html_report produces a valid HTML document with expected sections.
- generate_json_report produces valid JSON with expected keys.
- _traffic_light_class returns correct CSS classes based on targets.
- _fmt handles None, NaN, and valid numeric values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Lightweight stubs (no DWSIM dependency)
# ---------------------------------------------------------------------------


@dataclass
class FakeStreamResult:
    """Minimal stub for StreamResult."""

    temperature_C: float = 850.0
    pressure_kPa: float = 101.325
    mass_flow_kg_s: float = 0.5
    specific_enthalpy_kJ_kg: float = -1200.0
    volumetric_flow_Nm3_h: float = 1200.0
    mole_fractions: dict = field(
        default_factory=lambda: {
            "Carbon monoxide": 0.25,
            "Hydrogen": 0.30,
            "Carbon dioxide": 0.15,
            "Methane": 0.05,
            "Water": 0.10,
            "Nitrogen": 0.15,
        }
    )


@dataclass
class FakeEnergyStream:
    """Minimal stub for EnergyStreamResult."""

    energy_flow_kW: float = 50.0


@dataclass
class FakeFlowsheetResults:
    """Minimal stub for FlowsheetResults."""

    converged: bool = True
    streams: dict = field(default_factory=lambda: {"Final_Syngas": FakeStreamResult()})
    energy_streams: dict = field(
        default_factory=lambda: {"E_Gasifier_HeatLoss": FakeEnergyStream(-120.0)}
    )
    errors: list = field(default_factory=list)


@dataclass
class FakeMetrics:
    """Minimal stub for GasificationMetrics."""

    cold_gas_efficiency: float = 0.72
    carbon_conversion_efficiency: float = 0.95
    h2_co_ratio: float = 1.2
    specific_energy_consumption_kWh_t: float = 850.0
    tar_loading_mg_Nm3: float = 45.0
    mass_balance_closure: float = 0.998
    energy_balance_closure: float = 0.993
    syngas_lhv_mj_nm3: float = 5.5
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cold_gas_efficiency": self.cold_gas_efficiency,
            "carbon_conversion_efficiency": self.carbon_conversion_efficiency,
            "h2_co_ratio": self.h2_co_ratio,
            "specific_energy_consumption_kWh_t": self.specific_energy_consumption_kWh_t,
            "tar_loading_mg_Nm3": self.tar_loading_mg_Nm3,
            "mass_balance_closure": self.mass_balance_closure,
            "energy_balance_closure": self.energy_balance_closure,
            "syngas_lhv_mj_nm3": self.syngas_lhv_mj_nm3,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReporterHTMLGeneration:
    """Verify generate_html_report produces valid, structured HTML."""

    def test_html_report_creates_file(self, tmp_path: Path) -> None:
        from dwsim_model.results.reporter import generate_html_report

        out = tmp_path / "report.html"
        result_path = generate_html_report(
            results=FakeFlowsheetResults(),
            metrics=FakeMetrics(),
            output_path=out,
            scenario_name="Test Scenario",
        )
        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_html_contains_doctype(self, tmp_path: Path) -> None:
        from dwsim_model.results.reporter import generate_html_report

        out = tmp_path / "report.html"
        generate_html_report(
            results=FakeFlowsheetResults(),
            metrics=FakeMetrics(),
            output_path=out,
        )
        content = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_html_contains_scenario_name(self, tmp_path: Path) -> None:
        from dwsim_model.results.reporter import generate_html_report

        out = tmp_path / "report.html"
        generate_html_report(
            results=FakeFlowsheetResults(),
            metrics=FakeMetrics(),
            output_path=out,
            scenario_name="Alpha Beta Test",
        )
        content = out.read_text(encoding="utf-8")
        assert "Alpha Beta Test" in content

    def test_html_contains_converged_badge(self, tmp_path: Path) -> None:
        from dwsim_model.results.reporter import generate_html_report

        out = tmp_path / "report.html"
        generate_html_report(
            results=FakeFlowsheetResults(converged=True),
            metrics=FakeMetrics(),
            output_path=out,
        )
        content = out.read_text(encoding="utf-8")
        assert "CONVERGED" in content

    def test_html_contains_not_converged_badge(self, tmp_path: Path) -> None:
        from dwsim_model.results.reporter import generate_html_report

        out = tmp_path / "report.html"
        generate_html_report(
            results=FakeFlowsheetResults(converged=False),
            metrics=FakeMetrics(),
            output_path=out,
        )
        content = out.read_text(encoding="utf-8")
        assert "NOT CONVERGED" in content

    def test_html_contains_kpi_sections(self, tmp_path: Path) -> None:
        from dwsim_model.results.reporter import generate_html_report

        out = tmp_path / "report.html"
        generate_html_report(
            results=FakeFlowsheetResults(),
            metrics=FakeMetrics(),
            output_path=out,
        )
        content = out.read_text(encoding="utf-8")
        assert "Cold Gas Efficiency" in content
        assert "Carbon Conversion" in content
        assert "Syngas LHV" in content

    def test_html_contains_stream_table(self, tmp_path: Path) -> None:
        from dwsim_model.results.reporter import generate_html_report

        out = tmp_path / "report.html"
        generate_html_report(
            results=FakeFlowsheetResults(),
            metrics=FakeMetrics(),
            output_path=out,
        )
        content = out.read_text(encoding="utf-8")
        assert "Final_Syngas" in content
        assert "Stream Summary" in content

    def test_html_contains_chart_js_script(self, tmp_path: Path) -> None:
        from dwsim_model.results.reporter import generate_html_report

        out = tmp_path / "report.html"
        generate_html_report(
            results=FakeFlowsheetResults(),
            metrics=FakeMetrics(),
            output_path=out,
        )
        content = out.read_text(encoding="utf-8")
        assert "Chart.js" in content or "chart.umd" in content


class TestReporterJSONGeneration:
    """Verify generate_json_report produces valid JSON."""

    def test_json_report_creates_file(self, tmp_path: Path) -> None:
        from dwsim_model.results.reporter import generate_json_report

        out = tmp_path / "report.json"
        result_path = generate_json_report(
            results=FakeFlowsheetResults(),
            metrics=FakeMetrics(),
            output_path=out,
            scenario_name="JSON Test",
        )
        assert result_path.exists()

    def test_json_is_valid(self, tmp_path: Path) -> None:
        from dwsim_model.results.reporter import generate_json_report

        out = tmp_path / "report.json"
        generate_json_report(
            results=FakeFlowsheetResults(),
            metrics=FakeMetrics(),
            output_path=out,
        )
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_json_contains_expected_keys(self, tmp_path: Path) -> None:
        from dwsim_model.results.reporter import generate_json_report

        out = tmp_path / "report.json"
        generate_json_report(
            results=FakeFlowsheetResults(),
            metrics=FakeMetrics(),
            output_path=out,
            scenario_name="Key Test",
        )
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["scenario"] == "Key Test"
        assert "generated_at" in data
        assert data["converged"] is True
        assert "metrics" in data
        assert "streams" in data

    def test_json_streams_contain_final_syngas(self, tmp_path: Path) -> None:
        from dwsim_model.results.reporter import generate_json_report

        out = tmp_path / "report.json"
        generate_json_report(
            results=FakeFlowsheetResults(),
            metrics=FakeMetrics(),
            output_path=out,
        )
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "Final_Syngas" in data["streams"]
        syngas = data["streams"]["Final_Syngas"]
        assert syngas["temperature_C"] == pytest.approx(850.0)
        assert syngas["mole_fractions"]["Hydrogen"] == pytest.approx(0.30)

    def test_json_metrics_present(self, tmp_path: Path) -> None:
        from dwsim_model.results.reporter import generate_json_report

        out = tmp_path / "report.json"
        generate_json_report(
            results=FakeFlowsheetResults(),
            metrics=FakeMetrics(),
            output_path=out,
        )
        data = json.loads(out.read_text(encoding="utf-8"))
        m = data["metrics"]
        assert m["cold_gas_efficiency"] == pytest.approx(0.72)
        assert m["h2_co_ratio"] == pytest.approx(1.2)


class TestReporterHelpers:
    """Unit tests for internal helper functions."""

    def test_fmt_none_returns_dash(self) -> None:
        from dwsim_model.results.reporter import _fmt

        assert _fmt(None) == "—"

    def test_fmt_numeric(self) -> None:
        from dwsim_model.results.reporter import _fmt

        result = _fmt(3.14159, ".2f", " m")
        assert result == "3.14 m"

    def test_traffic_light_good(self) -> None:
        from dwsim_model.results.reporter import _traffic_light_class

        result = _traffic_light_class(
            "cold_gas_efficiency", 0.73, {"cold_gas_efficiency": 0.70}
        )
        assert result == "good"

    def test_traffic_light_bad(self) -> None:
        from dwsim_model.results.reporter import _traffic_light_class

        result = _traffic_light_class(
            "cold_gas_efficiency", 0.50, {"cold_gas_efficiency": 0.70}
        )
        assert result == "bad"

    def test_traffic_light_lower_better_good(self) -> None:
        from dwsim_model.results.reporter import _traffic_light_class

        result = _traffic_light_class(
            "tar_loading_mg_Nm3", 40.0, {"tar_loading_mg_Nm3": 50.0}
        )
        assert result == "good"

    def test_traffic_light_no_targets(self) -> None:
        from dwsim_model.results.reporter import _traffic_light_class

        result = _traffic_light_class("cold_gas_efficiency", 0.70, {})
        assert result == ""

    def test_stream_to_dict(self) -> None:
        from dwsim_model.results.reporter import _stream_to_dict

        stream = FakeStreamResult()
        d = _stream_to_dict(stream)
        assert d["temperature_C"] == pytest.approx(850.0)
        assert d["mole_fractions"]["Hydrogen"] == pytest.approx(0.30)
